# eqly_assessment/evaluate.py
import torch
import json
import argparse
from pathlib import Path
from models.teacher_model.model import TeacherEvaluator
from models.learner_model.model import LearnerModel
from data.preprocessor import VideoPreprocessor
from video_utils.metrics import calculate_metrics, log_metrics


def evaluate_video(
        video_path: str,
        teacher_model: TeacherEvaluator,
        learner_model: LearnerModel,
        config: dict
) -> dict:
    preprocessor = VideoPreprocessor(
        frame_size=tuple(config['data']['frame_size']),
        sample_rate=config['data']['sample_rate']
    )

    features = preprocessor.process_video(video_path)
    features = features.unsqueeze(0).to(config['device'])

    teacher_model.eval()
    learner_model.eval()

    with torch.no_grad():
        teacher_outputs = teacher_model(features)
        learner_outputs = learner_model(teacher_outputs['features'])

        understanding_ratio = (
                                      sum(learner_outputs['scores'].values()) / len(learner_outputs['scores'])
                              ) / (
                                      sum(teacher_outputs['scores'].values()) / len(teacher_outputs['scores'])
                              )

    return {
        'teacher_evaluation': teacher_outputs['scores'],
        'learner_understanding': learner_outputs['scores'],
        'understanding_ratio': understanding_ratio
    }


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    teacher_model = TeacherEvaluator(config['model'])
    teacher_model.load_checkpoint(args.teacher_checkpoint)
    teacher_model.to(config['device'])

    learner_model = LearnerModel(config['model'])
    learner_model.load_checkpoint(args.learner_checkpoint)
    learner_model.to(config['device'])

    # Evaluate video
    results = evaluate_video(args.video, teacher_model, learner_model, config)

    # Save results
    output_path = Path(args.output) if args.output else Path(args.video).with_suffix('.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--teacher-checkpoint', type=str, required=True)
    parser.add_argument('--learner-checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    main(args)