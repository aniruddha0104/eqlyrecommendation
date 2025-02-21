import torch
import argparse
import json
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from models.teacher_model.model import TeacherEvaluator
from models.teacher_model.trainer import TeacherTrainer
from models.learner_model.model import LearnerModel
from models.learner_model.trainer import LearnerTrainer
from data.loader import create_data_loaders
from video_utils.metrics import calculate_metrics, log_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(
        teacher_trainer: TeacherTrainer,
        learner_trainer: LearnerTrainer,
        train_loader: DataLoader,
        epoch: int,
        config: dict
):
    total_teacher_loss = 0
    total_learner_loss = 0
    steps = 0

    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(config['device'])
        labels = labels.to(config['device'])

        # Train teacher model
        teacher_outputs = teacher_trainer.train_step({'videos': videos, 'labels': labels})
        total_teacher_loss += teacher_outputs['loss']

        # Train learner model using teacher outputs
        learner_outputs = learner_trainer.train_step({'videos': videos, 'labels': labels}, teacher_outputs['outputs'])
        total_learner_loss += learner_outputs['loss']

        steps += 1

        if batch_idx % config['log_interval'] == 0:
            log_metrics({
                'teacher_loss': teacher_outputs['loss'],
                'learner_loss': learner_outputs['loss']
            }, batch_idx, f'Epoch {epoch}')

    return {
        'teacher_loss': total_teacher_loss / steps,
        'learner_loss': total_learner_loss / steps
    }


def validate(
        teacher_model: TeacherEvaluator,
        learner_model: LearnerModel,
        val_loader: DataLoader,
        config: dict
):
    teacher_model.eval()
    learner_model.eval()
    total_metrics = {'teacher': [], 'learner': []}

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(config['device'])
            labels = labels.to(config['device'])

            teacher_outputs = teacher_model(videos)
            learner_outputs = learner_model(teacher_outputs['features'])

            teacher_metrics = calculate_metrics(teacher_outputs['scores'], labels)
            learner_metrics = calculate_metrics(learner_outputs['scores'], labels)

            total_metrics['teacher'].append(teacher_metrics)
            total_metrics['learner'].append(learner_metrics)

    return {
        'teacher': {k: sum(d[k] for d in total_metrics['teacher']) / len(total_metrics['teacher'])
                    for k in total_metrics['teacher'][0]},
        'learner': {k: sum(d[k] for d in total_metrics['learner']) / len(total_metrics['learner'])
                    for k in total_metrics['learner'][0]}
    }


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    teacher_model = TeacherEvaluator(config).to(config['device'])
    learner_model = LearnerModel(config).to(config['device'])

    # Initialize trainers
    teacher_trainer = TeacherTrainer(teacher_model, config)
    learner_trainer = LearnerTrainer(learner_model, config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config['train_data'], config['val_data'], config)

    # Training loop
    best_metric = float('inf')
    for epoch in range(config['epochs']):
        train_metrics = train_epoch(teacher_trainer, learner_trainer, train_loader, epoch, config)
        val_metrics = validate(teacher_model, learner_model, val_loader, config)

        log_metrics({**train_metrics, **val_metrics}, epoch, 'Validation')

        # Save checkpoints
        if val_metrics['teacher']['loss'] < best_metric:
            best_metric = val_metrics['teacher']['loss']
            teacher_model.save_checkpoint(save_dir / 'best_teacher.pth')
            learner_model.save_checkpoint(save_dir / 'best_learner.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
