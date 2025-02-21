import asyncio
import json
import numpy as np
from typing import Dict
from teacher_evaluator import TeacherEvaluator
from models.learner_model.learning_model import LearnerModel  # Second AI implementation


class TwoAIAssessmentSystem:
    """
    System for assessing teaching using the Two-AI model approach:
    1. First AI (TeacherEvaluator) evaluates the teaching
    2. First AI transfers knowledge to Second AI (Learner)
    3. System calculates understanding ratio
    """

    def __init__(self, config: Dict = None):
        """Initialize the two-AI assessment system"""
        self.config = config or {}
        self.teacher_evaluator = TeacherEvaluator(self.config.get('teacher_config', {}))
        self.learner_model = LearnerModel(self.config.get('learner_config', {}))

    async def assess_teaching(self, session_data: Dict) -> Dict:
        """
        Perform the complete assessment process

        Args:
            session_data: Dictionary with video_frames, audio_data, transcript, and domain

        Returns:
            Complete assessment results
        """
        # Step 1: Teacher Evaluator analyzes the teaching
        teacher_evaluation = await self.teacher_evaluator.evaluate_teaching_session(session_data)

        # Step 2: Extract structured knowledge to transfer to Learner
        knowledge_transfer = self._prepare_knowledge_transfer(
            teacher_evaluation,
            session_data['transcript']
        )

        # Step 3: Learner Model learns from the transferred knowledge
        learner_understanding = self.learner_model.learn_concept(knowledge_transfer)

        # Step 4: Calculate the understanding ratio
        understanding_metrics = self._calculate_understanding_metrics(
            teacher_evaluation,
            learner_understanding,
            session_data['transcript']
        )

        # Step 5: Finalize assessment results
        final_results = self._combine_assessment_results(
            teacher_evaluation,
            learner_understanding,
            understanding_metrics
        )

        return final_results

    def _prepare_knowledge_transfer(self, evaluation: Dict, original_transcript: str) -> Dict:
        """
        Prepare knowledge for transfer from TeacherEvaluator to LearnerModel

        This simulates the first AI teaching the second AI
        """
        # Extract key concepts and their explanations
        concepts = evaluation['content_metrics']['coverage']
        content_components = evaluation['content_metrics']['components']

        # Structure the knowledge for transfer
        structured_knowledge = {
            'concepts': concepts,
            'explanations': {},
            'examples': [],
            'relationships': [],
            'accuracy_score': evaluation['accuracy'],
            'clarity_score': evaluation['audio_metrics']['clarity'],
            'structure_score': evaluation['content_metrics']['structure_score']
        }

        # Extract explanations and examples from content components
        for component in content_components:
            component_type = component['type']
            text = component['text']

            if component_type == 'CONCEPT':
                concept_name = text.split('-')[0].strip() if '-' in text else text
                explanation = text.split('-')[1].strip() if '-' in text else text
                structured_knowledge['explanations'][concept_name] = explanation

            elif component_type == 'EXAMPLE':
                structured_knowledge['examples'].append(text)

            elif component_type == 'RELATIONSHIP':
                structured_knowledge['relationships'].append(text)

        return structured_knowledge

    def _calculate_understanding_metrics(
            self,
            teacher_evaluation: Dict,
            learner_understanding: Dict,
            original_transcript: str
    ) -> Dict:
        """
        Calculate understanding metrics based on what was taught vs what was understood

        Understanding Ratio = (Knowledge understood by Learner / Input given by teacher) * 100
        """
        # Get metrics from both AIs
        teacher_concepts = set(teacher_evaluation['content_metrics']['coverage'])
        learner_concepts = set(learner_understanding['learned_concepts'])

        # Calculate concept overlap
        common_concepts = teacher_concepts.intersection(learner_concepts)
        concept_transfer_rate = len(common_concepts) / len(teacher_concepts) if teacher_concepts else 0

        # Calculate understanding depth
        understanding_depth = learner_understanding['understanding_depth']

        # Calculate meaning preservation
        meaning_preservation = learner_understanding['meaning_preservation']

        # Calculate understanding ratio
        understanding_ratio = (
                concept_transfer_rate * 0.4 +
                understanding_depth * 0.4 +
                meaning_preservation * 0.2
        )

        return {
            'understanding_ratio': understanding_ratio,
            'concept_transfer_rate': concept_transfer_rate,
            'understanding_depth': understanding_depth,
            'meaning_preservation': meaning_preservation,
            'concepts_understood': list(common_concepts),
            'concepts_missed': list(teacher_concepts - learner_concepts)
        }

    def _combine_assessment_results(
            self,
            teacher_evaluation: Dict,
            learner_understanding: Dict,
            understanding_metrics: Dict
    ) -> Dict:
        """Combine all metrics into final assessment results"""
        # Adjust the understanding ratio based on the learner's understanding
        adjusted_ratio = understanding_metrics['understanding_ratio']

        # Update the teacher evaluation with the new understanding metrics
        final_results = {**teacher_evaluation}
        final_results['understanding_ratio'] = adjusted_ratio
        final_results['understanding_metrics'] = understanding_metrics
        final_results['learner_model_metrics'] = {
            'learned_concepts': learner_understanding['learned_concepts'],
            'understanding_depth': learner_understanding['understanding_depth'],
            'application_ability': learner_understanding['application_ability']
        }

        # Recalculate overall score with the adjusted understanding ratio
        final_results['overall_score'] = self._calculate_final_score(
            teacher_evaluation,
            adjusted_ratio
        )

        # Update feedback based on the new understanding metrics
        final_results['feedback'] = self._generate_updated_feedback(
            teacher_evaluation['feedback'],
            understanding_metrics
        )

        return final_results

    def _calculate_final_score(self, teacher_evaluation: Dict, adjusted_ratio: float) -> float:
        """Calculate final score with adjusted understanding ratio"""
        weights = self.config.get('scoring_weights', {
            'teacher_evaluation': 0.6,
            'understanding_ratio': 0.4
        })

        teacher_score = teacher_evaluation['overall_score']

        return (
                teacher_score * weights['teacher_evaluation'] +
                adjusted_ratio * weights['understanding_ratio']
        )

    def _generate_updated_feedback(self, original_feedback: Dict, understanding_metrics: Dict) -> Dict:
        """Generate updated feedback based on understanding metrics"""
        updated_feedback = {**original_feedback}

        # Add understanding-specific feedback
        if understanding_metrics['understanding_ratio'] < 0.5:
            updated_feedback['areas_for_improvement'].append(
                "Knowledge transfer effectiveness is poor"
            )
            updated_feedback['recommendations'].append(
                "Focus on clearer explanations of core concepts"
            )

            # Update summary
            updated_feedback['summary'] = updated_feedback['summary'].replace(
                "Understanding ratio",
                f"Knowledge transfer is ineffective. Understanding ratio"
            )

        # Highlight missed concepts
        if understanding_metrics['concepts_missed']:
            missed_concepts = understanding_metrics['concepts_missed']
            if len(missed_concepts) > 2:
                concept_text = f"{missed_concepts[0]}, {missed_concepts[1]} and {len(missed_concepts) - 2} more"
            else:
                concept_text = ", ".join(missed_concepts)

            updated_feedback['areas_for_improvement'].append(
                f"Failed to effectively communicate: {concept_text}"
            )

        return updated_feedback


async def main():
    """Demo of two-AI model assessment system"""
    # Configuration
    config = {
        'teacher_config': {
            'knowledge_base': {'domain_path': 'data/domains'},
            'scoring_weights': {
                'visual': 0.25,
                'audio': 0.25,
                'content': 0.4,
                'authenticity': 0.1
            }
        },
        'learner_config': {
            'initial_knowledge': {},
            'learning_rate': 0.8
        },
        'scoring_weights': {
            'teacher_evaluation': 0.6,
            'understanding_ratio': 0.4
        }
    }

    # Sample data (in a real implementation, this would come from video processing)
    sample_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]
    sample_audio = np.random.randn(16000 * 5)  # 5 seconds of audio

    sample_transcript = """
    Neural networks consist of layers of interconnected nodes, similar to neurons in the brain.
    Each connection has a weight that adjusts as the network learns. The learning process has 
    two main phases: forward propagation and backpropagation. In forward propagation, input data 
    passes through the network, producing an output. During backpropagation, the network compares 
    this output to the expected result and adjusts connection weights to reduce errors.

    The power of neural networks lies in their ability to learn complex patterns. For example, 
    a neural network can learn to identify cats in images by recognizing combinations of features 
    like whiskers, pointed ears, and certain body shapes.
    """

    session_data = {
        'video_frames': sample_frames,
        'audio_data': sample_audio,
        'transcript': sample_transcript,
        'domain': 'data-science'
    }

    # Initialize and run assessment
    assessment_system = TwoAIAssessmentSystem(config)
    results = await assessment_system.assess_teaching(session_data)

    # Print summary results
    print("\n===== TWO-AI ASSESSMENT RESULTS =====")
    print(f"Overall Score: {results['overall_score']:.2f}")
    print(f"Understanding Ratio: {results['understanding_ratio']:.2f}")

    print("\nUnderstanding Metrics:")
    print(f"  Concept Transfer Rate: {results['understanding_metrics']['concept_transfer_rate']:.2f}")
    print(f"  Understanding Depth: {results['understanding_metrics']['understanding_depth']:.2f}")
    print(f"  Meaning Preservation: {results['understanding_metrics']['meaning_preservation']:.2f}")

    print("\nConcepts Understood:")
    for concept in results['understanding_metrics']['concepts_understood']:
        print(f"  - {concept}")

    if results['understanding_metrics']['concepts_missed']:
        print("\nConcepts Missed:")
        for concept in results['understanding_metrics']['concepts_missed']:
            print(f"  - {concept}")

    print("\nFeedback Summary:")
    print(results['feedback']['summary'])

    # Save detailed results
    with open('two_ai_assessment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to two_ai_assessment_results.json")


if __name__ == "__main__":
    asyncio.run(main())