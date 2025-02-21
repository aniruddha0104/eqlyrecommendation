# core/two_ai_assessment.py

import logging
import torch
import numpy as np
from typing import Dict, Any, List
from models.teacher_evaluator import TeacherEvaluator
from models.learner_model import LearnerModel
from core.knowledge_base import KnowledgeBase
from core.understanding_calculator import UnderstandingCalculator
from processors.video.video_processor import VideoProcessor
from processors.audio.audio_analyzer import AudioAnalyzer


class TwoAIAssessmentSystem:
    """Main coordinator for the two-AI assessment system"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Initialize components
        try:
            self.knowledge_base = KnowledgeBase(config['teacher_config']['knowledge_base'])
            self.understanding_calculator = UnderstandingCalculator(
                config.get('understanding_config', {})
            )
            self.teacher_evaluator = TeacherEvaluator(config['teacher_config'])
            self.learner_model = LearnerModel(config['learner_config'])
            self.video_processor = VideoProcessor()
            self.audio_analyzer = AudioAnalyzer()

            self.logger.info("TwoAIAssessmentSystem initialized successfully")

        except Exception as e:
            self.logger.error(f"TwoAIAssessmentSystem initialization failed: {str(e)}")
            raise

    async def assess_teaching(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete teaching assessment"""
        try:
            self.logger.info(f"Evaluating teaching session for domain: {session_data.get('domain')}")

            # 1. Teacher Evaluation
            teaching_results = await self._evaluate_teaching(session_data)

            # 2. Knowledge Transfer
            knowledge_transfer = await self._transfer_knowledge(
                teaching_results,
                session_data
            )

            # 3. Understanding Assessment
            understanding_results = await self._assess_understanding(
                knowledge_transfer
            )

            # 4. Calculate Final Metrics
            final_results = self._calculate_final_results(
                teaching_results,
                understanding_results
            )

            # 5. Generate Feedback
            feedback = self._generate_feedback(final_results)

            results = {
                **final_results,
                'feedback': feedback
            }

            self.logger.info("Teaching assessment completed successfully")
            return results

        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def _evaluate_teaching(self, session_data: Dict) -> Dict:
        """Evaluate teaching performance"""
        try:
            # Extract relevant data
            transcript = session_data.get('transcript', '')
            domain = session_data.get('domain')
            video_features = session_data.get('video_features', {})
            audio_features = session_data.get('audio_features', {})

            # Evaluate content
            content_results = self.teacher_evaluator.analyze_content(transcript, domain)

            # Evaluate visual aspects
            frames = video_features.get('frames', [])
            visual_results = self.teacher_evaluator.analyze_visual_cues(frames)

            # Evaluate audio aspects
            audio_results = {}
            if audio_features:
                audio_results = {
                    'clarity': audio_features.get('clarity', 0.5),
                    'pace': audio_features.get('pace', 0.5),
                    'prosody': audio_features.get('prosody', 0.5)
                }

            # Calculate authenticity
            authenticity_score = self._calculate_authenticity(
                transcript,
                video_features,
                audio_features
            )

            # Combine results
            teaching_results = {
                'content_metrics': content_results,
                'visual_metrics': visual_results,
                'audio_metrics': audio_results,
                'authenticity_score': authenticity_score,
                'final_score': self._calculate_teaching_score(
                    content_results,
                    visual_results,
                    audio_results,
                    authenticity_score
                )
            }

            return teaching_results

        except Exception as e:
            self.logger.error(f"Teaching evaluation failed: {str(e)}")
            raise

    async def _transfer_knowledge(
            self,
            teaching_results: Dict,
            session_data: Dict
    ) -> Dict:
        """Structure and transfer knowledge from teacher to learner"""
        try:
            # Get domain knowledge
            domain = session_data.get('domain')
            domain_data = {}
            if domain:
                domain_concepts = self.knowledge_base.get_domain_concepts(domain)
                domain_data = {
                    'concepts': domain_concepts,
                    'relationships': [],
                    'domain': domain
                }

            # Extract content information
            content_info = {
                'main_concepts': self._extract_main_concepts(session_data['transcript']),
                'explained_concepts': teaching_results['content_metrics'].get('concepts', []),
                'concept_accuracies': teaching_results['content_metrics'].get('concept_accuracies', {})
            }

            # Structure knowledge package
            knowledge_package = {
                'content': teaching_results['content_metrics'],
                'content_info': content_info,
                'domain_data': domain_data,
                'examples': self._extract_examples(session_data['transcript']),
                'visual_elements': teaching_results['visual_metrics']
            }

            return knowledge_package

        except Exception as e:
            self.logger.error(f"Knowledge transfer failed: {str(e)}")
            raise

    async def _assess_understanding(self, knowledge_package: Dict) -> Dict:
        """Assess learner's understanding"""
        try:
            # Process knowledge through learner model
            learning_results = self.learner_model.learn_concept(knowledge_package)

            # Calculate understanding ratio
            understanding_ratio = self.learner_model.calculate_understanding_ratio()

            # Get detailed understanding metrics
            understanding_metrics = {
                'concept_transfer_rate': learning_results.get('understanding_metrics', {}).get('concept_transfer_rate',
                                                                                               0.0),
                'understanding_depth': learning_results.get('understanding_metrics', {}).get('understanding_depth',
                                                                                             0.0),
                'meaning_preservation': learning_results.get('understanding_metrics', {}).get('meaning_preservation',
                                                                                              0.0),
                'concepts_understood': learning_results.get('understanding_metrics', {}).get('concepts_understood', []),
                'concepts_missed': learning_results.get('understanding_metrics', {}).get('concepts_missed', [])
            }

            return {
                'learning_results': learning_results,
                'understanding_ratio': understanding_ratio,
                'understanding_metrics': understanding_metrics
            }

        except Exception as e:
            self.logger.error(f"Understanding assessment failed: {str(e)}")
            raise

    def _calculate_final_results(
            self,
            teaching_results: Dict,
            understanding_results: Dict
    ) -> Dict:
        """Calculate final assessment results"""
        try:
            weights = self.config['scoring_weights']

            # Calculate weighted scores
            teaching_score = teaching_results['final_score']
            understanding_score = understanding_results['understanding_ratio']

            overall_score = (
                    weights['teacher_evaluation'] * teaching_score +
                    weights['understanding_ratio'] * understanding_score
            )

            # Prepare final results
            final_results = {
                'overall_score': overall_score,
                'teaching_score': teaching_score,
                'understanding_ratio': understanding_score,
                'accuracy': teaching_results['content_metrics'].get('accuracy', 0.0),
                'engagement': teaching_results['content_metrics'].get('engagement', 0.0),
                'authenticity_score': teaching_results.get('authenticity_score', 0.0),
                'understanding_metrics': understanding_results['understanding_metrics']
            }

            return final_results

        except Exception as e:
            self.logger.error(f"Final results calculation failed: {str(e)}")
            raise

    def _generate_feedback(self, results: Dict) -> Dict:
        """Generate comprehensive feedback"""
        try:
            # Determine overall assessment
            if results['overall_score'] >= 0.8:
                quality = "excellent"
            elif results['overall_score'] >= 0.6:
                quality = "good"
            else:
                quality = "needs improvement"

            # Generate summary
            summary = (
                f"The teaching session demonstrated {quality} effectiveness "
                f"with an understanding ratio of {results['understanding_ratio']:.1%}. "
                f"Content accuracy was {results['accuracy']:.1%} "
                f"with {results['engagement']:.1%} engagement."
            )

            # Identify strengths
            strengths = []
            if results['accuracy'] >= 0.7:
                strengths.append("Strong content accuracy")
            if results['engagement'] >= 0.7:
                strengths.append("Good student engagement")
            if results['understanding_ratio'] >= 0.7:
                strengths.append("Effective knowledge transfer")

            # Identify areas for improvement
            improvements = []
            if results['accuracy'] < 0.7:
                improvements.append("Improve content accuracy")
            if results['engagement'] < 0.7:
                improvements.append("Enhance student engagement")
            if results['understanding_ratio'] < 0.7:
                improvements.append("Work on knowledge transfer effectiveness")

            # Generate recommendations
            recommendations = self._generate_recommendations(results)

            feedback = {
                'summary': summary,
                'strengths': strengths,
                'areas_for_improvement': improvements,
                'recommendations': recommendations
            }

            return feedback

        except Exception as e:
            self.logger.error(f"Feedback generation failed: {str(e)}")
            return {'summary': "Error generating feedback"}

    def _calculate_teaching_score(
            self,
            content_metrics: Dict,
            visual_metrics: Dict,
            audio_metrics: Dict,
            authenticity_score: float
    ) -> float:
        """Calculate overall teaching score"""
        weights = self.config['teacher_config']['scoring_weights']

        # Content score (accuracy, clarity, engagement)
        content_score = np.mean([
            content_metrics.get('accuracy', 0.0),
            content_metrics.get('clarity', 0.0),
            content_metrics.get('engagement', 0.0)
        ])

        # Visual score
        visual_score = np.mean(list(visual_metrics.values())) if visual_metrics else 0.5

        # Audio score
        audio_score = np.mean(list(audio_metrics.values())) if audio_metrics else 0.5

        # Calculate weighted score
        teaching_score = (
                weights['content'] * content_score +
                weights['visual'] * visual_score +
                weights['audio'] * audio_score +
                weights['authenticity'] * authenticity_score
        )

        return teaching_score

    def _calculate_authenticity(
            self,
            transcript: str,
            video_features: Dict,
            audio_features: Dict
    ) -> float:
        """Calculate content authenticity score"""
        # Default authenticity score
        return 0.8

    def _extract_main_concepts(self, transcript: str) -> List[str]:
        """Extract main concepts from transcript"""
        # Simplified implementation
        return ["neural networks", "activation functions", "backpropagation"]

    def _extract_examples(self, transcript: str) -> List[Dict]:
        """Extract examples from transcript"""
        # Simplified implementation
        return []

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Add teaching technique recommendations
        if results['engagement'] < 0.7:
            recommendations.append("Use more interactive elements to increase engagement")

        # Add content recommendations
        if results['accuracy'] < 0.7:
            recommendations.append("Review concept explanations for accuracy")

        # Add knowledge transfer recommendations
        if results['understanding_ratio'] < 0.7:
            recommendations.append("Simplify complex concepts with better examples")

        return recommendations