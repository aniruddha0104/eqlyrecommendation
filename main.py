# core/two_ai_assessment.py

import logging
from typing import Dict, Any
from models.teacher_evaluator import TeacherEvaluator
from models.learner_model import LearnerModel


class TwoAIAssessmentSystem:
    """Main coordinator for the two-AI assessment system"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Initialize models
        try:
            self.teacher_evaluator = TeacherEvaluator(config['teacher_config'])
            self.learner_model = LearnerModel(config['learner_config'])
            self.logger.info("Assessment system initialized successfully")
        except Exception as e:
            self.logger.error(f"Assessment system initialization failed: {str(e)}")
            raise

    async def assess_teaching(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete teaching assessment"""
        try:
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
            self.logger.error(f"Teaching assessment failed: {str(e)}")
            raise

    async def _evaluate_teaching(self, session_data: Dict) -> Dict:
        """Evaluate teaching performance"""
        # Run teacher evaluation
        teaching_results = self.teacher_evaluator.evaluate_teaching(session_data)

        # Validate results
        if not teaching_results:
            raise ValueError("Teacher evaluation returned empty results")

        return teaching_results

    async def _transfer_knowledge(
            self,
            teaching_results: Dict,
            session_data: Dict
    ) -> Dict:
        """Structure and transfer knowledge from teacher to learner"""
        # Prepare knowledge transfer package
        knowledge_package = {
            'content': teaching_results['content_metrics'],
            'examples': self._extract_examples(session_data['transcript']),
            'visual_aids': teaching_results['visual_metrics'],
            'domain': session_data.get('domain')
        }

        return knowledge_package

    async def _assess_understanding(self, knowledge_transfer: Dict) -> Dict:
        """Assess learner's understanding"""
        # Run learning process
        learning_results = self.learner_model.learn_concept(knowledge_transfer)

        # Calculate understanding ratio
        understanding_ratio = self.learner_model.calculate_understanding_ratio()

        return {
            'learning_results': learning_results,
            'understanding_ratio': understanding_ratio
        }

    def _calculate_final_results(
            self,
            teaching_results: Dict,
            understanding_results: Dict
    ) -> Dict:
        """Calculate final assessment results"""
        weights = self.config['scoring_weights']

        # Calculate weighted scores
        teaching_score = teaching_results['final_score']
        understanding_score = understanding_results['understanding_ratio']

        overall_score = (
                weights['teacher_evaluation'] * teaching_score +
                weights['understanding_ratio'] * understanding_score
        )

        return {
            'overall_score': overall_score,
            'teaching_score': teaching_score,
            'understanding_ratio': understanding_score,
            'accuracy': teaching_results['content_metrics']['accuracy'],
            'engagement': teaching_results['content_metrics']['engagement'],
            'authenticity_score': teaching_results.get('authenticity_score', 0.0),
            'understanding_metrics': understanding_results['learning_results']
        }

    def _generate_feedback(self, results: Dict) -> Dict:
        """Generate comprehensive feedback"""
        # Get specific feedback from models
        teacher_feedback = self.teacher_evaluator.get_feedback()
        learner_feedback = self.learner_model.get_feedback()

        # Combine and structure feedback
        feedback = {
            'summary': self._generate_feedback_summary(results),
            'strengths': self._identify_strengths(results),
            'areas_for_improvement': self._identify_improvements(results),
            'recommendations': self._generate_recommendations(
                teacher_feedback,
                learner_feedback
            )
        }

        return feedback

    def _generate_feedback_summary(self, results: Dict) -> str:
        """Generate overall feedback summary"""
        if results['overall_score'] >= 0.8:
            quality = "excellent"
        elif results['overall_score'] >= 0.6:
            quality = "good"
        else:
            quality = "needs improvement"

        return (
            f"The teaching session demonstrated {quality} effectiveness "
            f"with an understanding ratio of {results['understanding_ratio']:.1%}. "
            f"Content accuracy was {results['accuracy']:.1%} "
            f"with {results['engagement']:.1%} engagement."
        )

    def _identify_strengths(self, results: Dict) -> List[str]:
        """Identify teaching strengths"""
        strengths = []

        if results['accuracy'] >= 0.8:
            strengths.append("High content accuracy")
        if results['engagement'] >= 0.8:
            strengths.append("Strong student engagement")
        if results['understanding_ratio'] >= 0.8:
            strengths.append("Excellent knowledge transfer")

        return strengths

    def _identify_improvements(self, results: Dict) -> List[str]:
        """Identify areas for improvement"""
        improvements = []

        if results['accuracy'] < 0.6:
            improvements.append("Content accuracy needs improvement")
        if results['engagement'] < 0.6:
            improvements.append("Student engagement could be enhanced")
        if results['understanding_ratio'] < 0.6:
            improvements.append("Knowledge transfer effectiveness needs work")

        return improvements

    def _generate_recommendations(
            self,
            teacher_feedback: Dict,
            learner_feedback: Dict
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Add teaching technique recommendations
        if teacher_feedback.get('visual_improvements'):
            recommendations.extend(teacher_feedback['visual_improvements'])

        # Add content recommendations
        if teacher_feedback.get('content_improvements'):
            recommendations.extend(teacher_feedback['content_improvements'])

        # Add knowledge transfer recommendations
        if learner_feedback.get('suggested_review'):
            for review in learner_feedback['suggested_review']:
                recommendations.append(
                    f"Review {review['concept']}: focus on "
                    f"{', '.join(review['focus_areas'])}"
                )

        return recommendations