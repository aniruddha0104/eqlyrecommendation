"""Understanding calculation module for the assessment platform.

This module implements the core logic for calculating understanding ratio
between the First AI (TeacherEvaluator) and Second AI (Learner).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import json
import re
import numpy as np
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnderstandingError(Exception):
    """Custom exception for understanding calculation errors"""
    pass


class KnowledgeComponent(Enum):
    """Types of knowledge components that can be evaluated"""
    CONCEPT = "concept"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    PRINCIPLE = "principle"
    FACT = "fact"
    EXAMPLE = "example"
    RELATIONSHIP = "relationship"
    APPLICATION = "application"


class AssessmentDimension(Enum):
    """Dimensions of assessment for understanding calculation"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    STRUCTURE = "structure"
    ENGAGEMENT = "engagement"
    AUTHENTICITY = "authenticity"
    EFFECTIVENESS = "effectiveness"


class UnderstandingCalculator:
    """Calculator for measuring knowledge transfer and understanding ratio"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the understanding calculator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = {
            'dimension_weights': {
                AssessmentDimension.ACCURACY.value: 0.25,
                AssessmentDimension.COMPLETENESS.value: 0.20,
                AssessmentDimension.CLARITY.value: 0.20,
                AssessmentDimension.STRUCTURE.value: 0.15,
                AssessmentDimension.ENGAGEMENT.value: 0.10,
                AssessmentDimension.AUTHENTICITY.value: 0.05,
                AssessmentDimension.EFFECTIVENESS.value: 0.05
            },
            'knowledge_weights': {
                KnowledgeComponent.CONCEPT.value: 0.20,
                KnowledgeComponent.DEFINITION.value: 0.15,
                KnowledgeComponent.PROCEDURE.value: 0.15,
                KnowledgeComponent.PRINCIPLE.value: 0.15,
                KnowledgeComponent.FACT.value: 0.10,
                KnowledgeComponent.EXAMPLE.value: 0.10,
                KnowledgeComponent.RELATIONSHIP.value: 0.10,
                KnowledgeComponent.APPLICATION.value: 0.05
            },
            'minimum_knowledge_threshold': 0.3,  # Minimum score to consider knowledge as transferred
            'maximum_understanding_ratio': 1.5,  # Cap on understanding ratio (150%)
            'similarity_threshold': 0.7,  # Threshold for considering concepts similar
            'assessment_scale': 10.0  # Scale for scoring (0-10)
        }

        if config:
            # Update config with provided values
            for key, value in config.items():
                if key in self.config:
                    if isinstance(value, dict) and isinstance(self.config[key], dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                else:
                    self.config[key] = value

        # Normalize weights if they don't sum to 1
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure all weight dictionaries sum to 1.0"""
        for weight_key in ['dimension_weights', 'knowledge_weights']:
            weights = self.config[weight_key]
            weight_sum = sum(weights.values())

            if abs(weight_sum - 1.0) > 0.001:  # If not close to 1.0
                # Normalize
                for key in weights:
                    weights[key] = weights[key] / weight_sum

    def calculate_understanding_ratio(self,
                                      teacher_evaluation: Dict[str, Any],
                                      learner_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate understanding ratio between teacher and learner.

        Args:
            teacher_evaluation: Evaluation data from the TeacherEvaluator
            learner_assessment: Assessment data from the Learner model

        Returns:
            Dictionary containing understanding metrics
        """
        try:
            # Validate inputs
            self._validate_inputs(teacher_evaluation, learner_assessment)

            # Extract knowledge components
            teacher_knowledge = teacher_evaluation['knowledge_components']
            learner_knowledge = learner_assessment['knowledge_components']

            # Calculate component-wise understanding scores
            component_scores = {}
            for component_type in KnowledgeComponent:
                type_value = component_type.value
                if type_value in teacher_knowledge and type_value in learner_knowledge:
                    component_scores[type_value] = self._compare_knowledge_components(
                        teacher_knowledge[type_value],
                        learner_knowledge[type_value]
                    )

            # Calculate dimension scores
            dimension_scores = {}
            for dimension in AssessmentDimension:
                dim_value = dimension.value
                if dim_value in teacher_evaluation['dimensions'] and dim_value in learner_assessment['dimensions']:
                    dimension_scores[dim_value] = self._calculate_dimension_score(
                        teacher_evaluation['dimensions'][dim_value],
                        learner_assessment['dimensions'][dim_value]
                    )

            # Calculate weighted understanding score
            knowledge_understanding = self._calculate_weighted_score(
                component_scores,
                self.config['knowledge_weights']
            )

            dimension_understanding = self._calculate_weighted_score(
                dimension_scores,
                self.config['dimension_weights']
            )

            # Final understanding ratio calculation
            # The idea is to measure how much knowledge was successfully transferred
            # from the teacher to the learner

            # Get the base content scores
            teacher_content_score = teacher_evaluation.get('content_score', 0)
            teacher_content_score = max(0.1, min(1.0, teacher_content_score))  # Clamp between 0.1 and 1.0

            # Calculate the understanding ratio
            understanding_ratio = (
                    (knowledge_understanding * 0.7 + dimension_understanding * 0.3) /
                    teacher_content_score
            )

            # Cap the understanding ratio
            understanding_ratio = min(
                self.config['maximum_understanding_ratio'],
                understanding_ratio
            )

            # Scale to a percentage
            understanding_percent = understanding_ratio * 100

            # Calculate detailed metrics
            detailed_metrics = self._calculate_detailed_metrics(
                teacher_evaluation,
                learner_assessment,
                component_scores,
                dimension_scores
            )

            # Compile results
            result = {
                'understanding_ratio': float(understanding_ratio),
                'understanding_percent': float(understanding_percent),
                'knowledge_understanding': float(knowledge_understanding),
                'dimension_understanding': float(dimension_understanding),
                'component_scores': component_scores,
                'dimension_scores': dimension_scores,
                'detailed_metrics': detailed_metrics,
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            logger.error(f"Understanding calculation failed: {str(e)}")
            raise UnderstandingError(f"Failed to calculate understanding ratio: {str(e)}")

    def _validate_inputs(self, teacher_evaluation: Dict[str, Any], learner_assessment: Dict[str, Any]):
        """Validate input data structure.

        Args:
            teacher_evaluation: Evaluation data from the TeacherEvaluator
            learner_assessment: Assessment data from the Learner model

        Raises:
            UnderstandingError: If inputs are invalid
        """
        # Check for required fields
        required_fields = ['knowledge_components', 'dimensions', 'content_score']

        for field in required_fields:
            if field not in teacher_evaluation:
                raise UnderstandingError(f"Missing required field '{field}' in teacher_evaluation")
            if field not in learner_assessment:
                raise UnderstandingError(f"Missing required field '{field}' in learner_assessment")

        # Validate knowledge components structure
        for evaluation in [teacher_evaluation, learner_assessment]:
            if not isinstance(evaluation['knowledge_components'], dict):
                raise UnderstandingError("'knowledge_components' must be a dictionary")

            # Check at least one knowledge component exists
            if len(evaluation['knowledge_components']) == 0:
                raise UnderstandingError("No knowledge components found in evaluation")

            # Validate dimensions structure
            if not isinstance(evaluation['dimensions'], dict):
                raise UnderstandingError("'dimensions' must be a dictionary")

            # Check content score is a number
            if not isinstance(evaluation.get('content_score', 0), (int, float)):
                raise UnderstandingError("'content_score' must be a number")

    def _compare_knowledge_components(self,
                                      teacher_components: List[Dict[str, Any]],
                                      learner_components: List[Dict[str, Any]]) -> float:
        """Compare knowledge components between teacher and learner.

        Args:
            teacher_components: Knowledge components from teacher
            learner_components: Knowledge components from learner

        Returns:
            Score representing understanding level (0.0-1.0)
        """
        if not teacher_components:
            return 0.0

        if not learner_components:
            return 0.0

        # Create sets of component keys for faster lookup
        teacher_keys = {comp['key'] for comp in teacher_components if 'key' in comp}

        total_score = 0.0
        matched_count = 0

        # For each learner component, find the best matching teacher component
        for learner_comp in learner_components:
            if 'key' not in learner_comp or not learner_comp.get('content'):
                continue

            learner_key = learner_comp['key']
            learner_content = learner_comp['content']

            # Direct key match
            if learner_key in teacher_keys:
                # Find the matching teacher component
                teacher_comp = next(
                    (tc for tc in teacher_components if tc.get('key') == learner_key),
                    None
                )

                if teacher_comp and teacher_comp.get('content'):
                    # Calculate similarity score
                    similarity = self._calculate_content_similarity(
                        teacher_comp['content'],
                        learner_content
                    )

                    # If similarity is above threshold, consider it matched
                    if similarity >= self.config['similarity_threshold']:
                        total_score += similarity
                        matched_count += 1
            else:
                # Try fuzzy matching if no direct key match
                best_match = None
                best_similarity = 0.0

                for teacher_comp in teacher_components:
                    if not teacher_comp.get('content'):
                        continue

                    similarity = self._calculate_content_similarity(
                        teacher_comp['content'],
                        learner_content
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = teacher_comp

                # If we found a good match, add to score
                if best_match and best_similarity >= self.config['similarity_threshold']:
                    total_score += best_similarity
                    matched_count += 1

        # Calculate final score
        if matched_count == 0:
            return 0.0

        # Weighted score based on matches and total components
        component_match_ratio = matched_count / len(teacher_components)
        average_match_score = total_score / matched_count

        # Combined score with higher weight to match quality
        final_score = (average_match_score * 0.7) + (component_match_ratio * 0.3)

        return float(final_score)

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between content strings.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple implementation using word set overlap
        # A more sophisticated implementation could use embeddings

        if not content1 or not content2:
            return 0.0

        # Normalize and tokenize
        words1 = self._tokenize_content(content1)
        words2 = self._tokenize_content(content2)

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        return float(intersection / union)

    def _tokenize_content(self, content: str) -> Set[str]:
        """Tokenize content into words for comparison.

        Args:
            content: Text content

        Returns:
            Set of normalized tokens
        """
        if not content:
            return set()

        # Convert to lowercase
        content = content.lower()

        # Remove punctuation and tokenize
        tokens = re.findall(r'\b\w+\b', content)

        # Remove stopwords (simplified version)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'in', 'on', 'at', 'to', 'for', 'with'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]

        return set(tokens)

    def _calculate_dimension_score(self, teacher_score: float, learner_score: float) -> float:
        """Calculate understanding score for a dimension.

        Args:
            teacher_score: Score from teacher evaluation
            learner_score: Score from learner assessment

        Returns:
            Normalized dimension score (0.0-1.0)
        """
        # Normalize scores to 0-1 range
        teacher_norm = teacher_score / self.config['assessment_scale']
        learner_norm = learner_score / self.config['assessment_scale']

        # Cap normalized scores
        teacher_norm = max(0.0, min(1.0, teacher_norm))
        learner_norm = max(0.0, min(1.0, learner_norm))

        # If teacher score is below threshold, dimension is not important
        if teacher_norm < self.config['minimum_knowledge_threshold']:
            return 1.0  # Not penalizing for dimensions that teacher didn't cover well

        # Ratio of learner's understanding to teacher's presentation
        if teacher_norm > 0:
            score = learner_norm / teacher_norm
        else:
            score = 0.0

        # Cap at maximum
        score = min(1.0, score)

        return float(score)

    def _calculate_weighted_score(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted score from individual component/dimension scores.

        Args:
            scores: Dictionary of scores by component/dimension
            weights: Dictionary of weights by component/dimension

        Returns:
            Weighted average score (0.0-1.0)
        """
        if not scores:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for key, score in scores.items():
            if key in weights:
                weight = weights[key]
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return float(total_score / total_weight)

    def _calculate_detailed_metrics(self,
                                    teacher_evaluation: Dict[str, Any],
                                    learner_assessment: Dict[str, Any],
                                    component_scores: Dict[str, float],
                                    dimension_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate detailed metrics for understanding analysis.

        Args:
            teacher_evaluation: Evaluation from teacher
            learner_assessment: Assessment from learner
            component_scores: Calculated component scores
            dimension_scores: Calculated dimension scores

        Returns:
            Dictionary of detailed metrics
        """
        # Calculate knowledge gap analysis
        knowledge_gaps = self._identify_knowledge_gaps(
            teacher_evaluation['knowledge_components'],
            learner_assessment['knowledge_components'],
            component_scores
        )

        # Calculate strength areas
        strengths = self._identify_strengths(component_scores, dimension_scores)

        # Calculate improvement recommendations
        recommendations = self._generate_recommendations(
            knowledge_gaps,
            dimension_scores
        )

        # Calculate effectiveness metrics
        teaching_effectiveness = self._calculate_teaching_effectiveness(
            teacher_evaluation,
            learner_assessment,
            component_scores
        )

        return {
            'knowledge_gaps': knowledge_gaps,
            'strengths': strengths,
            'recommendations': recommendations,
            'teaching_effectiveness': teaching_effectiveness
        }

    def _identify_knowledge_gaps(self,
                                 teacher_components: Dict[str, List[Dict[str, Any]]],
                                 learner_components: Dict[str, List[Dict[str, Any]]],
                                 component_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify knowledge gaps in learner's understanding.

        Args:
            teacher_components: Knowledge components from teacher
            learner_components: Knowledge components from learner
            component_scores: Calculated component scores

        Returns:
            List of identified knowledge gaps
        """
        gaps = []

        # Identify component types with low scores
        for component_type, score in component_scores.items():
            if score < self.config['minimum_knowledge_threshold']:
                if component_type in teacher_components:
                    # Find specific components that weren't well understood
                    missing_components = []

                    # Get keys from learner for faster lookup
                    learner_keys = {
                        comp['key'] for comp in learner_components.get(component_type, [])
                        if 'key' in comp
                    }

                    for teacher_comp in teacher_components[component_type]:
                        if 'key' not in teacher_comp:
                            continue

                        if teacher_comp['key'] not in learner_keys:
                            missing_components.append({
                                'key': teacher_comp['key'],
                                'description': teacher_comp.get('description', ''),
                                'importance': teacher_comp.get('importance', 0.5)
                            })

                    if missing_components:
                        gaps.append({
                            'component_type': component_type,
                            'score': score,
                            'missing_components': missing_components
                        })

        return gaps

    def _identify_strengths(self,
                            component_scores: Dict[str, float],
                            dimension_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Identify areas of strength in understanding.

        Args:
            component_scores: Calculated component scores
            dimension_scores: Calculated dimension scores

        Returns:
            Dictionary of strength areas
        """
        strengths = {
            'knowledge_components': [],
            'dimensions': []
        }

        # Identify strong knowledge components (top 3)
        strong_components = sorted(
            [(comp, score) for comp, score in component_scores.items() if score >= 0.7],
            key=lambda x: x[1],
            reverse=True
        )

        for comp, score in strong_components[:3]:  # Top 3 components
            strengths['knowledge_components'].append(comp)

        # Identify strong dimensions (top 2)
        strong_dimensions = sorted(
            [(dim, score) for dim, score in dimension_scores.items() if score >= 0.7],
            key=lambda x: x[1],
            reverse=True
        )

        for dim, score in strong_dimensions[:2]:  # Top 2 dimensions
            strengths['dimensions'].append(dim)

        return strengths

    def _generate_recommendations(self,
                                  knowledge_gaps: List[Dict[str, Any]],
                                  dimension_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations.

        Args:
            knowledge_gaps: Identified knowledge gaps
            dimension_scores: Calculated dimension scores

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        # Recommendations based on knowledge gaps
        if knowledge_gaps:
            for gap in knowledge_gaps:
                component_type = gap['component_type']
                missing_components = gap['missing_components']

                if missing_components:
                    high_importance = [comp for comp in missing_components if comp.get('importance', 0) > 0.7]

                    if high_importance:
                        # Recommend focusing on high importance components
                        keys = [comp['key'] for comp in high_importance[:3]]
                        recommendations.append(
                            f"Focus on improving {component_type} understanding, particularly: {', '.join(keys)}"
                        )
                    else:
                        recommendations.append(
                            f"Review {component_type} concepts to improve overall understanding"
                        )

        # Recommendations based on dimension scores
        weak_dimensions = [
            (dim, score) for dim, score in dimension_scores.items()
            if score < 0.6  # Threshold for weak dimensions
        ]

        dimension_recommendations = {
            'clarity': "Improve explanations with clearer language and examples",
            'structure': "Better organize content with logical progression",
            'completeness': "Ensure all key points are covered thoroughly",
            'engagement': "Use more engaging presentation techniques and interactions",
            'effectiveness': "Focus on more impactful teaching methods"
        }

        for dim, score in weak_dimensions:
            if dim in dimension_recommendations:
                recommendations.append(dimension_recommendations[dim])

        return recommendations

    def _calculate_teaching_effectiveness(self,
                                          teacher_evaluation: Dict[str, Any],
                                          learner_assessment: Dict[str, Any],
                                          component_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate teaching effectiveness metrics.

        Args:
            teacher_evaluation: Evaluation from teacher
            learner_assessment: Assessment from learner
            component_scores: Calculated component scores

        Returns:
            Dictionary of teaching effectiveness metrics
        """
        # Extract presentation quality from teacher evaluation
        presentation_quality = teacher_evaluation.get('presentation_quality', 0.5)

        # Calculate average component understanding
        avg_understanding = sum(component_scores.values()) / max(1, len(component_scores))

        # Calculate knowledge retention score
        retention_score = avg_understanding

        # Calculate efficiency as ratio of understanding to presentation quality
        if presentation_quality > 0:
            efficiency_score = min(1.0, avg_understanding / presentation_quality)
        else:
            efficiency_score = 0.0

        # Calculate impact as the weighted improvement in learner's understanding
        before_score = learner_assessment.get('initial_knowledge', 0.1)
        after_score = learner_assessment.get('final_knowledge', avg_understanding)

        impact_score = max(0.0, after_score - before_score)

        return {
            'overall_effectiveness': float(avg_understanding),
            'knowledge_retention': float(retention_score),
            'teaching_efficiency': float(efficiency_score),
            'learning_impact': float(impact_score)
        }