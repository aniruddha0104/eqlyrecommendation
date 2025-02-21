# components/learner_model.py

import os
import sys
import logging
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)
import random
from sentence_transformers import SentenceTransformer

# Add path fix if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class EnhancedLearnerModel:
    """
    Enhanced learner model using transformer-based question answering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced learner model"""
        self.config = {
            'output_directory': 'learner_results',
            'use_gpu': torch.cuda.is_available(),
            'qa_model': 'deepset/roberta-base-squad2',
            'question_generation_model': 't5-base',
            'understanding_threshold': 0.7,
            'learning_rate': 0.85,
            'test_question_count': 10,
            'concept_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }

        if config:
            self.config.update(config)

        self.logger = logging.getLogger("EnhancedLearnerModel")
        self.device = 0 if self.config['use_gpu'] else -1

        # Create output directory
        if not os.path.exists(self.config['output_directory']):
            os.makedirs(self.config['output_directory'])

        self.initialize_state()
        self.initialize_models()

    def initialize_state(self):
        """Initialize learner's knowledge state"""
        self.knowledge_state = {
            'learned_concepts': {},
            'confidence_levels': {},
            'learning_history': [],
            'test_results': [],
            'understanding_metrics': {
                'average_confidence': 0.0,
                'concept_coverage': 0.0,
                'reasoning_ability': 0.0,
                'application_skill': 0.0
            }
        }

    def initialize_models(self):
        """Initialize transformer models for learning and testing"""
        try:
            self.logger.info("Initializing learner models...")

            # Initialize QA model
            self.qa_model = pipeline(
                "question-answering",
                model=self.config['qa_model'],
                device=self.device
            )

            # Initialize question generation model
            self.question_generator = pipeline(
                "text2text-generation",
                model=self.config['question_generation_model'],
                device=self.device
            )

            # Initialize sentence embedding model if available
            try:
                self.embedding_model = SentenceTransformer(self.config['concept_embedding_model'])
            except ImportError:
                self.logger.warning("Sentence Transformers not available. Using simplified concept modeling.")
                self.embedding_model = None

            self.logger.info("Learner models initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize learner models: {str(e)}")
            self.logger.warning("Falling back to rule-based learning simulation")

    def learn_from_teacher(self,
                           teacher_knowledge: Dict[str, Any],
                           teaching_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn from knowledge provided by teacher model"""
        try:
            self.logger.info("Learning from teacher knowledge...")

            # Extract knowledge points
            knowledge_points = teacher_knowledge.get('knowledge_points', [])
            if not knowledge_points:
                self.logger.warning("No knowledge points provided by teacher")
                return self._generate_empty_learning_results()

            # Apply teaching effectiveness as a factor if available
            teaching_factor = 1.0
            if teaching_metrics:
                teaching_factor = teaching_metrics.get('overall_score', 0.7)

            # Process knowledge points
            learned_points = self._process_knowledge_points(knowledge_points, teaching_factor)

            # Process topic structure if available
            topic_groups = teacher_knowledge.get('topic_groups', {})
            if topic_groups:
                self._learn_topic_structure(topic_groups)

            # Generate context for testing
            learning_context = self._create_learning_context(learned_points, teacher_knowledge)

            # Test understanding
            understanding_results = self._test_understanding(learned_points, learning_context)

            # Calculate understanding metrics
            understanding_metrics = self._calculate_understanding_metrics(
                learned_points,
                understanding_results,
                topic_groups,
                teaching_factor
            )

            # Update knowledge state
            self._update_knowledge_state(learned_points, understanding_metrics, teacher_knowledge)

            # Create learning results
            learning_results = {
                'understanding_ratio': understanding_metrics['understanding_ratio'],
                'metrics': understanding_metrics,
                'test_results': understanding_results['test_summary'],
                'concepts_learned': len(learned_points),
                'topics_understood': len(topic_groups),
                'learning_timestamp': datetime.now().isoformat(),
                'teaching_effectiveness_factor': teaching_factor
            }

            # Save learning results
            self._save_learning_results(learning_results, teacher_knowledge)

            return learning_results

        except Exception as e:
            self.logger.error(f"Learning process failed: {str(e)}")
            return self._generate_empty_learning_results(error=str(e))

    def _process_knowledge_points(self,
                                  knowledge_points: List[Dict[str, Any]],
                                  teaching_factor: float) -> List[Dict[str, Any]]:
        """Process and learn from knowledge points"""
        # Sort by confidence and limit number of points
        sorted_points = sorted(
            knowledge_points,
            key=lambda x: x['confidence'],
            reverse=True
        )

        # Apply learning rate and teaching factor
        learned_points = []
        for point in sorted_points:
            # Calculate learning confidence based on point confidence and teaching factor
            learning_confidence = point['confidence'] * teaching_factor * self.config['learning_rate']

            # Apply natural variance in learning
            learning_confidence *= random.uniform(0.9, 1.0)

            # Store learned point
            learned_point = {
                'content': point['content'],
                'original_confidence': point['confidence'],
                'learned_confidence': min(learning_confidence, 0.98),  # Cap at 98%
                'context': point.get('context', {}),
                'learned_at': datetime.now().isoformat()
            }

            # Create embeddings if model available
            if hasattr(self, 'embedding_model') and self.embedding_model:
                try:
                    # Store concept embedding
                    embedding = self.embedding_model.encode(point['content'])
                    learned_point['embedding'] = embedding.tolist()
                except Exception:
                    # Continue without embedding
                    pass

            learned_points.append(learned_point)

            # Update knowledge state
            concept_key = self._get_concept_key(point['content'])
            self.knowledge_state['learned_concepts'][concept_key] = learned_point
            self.knowledge_state['confidence_levels'][concept_key] = learning_confidence

        return learned_points

    def _get_concept_key(self, content: str) -> str:
        """Generate a stable key for a concept"""
        # Simple hash-based key
        return f"concept_{hash(content) % 10000:04d}"

    def _learn_topic_structure(self, topic_groups: Dict[str, List[Dict[str, Any]]]):
        """Learn topic structure from grouped knowledge points"""
        for topic, points in topic_groups.items():
            # Store topic structure in knowledge state
            topic_key = self._get_concept_key(topic)
            self.knowledge_state['learned_concepts'][topic_key] = {
                'content': topic,
                'subtopics': [self._get_concept_key(p['content']) for p in points],
                'learned_at': datetime.now().isoformat()
            }

    def _create_learning_context(self,
                                 learned_points: List[Dict[str, Any]],
                                 teacher_knowledge: Dict[str, Any]) -> str:
        """Create context for testing understanding"""
        context = ""
        for point in learned_points:
            context += f"- {point['content']}\n"

        # Add summary if available
        if 'knowledge_summary' in teacher_knowledge:
            context += f"\nSummary:\n{teacher_knowledge['knowledge_summary']}"

        return context.strip()

    def _test_understanding(self,
                            learned_points: List[Dict[str, Any]],
                            learning_context: str) -> Dict[str, Any]:
        """Test learner's understanding of learned concepts"""
        try:
            test_questions = []
            correct_answers = 0

            # Generate questions
            for i in range(min(self.config['test_question_count'], len(learned_points))):
                point = learned_points[i]
                try:
                    # Generate question using model
                    generated_question = self.question_generator(
                        f"Generate a question about: {point['content']}",
                        max_length=50,
                        do_sample=True,
                        temperature=0.7
                    )[0]['generated_text']

                    # Clean up question
                    question = generated_question.strip()
                    if not question.endswith('?'):
                        question += '?'

                    test_questions.append({
                        'question': question,
                        'correct_answer': point['content']
                    })
                except Exception:
                    # Fallback to simple question
                    test_questions.append({
                        'question': f"What is {point['content'].split()[0]}?",
                        'correct_answer': point['content']
                    })

            # Answer questions
            test_results = []
            for question_data in test_questions:
                try:
                    answer = self.qa_model(
                        question=question_data['question'],
                        context=learning_context
                    )

                    is_correct = answer['answer'].strip() == question_data['correct_answer'].strip()
                    test_results.append({
                        'question': question_data['question'],
                        'answer_given': answer['answer'],
                        'correct_answer': question_data['correct_answer'],
                        'is_correct': is_correct
                    })

                    if is_correct:
                        correct_answers += 1
                except Exception:
                    test_results.append({
                        'question': question_data['question'],
                        'answer_given': "Error during answering",
                        'correct_answer': question_data['correct_answer'],
                        'is_correct': False
                    })

            # Calculate test summary
            total_questions = len(test_results)
            understanding_ratio = correct_answers / total_questions if total_questions > 0 else 0.0

            return {
                'test_questions': test_results,
                'test_summary': {
                    'total_questions': total_questions,
                    'correct_answers': correct_answers,
                    'understanding_ratio': round(understanding_ratio, 2)
                }
            }

        except Exception as e:
            self.logger.error(f"Understanding test failed: {str(e)}")
            return {
                'test_questions': [],
                'test_summary': {
                    'total_questions': 0,
                    'correct_answers': 0,
                    'understanding_ratio': 0.0
                }
            }

    def _calculate_understanding_metrics(self,
                                         learned_points: List[Dict[str, Any]],
                                         understanding_results: Dict[str, Any],
                                         topic_groups: Dict[str, List[Dict[str, Any]]],
                                         teaching_factor: float) -> Dict[str, Any]:
        """Calculate comprehensive understanding metrics"""
        try:
            # Calculate average confidence
            avg_confidence = sum(p['learned_confidence'] for p in learned_points) / len(learned_points)

            # Calculate concept coverage
            concept_coverage = len(learned_points) / max(len(topic_groups), 1)

            # Calculate reasoning ability based on test results
            reasoning_ability = understanding_results['test_summary']['understanding_ratio']

            # Calculate application skill (weighted by teaching factor)
            application_skill = (reasoning_ability * 0.6) + (avg_confidence * 0.4) * teaching_factor

            # Overall understanding ratio
            understanding_ratio = (avg_confidence * 0.3 +
                                   concept_coverage * 0.2 +
                                   reasoning_ability * 0.3 +
                                   application_skill * 0.2)

            return {
                'average_confidence': round(avg_confidence, 2),
                'concept_coverage': round(concept_coverage, 2),
                'reasoning_ability': round(reasoning_ability, 2),
                'application_skill': round(application_skill, 2),
                'understanding_ratio': round(understanding_ratio, 2)
            }

        except Exception as e:
            self.logger.error(f"Understanding metrics calculation failed: {str(e)}")
            return {
                'average_confidence': 0.0,
                'concept_coverage': 0.0,
                'reasoning_ability': 0.0,
                'application_skill': 0.0,
                'understanding_ratio': 0.0
            }

    def _update_knowledge_state(self,
                                learned_points: List[Dict[str, Any]],
                                understanding_metrics: Dict[str, Any],
                                teacher_knowledge: Dict[str, Any]):
        """Update learner's knowledge state with new information"""
        self.knowledge_state['learning_history'].append({
            'timestamp': datetime.now().isoformat(),
            'concepts_learned': [self._get_concept_key(p['content']) for p in learned_points],
            'understanding_metrics': understanding_metrics
        })

        # Update overall understanding metrics
        self.knowledge_state['understanding_metrics']['average_confidence'] = understanding_metrics[
            'average_confidence']
        self.knowledge_state['understanding_metrics']['concept_coverage'] = understanding_metrics['concept_coverage']
        self.knowledge_state['understanding_metrics']['reasoning_ability'] = understanding_metrics['reasoning_ability']
        self.knowledge_state['understanding_metrics']['application_skill'] = understanding_metrics['application_skill']

    def _save_learning_results(self,
                               learning_results: Dict[str, Any],
                               teacher_knowledge: Dict[str, Any]):
        """Save learning results to file"""
        output_path = os.path.join(
            self.config['output_directory'],
            f"{os.path.splitext(os.path.basename(teacher_knowledge['metadata']['video_path']))[0]}_learning.json"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(learning_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Learning results saved to: {output_path}")

    def _generate_empty_learning_results(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Generate empty learning results structure"""
        return {
            'understanding_ratio': 0.0,
            'metrics': {
                'average_confidence': 0.0,
                'concept_coverage': 0.0,
                'reasoning_ability': 0.0,
                'application_skill': 0.0,
                'understanding_ratio': 0.0
            },
            'test_results': {
                'total_questions': 0,
                'correct_answers': 0,
                'understanding_ratio': 0.0
            },
            'concepts_learned': 0,
            'topics_understood': 0,
            'learning_timestamp': datetime.now().isoformat(),
            'teaching_effectiveness_factor': 0.0,
            'error': error
        }