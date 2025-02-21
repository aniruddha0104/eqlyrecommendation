# models/learner_model.py

import logging
from typing import Dict, List, Any
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class LearnerModel:
    """Second AI model that learns from TeacherEvaluator"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.learning_rate = config['learning_rate']
        self.retention_rate = config['retention_rate']
        self.knowledge = {}
        self.understanding_metrics = {}
        self.initialize_models()

    def initialize_models(self):
        """Initialize required ML models"""
        try:
            # Initialize question answering model for understanding assessment
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                "bert-large-uncased-whole-word-masking-finetuned-squad"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-large-uncased-whole-word-masking-finetuned-squad"
            )
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

    def learn_concept(self, teacher_knowledge: Dict) -> Dict[str, float]:
        """Learn concept from teacher explanation"""
        try:
            # Process received knowledge
            processed_knowledge = self._process_knowledge(teacher_knowledge)

            # Update internal knowledge state
            self._update_knowledge(processed_knowledge)

            # Assess understanding
            understanding_metrics = self._assess_understanding(processed_knowledge)

            # Calculate retention
            retention_metrics = self._calculate_retention(processed_knowledge)

            # Combine metrics
            learning_results = {
                'understanding_metrics': understanding_metrics,
                'retention_metrics': retention_metrics,
                'overall_learning_score': self._calculate_learning_score(
                    understanding_metrics,
                    retention_metrics
                )
            }

            self.logger.info("Concept learning completed successfully")
            return learning_results

        except Exception as e:
            self.logger.error(f"Concept learning failed: {str(e)}")
            raise

    def _process_knowledge(self, teacher_knowledge: Dict) -> Dict:
        """Process and structure received knowledge"""
        processed = {
            'core_concepts': self._extract_core_concepts(
                teacher_knowledge['content']
            ),
            'relationships': self._extract_relationships(
                teacher_knowledge['content']
            ),
            'examples': teacher_knowledge.get('examples', []),
            'visual_elements': teacher_knowledge.get('visual_aids', {})
        }
        return processed

    def _update_knowledge(self, new_knowledge: Dict):
        """Update internal knowledge state"""
        # Apply learning rate to new knowledge
        learning_factor = self.learning_rate

        for concept in new_knowledge['core_concepts']:
            if concept not in self.knowledge:
                self.knowledge[concept] = {
                    'understanding': learning_factor,
                    'retention': 1.0,
                    'connections': []
                }
            else:
                # Reinforce existing knowledge
                self.knowledge[concept]['understanding'] = min(
                    1.0,
                    self.knowledge[concept]['understanding'] +
                    (1 - self.knowledge[concept]['understanding']) * learning_factor
                )

        # Update concept relationships
        for rel in new_knowledge['relationships']:
            source, target = rel['source'], rel['target']
            if source in self.knowledge:
                self.knowledge[source]['connections'].append(target)

    def _assess_understanding(self, knowledge: Dict) -> Dict[str, float]:
        """Assess understanding of learned concepts"""
        understanding_metrics = {
            'concept_transfer_rate': 0.0,
            'understanding_depth': 0.0,
            'meaning_preservation': 0.0,
            'concepts_understood': [],
            'concepts_missed': []
        }

        # Test understanding through question answering
        for concept in knowledge['core_concepts']:
            questions = self._generate_test_questions(concept)
            concept_score = self._test_concept_understanding(
                concept,
                questions
            )

            if concept_score > 0.7:
                understanding_metrics['concepts_understood'].append(concept)
            else:
                understanding_metrics['concepts_missed'].append(concept)

            understanding_metrics['concept_transfer_rate'] += concept_score

        # Calculate averages
        num_concepts = len(knowledge['core_concepts'])
        if num_concepts > 0:
            understanding_metrics['concept_transfer_rate'] /= num_concepts

        # Assess understanding depth
        understanding_metrics['understanding_depth'] = self._assess_understanding_depth(
            knowledge
        )

        # Assess meaning preservation
        understanding_metrics['meaning_preservation'] = self._assess_meaning_preservation(
            knowledge
        )

        return understanding_metrics

    def _generate_test_questions(self, concept: str) -> List[Dict]:
        """Generate test questions for concept understanding assessment"""
        # Template-based question generation
        questions = [
            {
                'question': f"What is {concept}?",
                'context': self.knowledge.get(concept, {}).get('description', ''),
                'expected_elements': ['definition', 'key characteristics']
            },
            {
                'question': f"How does {concept} work?",
                'context': self.knowledge.get(concept, {}).get('explanation', ''),
                'expected_elements': ['process', 'mechanism']
            },
            {
                'question': f"Why is {concept} important?",
                'context': self.knowledge.get(concept, {}).get('importance', ''),
                'expected_elements': ['benefits', 'applications']
            }
        ]
        return questions

    def _test_concept_understanding(self, concept: str, questions: List[Dict]) -> float:
        """Test understanding of a specific concept"""
        concept_score = 0.0

        for question in questions:
            # Prepare input for QA model
            inputs = self.tokenizer(
                question['question'],
                question['context'],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Get model outputs
            outputs = self.qa_model(**inputs)

            # Process answer
            answer_score = self._evaluate_answer(
                outputs,
                question['expected_elements']
            )
            concept_score += answer_score

        return concept_score / len(questions) if questions else 0.0

    def _evaluate_answer(self, model_outputs, expected_elements: List[str]) -> float:
        """Evaluate the quality of an answer"""
        # Extract answer from model outputs
        start_logits = model_outputs.start_logits
        end_logits = model_outputs.end_logits

        # Calculate confidence scores
        confidence = float(
            torch.softmax(start_logits, dim=1).max() *
            torch.softmax(end_logits, dim=1).max()
        )

        # Check for expected elements
        elements_score = self._check_expected_elements(
            model_outputs,
            expected_elements
        )

        return (confidence + elements_score) / 2

    def _check_expected_elements(
            self,
            model_outputs,
            expected_elements: List[str]
    ) -> float:
        """Check if answer contains expected elements"""
        score = 0.0
        answer_text = self._extract_answer_text(model_outputs)

        for element in expected_elements:
            if element.lower() in answer_text.lower():
                score += 1.0

        return score / len(expected_elements) if expected_elements else 0.0

    def _assess_understanding_depth(self, knowledge: Dict) -> float:
        """Assess depth of understanding"""
        depth_score = 0.0

        # Check concept relationships
        if 'relationships' in knowledge:
            num_relationships = len(knowledge['relationships'])
            valid_relationships = sum(
                1 for rel in knowledge['relationships']
                if self._validate_relationship(rel)
            )
            depth_score += valid_relationships / num_relationships if num_relationships > 0 else 0.0

        # Check example understanding
        if 'examples' in knowledge:
            num_examples = len(knowledge['examples'])
            understood_examples = sum(
                1 for example in knowledge['examples']
                if self._validate_example_understanding(example)
            )
            depth_score += understood_examples / num_examples if num_examples > 0 else 0.0

        return depth_score / 2

    def _validate_relationship(self, relationship: Dict) -> bool:
        """Validate understanding of a concept relationship"""
        source = relationship.get('source', '')
        target = relationship.get('target', '')

        # Check if both concepts exist in knowledge base
        if source not in self.knowledge or target not in self.knowledge:
            return False

        # Check if relationship is correctly understood
        source_connections = self.knowledge[source].get('connections', [])
        return target in source_connections

    def _validate_example_understanding(self, example: Dict) -> bool:
        """Validate understanding of an example"""
        concept = example.get('concept', '')
        example_text = example.get('text', '')

        if concept not in self.knowledge:
            return False

        # Generate question about the example
        question = f"How does this example demonstrate {concept}?"

        # Test understanding
        inputs = self.tokenizer(
            question,
            example_text,
            return_tensors="pt",
            truncation=True
        )

        outputs = self.qa_model(**inputs)
        confidence = float(
            torch.softmax(outputs.start_logits, dim=1).max() *
            torch.softmax(outputs.end_logits, dim=1).max()
        )

        return confidence > 0.7

    def calculate_understanding_ratio(self) -> float:
        """Calculate ratio of understood knowledge vs input"""
        if not self.knowledge:
            return 0.0

        total_understanding = sum(
            concept['understanding']
            for concept in self.knowledge.values()
        )
        return total_understanding / len(self.knowledge)

    def get_feedback(self) -> Dict[str, Any]:
        """Generate feedback about learning progress"""
        feedback = {
            'overall_understanding': self.calculate_understanding_ratio(),
            'strong_concepts': [],
            'weak_concepts': [],
            'missing_connections': [],
            'suggested_review': []
        }

        # Analyze concept understanding
        for concept, data in self.knowledge.items():
            if data['understanding'] > 0.8:
                feedback['strong_concepts'].append(concept)
            elif data['understanding'] < 0.6:
                feedback['weak_concepts'].append(concept)
                feedback['suggested_review'].append({
                    'concept': concept,
                    'focus_areas': self._identify_focus_areas(concept)
                })

        # Analyze concept relationships
        for concept, data in self.knowledge.items():
            expected_connections = set(data.get('expected_connections', []))
            actual_connections = set(data.get('connections', []))
            missing = expected_connections - actual_connections
            if missing:
                feedback['missing_connections'].append({
                    'concept': concept,
                    'missing': list(missing)
                })

        return feedback

    def _identify_focus_areas(self, concept: str) -> List[str]:
        """Identify areas needing review for a concept"""
        focus_areas = []
        concept_data = self.knowledge.get(concept, {})

        if concept_data.get('understanding', 0) < 0.6:
            focus_areas.append('basic_understanding')
        if not concept_data.get('connections', []):
            focus_areas.append('concept_relationships')
        if concept_data.get('retention', 1.0) < 0.7:
            focus_areas.append('retention')

        return focus_areas