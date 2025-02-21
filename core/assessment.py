from typing import Dict, List
from datetime import datetime
import uuid
from ..models.enterprise import Assessment
from ..db.database import Database


class AssessmentEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.db = Database(config['db_url'])

    async def evaluate(
            self,
            audio_features: Dict,
            video_features: Dict,
            org_id: str
    ) -> Assessment:
        scores = self._calculate_scores(audio_features, video_features)
        insights = self._generate_insights(scores)
        recommendations = self._generate_recommendations(scores)

        assessment = Assessment(
            id=str(uuid.uuid4()),
            org_id=org_id,
            timestamp=datetime.utcnow(),
            scores=scores,
            insights=insights,
            recommendations=recommendations,
            overall_score=sum(scores.values()) / len(scores)
        )

        await self.db.save_assessment(assessment)
        return assessment

    async def get_org_assessments(self, org_id: str) -> List[Assessment]:
        return await self.db.get_assessments_by_org(org_id)

    def _calculate_scores(self, audio_features: Dict, video_features: Dict) -> Dict[str, float]:
        return {
            'audio_clarity': self._evaluate_audio(audio_features),
            'visual_engagement': self._evaluate_visual(video_features),
            'overall_effectiveness': self._evaluate_overall(audio_features, video_features)
        }

    def _generate_insights(self, scores: Dict[str, float]) -> Dict[str, str]:
        return {
            'strengths': self._identify_strengths(scores),
            'areas_for_improvement': self._identify_improvements(scores),
            'key_observations': self._generate_observations(scores)
        }

    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        recommendations = []
        if scores['audio_clarity'] < 0.7:
            recommendations.append("Improve audio clarity and speaking pace")
        if scores['visual_engagement'] < 0.7:
            recommendations.append("Enhance visual engagement through better gestures")
        return recommendations