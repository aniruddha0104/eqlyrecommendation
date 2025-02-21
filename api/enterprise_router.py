# eqly_assessment/api/enterprise_router.py
import uuid

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, List
import jwt
from datetime import datetime

from scipy._lib.cobyqa import settings

from db.database import Database
from ..models.enterprise import EnterpriseConfig
from ..features.audio_processor import AudioProcessor
from ..features.video_processor import VideoProcessor
from ..core.assessment import AssessmentEngine

router = APIRouter(prefix="/enterprise")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class EnterpriseService:
    def __init__(self):
        self.config = EnterpriseConfig()
        self.audio_processor = AudioProcessor(self.config.audio)
        self.video_processor = VideoProcessor(self.config.video)
        self.assessment_engine = AssessmentEngine(self.config.assessment)

    async def process_video(self, video: UploadFile, org_id: str) -> Dict:
        video_path = await self._save_video(video)

        audio_features = await self.audio_processor.extract_features(video_path)
        video_features = await self.video_processor.extract_features(video_path)

        assessment = await self.assessment_engine.evaluate(
            audio_features=audio_features,
            video_features=video_features,
            org_id=org_id
        )

        return {
            'assessment_id': assessment.id,
            'scores': assessment.scores,
            'insights': assessment.insights,
            'recommendations': assessment.recommendations
        }

    async def get_org_insights(self, org_id: str) -> Dict:
        assessments = await self.assessment_engine.get_org_assessments(org_id)
        return {
            'total_assessments': len(assessments),
            'avg_score': sum(a.overall_score for a in assessments) / len(assessments),
            'trends': self._analyze_trends(assessments),
            'recommendations': self._generate_org_recommendations(assessments)
        }


@router.post("/assess")
async def assess_video(
        video: UploadFile = File(...),
        token: str = Depends(oauth2_scheme)
):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        org_id = payload.get("org_id")

        if not org_id:
            raise HTTPException(status_code=403, detail="Invalid organization")

        service = EnterpriseService()
        return await service.process_video(video, org_id)

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.get("/insights/{org_id}")
async def get_insights(
        org_id: str,
        token: str = Depends(oauth2_scheme)
):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        if payload.get("org_id") != org_id:
            raise HTTPException(status_code=403, detail="Access denied")

        service = EnterpriseService()
        return await service.get_org_insights(org_id)

    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# eqly_assessment/models/enterprise.py
from pydantic import BaseModel
from typing import Dict, List


class EnterpriseConfig(BaseModel):
    audio: Dict = {
        "sample_rate": 16000,
        "model_name": "facebook/wav2vec2-large-960h"
    }
    video: Dict = {
        "frame_rate": 30,
        "resolution": (1280, 720)
    }
    assessment: Dict = {
        "min_confidence": 0.7,
        "batch_size": 32
    }


class Assessment(BaseModel):
    id: str
    org_id: str
    timestamp: datetime
    scores: Dict[str, float]
    insights: Dict[str, str]
    recommendations: List[str]
    overall_score: float


# eqly_assessment/core/assessment.py
class AssessmentEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.db = Database()  # Replace with your database client

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