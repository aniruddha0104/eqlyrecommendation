# eqly_assessment/models/enterprise.py
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    model_name: str = "facebook/wav2vec2-large-960h"
    min_confidence: float = 0.7
    chunk_duration: float = 30.0

class VideoConfig(BaseModel):
    frame_rate: int = 30
    resolution: Tuple[int, int] = (1280, 720)
    batch_size: int = 32
    min_frames: int = 90

class AssessmentConfig(BaseModel):
    min_confidence: float = 0.7
    batch_size: int = 32
    teacher_weights: Dict[str, float] = {
        'content_mastery': 0.3,
        'explanation_clarity': 0.3,
        'engagement': 0.2,
        'structure': 0.2
    }
    learner_weights: Dict[str, float] = {
        'concept_grasp': 0.4,
        'knowledge_retention': 0.3,
        'application_ability': 0.3
    }

class EnterpriseConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    video: VideoConfig = VideoConfig()
    assessment: AssessmentConfig = AssessmentConfig()
    db_url: str = Field(..., description="MongoDB connection URL")
    api_key: Optional[str] = None
    org_id: Optional[str] = None

class Assessment(BaseModel):
    id: str
    org_id: str
    timestamp: datetime
    scores: Dict[str, float]
    insights: Dict[str, str]
    recommendations: List[str]
    overall_score: float
    metadata: Optional[Dict] = None