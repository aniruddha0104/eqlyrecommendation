from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List, Optional
import uuid

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