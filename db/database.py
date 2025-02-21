from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from ..models.enterprise import Assessment


class Database:
    def __init__(self, connection_url: str):
        self.client = AsyncIOMotorClient(connection_url)
        self.db = self.client.eqly_assessment

    async def save_assessment(self, assessment: Assessment) -> str:
        result = await self.db.assessments.insert_one(assessment.dict())
        return str(result.inserted_id)

    async def get_assessments_by_org(self, org_id: str) -> List[Assessment]:
        cursor = self.db.assessments.find({"org_id": org_id})
        return [Assessment(**doc) async for doc in cursor]