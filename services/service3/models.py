from pydantic import BaseModel
from typing import Optional


class TrainResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str           # pending | running | stopped | completed | timeout | failed
    results_count: int
    best_race_ms: Optional[int] = None
    last_distance: Optional[float] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ResultsResponse(BaseModel):
    files: list[str]
