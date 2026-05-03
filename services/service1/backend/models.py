from pydantic import BaseModel, Field
from typing import Literal


class Track(BaseModel):
    id: str
    label: str
    vcp_file: str


class DriverProfile(BaseModel):
    braking_aggression: float
    oversteer_understeer_score: float
    corner_entry_speed_ratio: float
    corner_entry_speed_level: str


class GenerateRequest(BaseModel):
    track: Literal["Bahrain", "Hockolicious"]
    best_lap_ms: int = Field(..., gt=0, description="Best lap time in milliseconds")
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    braking_aggression: float = Field(..., ge=0.0, le=1.0)
    oversteer_understeer_score: float = Field(..., ge=-5.0, le=5.0)
    corner_entry_speed_ratio: float = Field(..., ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    status: str
    hyperparameters: GenerateRequest
    message: str
