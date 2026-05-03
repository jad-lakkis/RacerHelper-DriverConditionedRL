from pydantic import BaseModel, Field


class DriverProfile(BaseModel):
    braking_aggression: float = Field(..., ge=0.0, le=1.0)
    oversteer_understeer_score: float = Field(..., ge=-5.0, le=5.0)
    corner_entry_speed_ratio: float = Field(..., ge=0.0, le=1.0)
    corner_entry_speed_level: str
