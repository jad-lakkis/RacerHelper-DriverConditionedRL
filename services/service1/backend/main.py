import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import Track, DriverProfile, GenerateRequest, GenerateResponse
from service2_client import extract_driver_profile

# Wired up when Service 3 is ready; set via docker-compose environment.
SERVICE3_URL = os.environ.get("SERVICE3_URL", "http://localhost:8003")

_raw_origins = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="RacerHelper — Service 1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

TRACKS: list[Track] = [
    Track(
        id="Bahrain",
        label="Bahrain Circuit",
        vcp_file="Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy",
    ),
    Track(
        id="Hockolicious",
        label="ESL-Hockolicious",
        vcp_file="ESL-Hockolicious_0.5m_cl2.npy",
    ),
]


@app.get("/api/tracks", response_model=list[Track])
def get_tracks():
    return TRACKS


@app.post("/api/driver-profile", response_model=DriverProfile)
async def driver_profile(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    content = await file.read()
    result = await extract_driver_profile(content, file.filename)
    return DriverProfile(**result)


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    return GenerateResponse(
        status="queued",
        hyperparameters=req,
        message="Service 3 not yet connected. Hyperparameters accepted and ready for dispatch.",
    )


# Serve frontend — registered last so API routes take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
