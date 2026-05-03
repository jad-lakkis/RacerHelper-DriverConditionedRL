import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models import Track, DriverProfile, GenerateRequest, GenerateResponse
from service2_client import extract_driver_profile
from service3_client import (
    start_training,
    get_job_status,
    stop_job,
    list_results,
    stream_result_file,
)

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
    result  = await extract_driver_profile(content, file.filename)
    return DriverProfile(**result)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    result = await start_training(req.model_dump())
    return GenerateResponse(
        status=result.get("status", "pending"),
        job_id=result.get("job_id"),
        hyperparameters=req,
        message="Training job started on Service 3.",
    )


# ── Job proxy routes ──────────────────────────────────────────────────────────

@app.get("/api/job/{job_id}/status")
async def job_status(job_id: str):
    return await get_job_status(job_id)


@app.post("/api/job/{job_id}/stop")
async def job_stop(job_id: str):
    return await stop_job(job_id)


@app.get("/api/job/{job_id}/result")
async def job_results(job_id: str):
    return await list_results(job_id)


@app.get("/api/job/{job_id}/result/{filename}")
async def job_download(job_id: str, filename: str):
    return await stream_result_file(job_id, filename)


# Serve frontend — registered last so API routes take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
