import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import DriverProfile
from driver_profile import run_extraction

_raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8001")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

app = FastAPI(title="RacerHelper — Driver Profile Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-profile", response_model=DriverProfile)
async def extract_profile(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = run_extraction(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return DriverProfile(**result)
