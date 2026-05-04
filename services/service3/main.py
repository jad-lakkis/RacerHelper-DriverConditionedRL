#!/usr/bin/env python3
"""
=============================================================
main.py — RacerHelper Training Service (FastAPI)
=============================================================
Runs on the Vast.ai KVM instance alongside the tmnf Docker
container. Accepts training requests, launches 2_start_training.sh,
monitors for completed runs, and returns results.

No epsilon check. No Service 2 communication.

Start with:
    pip install fastapi uvicorn httpx
    /home/user/.local/bin/uvicorn main:app --host 0.0.0.0 --port 8003


Environment variables (all optional):
    POLL_INTERVAL_MINUTES   How often to check for new runs (default: 10)
    BAHRAIN_GBX_PATH        Path to Bahrain map (default below)
    HOCKOLICIOUS_GBX_PATH   Path to Hockolicious map (default below)
    JOBS_DIR                Where job output is stored (default: /tmp/racerhelper_jobs)
=============================================================
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
JOBS_DIR    = Path(os.environ.get("JOBS_DIR", "/tmp/racerhelper_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL_S = int(os.environ.get("POLL_INTERVAL_MINUTES", "10")) * 60

_GBX_PATHS = {
    "Bahrain": os.environ.get(
        "BAHRAIN_GBX_PATH",
        "/home/user/RacerHelper-DriverConditionedRL/linesight/custom_maps/Bahrain_Circuit.Challenge.Gbx",
    ),
    "Hockolicious": os.environ.get(
        "HOCKOLICIOUS_GBX_PATH",
        "/home/user/maps/ESL-Hockolicious.Challenge.Gbx",
    ),
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RacerHelper — Training Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry
jobs: dict[str, dict] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    track: str
    best_lap_ms: int = Field(..., gt=0)
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    braking_aggression: float = Field(..., ge=0.0, le=1.0)
    oversteer_understeer_score: float = Field(..., ge=-5.0, le=5.0)
    corner_entry_speed_ratio: float = Field(..., ge=0.0, le=1.0)
    max_training_hours: float = Field(default=2.0, gt=0.0, le=24.0)


class TrainResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    results_count: int = 0
    best_race_ms: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None


class ResultsResponse(BaseModel):
    files: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    gbx_path = _GBX_PATHS.get(req.track)
    if not gbx_path:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown track: '{req.track}'. Available: {list(_GBX_PATHS.keys())}",
        )
    if not Path(gbx_path).exists():
        raise HTTPException(
            status_code=422,
            detail=f"Map file not found on host: {gbx_path}",
        )

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    jobs[job_id] = {
        "job_id":      job_id,
        "status":      "pending",
        "track":       req.track,
        "best_lap_ms": req.best_lap_ms,
        "max_hours":   req.max_training_hours,
        "process":     None,
        "output_dir":  str(job_dir / "results"),
        "best_race_ms": None,
        "message":     None,
        "error":       None,
        "start_time":  None,
    }

    background_tasks.add_task(_run_training, job_id, gbx_path, req.model_dump(), job_dir)
    return TrainResponse(job_id=job_id, status="pending")


@app.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not running (status: {job['status']}).",
        )
    await _terminate_process(job_id)
    await _collect_results(job_id, job_dir=JOBS_DIR / job_id)
    jobs[job_id]["status"] = "stopped"
    return {"job_id": job_id, "status": "stopped"}


@app.get("/train/{job_id}/status", response_model=StatusResponse)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job         = jobs[job_id]
    results_dir = Path(job["output_dir"])
    count       = len(list(results_dir.glob("**/*.inputs"))) if results_dir.exists() else 0
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        results_count=count,
        best_race_ms=job.get("best_race_ms"),
        message=job.get("message"),
        error=job.get("error"),
    )


@app.get("/train/{job_id}/result", response_model=ResultsResponse)
def list_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    results_dir = Path(jobs[job_id]["output_dir"])
    if not results_dir.exists():
        return ResultsResponse(files=[])
    files = sorted(
        f.name for f in results_dir.glob("**/*")
        if f.suffix in (".inputs", ".Gbx", ".gbx")
    )
    return ResultsResponse(files=files)


@app.get("/train/{job_id}/result/{filename}")
def download_result(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    results_dir = Path(jobs[job_id]["output_dir"])
    matches     = list(results_dir.glob(f"**/{filename}"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    return FileResponse(
        str(matches[0]),
        filename=filename,
        media_type="application/octet-stream",
    )


# ── Training orchestration ────────────────────────────────────────────────────

async def _run_training(job_id: str, gbx_path: str, params: dict, job_dir: Path):
    env = {
        **os.environ,
        "TRACK_NAME":                 params["track"],
        "RUN_NAME":                   job_id,
        "BRAKING_AGGRESSION":         str(params["braking_aggression"]),
        "RISK_TOLERANCE":             str(params["risk_tolerance"]),
        "OVERSTEER_UNDERSTEER_SCORE": str(params["oversteer_understeer_score"]),
        "CORNER_ENTRY_SPEED_RATIO":   str(params["corner_entry_speed_ratio"]),
    }
    script = str(SCRIPTS_DIR / "2_start_training.sh")

    jobs[job_id]["status"]     = "running"
    jobs[job_id]["start_time"] = time.monotonic()

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", script, gbx_path,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        jobs[job_id]["process"] = proc

        # Run training and monitor concurrently; stop whichever finishes first
        training_task = asyncio.create_task(proc.communicate())
        monitor_task  = asyncio.create_task(
            _monitor_training(job_id, params, job_dir)
        )

        done, pending = await asyncio.wait(
            {training_task, monitor_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        # If training ended on its own (crash/natural end)
        if jobs[job_id]["status"] == "running":
            stdout, _ = training_task.result() if training_task in done else (b"", None)
            if proc.returncode == 0:
                await _collect_results(job_id, job_dir)
                jobs[job_id]["status"] = "completed"
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"]  = (
                    stdout.decode(errors="replace")[-2000:] if stdout else "Unknown error"
                )

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(exc)


# ── Monitor loop ──────────────────────────────────────────────────────────────

async def _monitor_training(job_id: str, params: dict, job_dir: Path):
    """
    Polls the container every POLL_INTERVAL_MINUTES for new saved runs.
    Stopping conditions:
      1. A run beats best_lap_ms → collect results and mark completed.
      2. max_training_hours exceeded → timeout with best-so-far.
    No epsilon check. No Service 2 calls.
    """
    best_lap_ms  = params["best_lap_ms"]
    max_seconds  = params["max_training_hours"] * 3600
    known_runs:  set[str] = set()

    while jobs[job_id]["status"] == "running":
        await asyncio.sleep(POLL_INTERVAL_S)

        if jobs[job_id]["status"] != "running":
            return

        elapsed = time.monotonic() - (jobs[job_id]["start_time"] or time.monotonic())

        # ── Timeout ───────────────────────────────────────────────────────────
        if elapsed >= max_seconds:
            await _terminate_process(job_id)
            await _collect_results(job_id, job_dir)
            best = jobs[job_id]["best_race_ms"]
            jobs[job_id]["status"]  = "timeout"
            jobs[job_id]["message"] = (
                f"Training time limit reached. "
                + (f"Best lap so far: {best} ms (target: {best_lap_ms} ms)." if best else "No complete lap found.")
            )
            return

        # ── Scan for new runs that beat the target ────────────────────────────
        new_runs = await _scan_container_runs(known_runs, best_lap_ms, job_id)
        for container_path, race_time_ms in new_runs:
            known_runs.add(container_path)

            # Track overall best
            if jobs[job_id]["best_race_ms"] is None or race_time_ms < jobs[job_id]["best_race_ms"]:
                jobs[job_id]["best_race_ms"] = race_time_ms

            # Copy the .inputs file to results
            results_dir = Path(jobs[job_id]["output_dir"])
            results_dir.mkdir(exist_ok=True)
            local_path = str(results_dir / Path(container_path).name)
            await _docker_cp_from_container(container_path, local_path)

        # If we found at least one run beating the target, stop
        if new_runs:
            await _terminate_process(job_id)
            await _collect_results(job_id, job_dir)
            jobs[job_id]["status"]  = "completed"
            jobs[job_id]["message"] = (
                f"Found run beating target. "
                f"Best lap: {jobs[job_id]['best_race_ms']} ms (target: {best_lap_ms} ms)."
            )
            return


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _scan_container_runs(
    known_runs: set, best_lap_ms: int, job_id: str
) -> list[tuple[str, int]]:
    """
    List .inputs files in the container's save/{job_id}/best_runs dir
    with race_time < best_lap_ms.
    Filename format: {map_name}_{race_time_ms}_{timestamp}_{frames}_{explo|eval}.inputs
    """
    save_subdir = f"/home/wineuser/linesight/save/{job_id}/best_runs"
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "tmnf",
        "find", save_subdir, "-name", "*.inputs", "-type", "f",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if not stdout:
        return []

    results = []
    for line in stdout.decode().splitlines():
        path = line.strip()
        if not path or path in known_runs:
            continue
        parts = Path(path).stem.split("_")
        if len(parts) < 2:
            continue
        try:
            race_time = int(parts[1])
        except ValueError:
            continue
        if race_time < best_lap_ms:
            results.append((path, race_time))

    return results


async def _docker_cp_from_container(container_path: str, host_path: str):
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", f"tmnf:{container_path}", host_path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()


async def _collect_results(job_id: str, job_dir: Path):
    """Copy the job's entire save subdirectory out of the container."""
    results_dir = job_dir / "results"
    results_dir.mkdir(exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp",
        f"tmnf:/home/wineuser/linesight/save/{job_id}/.",
        str(results_dir),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()


async def _terminate_process(job_id: str):
    proc = jobs[job_id].get("process")
    if proc and proc.returncode is None:
        proc.terminate()
        await asyncio.sleep(3)
        if proc.returncode is None:
            proc.kill()