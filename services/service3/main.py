import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from models import TrainResponse, StatusResponse, ResultsResponse

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
JOBS_DIR    = Path(os.environ.get("JOBS_DIR", "/tmp/racerhelper_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config from environment ───────────────────────────────────────────────────
EPSILON              = float(os.environ.get("EPSILON", "0.15"))
POLL_INTERVAL_S      = int(os.environ.get("POLL_INTERVAL_MINUTES", "10")) * 60
SERVICE2_PUBLIC_URL  = os.environ.get("SERVICE2_PUBLIC_URL", "http://localhost:8002")

_GBX_PATHS = {
    "Bahrain":     os.environ.get("BAHRAIN_GBX_PATH",     "/home/user/maps/Bahrain_Circuit.Challenge.Gbx"),
    "Hockolicious": os.environ.get("HOCKOLICIOUS_GBX_PATH", "/home/user/maps/ESL-Hockolicious.Challenge.Gbx"),
}

_raw_origins = os.environ.get("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RacerHelper — Training Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry (sufficient for a single vast.ai instance)
jobs: dict[str, dict] = {}


# ── Request schema ────────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    track: str
    best_lap_ms: int = Field(..., gt=0)
    risk_tolerance: float = Field(..., ge=0.0, le=1.0)
    braking_aggression: float = Field(..., ge=0.0, le=1.0)
    oversteer_understeer_score: float = Field(..., ge=-5.0, le=5.0)
    corner_entry_speed_ratio: float = Field(..., ge=0.0, le=1.0)
    max_training_hours: float = Field(default=2.0, gt=0.0, le=24.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train", response_model=TrainResponse)
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    gbx_path = _GBX_PATHS.get(req.track)
    if not gbx_path:
        raise HTTPException(status_code=422, detail=f"Unknown track: '{req.track}'.")
    if not Path(gbx_path).exists():
        raise HTTPException(
            status_code=422,
            detail=f"Map file not found on host: {gbx_path}. Set {req.track.upper()}_GBX_PATH in env.",
        )

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    jobs[job_id] = {
        "job_id":        job_id,
        "status":        "pending",
        "track":         req.track,
        "best_lap_ms":   req.best_lap_ms,
        "max_hours":     req.max_training_hours,
        "user_profile":  {
            "braking_aggression":         req.braking_aggression,
            "oversteer_understeer_score": req.oversteer_understeer_score,
            "corner_entry_speed_ratio":   req.corner_entry_speed_ratio,
        },
        "process":       None,
        "output_dir":    str(job_dir / "results"),
        "result_replay": None,
        "best_race_ms":  None,
        "last_distance": None,
        "message":       None,
        "error":         None,
        "start_time":    None,
    }

    background_tasks.add_task(_run_training, job_id, gbx_path, req.model_dump(), job_dir)
    return TrainResponse(job_id=job_id, status="pending")


@app.post("/train/{job_id}/stop")
async def stop_training(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail=f"Job is not running (status: {job['status']}).")

    await _terminate_process(job_id)
    job_dir = JOBS_DIR / job_id
    await _collect_results(job_id, job_dir)
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
        last_distance=job.get("last_distance"),
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
    return FileResponse(str(matches[0]), filename=filename, media_type="application/octet-stream")


# ── Training orchestration ────────────────────────────────────────────────────

async def _run_training(job_id: str, gbx_path: str, params: dict, job_dir: Path):
    env = {
        **os.environ,
        "TRACK_NAME":                params["track"],
        "RUN_NAME":                  job_id,          # each job gets its own save/ subdirectory
        "BRAKING_AGGRESSION":        str(params["braking_aggression"]),
        "RISK_TOLERANCE":            str(params["risk_tolerance"]),
        "OVERSTEER_UNDERSTEER_SCORE":str(params["oversteer_understeer_score"]),
        "CORNER_ENTRY_SPEED_RATIO":  str(params["corner_entry_speed_ratio"]),
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

        # Run training and monitor concurrently; stop whichever wins
        training_task = asyncio.create_task(proc.communicate())
        monitor_task  = asyncio.create_task(_monitor_training(job_id, params, job_dir))

        done, pending = await asyncio.wait(
            {training_task, monitor_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

        # If training ended naturally (e.g. crash), collect what we have
        if jobs[job_id]["status"] == "running":
            stdout, _ = training_task.result() if training_task in done else (b"", None)
            if proc.returncode == 0:
                await _collect_results(job_id, job_dir)
                jobs[job_id]["status"] = "completed"
            else:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"]  = (stdout.decode(errors="replace")[-2000:] if stdout else "Unknown error")

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"]  = str(exc)


# ── Monitor loop ──────────────────────────────────────────────────────────────

async def _monitor_training(job_id: str, params: dict, job_dir: Path):
    """
    Polls the container every POLL_INTERVAL_MINUTES for new saved runs.
    Stopping conditions:
      1. A run beats best_lap_ms AND its driver profile is within EPSILON.
      2. max_training_hours is exceeded → timeout with best-so-far as fallback.
    """
    best_lap_ms  = params["best_lap_ms"]
    max_seconds  = params["max_training_hours"] * 3600
    user_profile = {
        "braking_aggression":         params["braking_aggression"],
        "oversteer_understeer_score":  params["oversteer_understeer_score"],
        "corner_entry_speed_ratio":    params["corner_entry_speed_ratio"],
    }
    known_runs: set[str] = set()

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
            if best is not None:
                jobs[job_id]["status"]  = "timeout"
                jobs[job_id]["message"] = (
                    f"Need more time. Best lap so far: {best} ms "
                    f"(target: {best_lap_ms} ms). Returning best result."
                )
            else:
                jobs[job_id]["status"]  = "timeout"
                jobs[job_id]["message"] = (
                    "Need more time. No complete lap found within the training window."
                )
            return

        # ── Scan for new good runs ────────────────────────────────────────────
        new_runs = await _scan_container_runs(known_runs, best_lap_ms, job_id)
        for container_path, race_time_ms in new_runs:
            known_runs.add(container_path)

            # Track overall best
            if jobs[job_id]["best_race_ms"] is None or race_time_ms < jobs[job_id]["best_race_ms"]:
                jobs[job_id]["best_race_ms"] = race_time_ms

            # Attempt replay → GBX → Service 2 profile check
            local_inputs = job_dir / Path(container_path).name
            await _docker_cp_from_container(container_path, str(local_inputs))

            replay_gbx = await _record_replay_gbx(str(local_inputs), job_dir)
            if replay_gbx is None:
                # TMInterface recording not yet implemented — skip EPSILON check
                continue

            agent_profile = await _call_service2(replay_gbx)
            if agent_profile is None:
                continue

            distance = _profile_distance(user_profile, agent_profile)
            jobs[job_id]["last_distance"] = round(distance, 4)

            if distance < EPSILON:
                # Style matches — copy replay to results and stop
                await _terminate_process(job_id)
                results_dir = Path(jobs[job_id]["output_dir"])
                results_dir.mkdir(exist_ok=True)
                import shutil
                shutil.copy2(str(replay_gbx), str(results_dir / replay_gbx.name))
                jobs[job_id]["result_replay"] = replay_gbx.name
                jobs[job_id]["status"]        = "completed"
                jobs[job_id]["message"]       = (
                    f"Style match found. Race time: {race_time_ms} ms, "
                    f"profile distance: {distance:.4f} (ε={EPSILON})."
                )
                return


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _scan_container_runs(known_runs: set, best_lap_ms: int, job_id: str) -> list[tuple[str, int]]:
    """List .inputs files in the container's save/{job_id}/best_runs dir with race_time < best_lap_ms."""
    save_subdir = f"/home/wineuser/linesight/save/{job_id}/best_runs"
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "tmnf",
        "find", save_subdir,
        "-name", "*.inputs", "-type", "f",
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
        # Filename format: {map_name}_{race_time_ms}_{MMDD_HHMMSS}_{frames}_{explo|eval}.inputs
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


async def _record_replay_gbx(inputs_host_path: str, job_dir: Path) -> Optional[Path]:
    """
    Run the .inputs through TMInterface to produce a .Replay.Gbx.
    Returns the path to the .Replay.Gbx on the host, or None if recording fails.
    See record_replay.sh for the TODO implementation.
    """
    output_gbx = job_dir / (Path(inputs_host_path).stem + ".Replay.Gbx")
    script = str(SCRIPTS_DIR / "record_replay.sh")
    proc = await asyncio.create_subprocess_exec(
        "bash", script, inputs_host_path, str(output_gbx),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()
    return output_gbx if output_gbx.exists() else None


async def _call_service2(replay_gbx: Path) -> Optional[dict]:
    """Send the .Replay.Gbx to Service 2 and return the extracted driver profile."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(replay_gbx, "rb") as f:
                response = await client.post(
                    f"{SERVICE2_PUBLIC_URL}/extract-profile",
                    files={"file": (replay_gbx.name, f, "application/octet-stream")},
                )
        if response.is_success:
            return response.json()
    except Exception:
        pass
    return None


def _profile_distance(user: dict, agent: dict) -> float:
    """Max absolute difference across the three driver metrics, all normalised to [0, 1]."""
    delta_braking = abs(agent["braking_aggression"]         - user["braking_aggression"])
    delta_ous     = abs(agent["oversteer_understeer_score"] - user["oversteer_understeer_score"]) / 10.0
    delta_corner  = abs(agent["corner_entry_speed_ratio"]   - user["corner_entry_speed_ratio"])
    return max(delta_braking, delta_ous, delta_corner)


async def _collect_results(job_id: str, job_dir: Path):
    results_dir = job_dir / "results"
    results_dir.mkdir(exist_ok=True)
    # Copy the job's own save subdirectory (RUN_NAME == job_id)
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", f"tmnf:/home/wineuser/linesight/save/{job_id}/.", str(results_dir),
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
