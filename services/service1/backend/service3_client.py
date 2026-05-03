import os

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

# Set to the public IP/hostname of your vast.ai instance.
SERVICE3_URL = os.environ.get("SERVICE3_URL", "http://localhost:8003")


async def start_training(payload: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{SERVICE3_URL}/train", json=payload)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Training service (Service 3) is unreachable. Check SERVICE3_URL and make sure it is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Training service timed out.")

    if not response.is_success:
        detail = response.json().get("detail", f"Service 3 returned {response.status_code}.")
        raise HTTPException(status_code=502, detail=detail)

    return response.json()


async def get_job_status(job_id: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SERVICE3_URL}/train/{job_id}/status")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training service unreachable.")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Service 3 returned {response.status_code}.")

    return response.json()


async def stop_job(job_id: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{SERVICE3_URL}/train/{job_id}/stop")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training service unreachable.")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Service 3 returned {response.status_code}.")

    return response.json()


async def list_results(job_id: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{SERVICE3_URL}/train/{job_id}/result")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training service unreachable.")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Service 3 returned {response.status_code}.")

    return response.json()


async def stream_result_file(job_id: str, filename: str) -> StreamingResponse:
    try:
        client   = httpx.AsyncClient(timeout=60.0)
        response = await client.get(f"{SERVICE3_URL}/train/{job_id}/result/{filename}")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Training service unreachable.")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="File not found.")
    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Service 3 returned {response.status_code}.")

    return StreamingResponse(
        response.aiter_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
