import os

import httpx
from fastapi import HTTPException

# In Docker, SERVICE2_URL is set to http://service2:8002 via docker-compose env.
# Falls back to localhost for local development.
SERVICE2_URL = os.environ.get("SERVICE2_URL", "http://localhost:8002")


async def extract_driver_profile(file_bytes: bytes, filename: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{SERVICE2_URL}/extract-profile",
                files={"file": (filename, file_bytes, "application/octet-stream")},
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Driver profile service (Service 2) is unreachable. Make sure it is running on port 8002.",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Driver profile service timed out.")

    if response.status_code == 422:
        detail = response.json().get("detail", "Invalid replay file.")
        raise HTTPException(status_code=422, detail=detail)

    if not response.is_success:
        raise HTTPException(
            status_code=502,
            detail=f"Driver profile service returned {response.status_code}.",
        )

    return response.json()
