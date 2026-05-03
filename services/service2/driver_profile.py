"""
Thin wrapper around linesight/scripts/extract_driver_profile.py.
Adds the scripts directory to sys.path so we import from the single source of truth.
"""

import sys
import tempfile
import os
from pathlib import Path

_env_scripts = os.environ.get("LINESIGHT_SCRIPTS_PATH")
_LINESIGHT_SCRIPTS = (
    Path(_env_scripts)
    if _env_scripts
    else Path(__file__).resolve().parent.parent.parent / "linesight" / "scripts"
)
if str(_LINESIGHT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_LINESIGHT_SCRIPTS))

from extract_driver_profile import extract_driver_profile as _extract  # noqa: E402


def run_extraction(file_bytes: bytes, filename: str) -> dict:
    """
    Save uploaded bytes to a temp file, run extraction, return the result dict.
    Raises ValueError on parse/format errors.
    """
    suffix = ".Replay.Gbx" if filename.lower().endswith(".replay.gbx") else Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = _extract(tmp_path)
    except SystemExit as exc:
        # extract_driver_profile calls sys.exit() on parse errors
        raise ValueError(str(exc)) from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse replay: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return result
