#!/bin/bash
# =============================================================
# record_replay.sh
# =============================================================
# Replays a .inputs file inside the tmnf container via
# TMInterface and captures the resulting .Replay.Gbx.
#
# Usage:
#   ./record_replay.sh <inputs_path_on_host> <output_replay_path_on_host>
#
# Returns exit code 0 on success, non-zero on failure.
#
# TODO: Implement the TMInterface replay-to-GBX recording.
#
# The implementation should:
#   1. Copy the .inputs file into the container
#   2. docker exec a Python script that uses the TMInterface
#      socket API to replay the inputs at 1x speed with
#      ghost recording enabled
#   3. Wait for the .Replay.Gbx to appear in the game's
#      replays directory:
#        /home/wineuser/.wine/drive_c/users/wineuser/Documents/
#        TmForever/Replays/
#   4. docker cp the .Replay.Gbx back to the host output path
#
# Reference: linesight/trackmania_rl/tmi_interaction/
#            game_instance_manager.py  (rollout logic)
#            linesight/scripts/run_to_video.py (replay example)
# =============================================================

set -euo pipefail

INPUTS_PATH="${1:?Usage: ./record_replay.sh <inputs_path> <output_gbx_path>}"
OUTPUT_GBX="${2:?Usage: ./record_replay.sh <inputs_path> <output_gbx_path>}"

echo "[record_replay] TODO: TMInterface replay recording not yet implemented."
echo "[record_replay] inputs: $INPUTS_PATH"
echo "[record_replay] output: $OUTPUT_GBX"

# Signal failure so the caller knows no GBX was produced
exit 1
