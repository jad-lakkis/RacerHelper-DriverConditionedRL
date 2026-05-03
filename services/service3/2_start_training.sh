#!/bin/bash
# =============================================================
# Script 2: 2_start_training.sh
# =============================================================
# Run this every time you want to start/resume training.
# Assumes Script 1 (1_setup_tmnf_docker.sh) has been run
# at least once and tmnf-vulkan:latest image exists.
#
# What this script does:
#   1.  Allows X display access from Docker (xhost +)
#   2.  Stops and removes any existing tmnf container
#   3.  Starts a fresh tmnf container with GPU + X display
#   4.  Installs liblzo2-dev system dependency
#   5.  Copies custom linesight from this repo into the container
#   6.  Copies the .Challenge.Gbx map into the container
#   7.  Installs linesight + dependencies inside the container
#   8.  Creates launch_game.sh and user_config.py
#   9.  Patches config.py with map path and driver hyperparameters
#  10.  Fixes ownership issues
#  11.  Creates XDG runtime dir
#  12.  Launches TMLoader (the game)
#  13.  Starts Linesight RL training
#
# Prerequisites:
#   - Script 1 has been run (tmnf-vulkan:latest exists)
#   - This repo (RacerHelper-DriverConditionedRL) is cloned on this machine
#   - Run on the HOST (user@ubuntu), NOT inside Docker
#   - KVM instance is running with DISPLAY=:0
#
# Usage:
#   ./2_start_training.sh <path/to/map.Challenge.Gbx>
#
# Environment variables (optional — all have defaults):
#   RACER_HELPER_PATH        Path to repo root (default: parent of this script's dir)
#   TRACK_NAME               Track name matching linesight config (default: Bahrain)
#   VCP_FILE                 VCP .npy filename in linesight/maps/ (auto-derived from TRACK_NAME)
#   GPU_COLLECTORS_COUNT     Game instances to run in parallel (default: 1)
#   BRAKING_AGGRESSION       [0,1]  (default: 0.5)
#   RISK_TOLERANCE           [0,1]  (default: 0.2)
#   OVERSTEER_UNDERSTEER_SCORE [-5,5] (default: 0.0)
#   CORNER_ENTRY_SPEED_RATIO [0,1]  (default: 0.84)
# =============================================================

set -euo pipefail

log() { echo ""; echo "==== $1 ===="; }

# =============================================================
# Argument & variable setup
# =============================================================
MAP_GBX="${1:?[ERROR] Usage: ./2_start_training.sh <path/to/map.Challenge.Gbx>}"

if [ ! -f "$MAP_GBX" ]; then
    echo "[ERROR] Map file not found: $MAP_GBX"
    exit 1
fi

# Default repo root: two levels above this script (services/service3/ → repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACER_HELPER_PATH="${RACER_HELPER_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LINESIGHT_PATH="$RACER_HELPER_PATH/linesight"

if [ ! -d "$LINESIGHT_PATH" ]; then
    echo "[ERROR] linesight not found at: $LINESIGHT_PATH"
    echo "        Set RACER_HELPER_PATH to the repo root."
    exit 1
fi

TRACK_NAME="${TRACK_NAME:-Bahrain}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
MAP_BASENAME=$(basename "$MAP_GBX")

# Derive VCP file from track name if not set explicitly
case "$TRACK_NAME" in
    "Bahrain")
        VCP_FILE="${VCP_FILE:-Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy}"
        ;;
    "Hockolicious")
        VCP_FILE="${VCP_FILE:-ESL-Hockolicious_0.5m_cl2.npy}"
        ;;
    *)
        if [ -z "${VCP_FILE:-}" ]; then
            echo "[ERROR] Unknown TRACK_NAME='$TRACK_NAME'. Set VCP_FILE manually."
            exit 1
        fi
        ;;
esac

echo ""
echo "============================================================"
echo "  RacerHelper Training — Configuration"
echo "============================================================"
echo "  Repo root    : $RACER_HELPER_PATH"
echo "  Run name     : $RUN_NAME"
echo "  Map GBX      : $MAP_GBX"
echo "  Track        : $TRACK_NAME"
echo "  VCP file     : $VCP_FILE"
echo "  Braking agg  : ${BRAKING_AGGRESSION:-0.5}"
echo "  Risk tol     : ${RISK_TOLERANCE:-0.2}"
echo "  OUS score    : ${OVERSTEER_UNDERSTEER_SCORE:-0.0}"
echo "  Corner speed : ${CORNER_ENTRY_SPEED_RATIO:-0.84}"
echo "============================================================"

# =============================================================
# STEP 1: Allow X display access from Docker
# =============================================================
log "Step 1: Allowing X display access"
xhost +
echo "xhost + done."

# =============================================================
# STEP 2: Stop and remove any existing tmnf container
# =============================================================
log "Step 2: Cleaning up existing container"
docker stop tmnf 2>/dev/null || true
docker rm   tmnf 2>/dev/null || true
echo "Old container removed (if any)."

# =============================================================
# STEP 3: Start the tmnf container
# =============================================================
log "Step 3: Starting tmnf container"
docker run --gpus all -d \
  --name tmnf \
  -e DISPLAY=:0 \
  -e WINEPREFIX=/home/wineuser/.wine \
  -e HOME=/home/wineuser \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro \
  --entrypoint bash \
  tmnf-vulkan:latest -c "
    while true; do
      wine /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/TmForever.exe 2>&1 | tee -a /tmp/tmnf.log
      echo 'Exited \$? - restarting in 3s'
      sleep 3
    done
  "
echo "Container started. Waiting for initialization..."
sleep 3

# =============================================================
# STEP 4: Install system dependency liblzo2-dev
# =============================================================
log "Step 4: Installing liblzo2-dev inside container"
docker exec -u 0 tmnf bash -c "apt-get update -qq && apt-get install -y liblzo2-dev"
echo "liblzo2-dev installed."

# =============================================================
# STEP 5: Copy custom linesight from this repo into the container
# This replaces the upstream git clone — we use our own fork.
# =============================================================
log "Step 5: Copying linesight from $LINESIGHT_PATH into container"
docker exec -u 0 tmnf bash -c "rm -rf /home/wineuser/linesight"
docker cp "$LINESIGHT_PATH" tmnf:/home/wineuser/linesight
docker exec -u 0 tmnf bash -c "chown -R wineuser:wineuser /home/wineuser/linesight"
echo "linesight copied."

# =============================================================
# STEP 6: Copy the .Challenge.Gbx map into the container
# Placed in the Wine TmForever Challenges directory so the
# game can load it and config.py can reference it by name.
# =============================================================
log "Step 6: Copying map '$MAP_BASENAME' into container"
WINE_CHALLENGES="/home/wineuser/.wine/drive_c/users/wineuser/Documents/TmForever/Tracks/Challenges"
docker exec -u 0 tmnf bash -c "mkdir -p '$WINE_CHALLENGES'"
docker cp "$MAP_GBX" "tmnf:$WINE_CHALLENGES/$MAP_BASENAME"
docker exec -u 0 tmnf bash -c "chown wineuser:wineuser '$WINE_CHALLENGES/$MAP_BASENAME'"
echo "Map copied to Wine Challenges."

# =============================================================
# STEP 7: Install linesight + create launch_game.sh + user_config.py
# =============================================================
log "Step 7: Setting up linesight inside container (as wineuser)"

docker exec -u wineuser tmnf bash -c '
set -euo pipefail

# --- Install uv ---
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source $HOME/.local/bin/env

# --- Python 3.11 venv ---
if [ ! -d /home/wineuser/linesight_env ]; then
    uv python install 3.11
    uv venv ~/linesight_env --python 3.11
fi
source ~/linesight_env/bin/activate

# --- Install linesight from the copied source ---
cd /home/wineuser/linesight
uv pip install -e .

# --- Create launch_game.sh ---
cat > /home/wineuser/linesight/scripts/launch_game.sh << '"'"'EOF'"'"'
#!/bin/bash
export DISPLAY=:0
export WINEARCH=win32
export WINEPREFIX=/home/wineuser/.wine
export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export WINEDLLOVERRIDES="d3d9=n;d3d11=n;dxgi=n;d3d10core=n"

mkdir -p "$XDG_RUNTIME_DIR"

exec wine /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/TMLoader.exe run TmForever "default" /configstring="set custom_port $1"
EOF
chmod +x /home/wineuser/linesight/scripts/launch_game.sh

# --- Create user_config.py ---
cat > /home/wineuser/linesight/config_files/user_config.py << '"'"'PYEOF'"'"'
from pathlib import Path
import os
import platform

username = "default"
_wine_docs = Path("/home/wineuser/.wine/drive_c/users/wineuser/Documents")
target_python_link_path = _wine_docs / "TMInterface" / "Plugins" / "Python_Link.as"
trackmania_base_path = _wine_docs / "TmForever"
base_tmi_port = 8478
linux_launch_game_path = "/home/wineuser/linesight/scripts/launch_game.sh"
windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
windows_TMLoader_profile_name = "default"
is_linux = platform.system() == "Linux"
PYEOF

# --- Add is_linux guard to config.py if missing ---
if ! grep -q "is_linux" /home/wineuser/linesight/config_files/config.py; then
    echo "
import platform
is_linux = platform.system() == \"Linux\"
" >> /home/wineuser/linesight/config_files/config.py
fi

echo "Linesight setup complete."
'

# =============================================================
# STEP 8: Patch config.py with map path and driver hyperparams
# A small Python script handles the multi-line map_cycle block
# cleanly — sed would be fragile here.
# =============================================================
log "Step 8: Patching config.py"

PATCH_SCRIPT=$(mktemp /tmp/patch_config_XXXXXX.py)
cat > "$PATCH_SCRIPT" << 'PYEOF'
import os
import re

CONFIG      = "/home/wineuser/linesight/config_files/config.py"
CONFIG_COPY = "/home/wineuser/linesight/config_files/config_copy.py"

with open(CONFIG) as f:
    content = f.read()

# ── Scalar driver conditioning params ────────────────────────
# Numeric scalar params
for key, env_var, default in [
    ("braking_aggression",         "BRAKING_AGGRESSION",         "0.5"),
    ("risk_tolerance",             "RISK_TOLERANCE",             "0.2"),
    ("oversteer_understeer_score", "OVERSTEER_UNDERSTEER_SCORE", "0.0"),
    ("corner_entry_speed_ratio",   "CORNER_ENTRY_SPEED_RATIO",   "0.84"),
    ("gpu_collectors_count",       "GPU_COLLECTORS_COUNT",       "1"),
]:
    val = os.environ.get(env_var, default)
    content = re.sub(
        rf"^({re.escape(key)}\s*=\s*).*$",
        rf"\g<1>{val}",
        content,
        flags=re.MULTILINE,
    )

# String param — must be quoted in the config file
run_name = os.environ.get("RUN_NAME", "default_run")
content = re.sub(
    r'^(run_name\s*=\s*).*$',
    rf'\g<1>"{run_name}"',
    content,
    flags=re.MULTILINE,
)

# ── map_cycle ─────────────────────────────────────────────────
# Use repeat() format so analyze_map_cycle's chain(*map_cycle)
# receives iterables of tuples, not raw tuples.
track = os.environ.get("TRACK_NAME",    "map")
wpath = os.environ.get("MAP_WINE_PATH", "current_map.Challenge.Gbx")
vcp   = os.environ.get("VCP_FILE",      "")

new_cycle = (
    "map_cycle = [\n"
    f'    repeat(("{track}", \'"My Challenges/{wpath}"\', "{vcp}", True,  True), 4),\n'
    f'    repeat(("{track}", \'"My Challenges/{wpath}"\', "{vcp}", False, True), 1),\n'
    "]"
)
# \s* before \] handles indented closing brackets
content = re.sub(
    r"map_cycle\s*=\s*\[.*?\n\s*\]",
    new_cycle,
    content,
    flags=re.DOTALL,
)

for path in (CONFIG, CONFIG_COPY):
    with open(path, "w") as f:
        f.write(content)

print(f"Config patched — run_name={run_name}, track={track}, map={wpath}, vcp={vcp}")
PYEOF

docker cp "$PATCH_SCRIPT" tmnf:/tmp/patch_config.py
rm "$PATCH_SCRIPT"

docker exec -u wineuser \
  -e RUN_NAME="$RUN_NAME" \
  -e BRAKING_AGGRESSION="${BRAKING_AGGRESSION:-0.5}" \
  -e RISK_TOLERANCE="${RISK_TOLERANCE:-0.2}" \
  -e OVERSTEER_UNDERSTEER_SCORE="${OVERSTEER_UNDERSTEER_SCORE:-0.0}" \
  -e CORNER_ENTRY_SPEED_RATIO="${CORNER_ENTRY_SPEED_RATIO:-0.84}" \
  -e GPU_COLLECTORS_COUNT="${GPU_COLLECTORS_COUNT:-1}" \
  -e TRACK_NAME="$TRACK_NAME" \
  -e MAP_WINE_PATH="$MAP_BASENAME" \
  -e VCP_FILE="$VCP_FILE" \
  tmnf bash -c "
    source /home/wineuser/.local/bin/env
    source /home/wineuser/linesight_env/bin/activate
    python /tmp/patch_config.py
    find /home/wineuser/linesight/config_files/__pycache__ -type f -delete 2>/dev/null || true
  "

# =============================================================
# STEP 9: Copy base model weights into container
# The models/ folder in the repo holds one pair of weights per
# track. Training fine-tunes from these rather than starting
# from a random initialisation.
# =============================================================
log "Step 9: Loading base model snapshot for track '$TRACK_NAME' → save/$RUN_NAME/"
MODELS_PATH="${MODELS_PATH:-$RACER_HELPER_PATH/models}"
BASE_MODEL_DIR="$MODELS_PATH/$TRACK_NAME"
SAVE_SUBDIR="/home/wineuser/linesight/save/$RUN_NAME"

# The full snapshot: weights, optimizer, scaler, accumulated_stats
BASE_MODEL_FILES=(
    "weights1.torch"
    "weights2.torch"
    "optimizer1.torch"
    "scaler.torch"
    "accumulated_stats.joblib"
)

all_present=true
for f in "${BASE_MODEL_FILES[@]}"; do
    if [ ! -f "$BASE_MODEL_DIR/$f" ]; then
        all_present=false
        break
    fi
done

if $all_present; then
    docker exec -u 0 tmnf bash -c "mkdir -p '$SAVE_SUBDIR'"
    for f in "${BASE_MODEL_FILES[@]}"; do
        docker cp "$BASE_MODEL_DIR/$f" "tmnf:$SAVE_SUBDIR/$f"
    done
    docker exec -u 0 tmnf bash -c "chown -R wineuser:wineuser '$SAVE_SUBDIR'"
    echo "Base model snapshot loaded into $SAVE_SUBDIR"
else
    docker exec -u 0 tmnf bash -c "mkdir -p '$SAVE_SUBDIR'"
    echo "[WARN] Incomplete snapshot at $BASE_MODEL_DIR — starting from random initialisation."
fi

# =============================================================
# STEP 10: Fix ownership issues
# =============================================================
log "Step 10: Fixing ownership"
docker exec -u 0 tmnf bash -c "
  if [ -d /home/wineuser/.triton ]; then
    chown -R wineuser:wineuser /home/wineuser/.triton
  fi
  chown -R wineuser:wineuser /home/wineuser
"

# =============================================================
# STEP 10: Create XDG runtime dir
# =============================================================
log "Step 11: Creating XDG runtime dir"
docker exec -u 0 tmnf bash -c "
  mkdir -p /tmp/runtime-wineuser
  chown wineuser:wineuser /tmp/runtime-wineuser
  chmod 700 /tmp/runtime-wineuser
"

# =============================================================
# STEP 11: Launch TMLoader
# =============================================================
log "Step 12: Launching TMLoader"
docker exec -d -u wineuser tmnf bash -c "
  export DISPLAY=:0
  export WINEPREFIX=/home/wineuser/.wine
  export HOME=/home/wineuser
  cd /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever
  wine TMLoader.exe 2>&1 >> /tmp/tmloader.log
"
echo "TMLoader launched. Waiting for it to initialize..."
sleep 5

# =============================================================
# STEP 12: Start Linesight RL training
# =============================================================
log "Step 13: Starting Linesight RL training"
echo ""
echo "============================================================"
echo "  Linesight RL Training Starting"
echo "============================================================"
echo "  IMPORTANT: Check the VNC/web desktop."
echo "  If there is a 'Stay offline' popup, click it to dismiss."
echo ""
echo "  Monitoring commands (new terminal):"
echo "    nvidia-smi"
echo "    docker exec tmnf cat /tmp/tmnf.log"
echo "    docker exec tmnf cat /tmp/tmloader.log"
echo ""
echo "  Press Ctrl+C to stop training."
echo "============================================================"
echo ""

docker exec -it -u wineuser tmnf bash -c "
  source /home/wineuser/.local/bin/env
  source /home/wineuser/linesight_env/bin/activate
  export DISPLAY=:0
  export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
  cd /home/wineuser/linesight
  python scripts/train.py
"
