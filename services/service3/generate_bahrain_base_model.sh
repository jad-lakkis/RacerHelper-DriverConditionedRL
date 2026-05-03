#!/bin/bash
# =============================================================
# generate_bahrain_base_model.sh
# =============================================================
# ONE-OFF SCRIPT — generates the initial Bahrain base model.
#
# Purpose:
#   Run a clean training session from scratch on Bahrain Circuit.
#   When you are happy with the result (Ctrl+C to stop), copy the
#   5 output files into models/Bahrain/ so that the production
#   fine-tuning pipeline has a base to start from.
#
# This script is NOT part of the production pipeline.
# It is only used once (or whenever you want to refresh the base).
#
# Prerequisites:
#   - 1_setup_tmnf_docker.sh has been run (tmnf-vulkan:latest exists)
#   - The Bahrain .Challenge.Gbx file is on this host
#   - Run on the HOST (user@ubuntu), NOT inside Docker
#
# Usage:
#   chmod +x generate_bahrain_base_model.sh
#   ./generate_bahrain_base_model.sh
#
# Optional env vars:
#   RACER_HELPER_PATH   repo root (default: two dirs above this script)
#   BAHRAIN_GBX_PATH    path to Bahrain_Circuit.Challenge.Gbx on this host
#   GPU_COLLECTORS_COUNT number of parallel game instances (default: 1)
#
# After training — copy results into the repo:
#   docker cp tmnf:/home/wineuser/linesight/save/bahrain_base/weights1.torch         models/Bahrain/
#   docker cp tmnf:/home/wineuser/linesight/save/bahrain_base/weights2.torch         models/Bahrain/
#   docker cp tmnf:/home/wineuser/linesight/save/bahrain_base/optimizer1.torch       models/Bahrain/
#   docker cp tmnf:/home/wineuser/linesight/save/bahrain_base/scaler.torch           models/Bahrain/
#   docker cp tmnf:/home/wineuser/linesight/save/bahrain_base/accumulated_stats.joblib models/Bahrain/
# =============================================================

set -euo pipefail

log() { echo ""; echo "==== $1 ===="; }

# =============================================================
# Variable setup
# =============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RACER_HELPER_PATH="${RACER_HELPER_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LINESIGHT_PATH="$RACER_HELPER_PATH/linesight"

BAHRAIN_GBX_PATH="${BAHRAIN_GBX_PATH:-/home/user/maps/Bahrain_Circuit.Challenge.Gbx}"
VCP_FILE="Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"
RUN_NAME="bahrain_base"
TRACK_NAME="Bahrain"
MAP_BASENAME="Bahrain_Circuit.Challenge.Gbx"

if [ ! -d "$LINESIGHT_PATH" ]; then
    echo "[ERROR] linesight not found at: $LINESIGHT_PATH"
    echo "        Set RACER_HELPER_PATH to the repo root."
    exit 1
fi

if [ ! -f "$BAHRAIN_GBX_PATH" ]; then
    echo "[ERROR] Bahrain GBX not found at: $BAHRAIN_GBX_PATH"
    echo "        Set BAHRAIN_GBX_PATH or place the file there."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Bahrain Base Model — Training from Scratch"
echo "============================================================"
echo "  Repo root  : $RACER_HELPER_PATH"
echo "  Map GBX    : $BAHRAIN_GBX_PATH"
echo "  Run name   : $RUN_NAME  →  save/$RUN_NAME/"
echo "  VCP file   : $VCP_FILE"
echo "============================================================"

# =============================================================
# STEP 1: Allow X display access from Docker
# =============================================================
log "Step 1: Allowing X display access"
xhost +

# =============================================================
# STEP 2: Stop and remove any existing tmnf container
# =============================================================
log "Step 2: Cleaning up existing container"
docker stop tmnf 2>/dev/null || true
docker rm   tmnf 2>/dev/null || true

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
# STEP 4: Install liblzo2-dev
# =============================================================
log "Step 4: Installing liblzo2-dev"
docker exec -u 0 tmnf bash -c "apt-get update -qq && apt-get install -y liblzo2-dev"

# =============================================================
# STEP 5: Copy linesight from repo into container
# =============================================================
log "Step 5: Copying linesight from $LINESIGHT_PATH"
docker exec -u 0 tmnf bash -c "rm -rf /home/wineuser/linesight"
docker cp "$LINESIGHT_PATH" tmnf:/home/wineuser/linesight
docker exec -u 0 tmnf bash -c "chown -R wineuser:wineuser /home/wineuser/linesight"

# =============================================================
# STEP 6: Copy Bahrain map into the Wine Challenges directory
# =============================================================
log "Step 6: Copying Bahrain map into container"
WINE_CHALLENGES="/home/wineuser/.wine/drive_c/users/wineuser/Documents/TmForever/Tracks/Challenges"
docker exec -u 0 tmnf bash -c "mkdir -p '$WINE_CHALLENGES'"
docker cp "$BAHRAIN_GBX_PATH" "tmnf:$WINE_CHALLENGES/$MAP_BASENAME"
docker exec -u 0 tmnf bash -c "chown wineuser:wineuser '$WINE_CHALLENGES/$MAP_BASENAME'"

# =============================================================
# STEP 7: Install linesight + create launch_game.sh + user_config.py
# =============================================================
log "Step 7: Setting up linesight inside container"

docker exec -u wineuser tmnf bash -c '
set -euo pipefail

if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source $HOME/.local/bin/env

if [ ! -d /home/wineuser/linesight_env ]; then
    uv python install 3.11
    uv venv ~/linesight_env --python 3.11
fi
source ~/linesight_env/bin/activate

cd /home/wineuser/linesight
uv pip install -e .

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

if ! grep -q "is_linux" /home/wineuser/linesight/config_files/config.py; then
    echo "
import platform
is_linux = platform.system() == \"Linux\"
" >> /home/wineuser/linesight/config_files/config.py
fi

echo "Linesight setup complete."
'

# =============================================================
# STEP 8: Patch config.py — run_name, map_cycle, collectors
#         Driver conditioning params are left at config defaults.
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

# ── Scalar params ─────────────────────────────────────────────
for key, env_var, default in [
    ("gpu_collectors_count", "GPU_COLLECTORS_COUNT", "1"),
]:
    val = os.environ.get(env_var, default)
    content = re.sub(
        rf"^({re.escape(key)}\s*=\s*).*$",
        rf"\g<1>{val}",
        content,
        flags=re.MULTILINE,
    )

# ── run_name (string — must be quoted) ────────────────────────
run_name = os.environ.get("RUN_NAME", "bahrain_base")
content = re.sub(
    r'^(run_name\s*=\s*).*$',
    rf'\g<1>"{run_name}"',
    content,
    flags=re.MULTILINE,
)

# ── map_cycle ─────────────────────────────────────────────────
# Use repeat() format so analyze_map_cycle's chain(*map_cycle)
# receives iterables of tuples, not raw tuples.
track = "Bahrain"
wpath = "Bahrain_Circuit.Challenge.Gbx"
vcp   = "Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"

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

print(f"Config patched — run_name={run_name}, track={track}")
PYEOF

docker cp "$PATCH_SCRIPT" tmnf:/tmp/patch_config.py
rm "$PATCH_SCRIPT"

docker exec -u wineuser \
  -e RUN_NAME="$RUN_NAME" \
  -e GPU_COLLECTORS_COUNT="${GPU_COLLECTORS_COUNT:-1}" \
  tmnf bash -c "
    source /home/wineuser/.local/bin/env
    source /home/wineuser/linesight_env/bin/activate
    python /tmp/patch_config.py
    find /home/wineuser/linesight/config_files/__pycache__ -type f -delete 2>/dev/null || true
  "

# =============================================================
# STEP 9: Fix ownership
# =============================================================
log "Step 9: Fixing ownership"
docker exec -u 0 tmnf bash -c "
  if [ -d /home/wineuser/.triton ]; then
    chown -R wineuser:wineuser /home/wineuser/.triton
  fi
  chown -R wineuser:wineuser /home/wineuser
"

# =============================================================
# STEP 10: Create XDG runtime dir
# =============================================================
log "Step 10: Creating XDG runtime dir"
docker exec -u 0 tmnf bash -c "
  mkdir -p /tmp/runtime-wineuser
  chown wineuser:wineuser /tmp/runtime-wineuser
  chmod 700 /tmp/runtime-wineuser
"

# =============================================================
# STEP 11: Launch TMLoader
# =============================================================
log "Step 11: Launching TMLoader"
docker exec -d -u wineuser tmnf bash -c "
  export DISPLAY=:0
  export WINEPREFIX=/home/wineuser/.wine
  export HOME=/home/wineuser
  cd /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever
  wine TMLoader.exe 2>&1 >> /tmp/tmloader.log
"
echo "Waiting for TMLoader to initialise..."
sleep 5

# =============================================================
# STEP 12: Start training
# =============================================================
log "Step 12: Starting Linesight RL training (Bahrain — scratch)"
echo ""
echo "============================================================"
echo "  Bahrain Base Model Training"
echo "  Saves to: save/$RUN_NAME/ inside the container"
echo ""
echo "  Press Ctrl+C when satisfied with the result."
echo ""
echo "  Then copy weights to the repo with:"
echo "    docker cp tmnf:/home/wineuser/linesight/save/$RUN_NAME/weights1.torch         models/Bahrain/"
echo "    docker cp tmnf:/home/wineuser/linesight/save/$RUN_NAME/weights2.torch         models/Bahrain/"
echo "    docker cp tmnf:/home/wineuser/linesight/save/$RUN_NAME/optimizer1.torch       models/Bahrain/"
echo "    docker cp tmnf:/home/wineuser/linesight/save/$RUN_NAME/scaler.torch           models/Bahrain/"
echo "    docker cp tmnf:/home/wineuser/linesight/save/$RUN_NAME/accumulated_stats.joblib models/Bahrain/"
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
