#!/bin/bash
# =============================================================
# generate_bahrain_base_model.sh
# =============================================================
# ONE-OFF SCRIPT — generates the initial Bahrain base model.
#
# First run (full setup, ~10-15 min):
#   ./generate_bahrain_base_model.sh
#
# Subsequent runs (fast path, ~1-2 min — reuses existing container):
#   ./generate_bahrain_base_model.sh
#
# Force a full rebuild (new container + reinstall everything):
#   FRESH=1 ./generate_bahrain_base_model.sh
#
# Optional env vars:
#   FRESH                   1 = full rebuild, 0 = reuse container (default: 0)
#   RACER_HELPER_PATH       repo root (default: two dirs above this script)
#   BAHRAIN_GBX_PATH        path to Bahrain_Circuit.Challenge.Gbx on this host
#   GPU_COLLECTORS_COUNT    parallel game instances (default: 1)
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
FRESH="${FRESH:-0}"
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

# Check if the container already exists
CONTAINER_EXISTS=0
if docker ps -a --format '{{.Names}}' | grep -q '^tmnf$'; then
    CONTAINER_EXISTS=1
fi

echo ""
echo "============================================================"
echo "  Bahrain Base Model — Training from Scratch"
if [ "$FRESH" = "1" ] || [ "$CONTAINER_EXISTS" = "0" ]; then
echo "  Mode       : FULL SETUP (fresh container)"
else
echo "  Mode       : FAST (reusing existing container — FRESH=1 to rebuild)"
fi
echo "  Repo root  : $RACER_HELPER_PATH"
echo "  Map GBX    : $BAHRAIN_GBX_PATH"
echo "  Run name   : $RUN_NAME  →  save/$RUN_NAME/"
echo "============================================================"

xhost + 2>/dev/null || true

# =============================================================
# Full setup path — only runs on first run or FRESH=1
# =============================================================
if [ "$FRESH" = "1" ] || [ "$CONTAINER_EXISTS" = "0" ]; then

    log "Cleaning up existing container"
    docker stop tmnf 2>/dev/null || true
    docker rm   tmnf 2>/dev/null || true

    log "Starting fresh tmnf container"
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
    echo "Waiting for container to initialise..."
    sleep 3

    log "Installing liblzo2-dev"
    docker exec -u 0 tmnf bash -c "apt-get update -qq && apt-get install -y liblzo2-dev"

    INSTALL_DEPS=1

else
    # ── Fast path ─────────────────────────────────────────────
    if ! docker ps --format '{{.Names}}' | grep -q '^tmnf$'; then
        log "Container exists but is stopped — restarting it"
        docker start tmnf
        sleep 3
    else
        log "Container already running — skipping setup"
    fi

    INSTALL_DEPS=0
fi

# =============================================================
# Always: detect display — use host X11 if available,
# otherwise start Xvfb inside the container (headless/vast.ai)
# =============================================================
log "Setting up display"

if docker exec tmnf bash -c "DISPLAY=:0 xdpyinfo" > /dev/null 2>&1; then
    WINE_DISPLAY=":0"
    echo "Host X11 display :0 is reachable — using it"
else
    WINE_DISPLAY=":99"
    echo "Host display :0 not reachable — using Xvfb :99 inside container"
    if ! docker exec tmnf bash -c "pgrep -f 'Xvfb :99'" > /dev/null 2>&1; then
        docker exec -u 0 tmnf bash -c "
            Xvfb :99 -screen 0 1920x1080x24 -ac &
            sleep 2
            echo 'Xvfb :99 started'
        "
    else
        echo "Xvfb :99 already running"
    fi
fi

echo "WINE_DISPLAY=$WINE_DISPLAY"

# =============================================================
# Always: copy fresh linesight and the map GBX
# (linesight code may have changed between runs)
# =============================================================
log "Copying linesight from repo"
docker exec -u 0 tmnf bash -c "rm -rf /home/wineuser/linesight"
docker cp "$LINESIGHT_PATH" tmnf:/home/wineuser/linesight
docker exec -u 0 tmnf bash -c "chown -R wineuser:wineuser /home/wineuser/linesight"

log "Copying Bahrain map into container"
WINE_CHALLENGES="/home/wineuser/.wine/drive_c/users/wineuser/Documents/TmForever/Tracks/Challenges"
docker exec -u 0 tmnf bash -c "mkdir -p '$WINE_CHALLENGES'"
docker cp "$BAHRAIN_GBX_PATH" "tmnf:$WINE_CHALLENGES/$MAP_BASENAME"
docker exec -u 0 tmnf bash -c "chown wineuser:wineuser '$WINE_CHALLENGES/$MAP_BASENAME'"

# =============================================================
# Install Python env + linesight on first run only
# On subsequent runs: skip pip install, just recreate the
# generated config files that were wiped by the linesight copy
# =============================================================
if [ "$INSTALL_DEPS" = "1" ]; then

    log "Installing linesight + dependencies (first run — this takes a while)"
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

echo "Dependencies installed."
'

fi

# =============================================================
# Always: recreate generated files that live inside linesight/
# (they were wiped when we copied fresh linesight above)
# DISPLAY is intentionally omitted from launch_game.sh so it
# inherits the value set by the training process at runtime.
# =============================================================
log "Creating launch_game.sh and user_config.py"
docker exec -u wineuser tmnf bash -c '
set -euo pipefail
source $HOME/.local/bin/env
source ~/linesight_env/bin/activate

cat > /home/wineuser/linesight/scripts/launch_game.sh << '"'"'EOF'"'"'
#!/bin/bash
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
'

# =============================================================
# Always: patch config.py
# =============================================================
log "Patching config.py"

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
# Use repeat() so analyze_map_cycle's chain(*map_cycle) receives
# iterables of tuples, not raw tuple elements.
track = "Bahrain"
wpath = "Bahrain_Circuit.Challenge.Gbx"
vcp   = "Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"

new_cycle = (
    "map_cycle = [\n"
    f'    repeat(("{track}", \'"My Challenges/{wpath}"\', "{vcp}", True,  True), 4),\n'
    f'    repeat(("{track}", \'"My Challenges/{wpath}"\', "{vcp}", False, True), 1),\n'
    "]"
)
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
# Always: fix ownership
# =============================================================
log "Fixing ownership"
docker exec -u 0 tmnf bash -c "
  [ -d /home/wineuser/.triton ] && chown -R wineuser:wineuser /home/wineuser/.triton || true
  chown -R wineuser:wineuser /home/wineuser
"

# =============================================================
# Always: XDG runtime dir
# =============================================================
docker exec -u 0 tmnf bash -c "
  mkdir -p /tmp/runtime-wineuser
  chown wineuser:wineuser /tmp/runtime-wineuser
  chmod 700 /tmp/runtime-wineuser
"

# =============================================================
# Launch TMLoader — always restart when using Xvfb so any
# previously running instance (wrong display) is replaced.
# On host-display mode, skip if already running.
# =============================================================
if [ "$WINE_DISPLAY" = ":99" ]; then
    log "Restarting TMLoader with Xvfb display $WINE_DISPLAY"
    docker exec -u 0 tmnf bash -c "pkill -f 'TMLoader.exe' 2>/dev/null || true"
    sleep 2
    docker exec -d -u wineuser tmnf bash -c "
      export DISPLAY=$WINE_DISPLAY
      export WINEPREFIX=/home/wineuser/.wine
      export HOME=/home/wineuser
      export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
      cd /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever
      wine TMLoader.exe 2>&1 >> /tmp/tmloader.log
    "
    echo "Waiting for TMLoader to initialise..."
    sleep 5
elif ! docker exec tmnf bash -c "pgrep -f 'TMLoader.exe'" > /dev/null 2>&1; then
    log "Launching TMLoader"
    docker exec -d -u wineuser tmnf bash -c "
      export DISPLAY=$WINE_DISPLAY
      export WINEPREFIX=/home/wineuser/.wine
      export HOME=/home/wineuser
      export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
      cd /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever
      wine TMLoader.exe 2>&1 >> /tmp/tmloader.log
    "
    echo "Waiting for TMLoader to initialise..."
    sleep 5
else
    log "TMLoader already running — skipping"
fi

# =============================================================
# Start training
# =============================================================
log "Starting Linesight RL training (Bahrain — scratch)"
echo ""
echo "============================================================"
echo "  Bahrain Base Model Training"
echo "  Saves to: save/$RUN_NAME/ inside the container"
echo "  Display  : $WINE_DISPLAY"
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
  export DISPLAY=$WINE_DISPLAY
  export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
  cd /home/wineuser/linesight
  python scripts/train.py
"
