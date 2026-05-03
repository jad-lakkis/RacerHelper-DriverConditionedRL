#!/bin/bash
# =============================================================
# Script 2: 2_start_training.sh
# =============================================================
# Run this every time you want to start training.
# Assumes Script 1 (1_setup_tmnf_docker.sh) has been run
# at least once and tmnf-vulkan:latest image exists.
#
# What this script does:
#   1. Allows X display access from Docker (xhost +)
#   2. Stops and removes any existing tmnf container
#   3. Starts a fresh tmnf container with GPU + X display
#   4. Installs liblzo2-dev system dependency
#   5. Installs uv + Python 3.11 + Linesight inside container
#   6. Creates launch_game.sh and user_config.py
#   7. Patches config files
#   8. Fixes ownership issues
#   9. Launches TMLoader (the game)
#  10. Starts Linesight RL training
#
# Prerequisites:
#   - Script 1 has been run (tmnf-vulkan:latest exists)
#   - Run on the HOST (user@ubuntu), NOT inside Docker
#   - KVM instance is running with DISPLAY=:0
#
# Usage:
#   chmod +x 2_start_training.sh && ./2_start_training.sh
# =============================================================

set -euo pipefail

log() { echo ""; echo "==== $1 ===="; }

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
docker rm tmnf 2>/dev/null || true
echo "Old container removed (if any)."

# =============================================================
# STEP 3: Start the tmnf container
# Key flags:
#   --gpus all              GPU access for DXVK/Vulkan rendering
#   -e DISPLAY=:0           Use host X display (visible in web desktop)
#   -v /tmp/.X11-unix       Mount host X11 socket
#   -v /usr/share/vulkan    Mount NVIDIA Vulkan ICD from host
#   --entrypoint bash       Override container Xorg entrypoint
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
# Required by python-lzo which is required by pygbx (Linesight)
# =============================================================
log "Step 4: Installing liblzo2-dev inside container"
docker exec -u 0 tmnf bash -c "apt-get update -qq && apt-get install -y liblzo2-dev"
echo "liblzo2-dev installed."

# =============================================================
# STEP 5: Full Linesight setup inside container as wineuser
# =============================================================
log "Step 5: Setting up Linesight inside container (as wineuser)"

docker exec -u wineuser tmnf bash -c '
set -euo pipefail

# --- Install uv (fast Python package manager) ---
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source $HOME/.local/bin/env

# --- Install Python 3.11 and create venv ---
if [ ! -d /home/wineuser/linesight_env ]; then
    uv python install 3.11
    uv venv ~/linesight_env --python 3.11
fi
source ~/linesight_env/bin/activate

# --- Clone Linesight ---
cd /home/wineuser
if [ ! -d linesight ]; then
    git clone https://github.com/pb4git/linesight
fi
cd linesight

# --- Install Linesight and dependencies ---
uv pip install -e .

# --- Create launch_game.sh ---
# This is called by the training script to launch the game
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

# --- Patch config.py: add is_linux if missing ---
if ! grep -q "is_linux" /home/wineuser/linesight/config_files/config.py; then
    echo "
import platform
is_linux = platform.system() == \"Linux\"
" >> /home/wineuser/linesight/config_files/config.py
fi

# --- Copy config and set 1 game instance (more stable) ---
cp /home/wineuser/linesight/config_files/config.py \
   /home/wineuser/linesight/config_files/config_copy.py

sed -i "s/gpu_collectors_count = 2/gpu_collectors_count = 1/" \
    /home/wineuser/linesight/config_files/config.py
sed -i "s/gpu_collectors_count = 2/gpu_collectors_count = 1/" \
    /home/wineuser/linesight/config_files/config_copy.py

# --- Clear Python cache ---
find /home/wineuser/linesight/config_files/__pycache__ -type f -delete 2>/dev/null || true

echo "Linesight setup complete."
'

# =============================================================
# STEP 6: Fix ownership issues
# .triton may have been created as root in earlier runs
# =============================================================
log "Step 6: Fixing ownership"
docker exec -u 0 tmnf bash -c "
  if [ -d /home/wineuser/.triton ]; then
    chown -R wineuser:wineuser /home/wineuser/.triton
  fi
  chown -R wineuser:wineuser /home/wineuser
"

# =============================================================
# STEP 7: Create XDG runtime dir with correct ownership
# Required by Wine/Vulkan inside the container
# =============================================================
log "Step 7: Creating XDG runtime dir"
docker exec -u 0 tmnf bash -c "
  mkdir -p /tmp/runtime-wineuser
  chown wineuser:wineuser /tmp/runtime-wineuser
  chmod 700 /tmp/runtime-wineuser
"

# =============================================================
# STEP 8: Launch TMLoader (the game launcher)
# Runs in background (-d). Check VNC desktop to see the window.
# If there is a 'Stay offline' popup, click it to dismiss.
# =============================================================
log "Step 8: Launching TMLoader"
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
# STEP 9: Start Linesight RL training
# This runs in the foreground so you can see training output.
# Press Ctrl+C to stop.
# =============================================================
log "Step 9: Starting Linesight RL training"
echo ""
echo "============================================================"
echo "  Linesight RL Training Starting"
echo "============================================================"
echo ""
echo "  IMPORTANT: Check the VNC/web desktop."
echo "  If there is a 'Stay offline' popup, click it to dismiss."
echo "  Training will begin automatically after that."
echo ""
echo "  Useful monitoring commands (open a new terminal):"
echo "    nvidia-smi                          # GPU usage"
echo "    docker exec tmnf cat /tmp/tmnf.log  # Game logs"
echo "    docker exec tmnf cat /tmp/tmloader.log  # TMLoader logs"
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
