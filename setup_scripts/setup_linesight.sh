#!/bin/bash
# =============================================================
# Script 2: setup_linesight.sh
# =============================================================
# Installs Linesight RL framework and starts training.
# Prerequisites: setup_tmnf.sh must have been run first.
#
# Usage:
#   chmod +x setup_linesight.sh && ./setup_linesight.sh
#
# IMPORTANT: After the game launches, check VNC.
#   If there's a "Stay offline" popup, click it to dismiss.
#   Training will begin automatically after that.
# =============================================================

set -euo pipefail

USERNAME="wineuser"

log() { echo ""; echo "==== $1 ===="; }

# ---- Step 1: Install Linesight ----
log "Step 1/5: Installing Linesight"
cd /home/${USERNAME}

if [ ! -d linesight ]; then
    git clone https://github.com/pb4git/linesight
fi
cd linesight

# Install dependencies (torch should already be present from RunPod template)
pip install -e . 2>&1 | tail -5
echo "Linesight installed."

# ---- Step 2: Create launch script ----
log "Step 2/5: Creating game launch script"
cat > /home/${USERNAME}/linesight/scripts/launch_game.sh << 'EOF'
#!/bin/bash
export DISPLAY=:1
export WINEARCH=win32
export WINEPREFIX=/home/wineuser/.wine
export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export WINEDLLOVERRIDES="d3d9=n;d3d11=n;dxgi=n;d3d10core=n"

mkdir -p "$XDG_RUNTIME_DIR"

exec wine /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/TMLoader.exe run TmForever "default" /configstring="set custom_port $1"
EOF
chmod +x /home/${USERNAME}/linesight/scripts/launch_game.sh

# ---- Step 3: Configure user_config.py ----
log "Step 3/5: Configuring user_config.py"
cat > /home/${USERNAME}/linesight/config_files/user_config.py << 'PYEOF'
from pathlib import Path
import os
import platform

username = "default"  # TMNF profile name

# Wine prefix Documents path
_wine_docs = Path("/home/wineuser/.wine/drive_c/users/wineuser/Documents")

# Path where Python_Link.as is placed
target_python_link_path = _wine_docs / "TMInterface" / "Plugins" / "Python_Link.as"

# TrackMania base path
trackmania_base_path = _wine_docs / "TmForever"

# Communication port for TMInterface
base_tmi_port = 8478

# Linux launch script
linux_launch_game_path = "/home/wineuser/linesight/scripts/launch_game.sh"

# Windows paths (unused on Linux)
windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
windows_TMLoader_profile_name = "default"

# Platform detection
is_linux = platform.system() == "Linux"
PYEOF

# ---- Step 4: Patch config.py / config_copy.py ----
log "Step 4/5: Patching config files"

# Add is_linux if missing
if ! grep -q "is_linux" /home/${USERNAME}/linesight/config_files/config.py; then
    echo '
import platform
is_linux = platform.system() == "Linux"
' >> /home/${USERNAME}/linesight/config_files/config.py
fi

# Copy config.py to config_copy.py
cp /home/${USERNAME}/linesight/config_files/config.py /home/${USERNAME}/linesight/config_files/config_copy.py

# Set 1 game instance (more stable, can increase to 2 later)
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' /home/${USERNAME}/linesight/config_files/config.py
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' /home/${USERNAME}/linesight/config_files/config_copy.py

# Clear Python cache
find /home/${USERNAME}/linesight/config_files/__pycache__ -type f -delete 2>/dev/null || true

# Fix ownership
chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

echo "Config files patched."

# ---- Step 5: Start VNC on :1 and launch training ----
log "Step 5/5: Starting training"

# Start VNC on Xorg display for monitoring the game
x11vnc -display :1 -passwd mypasswd -shared -forever -repeat -xkb -rfbport 5901 &>/dev/null &

echo ""
echo "============================================================"
echo "  Starting Linesight RL Training"
echo "============================================================"
echo ""
echo "  IMPORTANT: After the game launches, check VNC on port 5901."
echo "  If there is a 'Stay offline' popup, click it to dismiss."
echo "  Training will start automatically after that."
echo ""
echo "  To monitor via VNC (from your local machine):"
echo "    ssh -L 5901:localhost:5901 root@<RUNPOD_IP> -p <PORT> -i <KEY>"
echo "    Connect Remmina (VNC) to localhost:5901 (password: mypasswd)"
echo ""
echo "  To monitor training metrics:"
echo "    (in another terminal) cd /home/wineuser/linesight"
echo "    tensorboard --logdir=tensorboard --bind_all"
echo "    ssh -L 6006:localhost:6006 root@<RUNPOD_IP> -p <PORT> -i <KEY>"
echo "    Open http://localhost:6006 in your browser"
echo ""
echo "  GPU rendering: DXVK -> Vulkan -> NVIDIA GPU"
echo "  Neural network training: PyTorch -> CUDA -> NVIDIA GPU"
echo "============================================================"
echo ""

# Launch training as wineuser
su wineuser -c 'cd /home/wineuser/linesight && DISPLAY=:1 python scripts/train.py'