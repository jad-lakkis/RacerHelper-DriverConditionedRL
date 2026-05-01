#!/bin/bash
# =============================================================
# Custom Script 2: setup_linesight_custom.sh
# Uses your RacerHelper-DriverConditionedRL repo instead of
# cloning the original pb4git/linesight repo.
# Prerequisite: setup_tmnf.sh already ran successfully.
# =============================================================

set -euo pipefail

USERNAME="wineuser"

# CHANGE THIS ONLY IF YOUR GITHUB URL IS DIFFERENT
REPO_URL="https://github.com/jad-lakkis/RacerHelper-DriverConditionedRL.git"

REPO_DIR="/home/${USERNAME}/RacerHelper-DriverConditionedRL"
LINESIGHT_DIR="${REPO_DIR}/linesight"

log() {
    echo ""
    echo "==== $1 ===="
}

# ---- Step 1: Clone or update your repo ----
log "Step 1/5: Cloning/updating your RacerHelper repo"

cd /home/${USERNAME}

if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone "${REPO_URL}" "${REPO_DIR}"
else
    cd "${REPO_DIR}"
    git pull
fi

# Verify expected Linesight structure
if [ ! -d "${LINESIGHT_DIR}" ]; then
    echo "ERROR: Could not find ${LINESIGHT_DIR}"
    echo "Your repo must contain a linesight/ folder."
    exit 1
fi

if [ ! -f "${LINESIGHT_DIR}/scripts/train.py" ]; then
    echo "ERROR: Could not find ${LINESIGHT_DIR}/scripts/train.py"
    exit 1
fi

if [ ! -f "${LINESIGHT_DIR}/config_files/config.py" ]; then
    echo "ERROR: Could not find ${LINESIGHT_DIR}/config_files/config.py"
    exit 1
fi

cd "${LINESIGHT_DIR}"

# Install dependencies/package
pip install -e . 2>&1 | tail -5
echo "Your modified Linesight repo installed."

# ---- Step 2: Create game launch script ----
log "Step 2/5: Creating game launch script"

cat > "${LINESIGHT_DIR}/scripts/launch_game.sh" << 'EOF'
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

chmod +x "${LINESIGHT_DIR}/scripts/launch_game.sh"

# ---- Step 3: Configure user_config.py ----
log "Step 3/5: Configuring user_config.py"

cat > "${LINESIGHT_DIR}/config_files/user_config.py" << PYEOF
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
linux_launch_game_path = "${LINESIGHT_DIR}/scripts/launch_game.sh"

# Windows paths, unused on Linux
windows_TMLoader_path = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
windows_TMLoader_profile_name = "default"

# Platform detection
is_linux = platform.system() == "Linux"
PYEOF

# ---- Step 4: Patch config.py / config_copy.py ----
log "Step 4/5: Patching config files"

if ! grep -q "is_linux" "${LINESIGHT_DIR}/config_files/config.py"; then
    echo '
import platform
is_linux = platform.system() == "Linux"
' >> "${LINESIGHT_DIR}/config_files/config.py"
fi

cp "${LINESIGHT_DIR}/config_files/config.py" "${LINESIGHT_DIR}/config_files/config_copy.py"

# Use one game instance first for stability
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' "${LINESIGHT_DIR}/config_files/config.py"
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' "${LINESIGHT_DIR}/config_files/config_copy.py"

find "${LINESIGHT_DIR}/config_files/__pycache__" -type f -delete 2>/dev/null || true

chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}"

echo "Config files patched."

# ---- Step 5: Start VNC on :1 and launch training ----
log "Step 5/5: Starting training"

x11vnc -display :1 -passwd mypasswd -shared -forever -repeat -xkb -rfbport 5901 &>/dev/null &

echo ""
echo "============================================================"
echo "  Starting Custom Linesight Training"
echo "============================================================"
echo ""
echo "  Repo:"
echo "    ${REPO_DIR}"
echo ""
echo "  Linesight path:"
echo "    ${LINESIGHT_DIR}"
echo ""
echo "  IMPORTANT:"
echo "    Check VNC on port 5901."
echo "    If TrackMania shows a 'Stay offline' popup, click it."
echo ""
echo "  TensorBoard:"
echo "    cd ${LINESIGHT_DIR}"
echo "    tensorboard --logdir=tensorboard --bind_all"
echo ""
echo "============================================================"
echo ""

su ${USERNAME} -c "cd ${LINESIGHT_DIR} && DISPLAY=:1 python scripts/train.py"
