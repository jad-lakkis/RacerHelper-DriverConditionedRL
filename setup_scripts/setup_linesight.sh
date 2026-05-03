#!/bin/bash
# =============================================================
# Custom Script 2: setup_linesight_custom.sh
# Uses your RacerHelper-DriverConditionedRL repo instead of
# cloning the original pb4git/linesight repo.
# Prerequisite: setup_tmnf.sh already ran successfully.
# =============================================================

set -euo pipefail

USERNAME="wineuser"

# Your modified repo
REPO_URL="https://github.com/jad-lakkis/RacerHelper-DriverConditionedRL.git"

REPO_DIR="/home/${USERNAME}/RacerHelper-DriverConditionedRL"
LINESIGHT_DIR="${REPO_DIR}/linesight"

# =============================================================
# Custom Bahrain settings
# =============================================================
CUSTOM_MAP_SHORT_NAME="Bahrain"

# Actual TrackMania map file stored inside your repo
CUSTOM_MAP_FILE="Bahrain_Circuit.Challenge.Gbx"
CUSTOM_MAP_REPO_PATH="${LINESIGHT_DIR}/custom_maps/${CUSTOM_MAP_FILE}"

# Reference line generated from your replay
CUSTOM_REFERENCE_LINE="Bahrain_Circuit_0.5m_jadlakkisjad_012629.npy"
CUSTOM_REFERENCE_PATH="${LINESIGHT_DIR}/maps/${CUSTOM_REFERENCE_LINE}"

# Where TrackMania should see the map inside Wine
TM_CHALLENGES_DIR="/home/${USERNAME}/.wine/drive_c/users/${USERNAME}/Documents/TmForever/Tracks/Challenges/My Challenges"

# This is the path used by Linesight/TMInterface map command
CUSTOM_MAP_TMI_PATH="My Challenges/${CUSTOM_MAP_FILE}"

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

# =============================================================
# Custom Bahrain check and copy
# =============================================================
log "Checking Bahrain map and reference line"

if [ ! -f "${CUSTOM_REFERENCE_PATH}" ]; then
    echo "ERROR: Missing Bahrain reference line:"
    echo "  ${CUSTOM_REFERENCE_PATH}"
    echo ""
    echo "Expected file:"
    echo "  linesight/maps/${CUSTOM_REFERENCE_LINE}"
    exit 1
fi

if [ ! -f "${CUSTOM_MAP_REPO_PATH}" ]; then
    echo "ERROR: Missing Bahrain map file:"
    echo "  ${CUSTOM_MAP_REPO_PATH}"
    echo ""
    echo "Put the actual .Challenge.Gbx map file here:"
    echo "  linesight/custom_maps/${CUSTOM_MAP_FILE}"
    exit 1
fi

mkdir -p "${TM_CHALLENGES_DIR}"

cp "${CUSTOM_MAP_REPO_PATH}" "${TM_CHALLENGES_DIR}/${CUSTOM_MAP_FILE}"

chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}/.wine/drive_c/users/${USERNAME}/Documents/TmForever/Tracks/Challenges"

echo "Bahrain map copied to:"
echo "  ${TM_CHALLENGES_DIR}/${CUSTOM_MAP_FILE}"
echo ""
echo "Bahrain reference line found:"
echo "  ${CUSTOM_REFERENCE_PATH}"

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
export WINEDLLOVERRIDES="d3d9=n;d3d11=n;dxgi=n;d3d10core=n"

mkdir -p "$XDG_RUNTIME_DIR"

# vglrun bridges VirtualGL so Wine's 32-bit DXVK can reach the NVIDIA GPU.
# Without it the 64-bit-only VK ICD path causes Vulkan device enumeration to
# fail inside the 32-bit Wine process, leaving the game windowless.
exec vglrun wine /home/wineuser/.wine/drive_c/Program_Files_x86/TmNationsForever/TMLoader.exe run TmForever "default" /configstring="set custom_port $1"
EOF

chmod +x "${LINESIGHT_DIR}/scripts/launch_game.sh"

# ---- Step 3: user_config.py already lives in the repo (linesight/config_files/user_config.py) ----
log "Step 3/5: user_config.py already configured in repo — skipping"

# ---- Step 4: Sync config_copy.py from repo config.py ----
log "Step 4/5: Syncing config_copy.py"

cp "${LINESIGHT_DIR}/config_files/config.py" "${LINESIGHT_DIR}/config_files/config_copy.py"

find "${LINESIGHT_DIR}/config_files/__pycache__" -type f -delete 2>/dev/null || true

chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}"

echo "config_copy.py synced from config.py."

# ---- Step 5: Start VNC on :1 and launch training ----
log "Step 5/5: Starting training"

x11vnc -display :1 -passwd mypasswd -shared -forever -repeat -xkb -rfbport 5901 &>/dev/null &

echo ""
echo "============================================================"
echo "  Starting Custom Linesight Training on Bahrain"
echo "============================================================"
echo ""
echo "  Repo:"
echo "    ${REPO_DIR}"
echo ""
echo "  Linesight path:"
echo "    ${LINESIGHT_DIR}"
echo ""
echo "  Map:"
echo "    ${CUSTOM_MAP_TMI_PATH}"
echo ""
echo "  Reference line:"
echo "    ${CUSTOM_REFERENCE_LINE}"
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