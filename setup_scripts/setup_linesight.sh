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
log "Step 4/5: Patching config files for Bahrain"

if ! grep -q "is_linux" "${LINESIGHT_DIR}/config_files/config.py"; then
    echo '
import platform
is_linux = platform.system() == "Linux"
' >> "${LINESIGHT_DIR}/config_files/config.py"
fi

# Patch map_cycle safely:
# - old map_cycle is commented, not deleted
# - new Bahrain map_cycle is added
CUSTOM_MAP_TMI_PATH="${CUSTOM_MAP_TMI_PATH}" \
CUSTOM_REFERENCE_LINE="${CUSTOM_REFERENCE_LINE}" \
CUSTOM_MAP_SHORT_NAME="${CUSTOM_MAP_SHORT_NAME}" \
python - "${LINESIGHT_DIR}/config_files/config.py" << 'PYEOF'
from pathlib import Path
import os
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

map_name = os.environ["CUSTOM_MAP_SHORT_NAME"]
map_path = os.environ["CUSTOM_MAP_TMI_PATH"]
ref_line = os.environ["CUSTOM_REFERENCE_LINE"]

custom_block = f'''
# =============================================================
# CUSTOM BAHRAIN MAP CYCLE BEGIN
# Old map_cycle above is kept/commented as reference.
# This block overrides the previous training map.
# =============================================================
map_cycle = [
    repeat(("{map_name}", '"{map_path}"', "{ref_line}", True, True), 4),
    repeat(("{map_name}", '"{map_path}"', "{ref_line}", False, True), 1),
]

# Bahrain first test schedule speed
global_schedule_speed = 1.0
# =============================================================
# CUSTOM BAHRAIN MAP CYCLE END
# =============================================================
'''

begin = "# =============================================================\n# CUSTOM BAHRAIN MAP CYCLE BEGIN"
end = "# =============================================================\n# CUSTOM BAHRAIN MAP CYCLE END\n# ============================================================="

# Remove old custom block if script is re-run
if "CUSTOM BAHRAIN MAP CYCLE BEGIN" in text:
    text = re.sub(
        r'\n?# =============================================================\n# CUSTOM BAHRAIN MAP CYCLE BEGIN.*?# =============================================================\n# CUSTOM BAHRAIN MAP CYCLE END\n# =============================================================\n?',
        "\n",
        text,
        flags=re.DOTALL
    )

lines = text.splitlines()

# Comment the last active map_cycle block, but do not delete it
map_cycle_indices = [
    i for i, line in enumerate(lines)
    if re.match(r'^\s*map_cycle\s*=', line)
]

if map_cycle_indices:
    start = map_cycle_indices[-1]

    # If it is not already commented, comment the block
    if not lines[start].lstrip().startswith("#"):
        depth = 0
        end_idx = start

        for j in range(start, len(lines)):
            depth += lines[j].count("[")
            depth -= lines[j].count("]")
            end_idx = j
            if j > start and depth <= 0:
                break

        for j in range(start, end_idx + 1):
            lines[j] = "# OLD MAP_CYCLE: " + lines[j]

# Comment the last active global_schedule_speed line, but do not delete it
gss_indices = [
    i for i, line in enumerate(lines)
    if re.match(r'^\s*global_schedule_speed\s*=', line)
]

if gss_indices:
    i = gss_indices[-1]
    if not lines[i].lstrip().startswith("#"):
        lines[i] = "# OLD global_schedule_speed: " + lines[i]

text = "\n".join(lines).rstrip() + "\n" + custom_block + "\n"

path.write_text(text)
PYEOF

# Make config_copy.py the same as patched config.py
cp "${LINESIGHT_DIR}/config_files/config.py" "${LINESIGHT_DIR}/config_files/config_copy.py"

# Use one game instance first for stability
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' "${LINESIGHT_DIR}/config_files/config.py"
sed -i 's/gpu_collectors_count = 2/gpu_collectors_count = 1/' "${LINESIGHT_DIR}/config_files/config_copy.py"

find "${LINESIGHT_DIR}/config_files/__pycache__" -type f -delete 2>/dev/null || true

chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}"

echo "Config files patched for Bahrain."
echo ""
echo "Training map path:"
echo "  ${CUSTOM_MAP_TMI_PATH}"
echo ""
echo "Reference line:"
echo "  ${CUSTOM_REFERENCE_LINE}"

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