#!/bin/bash
# =============================================================
# Script 1: setup_tmnf.sh
# =============================================================
# Installs TMNF + TMLoader + TMInterface + display infrastructure
# on a RunPod GPU pod (Ubuntu 22.04 + PyTorch template).
#
# Usage:
#   chmod +x setup_tmnf.sh && ./setup_tmnf.sh
#
# This script does NOT launch the game or training.
# Use setup_linesight.sh (Script 2) for that.
# =============================================================

set -euo pipefail

USERNAME="wineuser"
PASSWD="mypasswd"
TM_INSTALLER_URL="https://nadeo-download.cdn.ubi.com/trackmaniaforever/tmnationsforever_setup.exe"
REPO_URL="https://github.com/SgSiegens/TMNF-Docker"
DXVK_VERSION="2.6.1"
VIRTUALGL_VERSION="3.1"

log() { echo ""; echo "==== $1 ===="; }

# ---- Pre-flight checks ----
log "Pre-flight checks"
[ "$(id -u)" -ne 0 ] && echo "ERROR: Run as root." >&2 && exit 1
command -v nvidia-smi &>/dev/null || { echo "ERROR: No nvidia-smi. Need a GPU pod." >&2; exit 1; }
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---- Step 1: X11 socket + cleanup ----
log "Step 1/11: Preparing X11 socket directory"
mkdir -p /tmp/.X11-unix
chown root:root /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
pkill Xvfb 2>/dev/null || true
pkill Xorg 2>/dev/null || true
pkill fluxbox 2>/dev/null || true
pkill x11vnc 2>/dev/null || true
pkill -f TmForever 2>/dev/null || true
rm -f /tmp/.X*-lock
sleep 1
echo "Done."

# ---- Step 2: System packages ----
log "Step 2/11: Installing system packages"
dpkg --add-architecture i386
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    locales software-properties-common ca-certificates curl wget git nano htop \
    net-tools xvfb x11-apps x11-utils x11-xserver-utils x11vnc xdotool \
    fluxbox wmctrl cabextract gnupg gosu gpg-agent sudo winbind wine32 \
    dbus-x11 libgl1-mesa-dri libgl1-mesa-dri:i386 mesa-utils \
    libegl1-mesa:i386 libgl1-mesa-glx:i386 vulkan-tools libvulkan1 \
    liblzo2-dev xserver-xorg-core 2>&1 | tail -5

# WineHQ
echo "  Installing WineHQ stable..."
wget -nv -O- https://dl.winehq.org/wine-builds/winehq.key | apt-key add - 2>/dev/null
CODENAME=$(grep VERSION_CODENAME= /etc/os-release | cut -d= -f2)
# Avoid duplicate entries
grep -q "winehq" /etc/apt/sources.list || echo "deb https://dl.winehq.org/wine-builds/ubuntu/ ${CODENAME} main" >> /etc/apt/sources.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --install-recommends winehq-stable 2>&1 | tail -5
command -v wineboot &>/dev/null || { echo "ERROR: wineboot not found!" >&2; exit 1; }

# 32-bit NVIDIA libraries (needed for DXVK in 32-bit Wine)
# Must match the EXACT host driver version or Vulkan device enumeration fails
echo "  Installing 32-bit NVIDIA libraries..."
NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
NVIDIA_MAJOR=$(echo "$NVIDIA_DRIVER_VERSION" | cut -d. -f1)
echo "  Detected NVIDIA driver: ${NVIDIA_DRIVER_VERSION} (major: ${NVIDIA_MAJOR})"
# Try exact version match first (from Ubuntu repos), then fall back to NVIDIA repo version
apt-get install -y --allow-downgrades "libnvidia-gl-${NVIDIA_MAJOR}:i386=${NVIDIA_DRIVER_VERSION}-0ubuntu0.22.04.1" 2>/dev/null \
    || apt-get install -y --allow-downgrades "libnvidia-gl-${NVIDIA_MAJOR}:i386=${NVIDIA_DRIVER_VERSION}-1ubuntu1" 2>/dev/null \
    || apt-get install -y "libnvidia-gl-${NVIDIA_MAJOR}:i386" 2>/dev/null \
    || echo "  WARNING: Could not install matching libnvidia-gl-${NVIDIA_MAJOR}:i386. GPU rendering may not work."
echo "  Installed: $(dpkg -l | grep libnvidia-gl | grep i386 | awk '{print $3}')"

# VirtualGL
echo "  Installing VirtualGL..."
curl -fsSL -O "https://sourceforge.net/projects/virtualgl/files/virtualgl_${VIRTUALGL_VERSION}_amd64.deb"
curl -fsSL -O "https://sourceforge.net/projects/virtualgl/files/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb"
apt-get install -y --no-install-recommends ./virtualgl_${VIRTUALGL_VERSION}_amd64.deb ./virtualgl32_${VIRTUALGL_VERSION}_amd64.deb 2>&1 | tail -3
rm -f virtualgl_*.deb

# Winetricks
[ ! -f /usr/bin/winetricks ] && wget -q -O /usr/bin/winetricks https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks && chmod +x /usr/bin/winetricks

locale-gen en_US.UTF-8 >/dev/null 2>&1
export LANG=en_US.UTF-8
echo "System packages installed. Wine: $(wine --version)"

# ---- Step 3: Xorg NVIDIA driver symlinks ----
log "Step 3/11: Setting up NVIDIA Xorg driver"
mkdir -p /usr/lib/xorg/modules/drivers /usr/lib/xorg/modules/extensions
ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/nvidia_drv.so /usr/lib/xorg/modules/drivers/nvidia_drv.so
ln -sf /usr/lib/x86_64-linux-gnu/nvidia/xorg/libglxserver_nvidia.so /usr/lib/xorg/modules/extensions/libglxserver_nvidia.so
echo "NVIDIA Xorg driver symlinks created."

# ---- Step 4: Create wineuser ----
log "Step 4/11: Creating wineuser"
if ! id "${USERNAME}" &>/dev/null; then
    useradd -m -s /bin/bash ${USERNAME}
    echo "${USERNAME}:${PASSWD}" | chpasswd
    usermod -aG sudo ${USERNAME}
fi
mkdir -p /home/${USERNAME}/app
chown ${USERNAME}:${USERNAME} /home/${USERNAME}/app
echo "User '${USERNAME}' ready."

# ---- Step 5: Wine prefix + winetricks + TMNF ----
log "Step 5/11: Setting up Wine + TMNF (takes several minutes)"
sudo -u ${USERNAME} bash <<'WINEBLOCK'
set -e
export WINEARCH=win32
export WINEPREFIX=/home/wineuser/.wine
export DISPLAY=:99
export HOME=/home/wineuser

Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2

echo "  -> Initializing Wine prefix (32-bit)..."
timeout 60 wineboot --init 2>&1 | grep -v "^00" || true
timeout 30 wineserver --wait 2>/dev/null || echo "  (wineserver timed out — normal on first run)"

echo "  -> Installing winetricks: corefonts, vcrun6, d3dx9_43..."
winetricks -q corefonts 2>&1 | grep -E "(Executing w_do_call|Downloading)" || true
wineserver --wait
winetricks -q vcrun6 2>&1 | grep -E "(Executing w_do_call|Downloading)" || true
wineserver --wait
winetricks -q d3dx9_43 2>&1 | grep -E "(Executing w_do_call|Downloading)" || true
wineserver --wait

echo "  -> Downloading TMNF..."
cd /home/wineuser/app
wget -q -O tmnationsforever_setup.exe "https://nadeo-download.cdn.ubi.com/trackmaniaforever/tmnationsforever_setup.exe"

echo "  -> Installing TMNF (silent)..."
wine tmnationsforever_setup.exe \
    /VERYSILENT /SUPPRESSMSGBOXES /NOCANCEL /NORESTART /SP- \
    /DIR="C:\\Program_Files_x86\\TmNationsForever" 2>&1 | grep -v "^00" || true
wineserver --wait
rm -f tmnationsforever_setup.exe

[ -f "$WINEPREFIX/drive_c/Program_Files_x86/TmNationsForever/TmForever.exe" ] \
    && echo "  -> TMNF installed successfully." \
    || { echo "  -> ERROR: TmForever.exe not found!" >&2; kill $XVFB_PID 2>/dev/null; exit 1; }

rm -rf "${WINEPREFIX}/drive_c/users/wineuser/"*{Downloads,Music,Pictures,Videos,Templates,Public}* 2>/dev/null || true
kill $XVFB_PID 2>/dev/null || true
WINEBLOCK
echo "TMNF installation done."

# ---- Step 6: DXVK ----
log "Step 6/11: Installing DXVK ${DXVK_VERSION}"
mkdir -p /tmp/dxvk && cd /tmp/dxvk
curl -sL "https://github.com/doitsujin/dxvk/releases/download/v${DXVK_VERSION}/dxvk-${DXVK_VERSION}.tar.gz" -o dxvk.tar.gz
tar -xzf dxvk.tar.gz
cp dxvk-${DXVK_VERSION}/x32/*.dll /home/${USERNAME}/.wine/drive_c/windows/system32/
chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.wine/drive_c/windows/system32/d3d*.dll
chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.wine/drive_c/windows/system32/dxgi.dll
rm -rf /tmp/dxvk && cd /

# Register DXVK DLL overrides
sudo -u ${USERNAME} bash -c '
export WINEARCH=win32 WINEPREFIX=/home/wineuser/.wine DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2
for dll in d3d9 d3d11 dxgi d3d10core; do
    wine reg add "HKEY_CURRENT_USER\Software\Wine\DllOverrides" /v $dll /d native /f 2>/dev/null
done
wineserver --wait
kill $XVFB_PID 2>/dev/null || true
'
echo "DXVK installed and registered."

# ---- Step 7: Game-data (TMLoader, TMInterface, profiles) ----
log "Step 7/11: Copying game-data from TMNF-Docker repo"
WINEPREFIX="/home/${USERNAME}/.wine"
[ ! -d /tmp/tmnf-repo ] && git clone --quiet "${REPO_URL}" /tmp/tmnf-repo
mkdir -p "${WINEPREFIX}/drive_c/users/${USERNAME}/Documents/TMInterface"
mkdir -p "${WINEPREFIX}/drive_c/users/${USERNAME}/Documents/TmForever"
cp -r /tmp/tmnf-repo/game-data/TMInterface/* "${WINEPREFIX}/drive_c/users/${USERNAME}/Documents/TMInterface/" 2>/dev/null || true
cp -r /tmp/tmnf-repo/game-data/TmForever/* "${WINEPREFIX}/drive_c/users/${USERNAME}/Documents/TmForever/" 2>/dev/null || true
cp -r /tmp/tmnf-repo/game-data/TMLoader/* "${WINEPREFIX}/drive_c/Program_Files_x86/TmNationsForever/" 2>/dev/null || true
chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}
echo "Game-data copied."

# ---- Step 8: Xorg config (for GPU Vulkan rendering) ----
log "Step 8/11: Creating Xorg config"
GPU_BUS_HEX=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1)
# Convert hex bus ID to decimal: 00000000:D6:00.0 -> PCI:214:0:0
BUS_NUM=$(echo "$GPU_BUS_HEX" | sed 's/.*://' | cut -d: -f1 | xargs printf "%d" 2>/dev/null || echo "0")
# More robust parsing
BUS_PARTS=$(echo "$GPU_BUS_HEX" | grep -oP '[0-9A-Fa-f]+:[0-9A-Fa-f]+\.[0-9A-Fa-f]+$')
BUS_DEC=$(printf "%d" "0x$(echo $BUS_PARTS | cut -d: -f1)" 2>/dev/null || echo "0")
DEV_DEC=$(printf "%d" "0x$(echo $BUS_PARTS | cut -d: -f2 | cut -d. -f1)" 2>/dev/null || echo "0")
FUNC_DEC=$(printf "%d" "0x$(echo $BUS_PARTS | cut -d. -f2)" 2>/dev/null || echo "0")

cat > /workspace/xorg-gpu.conf << XEOF
Section "ServerLayout"
    Identifier     "Layout0"
    Screen      0  "Screen0"
EndSection
Section "Device"
    Identifier     "Device0"
    Driver         "nvidia"
    BusID          "PCI:${BUS_DEC}:${DEV_DEC}:${FUNC_DEC}"
    Option         "AllowEmptyInitialConfiguration" "True"
    Option         "UseDisplayDevice" "none"
EndSection
Section "Screen"
    Identifier     "Screen0"
    Device         "Device0"
    DefaultDepth    24
    SubSection     "Display"
        Virtual     1920 1080
        Depth       24
    EndSubSection
EndSection
Section "ServerFlags"
    Option         "DontVTSwitch" "True"
    Option         "AllowMouseOpenFail" "True"
    Option         "AutoAddGPU" "False"
    Option         "AutoAddDevices" "False"
EndSection
XEOF
echo "Xorg config created with BusID PCI:${BUS_DEC}:${DEV_DEC}:${FUNC_DEC}"

# ---- Step 9: Start display servers ----
log "Step 9/11: Starting display servers (Xvfb :0 for VNC, Xorg :1 for GPU)"

# Xvfb on :0 for VNC viewing
Xvfb :0 -screen 0 1920x1080x24 &
sleep 2
export DISPLAY=:0
fluxbox &>/dev/null &
x11vnc -display :0 -passwd ${PASSWD} -shared -forever -repeat -xkb -rfbport 5900 &>/dev/null &
echo "Xvfb + VNC running on :0 (port 5900, password: ${PASSWD})"

# Xorg on :1 for GPU rendering (Vulkan/DXVK)
ln -snf /dev/ptmx /dev/tty7 2>/dev/null || true
Xorg :1 vt7 -noreset -novtswitch -sharevts -nolisten tcp -config /workspace/xorg-gpu.conf 2>/dev/null &
sleep 3
if [ -S /tmp/.X11-unix/X1 ]; then
    echo "Xorg running on :1 (GPU accelerated)"
    DISPLAY=:1 fluxbox &>/dev/null &
else
    echo "WARNING: Xorg failed to start on :1. GPU rendering unavailable."
    echo "  Game will fall back to CPU software rendering on :0."
fi

# ---- Step 10: Wineuser environment ----
log "Step 10/11: Configuring wineuser environment"
cat > /home/${USERNAME}/.tmnf_env << 'ENVEOF'
export WINEARCH=win32
export WINEPREFIX=/home/wineuser/.wine
export DISPLAY=:1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp/runtime-wineuser
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null
ENVEOF
grep -q "tmnf_env" /home/${USERNAME}/.bashrc 2>/dev/null || echo 'source /home/wineuser/.tmnf_env' >> /home/${USERNAME}/.bashrc

sudo -u ${USERNAME} bash -c '
    mkdir -p /home/wineuser/.fluxbox
    echo "session.session0: true" > /home/wineuser/.fluxbox/init
    touch /home/wineuser/.fluxbox/menu /home/wineuser/.fluxbox/keys /home/wineuser/.fluxbox/apps
'
chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}
echo "Environment configured."

# ---- Step 11: xhost + final ----
log "Step 11/11: Final configuration"
DISPLAY=:0 xhost +local:${USERNAME} 2>/dev/null || true
DISPLAY=:1 xhost +local:${USERNAME} 2>/dev/null || true

echo ""
echo "============================================================"
echo "  TMNF Environment Ready!"
echo "============================================================"
echo ""
echo "  Display :0 = Xvfb + VNC (for remote viewing)"
echo "  Display :1 = Xorg + NVIDIA GPU (for game rendering)"
echo ""
echo "  VNC access (from your local machine):"
echo "    ssh -L 5900:localhost:5900 root@<RUNPOD_IP> -p <PORT> -i <KEY>"
echo "    Then connect Remmina to localhost:5900 (password: ${PASSWD})"
echo ""
echo "  To also see the GPU display (:1) via VNC:"
echo "    x11vnc -display :1 -passwd ${PASSWD} -shared -forever -rfbport 5901 &"
echo "    ssh -L 5901:localhost:5901 root@<RUNPOD_IP> -p <PORT> -i <KEY>"
echo "    Connect Remmina to localhost:5901"
echo ""
echo "  Next: Run setup_linesight.sh to install Linesight and start training."
echo "============================================================"

# Keep background processes alive
wait