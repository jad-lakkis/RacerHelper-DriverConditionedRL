#!/bin/bash
# =============================================================
# Script 1: 1_setup_tmnf_docker.sh
# =============================================================
# ONE-TIME SETUP — Run this once when you first get a new
# Vast.ai KVM instance. It clones the TMNF-Docker repo,
# patches the Dockerfiles, and builds the tmnf-vulkan:latest
# Docker image.
#
# After this script completes, run:
#   ./2_start_training.sh
#
# Prerequisites:
#   - Fresh Vast.ai KVM Ubuntu Desktop instance
#   - SSH into the instance as user@<IP>
#   - Run this script on the HOST (user@ubuntu), NOT in Docker
#
# Usage:
#   chmod +x 1_setup_tmnf_docker.sh && ./1_setup_tmnf_docker.sh
# =============================================================

set -euo pipefail

log() { echo ""; echo "==== $1 ===="; }

# =============================================================
# STEP 1: Clone TMNF-Docker repo
# =============================================================
log "Step 1: Cloning TMNF-Docker repo"
cd /home/user

if [ ! -d TMNF-Docker ]; then
    git clone https://github.com/SgSiegens/TMNF-Docker
else
    echo "TMNF-Docker already cloned, skipping."
fi
cd TMNF-Docker

# =============================================================
# STEP 2: Patch Dockerfile.base
# Remove broken launchpad.net vdpau download that fails
# =============================================================
log "Step 2: Patching Dockerfile.base"

if grep -q "vdpau-va-driver.deb" Dockerfile.base; then
    sed -i \
      -e 's|    curl -fsSL -o /tmp/vdpau-va-driver.deb.*\&\& \\||g' \
      -e 's|    apt-get install --no-install-recommends -y /tmp/vdpau-va-driver.deb.*\&\& \\||g' \
      Dockerfile.base
    echo "Dockerfile.base patched."
else
    echo "Dockerfile.base already patched, skipping."
fi

# =============================================================
# STEP 3: Patch Dockerfile.vulkan
# Add timeout to wineboot so it doesn't hang indefinitely
# =============================================================
log "Step 3: Patching Dockerfile.vulkan"

if grep -q "^RUN xvfb-run --auto-servernum wineboot" Dockerfile.vulkan; then
    sed -i \
      's/RUN xvfb-run --auto-servernum wineboot \&\& \\/RUN timeout 60 xvfb-run --auto-servernum wineboot || true \&\& \\/' \
      Dockerfile.vulkan
    echo "Dockerfile.vulkan patched."
else
    echo "Dockerfile.vulkan already patched, skipping."
fi

# =============================================================
# STEP 4: Build the tmnf-vulkan Docker image
# Uses jadlakkis/tmnf-docker-base:latest as base (pre-built)
# because building Dockerfile.base from scratch hangs at
# wineboot --init inside Docker.
# Build takes ~5-10 minutes.
# =============================================================
log "Step 4: Building tmnf-vulkan:latest Docker image (~5-10 min)"

if docker images | grep -q "tmnf-vulkan"; then
    echo "tmnf-vulkan:latest already exists, skipping build."
    echo "To force rebuild, run: docker rmi tmnf-vulkan:latest"
else
    docker build \
      --build-arg BASE_IMAGE=jadlakkis/tmnf-docker-base:latest \
      -t tmnf-vulkan:latest \
      -f Dockerfile.vulkan \
      . 2>&1 | tee /tmp/build-vulkan.log
    echo "Build complete. Logs at /tmp/build-vulkan.log"
fi

# =============================================================
# Done
# =============================================================
echo ""
echo "============================================================"
echo "  ONE-TIME SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Docker image tmnf-vulkan:latest is ready."
echo ""
echo "  Next step: run the training script:"
echo "    chmod +x 2_start_training.sh && ./2_start_training.sh"
echo "============================================================"
