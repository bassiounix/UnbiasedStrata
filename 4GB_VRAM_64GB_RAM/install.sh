#!/usr/bin/env bash
# install.sh - put this folder's tools on the PATH and write their settings.
#
#   ./install.sh [--model-dir DIR] [--system] [--force-config]
#
# Always (no root):
#   ~/.local/bin/strata-agent            the launcher (the measured 4 GB / 64 GB configuration)
#   ~/.local/bin/gpu-mode                host/VM ownership of the GPU, status, PCIe link retrain
#   ~/.local/share/gpu-mode/pcie-probe   4-second host->GPU bandwidth probe (built here with nvcc)
#   ~/.config/strata-agent.conf          REPO= and MODEL_DIR= for strata-agent (kept if it exists)
#   ~/.config/gpu-mode.conf              GPU= and VFIO_IDS= detected now (kept if it exists)
# With --system (asks for sudo):
#   /usr/local/sbin/gpu-link-retrain     root-owned, retrains the GPU's PCIe port, takes no input
#   /etc/sudoers.d/gpu-link-retrain      lets your user run exactly that helper without a password
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
MODEL_DIR=$HOME/dev/models/swift-iq3_xxs
SYSTEM=0; FORCE_CONFIG=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir) MODEL_DIR=$2; shift ;;
        --system) SYSTEM=1 ;;
        --force-config) FORCE_CONFIG=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "install: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "$HOME/.local/bin" "$HOME/.local/share/gpu-mode" "$HOME/.config"
install -m 755 "$HERE/bin/strata-agent" "$HERE/bin/gpu-mode" "$HOME/.local/bin/"
echo "install: ~/.local/bin/strata-agent, ~/.local/bin/gpu-mode"

# ---- settings
if [[ $FORCE_CONFIG == 1 || ! -e $HOME/.config/strata-agent.conf ]]; then
    cat > "$HOME/.config/strata-agent.conf" <<EOF
# strata-agent settings (4GB_VRAM_64GB_RAM/install.sh); environment variables of the same name override them
REPO=$REPO
MODEL_DIR=$MODEL_DIR
#HOST=127.0.0.1
#PORT=18090
#CONTEXT=262144
#CHUNK=4096
# 1 = the experts' RAM copy page-locked (~26% faster prompt reading, ~43 GiB locked in all); 0 = ordinary RAM
#PIN_EXPERTS=1
#VISION=1
#VISION_THREADS=12
# watch the PCIe link while serving and warn when it drops to Gen 1 (reads nvidia-smi only)
#WATCH_LINK=1
# 1 = retrain a stuck link by itself through the root helper (install.sh --system). Off by default: a retrain once
# made the GPU fall off the bus (a reboot to recover)
#AUTO_RETRAIN=0
EOF
    echo "install: ~/.config/strata-agent.conf (REPO=$REPO MODEL_DIR=$MODEL_DIR)"
else
    echo "install: ~/.config/strata-agent.conf kept (--force-config rewrites it)"
fi
GPU=$(lspci -Dn -d 10de::0300 2>/dev/null | awk 'NR == 1 { print $1 }' || true)
if [[ -z $GPU ]]; then
    echo "install: WARNING: no NVIDIA VGA controller found (lspci): gpu-mode and the probe need one" >&2
elif [[ $FORCE_CONFIG == 1 || ! -e $HOME/.config/gpu-mode.conf ]]; then
    IDS=$(lspci -Dn -s "${GPU%.*}" | awk '{ print $3 }' | paste -sd, -)
    printf '# gpu-mode: the GPU and every function in its slot (vfio-pci.ids in vm mode)\nGPU=%s\nVFIO_IDS=%s\n' \
        "$GPU" "$IDS" > "$HOME/.config/gpu-mode.conf"
    echo "install: ~/.config/gpu-mode.conf (GPU=$GPU VFIO_IDS=$IDS) - check these against  lspci -nn"
else
    echo "install: ~/.config/gpu-mode.conf kept"
fi

# ---- the PCIe probe (CUDA): nvcc and a host compiler it accepts (newest first, as the repo's Makefile probes)
NVCC=${NVCC:-$(command -v nvcc || ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)}
if [[ -z $NVCC || ! -x $NVCC ]]; then
    echo "install: WARNING: nvcc not found: the PCIe probe is not built (strata-agent then skips the link check)" >&2
else
    t=$(mktemp -d)
    printf 'int main(){return 0;}\n' > "$t/t.cu"
    CCBIN=""
    for c in g++-15 g++-14 g++-13 g++-12 g++; do
        p=$(command -v "$c" 2>/dev/null) || continue
        if "$NVCC" -ccbin "$p" -c "$t/t.cu" -o "$t/t.o" >/dev/null 2>&1; then CCBIN=$p; break; fi
    done
    rm -rf "$t"
    cp "$HERE/tools/pcie-probe.cu" "$HOME/.local/share/gpu-mode/pcie-probe.cu"
    if "$NVCC" -O2 -arch=native ${CCBIN:+-ccbin "$CCBIN"} -o "$HOME/.local/share/gpu-mode/pcie-probe" \
            "$HOME/.local/share/gpu-mode/pcie-probe.cu" 2>/dev/null; then
        echo "install: ~/.local/share/gpu-mode/pcie-probe (host->GPU now: $("$HOME/.local/share/gpu-mode/pcie-probe") GB/s)"
    else
        echo "install: WARNING: building the PCIe probe failed ($NVCC${CCBIN:+, -ccbin $CCBIN})" >&2
    fi
fi

# ---- the root helper for the stuck PCIe link
if [[ $SYSTEM == 1 ]]; then
    [[ -n $GPU ]] || { echo "install: --system needs the GPU's PCI address" >&2; exit 1; }
    PORT=$(basename "$(readlink -f "/sys/bus/pci/devices/$GPU/..")")
    t=$(mktemp -d)
    sed -e "s|@PORT@|$PORT|g" -e "s|@GPU@|$GPU|g" "$HERE/system/gpu-link-retrain.in" > "$t/gpu-link-retrain"
    sed -e "s|@USER@|$(id -un)|g" "$HERE/system/sudoers-gpu-link-retrain.in" > "$t/sudoers"
    sudo install -o root -g root -m 755 "$t/gpu-link-retrain" /usr/local/sbin/gpu-link-retrain
    sudo install -o root -g root -m 440 "$t/sudoers" /etc/sudoers.d/gpu-link-retrain
    sudo visudo -cf /etc/sudoers.d/gpu-link-retrain
    rm -rf "$t"
    echo "install: /usr/local/sbin/gpu-link-retrain (port $PORT) and /etc/sudoers.d/gpu-link-retrain"
fi

case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "install: add ~/.local/bin to PATH (e.g. in ~/.bashrc: export PATH=\$HOME/.local/bin:\$PATH)" ;;
esac
echo "install: done.  Next: strata-agent   (README.md: Run)"
