# Qwen3.8-Flash-Next on a 4 GB GPU with 64 GB of RAM

This folder makes **Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO abliterated, IQ3_XXS**
([SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
a 76 GB model) serve on a laptop with a **4 GB RTX 3050 Ti** and **64 GB of RAM**, tuned for agent use: long
system prompts, many turns, images. Everything here was measured on that laptop on 2026-11-08.

| What | Result |
|---|---|
| Context window | 262,144 tokens, the default (the speeds below were measured at 131,072 - see [Context size](#context-size-262k-default-and-128k)) |
| Cold read of a 25K-token system prompt | **460-478 tok/s** (52-54 s) |
| Agent turn with a ~3K-token tool output | 385-394 tok/s (7.6-8.0 s) |
| Next turn / new conversation with the same system prompt | the prompt is reused from the cache: ~25 new tokens, ~1 s |
| Fresh prompts of 1K to 75K tokens | 158-184 tok/s with the earlier 768-token chunks (not re-measured at 4096) |
| Decode | ~17-19 tok/s (with 768-token chunks: 20.6 near an empty context, 16.3 at 75K tokens) |
| Images | yes: encoded on the CPU, ~8 s for a 260-token picture |
| VRAM left free while serving | ~212 MiB of 4,096 (~220 at 128K) |
| RAM | ~44 GiB page-locked (the experts, the K/V, and copies of what the GPU lends out); ~45 GiB must be free at start |

The starting point was an engine that stopped at load with `native head upload: out of memory (236 MiB of 3767 MiB
VRAM free)` at a 2,048-token context. Getting from there to the table above took seven engine changes (all opt-in,
[listed below](#engine-changes-in-this-fork)), a launcher, and fixing a PCIe link that the NVIDIA driver kept at a
fraction of its speed.

**Contents**

1. [The machine](#the-machine)
2. [Quick start](#quick-start)
3. [Step 1 - system packages](#step-1---system-packages)
4. [Step 2 - build the engine](#step-2---build-the-engine)
5. [Step 3 - get the model](#step-3---get-the-model)
6. [Step 4 - install the tools](#step-4---install-the-tools)
7. [Step 5 - give the GPU to the host (gpu-mode)](#step-5---give-the-gpu-to-the-host-gpu-mode)
8. [Step 6 - run](#step-6---run)
9. [Step 7 - benchmark](#step-7---benchmark)
10. [The PCIe link problem](#the-pcie-link-problem)
11. [How the configuration was found](#how-the-configuration-was-found)
12. [Engine changes in this fork](#engine-changes-in-this-fork)
13. [Context size: 262K (default) and 128K](#context-size-262k-default-and-128k)
14. [Troubleshooting](#troubleshooting)
15. [Undo everything](#undo-everything)
16. [Files in this folder](#files-in-this-folder)

## The machine

| Part | Version |
|---|---|
| Laptop | HP (subsystem 103c:8a26) |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU, GA107M, 4 GB (3,767 MiB usable by CUDA), PCIe Gen 4 x8, 60 W on AC / 20 W on battery |
| CPU | Intel Core i7-12700H (6 P-cores + 8 E-cores, 20 threads, AVX2, no AVX-512) |
| RAM | 64 GB (62 GiB visible), 64 GB swap file |
| Disk | NVMe; the model needs ~78 GB in the Hugging Face cache + 2.3 GB for the pack and MTP files |
| OS | Fedora Linux 44, kernel 7.1.13-200.fc44, GNOME on Wayland (desktop on the Intel iGPU) |
| NVIDIA driver | 610.57.04 from NVIDIA's `cuda-fedora44` repository (`cuda-drivers`: `nvidia-driver`, `kmod-nvidia-latest-dkms`), GSP firmware on |
| CUDA | 13.3 (`cuda-toolkit-13-3`, in `/usr/local/cuda-13.3`) |
| Compilers | GCC 16 is the system default; nvcc 13.3 needs GCC 15 or older, so `gcc15` / `gcc15-c++` (15.3.1) are installed |
| Engine | this fork, branch `4gb-vram` (upstream UnbiasedStrata engine 0.1.45) |

Other 4 GB / 64 GB machines should work the same way. What matters: CUDA compute capability 7.5 or newer (the build
here is for 8.6), about 45 GiB of RAM free when the engine starts (about 44 GiB of it is page-locked), and a PCIe link
that actually runs at its rated speed under load (see [the PCIe link problem](#the-pcie-link-problem)).

## Quick start

```bash
# 1-2: packages, then the engine (about 10 minutes of building)
git clone https://github.com/bassiounix/UnbiasedStrata ~/dev/UnbiasedStrata
cd ~/dev/UnbiasedStrata && git checkout 4gb-vram && git submodule update --init
make build CUDA_ARCHS=86                      # your GPU's compute capability, without the dot

# 3: the model (~76 GB download) -> ~/dev/models/swift-iq3_xxs
./4GB_VRAM_64GB_RAM/prepare-model.sh

# 4: the tools on the PATH (+ the root helper for retraining the PCIe link)
./4GB_VRAM_64GB_RAM/install.sh --system

# 5: if the GPU was set up for VM passthrough: give it to the host, reboot
gpu-mode status
gpu-mode host && sudo reboot

# 6: serve (OpenAI-compatible API on http://127.0.0.1:18090/v1)
strata-agent

# 7: in another terminal, once it says "listening on"
python3 4GB_VRAM_64GB_RAM/bench/bench.py agent
```

## Step 1 - system packages

On Fedora 44 (adapt the package names for other distributions):

```bash
# NVIDIA driver + CUDA 13.3 from NVIDIA's repository (the driver must be >= 580 for CUDA 13)
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install cuda-toolkit-13-3 cuda-drivers       # on the laptop: nvidia-driver 610.57.04 + kmod-nvidia-latest-dkms
# build tools: nvcc 13.3 accepts GCC <= 15, Fedora 44's default is 16
sudo dnf install git cmake make gcc15 gcc15-c++ python3 python3-pip pciutils
# the Hugging Face CLI, for the download
pip install --user -U huggingface_hub                  # gives `hf`; or: uv tool install huggingface_hub
```

Check before going on:

```bash
nvidia-smi                                     # driver 580 or newer, the GPU listed
/usr/local/cuda-13.3/bin/nvcc --version        # release 13.3
g++-15 --version                               # GCC 15
hf version
free -g                                        # ~44 GiB or more "available" when nothing else runs
```

Put CUDA on the PATH if it is not (`export PATH=/usr/local/cuda-13.3/bin:$PATH` in `~/.bashrc`).

## Step 2 - build the engine

```bash
git clone https://github.com/bassiounix/UnbiasedStrata ~/dev/UnbiasedStrata
cd ~/dev/UnbiasedStrata
git checkout 4gb-vram
git submodule update --init                    # third_party/llama.cpp: the front end and gguf-py
make build CUDA_ARCHS=86                       # RTX 30xx = 86, RTX 20xx = 75, RTX 40xx = 89
ls build/bin/                                  # strata  strata-poly  strata-dsv41  unbiased-strata  libstrata_kq.so
```

`make build` probes for a host compiler nvcc accepts (newest first: `g++-15`, `g++-14`, ...) and configures CMake
with it; on this laptop that resolved to `/usr/bin/g++-15`. The equivalent CMake configuration is in
`build/CMakeCache.txt` (`CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=86`,
`CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`, `CMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc`). After editing the
engine, `make -C build strata -j` rebuilds just the engine.

## Step 3 - get the model

The files come from
[SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF](https://huggingface.co/SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF),
revision `3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3` (the one measured here):

| File in the repository | Size | Used for |
|---|---:|---|
| `IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf` | 39,771,812,960 | the model, shard 1 (also the vocabulary, chat template, PLE table) |
| `IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf` | 36,056,143,040 | the model, shard 2 |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,542,944 | the vision projector (images) |
| `mtp-q2_0.gguf` | ~889 MB | the MTP draft layer (optional; it gives no speed-up on a 4 GB card) |

`prepare-model.sh` downloads them and builds the directory the launcher reads:

```bash
./4GB_VRAM_64GB_RAM/prepare-model.sh                       # -> ~/dev/models/swift-iq3_xxs
./4GB_VRAM_64GB_RAM/prepare-model.sh /data/swift-iq3_xxs   # anywhere else (then install.sh --model-dir /data/swift-iq3_xxs)
./4GB_VRAM_64GB_RAM/prepare-model.sh --no-mtp              # skip the MTP files (they are not used by strata-agent)
```

The result (the GGUFs stay in the Hugging Face cache, `~/.cache/huggingface/hub`, and are linked):

```
~/dev/models/swift-iq3_xxs/
├── Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf -> ~/.cache/huggingface/hub/models--SC117--.../snapshots/3db919a.../IQ3_XXS/...   (38 GB)
├── Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf -> .../IQ3_XXS/...   (34 GB)
├── mmproj-Qwen3.8-Flash-Next-BF16.gguf -> .../mmproj-Qwen3.8-Flash-Next-BF16.gguf                  (866 MB)
├── pack/                                      tools/iq_pack.py --gguf <shard 1>               (1.5 GB)
│   ├── dense.bin                              the dense weights the engine loads onto the GPU
│   ├── index.txt                              the tensor table; experts and quantized projections stay in the GGUF
│   ├── conversions.json                       what was converted, with source hashes
│   ├── native_experts.txt                     which shard holds each layer's experts
│   └── tokenizer/                             chat_template.jinja merges.txt tokenizer.json token_type.json vocab.json
└── mtp/                                       optional
    ├── mtp-q2_0.gguf -> .../mtp-q2_0.gguf                                                    (848 MB)
    └── rt/                                    tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
        ├── dense.bin  dense.txt  experts.bin
        └── draft_vocab.bin                    a copy of the repository's data/draft_vocab.bin
```

What the script runs, if you prefer doing it by hand:

```bash
R=SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF; REV=3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3
N=Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS
SNAP=$(hf download $R IQ3_XXS/$N-00001-of-00002.gguf IQ3_XXS/$N-00002-of-00002.gguf \
         mmproj-Qwen3.8-Flash-Next-BF16.gguf mtp-q2_0.gguf --revision $REV --quiet | tail -1)
M=~/dev/models/swift-iq3_xxs; mkdir -p $M/mtp
ln -s $SNAP/IQ3_XXS/$N-0000{1,2}-of-00002.gguf $SNAP/mmproj-Qwen3.8-Flash-Next-BF16.gguf $M/
ln -s $SNAP/mtp-q2_0.gguf $M/mtp/
cd ~/dev/UnbiasedStrata
python3 -m venv .venv-tools && .venv-tools/bin/pip install -r tools/requirements.txt    # numpy, regex
.venv-tools/bin/python tools/iq_pack.py --gguf $M/$N-00001-of-00002.gguf --out $M/pack
.venv-tools/bin/python tools/mtp_rt.py --gguf $M/mtp/mtp-q2_0.gguf --out $M/mtp/rt
cp data/draft_vocab.bin $M/mtp/rt/
```

Both builds are deterministic: the script, run on a second directory, produced a `pack/` and `mtp/rt/` byte-identical
to the ones every measurement here used. The expert profile (`data/expert-profile.bin`, which experts are used most)
is part of the repository.

## Step 4 - install the tools

```bash
./4GB_VRAM_64GB_RAM/install.sh                 # user part only
./4GB_VRAM_64GB_RAM/install.sh --system        # also the root helper for retraining the PCIe link (asks for sudo)
./4GB_VRAM_64GB_RAM/install.sh --model-dir /data/swift-iq3_xxs --force-config   # another model directory
```

What it puts where:

| Path | What |
|---|---|
| `~/.local/bin/strata-agent` | the launcher: the measured configuration, a PCIe check at start and a watch on the link while serving |
| `~/.local/bin/gpu-mode` | who owns the GPU at boot (host / VM), status, PCIe link retrain |
| `~/.local/share/gpu-mode/pcie-probe` (+ `.cu`) | 4-second host-to-GPU bandwidth probe, built with nvcc for your GPU |
| `~/.config/strata-agent.conf` | `REPO=` and `MODEL_DIR=` (plus commented defaults: HOST, PORT, CONTEXT, CHUNK, VISION, VISION_THREADS) |
| `~/.config/gpu-mode.conf` | `GPU=` (PCI address) and `VFIO_IDS=` (every function of the GPU's slot), detected with `lspci` |
| `/usr/local/sbin/gpu-link-retrain` | `--system` only: root-owned script that retrains the GPU's PCIe port; no input, no config |
| `/etc/sudoers.d/gpu-link-retrain` | `--system` only: `<you> ALL=(root) NOPASSWD: /usr/local/sbin/gpu-link-retrain` |

`~/.local/bin` must be on the PATH (`install.sh` says so if it is not). Existing config files are kept unless
`--force-config` is given. Check `~/.config/gpu-mode.conf` against `lspci -nn`: on the laptop it is
`GPU=0000:01:00.0` and `VFIO_IDS=10de:25a0,10de:2291` (the GPU and its HDMI audio).

## Step 5 - give the GPU to the host (gpu-mode)

Skip this step if you never set the GPU up for virtual machines (`gpu-mode status` then says `this boot: host` and
`GNOME: may use the NVIDIA GPU`).

### Background

The laptop was set up, following a VFIO GPU-passthrough guide, to hand the RTX 3050 Ti to a Windows VM (libvirt,
`win11`, with the GPU at `01:00.0`, its audio at `01:00.1` and an NVMe at `07:00.0` as `managed='yes'` host
devices, and Looking Glass). That setup consisted of:

- kernel arguments `intel_iommu=on iommu=pt rd.driver.pre=vfio-pci vfio-pci.ids=10de:25a0,10de:2291` (vfio-pci claims
  the GPU at boot);
- `/etc/dracut.conf.d/local.conf`: `add_drivers+=" vfio vfio_iommu_type1 vfio_pci "` (vfio in the initramfs);
- `/etc/modprobe.d/kvmfr.conf` and `/etc/modules-load.d/kvmfr.conf` (Looking Glass's shared-memory module);
- `/etc/udev/rules.d/61-mutter-ignore-nvidia.rules`, which tags every NVIDIA DRM card `mutter-device-ignore` so GNOME
  never holds the GPU (otherwise it cannot be detached for the VM).

To use the GPU on the host, it was handed back at runtime:
`sudo virsh nodedev-reattach pci_0000_01_00_0; sudo rmmod vfio_pci vfio_pci_core vfio_iommu_type1; sudo modprobe -i nvidia_modeset nvidia_uvm nvidia`.
That works, but it left the PCIe link at Gen 1 (`nvidia-smi` reported `pcie.link.gen.max = 1`, 1.7 GB/s instead of
~13), and with the udev rule in place GNOME ignored the GPU, so the HDMI port (wired to the NVIDIA GPU on this
laptop) gave sound but no picture.

### What gpu-mode does

```
gpu-mode status              this boot, the next boot, driver, PCIe link, power cap, GNOME, battery
gpu-mode host                next boot: the NVIDIA driver takes the GPU at boot (full speed, HDMI works)
gpu-mode vm                  next boot: vfio-pci claims the GPU at boot, for passing it to a VM
gpu-mode host|vm --dry-run   show the edits without writing anything
gpu-mode retrain             the PCIe link is stuck at Gen 1 under load: retrain the GPU's port at full speed
```

`host` and `vm` move exactly two things (and take sudo for it):

1. the two boot arguments `rd.driver.pre=vfio-pci vfio-pci.ids=<VFIO_IDS>`, in all three places Fedora reads them:
   `/etc/kernel/cmdline` (copied into every kernel installed later - if this one is missed, the next kernel update
   silently switches back), `GRUB_CMDLINE_LINUX` in `/etc/default/grub`, and the installed boot entries
   (`grubby --update-kernel=ALL`);
2. the udev rule: `host` renames `/etc/udev/rules.d/61-mutter-ignore-nvidia.rules` to `*.gpu-mode-off` (GNOME may use
   the GPU, HDMI works), `vm` puts it back (or writes it if it never existed).

Everything else (IOMMU arguments, `nvidia-drm.modeset=1`, the initramfs, Looking Glass, the libvirt VM) stays, so a
VM can be used again after `gpu-mode vm && sudo reboot`. Backups of the two files are written as
`*.gpu-mode-<timestamp>`. `gpu-mode` is written for Fedora (BLS boot entries, `grubby`); on another distribution only
`status` and `retrain` apply as they are.

```bash
gpu-mode host --dry-run      # read what it will change
gpu-mode host && sudo reboot
gpu-mode status              # this boot: host, GPU driver: nvidia, GNOME: may use the NVIDIA GPU
```

Switching to a VM without a reboot is possible (libvirt detaches a `managed='yes'` device itself) but not worth it:
every program that opened the GPU (on this desktop: xdg-desktop-portal, Nautilus, the terminal, editors) has to be
closed and the NVIDIA modules unloaded first, and handing the GPU back afterwards is exactly what left the link at
Gen 1.

### Notes

- `lspci` shows `Kernel modules: nouveau, nvidia_drm, nvidia` for the GPU: that is the list of drivers that *could*
  take it; `Kernel driver in use: nvidia` is what matters (`nouveau` is blacklisted on the kernel command line).
- The GPU's HDMI audio function (`01:00.1`) is driven by `snd_hda_intel` with the `snd_hda_codec_nvhdmi` codec
  driver - correct on every NVIDIA card: the `nvidia` module only claims display-class devices.
- `nvidia-smi` showing `8MiB / 4096MiB` is memory *used*, not the card's size.
- On battery the firmware caps the GPU at 20 W (`enforced.power.limit`), which also keeps the PCIe link at Gen 2 or
  below: plug in the charger. `gpu-mode status` and `strata-agent` warn about it.
- Once GNOME may use the GPU, an external screen on the HDMI port takes some of the GPU's 4 GB for its images; the
  configuration here leaves ~220 MiB spare at rest. If a screen takes more, the engine steps down to a smaller prompt
  chunk (3072, 2048, ...) and `strata-agent` says so. Measured with no external screen attached.

## Step 6 - run

```bash
strata-agent
```

It prints, before and while loading:

```
strata-agent: PCIe host->GPU 13.0 GB/s          # ~13 = healthy; under 5 it warns (see the PCIe link problem)
strata generate: PCIe probe: 13.0 GB/s host->device ...
FileExpertSource: ... resident 39.96 GiB ...                                         # all experts in RAM, page-locked
strata serve: ~220 MiB of VRAM free with everything loaded - LOW: ...                 # expected: ignore the advice
... listening on http://127.0.0.1:18090, context 262144 tokens
```

The engine's "LOW: add --vram-reserve-mib ..." advice is written for bigger cards; following it does not fit here.
Loading takes about a minute: 64 s at the 262K default (one run), 43-59 s at 128K (three runs), ~34 s at 128K with
`PIN_EXPERTS=0`. Page-locking the 40 GiB copy of the experts is ~42 s of it.
If the experts cannot be page-locked (not enough free RAM), the engine falls back to ordinary RAM by itself and
`strata-agent` warns that prompt reading is then ~20% slower.

The server speaks the OpenAI chat API (as llama.cpp's server does): `http://127.0.0.1:18090/v1/chat/completions`,
model name `swift-iq3_xxs`, streaming and not, plus `/v1/models`, `/health`, `/tokenize`. Images are `image_url`
parts (data URIs work). Reasoning is on by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off
per request. One request is served at a time.

Settings: `~/.config/strata-agent.conf`, or the same names in the environment for one run, e.g.
`PORT=8080 strata-agent`, `VISION=0 strata-agent`, `PIN_EXPERTS=0 strata-agent`, `CHUNK=2048 strata-agent`. The
settings are REPO, MODEL_DIR, HOST, PORT, CONTEXT (262144), CHUNK (4096), PIN_EXPERTS (1), VISION (1),
VISION_THREADS (12), WATCH_LINK (1) and AUTO_RETRAIN (0). Extra arguments are
passed to the engine. What every engine option and variable in the launcher does is commented in
[bin/strata-agent](bin/strata-agent) and explained in [How the configuration was found](#how-the-configuration-was-found).

The exact command it runs (with the default settings):

```bash
cd ~/dev/UnbiasedStrata
M=~/dev/models/swift-iq3_xxs
SH1=$M/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf
CUDA_MODULE_LOADING=LAZY STRATA_HC_REQ8=1 \
STRATA_IDX_HOST=1 STRATA_PREFILL_IN_HEAD=1 STRATA_PREFILL_IN_DENSE=1 \
STRATA_PF_GEMM_SCRATCH_MIB=16 STRATA_PF_GEMM_WS_MIB=4 \
build/bin/unbiased-strata --engine build/bin/strata -m $SH1 \
  --host 127.0.0.1 --port 18090 --alias swift-iq3_xxs \
  --mmproj $M/mmproj-Qwen3.8-Flash-Next-BF16.gguf --threads 12 \
  -- --pack $M/pack --native $SH1 \
     --expert-profile data/expert-profile.bin --expert-cache 1 --vram-reserve-mib 150 \
     --prefill 4096 --spec 2 --suffix-draft 0 --short-read 0 \
     --max-context 262144 --kv q4_0 --kv-resident 20480 --resident-experts --vision
```

### Images

The vision projector runs in the front end on the CPU (`--mmproj ... --threads 12`); the engine takes the encoded
rows (`--vision`, ~2 MiB of VRAM). `--mmproj-gpu` would need ~1 GB of VRAM, which a 4 GB card does not have. A
260-token picture encoded in 7.9 s with 12 threads (9.0 s with 6, the default; 8.6 s with 14 - the E-cores slow it
down); the front end keeps the rows of the last 64 pictures, so a resent picture is not encoded again. The repo's
docs recommend a Q8_0 projector for CPU use; `llama-quantize` (this version and the bundled llama.cpp) refuses
projector files (`unsupported model architecture: 'clip'`), so the BF16 one is used. Checked with a generated picture
("PURPLE LLAMA 7319", a red circle, a blue square): text and shapes read back exactly.

## Step 7 - benchmark

With `strata-agent` running:

```bash
python3 4GB_VRAM_64GB_RAM/bench/bench.py agent      # ~5 min: the agent session below
python3 4GB_VRAM_64GB_RAM/bench/bench.py context    # ~15 min: fresh 1K..64K-token prompts
python3 4GB_VRAM_64GB_RAM/bench/bench.py context --lengths 2048 8192 --gen 128 --json out.json
```

It needs only Python's standard library. Prompts are built from the repository's `docs/*.md`; each fresh prompt
starts with a random tag, so it is read cold without restarting the server. The numbers are the server's own
`timings` from each response. `agent` is a 25K-token system prompt with a task, the next turn, a new conversation with
the same system prompt, then three turns that each add a ~2.5K-token tool output. `context` sends prompts of 1K, 4K,
16K, 32K and 64K tokens and generates 256 tokens after each.

Run it on AC power, with the PCIe link checked (`strata-agent` prints it), and nothing else using the GPU or a lot of
RAM. Results on the laptop: [Measured results](#measured-results).

## The PCIe link problem

The single biggest factor in prompt-reading speed on a 4 GB card is the PCIe link: the 40 GB of experts live in RAM,
and every prompt chunk moves a large share of them to the GPU.

**Symptom:** prompt reading ~77 tok/s instead of ~460-480 (with the earlier 768-token chunks: ~35-40 instead of
~180), and the head and projections take ~2.3 s instead of ~0.17 s to be written back after each prompt; `strata-agent` or the engine prints a PCIe probe of
~1 GB/s instead of ~13; `nvidia-smi --query-gpu=pcie.link.gen.current` stays at 1 *under load* (at idle it is
always 1 - that is normal power saving).

**What was found:**

- The link trains at 16 GT/s x8 at boot (kernel log: `126.024 Gb/s available PCIe bandwidth, limited by 16.0 GT/s
  PCIe x8 link`). The NVIDIA driver then lowers it when idle and raises it under load - but sometimes it lowers it and
  never raises it again, even at P0 and 100% load: the link stays at 2.5 GT/s, ~0.9-1.7 GB/s.
- Both ends were set to target 16 GT/s when it was stuck (`lspci -vvv`: `LnkCtl2: Target Link Speed: 16GT/s` on
  `00:01.0` and `01:00.0`, `LnkSta: ... (downgraded)`), there were no link errors (`nvidia-smi -q`: 0 replays), no
  throttling (clock event reason counters at 0), AC power, normal temperatures.
- It happened after the vfio-pci to NVIDIA runtime handoff, once right after a boot, once a few minutes after a boot
  with the GPU idle, and twice in the middle of heavy benchmark runs; between those it stayed fine for hours.
- **Ruled out:** runtime power-off of the GPU (D3cold): it also happened with `power/control=on` and no power-off
  since boot; NVIDIA persistence mode (`nvidia-smi -pm 1`): it happened with it on too; battery; link errors; the
  kernel's thermal zones (the `PCIe_Port_Link_Speed_*` cooling devices' `cur_state` only mirrors the current speed).
  The trigger is not known.

**Fix:** retrain the link. Writing 0 to the GPU port's PCIe link-speed cooling device makes the kernel's bandwidth
controller (Linux 6.13+) set the target to full speed and retrain; after that the driver's own up/down switching works
again until the next time:

```bash
gpu-mode retrain                                  # asks for sudo, or uses the helper without a password
sudo -n /usr/local/sbin/gpu-link-retrain          # the helper (install.sh --system)
~/.local/share/gpu-mode/pcie-probe                # ~13 again
```

The write sometimes reports `Resource temporarily unavailable` or blocks for a while; the retrain happens anyway
(both are handled).

**A retrain is not always safe.** It worked about ten times - idle, and three times in the middle of a request with
the answer unaffected - but once the bandwidth probe that ran right after a retrain made the GPU fall off the bus
(kernel: `NVRM: Xid (PCI:0000:01:00): 79, pid=..., name=pcie-probe, GPU has fallen off the bus`, then `Xid 154,
GPU recovery action changed ... (PF FLR)`; `nvidia-smi`: `Unable to determine the device handle for GPU0`). Only a
reboot brings it back. So retraining is your decision, made between requests, not something done behind your back.

**How strata-agent deals with it:**

- **At start** (4 s): it measures the link with `pcie-probe`. Under 5 GB/s it warns and tells you to run
  `gpu-mode retrain` and start again.
- **While serving**: it watches `nvidia-smi` every 2 s (GPU utilization and link generation; nothing is done to the
  GPU) and warns when the GPU is busy (10% or more) but the link stays at Gen 1 for 5 samples in a row. On a healthy
  link every busy sample was at Gen 4 (144 of 144, prompt reading and decoding), and idle samples at Gen 1 are
  normal power saving, so they are ignored. `WATCH_LINK=0` turns it off.
- **`AUTO_RETRAIN=1`** (environment or `~/.config/strata-agent.conf`, needs `install.sh --system`) lets it retrain by
  itself: at start (then it waits 5 s before measuring again) and while serving, at most once every 5 minutes. Off
  by default because of the crash above.

How often it happens: on 2026-11-08, after the vfio handoff, once right after boot, once a few minutes after boot,
in the afternoon several times within a minute of a healthy start (one run read at 41 tok/s although the start
measured 13 GB/s), and twice more during later runs, once in the middle of plain decoding.

## How the configuration was found

### The VRAM budget

The card has 3,767 MiB for CUDA. What has to be there at 128K context:

| Item | VRAM |
|---|---:|
| CUDA context | ~100 MiB |
| Pack weights on the GPU (routers, PLE key, indexer projections, ...) with int8 hyper-connections | 266 MiB |
| Native projection matrices (attention, GDN, shared experts) | 1,649 MiB |
| Output head (Q5_K, 248,320 x 2,560) - also lent to the prompt path while a prompt is read | 417 MiB |
| Session: GDN state, buffers, 20,480 K/V cells per attention layer (q4_0) | ~330 MiB |
| Prompt path beyond the head's VRAM (cuBLAS, kernels, the part of a 768-token chunk that does not fit) | ~200 MiB |
| Verify window (2 rows) and its graphs, 1 expert cache slot | ~60 MiB |
| **Left free** | **~110 MiB** |

That table is the 768-token configuration. The current default differs in three ways: with
`STRATA_PREFILL_IN_DENSE=1` the 300 projection matrices are one 1.6 GiB block (118 MiB less than 300 separate
allocations), and that block is lent to the prompt path too while a prompt is read; the page-locked experts add
~80 MiB of page tables; and the prompt path's own part grows with the 4096-token chunk. At rest ~220 MiB is free.

In RAM: the experts (39.96 GiB, page-locked), the K/V of all cells (pinned), the indexer keys and RoPE table (pinned),
the head's copy (417 MiB, pinned), the projections' copy (1.63 GiB, pinned), the vision projector (~1 GB in the front
end). About 43 GiB in all is page-locked.

### From "out of memory at load" to serving

| Step | Change | Why |
|---|---|---|
| 1 | `STRATA_HC_REQ8=1` | the hyper-connection projections as int8: 1,200 -> 675 MiB, so the 417 MiB head fits |
| 2 | `CUDA_MODULE_LOADING=LAZY` | ~50 MiB less taken by CUDA kernels |
| 3 | `--expert-cache 1 --vram-reserve-mib ...` | the verify path refuses to run without an expert cache; `--expert-cache 0` means "auto", which finds no room; the default 700 MiB reserve sized an explicit cache to 0 |
| 4 | `--resident-experts` + `STRATA_RESIDENT_PIN=0` | the experts the GPU does not hold kept in RAM, pageable: decode 7.8 -> 14.5 tok/s, prompt 2.9 -> 4.7 (at the time, on the slow link); pinning them costs ~80 MiB of VRAM in page tables |
| 5 | `--suffix-draft 0` | prompt-lookup drafting makes the verify window 4 rows; its graph did not fit |
| 6 | `--kv q4_0 --kv-resident 20480` | 4-bit K/V, streamed from RAM beyond 20,480 cells per layer (the engine's minimum) |
| 7 | `STRATA_IDX_HOST=1` *(new)* | at 262K the indexer's pooled keys (383 MiB) and RoPE table (64 MiB) move to pinned RAM |
| 8 | `STRATA_PREFILL_IN_HEAD=1` *(new)* | the prompt path's buffers (and its 151 MiB K/V staging pool at 262K) live in the output head's VRAM while a prompt is read; the head is written back from RAM before decoding (34 ms on a healthy link) |
| 9 | the PCIe link fixed | prompt reading ~40 -> 83 tok/s at the same settings |
| 10 | 128K context, `--prefill 768`, `STRATA_PF_GEMM_SCRATCH_MIB=16 STRATA_PF_GEMM_WS_MIB=4` *(new)* | a bigger chunk reads faster; the smaller GEMM scratch/workspace (76 MiB) is what makes 768 fit |
| 11 | `STRATA_PREFILL_IN_DENSE=1` *(new)*, `--prefill 4096` | the dense projections' 1.6 GiB of VRAM is lent to the prompt path too (each layer's projections are streamed into two slots from a pinned copy while a prompt is read, and the block is written back afterwards, ~0.17 s with the head): a 4096-token chunk fits, 182 -> 380 tok/s |
| 12 | page-locked experts (`STRATA_RESIDENT_PIN` unset, `PIN_EXPERTS=1`) | the GPU copies the experts straight from RAM instead of through CPU staging copies (16 s of a 25K read): 380 -> 462-478 tok/s |

### Measured results

All on a healthy link (13 GB/s), AC power, the 25K-token agent session (`bench.py agent` or its predecessor):

| Configuration | Cold 25K read | Decode | VRAM free | Stable |
|---|---:|---:|---:|---|
| 262K context, chunk 128, drafting on | 83.3 tok/s | 18.6 tok/s | 112 MiB | yes |
| 262K context, chunk 256, drafting off | - | - | 50 MiB | no: verify graph out of memory at the first token |
| 128K context, chunk 512, drafting off | 152.6 tok/s | 19.1 tok/s | 126 MiB | yes |
| 128K context, chunk 512, drafting on | 147.3 tok/s | 17.7-19.4 tok/s | 114 MiB | yes |
| 128K context, chunk 512, all experts on the GPU (`STRATA_PREFILL_CPU_SHARE=0`) | 114.8 tok/s | 16.9 tok/s | 126 MiB | yes |
| 128K context, chunk 768, drafting on | 184.2 tok/s | 18.5 tok/s | 98 MiB | no: verify graph out of memory on the 2nd request |
| **128K context, chunk 768, drafting off (strata-agent)** | **186.5 tok/s** | **18.8-19.9 tok/s** | **110 MiB** | **yes** |
| 128K, chunk 768 + MTP (weights in RAM, full head, `--spec 2`) | 182.8 tok/s | 19.4 tok/s (63% of drafts accepted) | 72 MiB | yes |
| 128K, chunk 1024 or 2048 (without `STRATA_PREFILL_IN_DENSE`) | - | - | - | does not fit: the engine steps down to 512 |
| 128K, `STRATA_PREFILL_IN_DENSE=1`, chunk 4096, pageable experts | 380 tok/s | 19.6 tok/s | 300 MiB | yes |
| **128K, `STRATA_PREFILL_IN_DENSE=1`, chunk 4096, page-locked experts (strata-agent now)** | **462-478 tok/s** | **17.2 tok/s** (controlled; 17.3 for the 768 setup on the same prompts) | **220 MiB** | **yes** |
| the same + `STRATA_PREFILL_EQUAL=1` | 462 tok/s | | | no gain, not used |

The 768-token configuration (strata-agent's earlier default) through a whole agent session:

| Request | Read fresh | Reused | Read time | Prompt reading | Decode |
|---|---:|---:|---:|---:|---:|
| Cold: 25K system prompt + task | 24,941 | 0 | 133.7 s | 186.5 tok/s | 18.9 tok/s |
| Next turn, same conversation | 26 | 25,063 | 0.9 s | - | 19.3 tok/s |
| New conversation, same system prompt | 27 | 24,915 | 0.9 s | - | 19.9 tok/s |
| Agent turn 1: +2.5K-token tool output | 2,494 | 24,915 | 15.7 s | 158.7 tok/s | 19.1 tok/s |
| Agent turn 2 | 2,477 | 27,509 | 15.6 s | 158.5 tok/s | 19.1 tok/s |
| Agent turn 3 | 2,343 | 30,086 | 14.1 s | 165.8 tok/s | 18.8 tok/s |

So a long system prompt is read once per server start; after that every turn and every new conversation with the
same system prompt only reads what is new. `bench.py` results from the final scripts: [below](#benchmark-runs-of-the-final-scripts).

### Benchmark runs of the final scripts

**Current default** (chunk 4096, `STRATA_PREFILL_IN_DENSE=1`, page-locked experts), `bench.py agent`:

| Request | Tokens read fresh | Read time | Prompt reading |
|---|---:|---:|---:|
| cold: ~25K-token system prompt + task | 24,955 | 54.2 s | 460 tok/s (from count and time) |
| next turn, same conversation | 22 | 1.0 s | - |
| new conversation, same system prompt | 27 | 1.0 s | - |
| agent turn 1: +~2.5K-token tool output | 3,103 | 8.0 s | 391.7 tok/s |
| agent turn 2 | 2,989 | 7.6 s | 394.2 tok/s |
| agent turn 3 | 2,897 | 7.6 s | 385.1 tok/s |

Decode with this default: ~17-19 tok/s. With `STRATA_PREFILL_IN_DENSE` on and off at the same chunk size the greedy
output was byte-identical; a different chunk size changes the rounding, as any chunk change does.

Why it stops near 480: every chunk of more than ~300 tokens touches all 24,576 experts (the top 10 of 512 per token and
layer), so 42.9 GB crosses PCIe per chunk - 3.3 s at 12.9 GB/s, whatever the chunk size; a bigger chunk spreads it over
more tokens. At 4096 the GPU's own work is ~1.77 ms per token (a ceiling of ~565 tok/s) and the PCIe wait is ~7 s of a
51 s read.

**The earlier default** (chunk 768, pageable experts): `install.sh`, then `strata-agent` (which found the link stuck at 1.3 GB/s, retrained it through the helper and started
at 13.0 GB/s; 108 MiB of VRAM free with vision on), then `bench.py agent` and `bench.py context`, 2026-11-08:

```
$ python3 4GB_VRAM_64GB_RAM/bench/bench.py agent
```

| Request | Tokens read fresh | Reused from cache | Read time | Prompt reading | Generated | Decode |
|---|---:|---:|---:|---:|---:|---:|
| cold: ~25K-token system prompt + task | 24,955 | 0 | 137.2 s | 181.9 tok/s | 128 | 19.2 tok/s |
| next turn, same conversation | 23 | 25,082 | 1.0 s | 22.8 tok/s | 25 | 18.7 tok/s |
| new conversation, same system prompt | 27 | 24,929 | 1.0 s | 26.4 tok/s | 59 | 19.2 tok/s |
| agent turn 1: +~2.5K-token tool output | 3,151 | 24,929 | 19.4 s | 162.8 tok/s | 55 | 19.0 tok/s |
| agent turn 2: +~2.5K-token tool output | 2,989 | 28,134 | 17.4 s | 171.9 tok/s | 55 | 18.8 tok/s |
| agent turn 3: +~2.5K-token tool output | 2,897 | 31,177 | 17.1 s | 169.2 tok/s | 55 | 19.0 tok/s |

```
$ python3 4GB_VRAM_64GB_RAM/bench/bench.py context
```

| Request | Tokens read fresh | Reused from cache | Read time | Prompt reading | Generated | Decode |
|---|---:|---:|---:|---:|---:|---:|
| fresh ~1K-token prompt | 926 | 0 | 5.9 s | 158.2 tok/s | 256 | 20.6 tok/s |
| fresh ~4K-token prompt | 3,396 | 0 | 19.3 s | 176.2 tok/s | 256 | 19.0 tok/s |
| fresh ~16K-token prompt | 15,488 | 0 | 84.1 s | 184.1 tok/s | 256 | 18.1 tok/s |
| fresh ~32K-token prompt | 34,010 | 0 | 194.0 s | 175.3 tok/s | 256 | 17.8 tok/s |
| fresh ~64K-token prompt | 75,611 | 0 | 427.2 s | 177.0 tok/s | 256 | 16.3 tok/s |

The token counts are what the server read: `bench.py` sizes prompts by characters (3.2 per token on the docs), and the
later docs pack more tokens per character, so "~64K" came out at 75,611. Prompts of 1-2 chunks read slower (a chunk
has a fixed cost); from a few thousand tokens on, prompt reading holds at ~175-185 tok/s. Decode eases from ~20.6 to
~16 tok/s as the context grows (more K/V and indexer keys to read). The same run on a stuck link reads at ~35-40 tok/s:
check the PCIe line `strata-agent` prints before trusting a benchmark.

What did not help, and why:

- **MTP** (the `mtp/` folder): works on the 4 GB card after three changes (its experts and dense weights in RAM,
  drafting from the main head), but decode stays at ~19 tok/s. Decode here is bound by reading each token's experts
  from RAM: verifying 2 tokens costs ~1.4x verifying 1, plus ~10 ms per draft, so a 63% acceptance does not pay.
  It only costs VRAM (110 -> 72 MiB free).
- **Prompt-lookup drafting** (`--suffix-draft`): +7% decode (62 of 96 drafts accepted on a repetitive prompt), but its
  4-row verify window does not fit next to a 768-token chunk.
- **All experts on the GPU** during prompt reading: slower than letting the CPU take a share (the default split).
  On a stuck (Gen 1) link the opposite holds: `STRATA_PREFILL_CPU_SHARE=1` (all to the CPU) was faster there.
- **Pinned experts, bigger chunks**: at first neither worked (pinning costs ~80 MiB of VRAM, and 1024 was ~90 MiB
  short at 128K). With `STRATA_PREFILL_IN_DENSE` both fit and both are now the default.
- **Disabling the GPU's runtime power-off, persistence mode**: did not stop the PCIe link from getting stuck.

GPU utilization during prompt reading was ~50% with 768-token chunks: the GPU waits for experts to arrive over PCIe
and for the CPU's share (14 threads busy). 100% is not reachable with the experts in RAM; bigger chunks raise it.

## Engine changes in this fork

All opt-in through environment variables; without them the engine behaves as upstream. Greedy outputs were checked
byte-identical with `STRATA_IDX_HOST`, `STRATA_PREFILL_IN_HEAD` and `STRATA_PREFILL_IN_DENSE` on and off (they move
where bytes live, not what is computed).

| Variable | File | What it does |
|---|---|---|
| `STRATA_IDX_HOST=1` | `src/core/layer.cpp` | the indexer's pooled keys (`max_cells/4` x 128 floats per attention layer: 32 MiB a layer at 262K) and the RoPE cos/sin table (64 MiB at 262K) in pinned, device-mapped RAM; kernels read them through the same pointers |
| `STRATA_PREFILL_IN_HEAD=1` | `src/program/generate.cpp`, `src/core/native_head.cpp`, `include/strata/core/native_head.hpp` | the prompt path's buffers are carved inside the native output head's VRAM (the rest allocated beside it); the head's blocks are copied to pinned RAM at start and written back at the same address after every prompt read, so the captured verify graphs stay valid. Needs `--serve`, one GPU, `--short-read 0`, no `--batch`, no `--prefill-head-alias` |
| `STRATA_PF_GEMM_SCRATCH_MIB=N` (8-64, default 64) | `src/prefill/prefill.cpp` | the prompt path's FP16 dequantization scratch; a smaller one makes the dense GEMMs run in row blocks (same values) |
| `STRATA_PF_GEMM_WS_MIB=N` (1-32, default 32) | `src/prefill/prefill.cpp` | the prompt path's cuBLAS workspace |
| `STRATA_PREFILL_IN_DENSE=1` | `src/core/native_dense.cpp`, `include/strata/core/native_dense.hpp`, `src/prefill/prefill.cpp`, `include/strata/prefill/prefill.hpp`, `src/program/generate.cpp` | the 300 native projection matrices in one 1.6 GiB block (118 MiB less than separate allocations), lent to the prompt path while a prompt is read: each layer's projections are streamed into two slots of it from a pinned RAM copy, and the block is written back after the prompt. Needs `STRATA_PREFILL_IN_HEAD=1`, one GPU, no `--mtp`. Greedy output byte-identical on and off at the same chunk size |
| `STRATA_MTP_EXPERTS_HOST=1` | `src/core/mtp.cpp`, `include/strata/core/mtp.hpp` | the MTP draft layer's 512 experts (675 MiB) in pinned RAM instead of VRAM |
| `STRATA_MTP_DENSE_HOST=1` | `src/core/mtp.cpp`, `include/strata/core/mtp.hpp` | the MTP draft layer's dense weights (110 MiB) in pinned RAM |

Upstream variables used: `STRATA_HC_REQ8`, `STRATA_RESIDENT_PIN`, `STRATA_PREFILL_CPU_SHARE`, `STRATA_MTP_FULL_HEAD`,
`STRATA_TRACE=1` (prints the free VRAM after each startup step - the tool used to find all of the above).

MTP (no speed-up measured) does not combine with `STRATA_PREFILL_IN_DENSE`, which `strata-agent` now sets: run the
engine command from [Step 6](#step-6---run) by hand, without `STRATA_PREFILL_IN_DENSE=1`, with `--prefill 768`, and add
`STRATA_MTP_EXPERTS_HOST=1 STRATA_MTP_DENSE_HOST=1 STRATA_MTP_FULL_HEAD=1` to the environment and
`--mtp ~/dev/models/swift-iq3_xxs/mtp/rt --vram-reserve-mib 50` to the engine options.

## Context size: 262K (default) and 128K

`strata-agent` serves the full 262,144-token window by default. With `STRATA_PREFILL_IN_DENSE=1` and page-locked experts
it fits with the same 4096-token chunks as 128K (no step-down): 212 MiB of VRAM free at rest (220 at 128K), 2.13 GiB of
pinned RAM for the K/V (1.06 GiB at 128K, ~44 GiB page-locked in all), ready in 64 s, answers coherent. Its prompt-reading
speed on long prompts is not measured yet (the link was stuck at Gen 1 during that test); every speed in this README
was measured at 128K. To run at 128K (a little more VRAM and RAM to spare, a faster start):

```bash
CONTEXT=131072 strata-agent
```

Before `STRATA_PREFILL_IN_DENSE` and page-locked experts only a 128-token chunk fit at 262K (a cold 25K read at 83 tok/s
instead of 186). The window itself was checked then: a passphrase hidden in the middle of a 25,740-token prompt was
found, and the follow-up turn reused all 25,748 cached tokens. Decode slows as a conversation gets long, because the
indexer keys and the K/V beyond 20,480 cells are read over PCIe (13.2 tok/s at 26K tokens, measured on a slow link).

## Troubleshooting

| Message / symptom | Cause | What to do |
|---|---|---|
| `native head upload: out of memory` | the engine without this configuration | use `strata-agent` (`STRATA_HC_REQ8=1` and the rest) |
| `verify: needs the profile-filled VRAM expert tier` | `--expert-cache 0` (= auto) found no room, or the reserve was larger than the free VRAM | `--expert-cache 1` with a reserve below the free VRAM (the launcher's 150) |
| `verify: instantiate: out of memory` at the first answer | less than ~55-60 MiB free after loading | a smaller chunk, drafting off, no external screen on the NVIDIA GPU |
| `device buffers for a chunk of N tokens do not fit` and a smaller chunk | the chunk is too big for the free VRAM (e.g. an external screen on the NVIDIA GPU); the engine steps down (4096 -> 3072 -> 2048 ...) | free the VRAM, or set the CHUNK that fits |
| `*** strata-agent: the experts could not be page-locked` | less RAM free than ~44 GiB: the engine fell back to ordinary RAM | close memory-heavy programs and restart (~20% faster prompts), or `PIN_EXPERTS=0` to skip the attempt |
| prompt reading ~77 tok/s (or ~35-40 with 768-token chunks), probe ~1 GB/s | the PCIe link is stuck | `gpu-mode retrain`, restart `strata-agent` ([details](#the-pcie-link-problem)) |
| `nvidia-smi`: `Unable to determine the device handle for GPU0 ... Unknown Error`; kernel `NVRM: Xid ... 79, GPU has fallen off the bus` | the GPU crashed (seen once, right after a link retrain) | reboot; then check `gpu-mode status` and the probe |
| `the whole resident RAM mode does not fit` | less than ~44 GiB available at start; part of the experts is then read from disk | close memory-heavy programs; the engine counts page cache only if the kernel's `MemAvailable` does |
| the same, inside a container or an app sandbox | a cgroup memory limit counts the page cache of the model files the process read | drop those files' cache before starting (the snippet below) or raise the limit |
| GPU at 20 W, link at Gen 2 | on battery | plug in the charger |
| HDMI: sound, no picture | `61-mutter-ignore-nvidia.rules` makes GNOME ignore the NVIDIA GPU | `gpu-mode host`, log out and in (or reboot) |
| `this engine was started without --vision` | images sent without `--vision` | `VISION=1` (the default) needs the mmproj file in the model directory |

Dropping the model files' page cache without root (only needed under a cgroup limit):

```bash
python3 -c "
import os, glob
M = os.path.expanduser('~/dev/models/swift-iq3_xxs')
for f in glob.glob(M + '/*.gguf') + [M + '/pack/dense.bin']:
    fd = os.open(os.path.realpath(f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)"
```

## Undo everything

```bash
gpu-mode vm && sudo reboot                  # back to the VFIO passthrough setup (only if you used gpu-mode host)
rm ~/.local/bin/strata-agent ~/.local/bin/gpu-mode ~/.config/strata-agent.conf ~/.config/gpu-mode.conf
rm -r ~/.local/share/gpu-mode
sudo rm /usr/local/sbin/gpu-link-retrain /etc/sudoers.d/gpu-link-retrain
rm -r ~/dev/models/swift-iq3_xxs            # the links, pack and MTP files (the GGUFs stay in ~/.cache/huggingface)
hf cache delete                             # interactive; or remove ~/.cache/huggingface/hub/models--SC117--Swift-1.5-*
```

`/etc/kernel/cmdline.gpu-mode-*` and `/etc/default/grub.gpu-mode-*` are the backups `gpu-mode` wrote.

## Files in this folder

| File | What |
|---|---|
| `README.md` | this guide |
| `prepare-model.sh` | download from Hugging Face, link, build `pack/` and `mtp/rt/` |
| `install.sh` | put the tools on the PATH, write their settings, build the probe, `--system`: the root helper |
| `bin/strata-agent` | the launcher |
| `bin/gpu-mode` | host/VM GPU ownership, status, PCIe retrain |
| `system/gpu-link-retrain.in` | the root helper (install.sh fills in the PCIe port) |
| `system/sudoers-gpu-link-retrain.in` | its sudoers line (install.sh fills in the user) |
| `tools/pcie-probe.cu` | the bandwidth probe |
| `bench/bench.py` | the benchmark (agent session, context sweep) |
