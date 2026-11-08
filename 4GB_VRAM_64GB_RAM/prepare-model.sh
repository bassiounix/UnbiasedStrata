#!/usr/bin/env bash
# prepare-model.sh - download Swift 1.5 Qwen3.8-Flash-Next IQ3_XXS and build the model directory strata-agent reads.
#
#   ./prepare-model.sh [MODEL_DIR] [--no-download] [--no-mtp] [--force]
#
# MODEL_DIR defaults to ~/dev/models/swift-iq3_xxs.  The result (sizes as built on 2026-11-08):
#
#   MODEL_DIR/
#   ├── Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00001-of-00002.gguf  -> HF cache   (38 GB, shard 1)
#   ├── Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-00002-of-00002.gguf  -> HF cache   (34 GB, shard 2)
#   ├── mmproj-Qwen3.8-Flash-Next-BF16.gguf                                       -> HF cache   (866 MB, vision)
#   ├── pack/                     tools/iq_pack.py from shard 1                                 (1.5 GB)
#   │   ├── dense.bin  index.txt  conversions.json  native_experts.txt
#   │   └── tokenizer/  (chat_template.jinja merges.txt tokenizer.json token_type.json vocab.json)
#   └── mtp/                      optional: the MTP draft layer (measured: no speed-up on a 4 GB card, see README)
#       ├── mtp-q2_0.gguf                                                         -> HF cache   (848 MB)
#       └── rt/  (dense.bin dense.txt experts.bin draft_vocab.bin)  tools/mtp_rt.py + data/draft_vocab.bin
#
# The GGUFs stay in the Hugging Face cache (~/.cache/huggingface/hub, ~78 GB) and are linked, not copied.
# Needs: the `hf` CLI (pip install -U huggingface_hub, or uv tool install huggingface_hub), python3 with venv,
# and this repository's llama.cpp submodule (git submodule update --init) for gguf-py.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
HF_REPO=SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
REVISION=3db919a964b1f0ae0d1000ba7be90b3e38b1f7a3   # the snapshot every number in README.md was measured on
NAME=Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS
MMPROJ=mmproj-Qwen3.8-Flash-Next-BF16.gguf

MODEL_DIR=$HOME/dev/models/swift-iq3_xxs
DOWNLOAD=1; MTP=1; FORCE=0
for a in "$@"; do
    case $a in
        --no-download) DOWNLOAD=0 ;;
        --no-mtp) MTP=0 ;;
        --force) FORCE=1 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "prepare-model: unknown option $a" >&2; exit 2 ;;
        *) MODEL_DIR=$a ;;
    esac
done
mkdir -p "$MODEL_DIR"
MODEL_DIR=$(cd "$MODEL_DIR" && pwd)
echo "prepare-model: model directory $MODEL_DIR"

# ---- 1. the files from Hugging Face (into its cache), linked into MODEL_DIR
FILES=("IQ3_XXS/$NAME-00001-of-00002.gguf" "IQ3_XXS/$NAME-00002-of-00002.gguf" "$MMPROJ")
[[ $MTP == 1 ]] && FILES+=("mtp-q2_0.gguf")
HF=$(command -v hf || true)
[[ -n $HF ]] || { echo "prepare-model: the hf CLI is missing: pip install -U huggingface_hub" >&2; exit 1; }
if [[ $DOWNLOAD == 1 ]]; then
    echo "prepare-model: downloading ${FILES[*]} from $HF_REPO@${REVISION:0:8} (~$([[ $MTP == 1 ]] && echo 76 || echo 75) GB the first time)"
    SNAP=$("$HF" download "$HF_REPO" "${FILES[@]}" --revision "$REVISION" --quiet | tail -1)
else
    SNAP=$(ls -d "$HOME/.cache/huggingface/hub/models--${HF_REPO/\//--}/snapshots/$REVISION" 2>/dev/null || true)
fi
[[ -d $SNAP ]] || { echo "prepare-model: no snapshot directory ($SNAP)" >&2; exit 1; }
ln -sfn "$SNAP/IQ3_XXS/$NAME-00001-of-00002.gguf" "$MODEL_DIR/"
ln -sfn "$SNAP/IQ3_XXS/$NAME-00002-of-00002.gguf" "$MODEL_DIR/"
ln -sfn "$SNAP/$MMPROJ" "$MODEL_DIR/"
SH1=$MODEL_DIR/$NAME-00001-of-00002.gguf

# ---- 2. the offline tools' Python (numpy, regex) and gguf-py from the llama.cpp submodule
[[ -f $REPO/third_party/llama.cpp/gguf-py/gguf/__init__.py ]] || {
    echo "prepare-model: third_party/llama.cpp is empty: run  git -C $REPO submodule update --init" >&2; exit 1; }
VENV=$REPO/.venv-tools
[[ -x $VENV/bin/python ]] || python3 -m venv "$VENV"
# tools/requirements.txt (numpy, regex) plus what gguf-py itself imports (pyyaml, tqdm, requests; mtp_rt.py needs it)
"$VENV/bin/pip" install -q --disable-pip-version-check -r "$REPO/tools/requirements.txt" pyyaml tqdm requests
PY=$VENV/bin/python

# ---- 3. the pack: dense weights + index + tokenizer (the experts stay in the GGUF shards)
if [[ $FORCE == 1 || ! -f $MODEL_DIR/pack/index.txt ]]; then
    echo "prepare-model: building $MODEL_DIR/pack (tools/iq_pack.py, a few minutes)"
    (cd "$REPO" && "$PY" tools/iq_pack.py --gguf "$SH1" --out "$MODEL_DIR/pack")
else
    echo "prepare-model: $MODEL_DIR/pack exists (--force rebuilds it)"
fi

# ---- 4. optional: the MTP draft layer's runtime files
if [[ $MTP == 1 ]]; then
    mkdir -p "$MODEL_DIR/mtp"
    ln -sfn "$SNAP/mtp-q2_0.gguf" "$MODEL_DIR/mtp/"
    if [[ $FORCE == 1 || ! -f $MODEL_DIR/mtp/rt/experts.bin ]]; then
        echo "prepare-model: building $MODEL_DIR/mtp/rt (tools/mtp_rt.py)"
        (cd "$REPO" && "$PY" tools/mtp_rt.py --gguf "$MODEL_DIR/mtp/mtp-q2_0.gguf" --out "$MODEL_DIR/mtp/rt")
        cp "$REPO/data/draft_vocab.bin" "$MODEL_DIR/mtp/rt/draft_vocab.bin"
    else
        echo "prepare-model: $MODEL_DIR/mtp/rt exists (--force rebuilds it)"
    fi
fi

echo "prepare-model: done."
(cd "$MODEL_DIR" && find . -maxdepth 3 | sort | sed 's/^\.\///' | grep -v '^\.$')
