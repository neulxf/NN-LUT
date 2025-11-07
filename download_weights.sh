#!/usr/bin/env bash
set -euo pipefail

# Download 7B/8B class weights into ./weights/ to match models/config.py keys.
# Default: download all three (Qwen2.5-7B-Instruct, LLaMA3-8B, Qwen3-8B).
# Usage examples:
#   bash download_weights.sh                              # download all
#   bash download_weights.sh --only qwen2.5-7b            # only Qwen2.5-7B-Instruct
#   HF_TOKEN=xxxx bash download_weights.sh                # use token from env
#   bash download_weights.sh --use-mirror                 # use HF mirror endpoint

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

WEIGHTS_DIR="$DIR/weights"
mkdir -p "$WEIGHTS_DIR"

ONLY=""
USE_MIRROR="false"
if [[ "${1:-}" == "--only" && -n "${2:-}" ]]; then
  ONLY="$2"; shift 2
fi
if [[ "${1:-}" == "--use-mirror" ]]; then
  USE_MIRROR="true"; shift 1
fi

need_cmd() { command -v "$1" >/dev/null 2>&1; }

echo "[INFO] Python: $(python -V 2>/dev/null || echo 'not found')"

ensure_cli() {
  # Prefer new `hf` CLI; fallback to `huggingface-cli` if needed
  if ! need_cmd hf; then
    echo "[INFO] Installing/Upgrading huggingface_hub (provides 'hf' CLI)"
    pip install --upgrade "huggingface_hub[cli]" >/dev/null
  fi
  if ! need_cmd hf; then
    echo "[WARN] 'hf' CLI not found after install. Will try 'huggingface-cli' as fallback."
    if ! need_cmd huggingface-cli; then
      pip install --upgrade "huggingface_hub[cli]" >/dev/null
    fi
  fi
}
ensure_cli

# Enable faster & resilient transfer if available
export HF_HUB_ENABLE_HF_TRANSFER=1
if ! python -c "import hf_transfer" >/dev/null 2>&1; then
  pip install -q hf-transfer || true
fi

HF_TOKEN="${HF_TOKEN:-}"
if [[ -n "$HF_TOKEN" ]]; then
  echo "[INFO] Using HF_TOKEN from environment for authenticated downloads."
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if [[ "$USE_MIRROR" == "true" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
  echo "[INFO] Using mirror endpoint: $HF_ENDPOINT"
fi

download_model() {
  local repo="$1"    # e.g., Qwen/Qwen2.5-7B-Instruct
  local outdir="$2"  # e.g., weights/Qwen2.5-7B-Instruct
  local name="$3"    # human-readable name

  echo "[INFO] Downloading $name → $outdir"
  mkdir -p "$outdir"
  # Try with `hf download` (new CLI). Retry a few times in case of transient SSL EOF.
  local tries=0 max_tries=4
  local delay=5
  while (( tries < max_tries )); do
    if command -v hf >/dev/null 2>&1; then
      # Minimal flags for broader CLI compatibility
      if hf download "$repo" --local-dir "$outdir" --repo-type model --quiet; then
        return 0
      fi
    else
      # Fallback to deprecated huggingface-cli
      if huggingface-cli download "$repo" --local-dir "$outdir" --local-dir-use-symlinks False --resume-download --quiet; then
        return 0
      fi
    fi
    tries=$((tries+1))
    echo "[WARN] ($tries/$max_tries) Download error for $name. Retrying in ${delay}s..."
    sleep "$delay"
    delay=$((delay*2))
  done
  echo "[ERROR] Download failed for $name ($repo)."
  echo "        If this model requires license acceptance or access, ensure your token has permission:"
  echo "        1) Export token: export HF_TOKEN=xxxxxxxx"
  echo "        2) Use mirror:  bash download_weights.sh --use-mirror --only $(basename "$outdir")"
  echo "        3) Or try later due to network/SSL issues."
  return 1
}

# Map models to repos/paths expected by models/config.py
download_qwen2_5_7b() {
  download_model "Qwen/Qwen2.5-7B-Instruct" "$WEIGHTS_DIR/Qwen2.5-7B-Instruct" "Qwen2.5-7B-Instruct"
}

download_llama3_8b() {
  # LLaMA3-8B (requires license acceptance on Hugging Face)
  download_model "meta-llama/Meta-Llama-3-8B" "$WEIGHTS_DIR/llama3-8b-hf" "LLaMA3-8B"
}

download_qwen3_8b() {
  download_model "Qwen/Qwen3-8B" "$WEIGHTS_DIR/Qwen3-8B" "Qwen3-8B"
}

case "$ONLY" in
  "")
    download_qwen2_5_7b || true
    download_llama3_8b || true
    download_qwen3_8b || true
    ;;
  qwen2.5-7b|Qwen2.5-7B-Instruct)
    download_qwen2_5_7b
    ;;
  llama3-8b|llama3-8b-hf)
    download_llama3_8b
    ;;
  qwen3-8b|Qwen3-8B)
    download_qwen3_8b
    ;;
  *)
    echo "[ERROR] Unknown model for --only: $ONLY"
    echo "        Supported: qwen2.5-7b | llama3-8b | qwen3-8b"
    exit 2
    ;;
esac

echo "[OK] Done. Local folders ready:"
echo " - $WEIGHTS_DIR/Qwen2.5-7B-Instruct  (for models/config.py: Qwen2.5-7B-Instruct)"
echo " - $WEIGHTS_DIR/llama3-8b-hf        (for models/config.py: llama3-8b-hf)"
echo " - $WEIGHTS_DIR/Qwen3-8B            (for models/config.py: Qwen3-8B-bf16 / Qwen3-8B-float)"


