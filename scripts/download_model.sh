#!/usr/bin/env bash
# ==============================================================================
# ProdML Model Artifacts Downloader
# Downloads the fine-tuned AraBERT sentiment model from Google Drive into outputs/final_model
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${1:-$PROJECT_ROOT/outputs/final_model}"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1yeJcg-nrci3BDq01p9-pjtciv1c4g9rc"

echo "=== ProdML Model Downloader ==="
echo "Target directory: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

# Check if model files already exist
if [ -f "$TARGET_DIR/config.json" ] && \
   [ -f "$TARGET_DIR/model.safetensors" ] && \
   [ -f "$TARGET_DIR/tokenizer.json" ] && \
   [ -f "$TARGET_DIR/tokenizer_config.json" ]; then
    echo "✓ All model artifacts already exist in $TARGET_DIR. Skipping download."
    ls -lh "$TARGET_DIR"
    exit 0
fi

# Ensure gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing gdown via pip..."
    pip install --quiet --no-cache-dir gdown || uv pip install --quiet gdown || python3 -m pip install --quiet gdown
fi

echo "Downloading fine-tuned weights from Google Drive..."
python3 -m gdown --folder "$GDRIVE_FOLDER" -O "$TARGET_DIR" --remaining-ok 2>/dev/null || \
python3 -m gdown --folder "$GDRIVE_FOLDER" -O "$TARGET_DIR"

echo "=== Verification ==="
if [ -f "$TARGET_DIR/model.safetensors" ]; then
    echo "✓ Model artifacts successfully verified:"
    ls -lh "$TARGET_DIR"
else
    echo "⚠ Download verification failed. Attempting direct file ID downloads..."
    python3 -m gdown "https://drive.google.com/uc?id=1T1KBobXpqQ3I6D1P2eWMa14oEHZ0vL3L" -O "$TARGET_DIR/config.json"
    python3 -m gdown "https://drive.google.com/uc?id=1-LjK_ineRXRMqs6ZWszISimuQpzYJQ4i" -O "$TARGET_DIR/model.safetensors"
    python3 -m gdown "https://drive.google.com/uc?id=1udNgNrk-HbpLDbk4jUzU3zTGB4UPIM5I" -O "$TARGET_DIR/tokenizer.json"
    python3 -m gdown "https://drive.google.com/uc?id=1s43JYX99ldlbQH7NVEXWiNvAnQNUX0u6" -O "$TARGET_DIR/tokenizer_config.json"
    ls -lh "$TARGET_DIR"
fi
echo "Done!"
