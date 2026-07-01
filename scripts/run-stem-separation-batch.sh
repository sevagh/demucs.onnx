#!/usr/bin/env bash
# Run DemucsONNX stem separation over every wav in /audio that hasn't been
# processed yet (skips ones that already have output stems in demucs-onnx-out).
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="./onnx-models/htdemucs.onnx"
AUDIO_DIR="../audio"
OUT_DIR="$AUDIO_DIR/demucs-onnx-out"
BIN="./build/build-cli/demucs"

shopt -s nullglob
for wav in "$AUDIO_DIR"/*.wav; do
    base="$(basename "$wav" .wav)"
    if [[ -f "$OUT_DIR/${base}_0_drums.wav" ]]; then
        echo "skip (already done): $base"
        continue
    fi
    echo "=== processing: $base ==="
    "$BIN" "$MODEL" "$wav" "$OUT_DIR"
done

echo "all done"
