#!/usr/bin/env bash
# Run DemucsONNX stem separation over every supported audio file in /audio that
# hasn't been processed yet (skips ones that already have output stems in
# demucs-onnx-out).
#
# Natively decoded by libnyquist: wav, flac, mp3, ogg, opus.
# AIFF (.aif/.aiff) has no libnyquist decoder, so it is transcoded to a
# temporary wav via `afconvert` (macOS) before separation.
# Any sample rate is accepted: the CLI resamples to 44.1 kHz internally.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="./onnx-models/htdemucs.onnx"
AUDIO_DIR="../audio"
OUT_DIR="$AUDIO_DIR/demucs-onnx-out"
BIN="./build/build-cli/demucs"

# Extensions libnyquist can decode directly.
NATIVE_EXTS=(wav wave flac mp3 ogg opus)
# Extensions we transcode to wav first (no libnyquist decoder).
CONVERT_EXTS=(aif aiff)

separate() {
    # $1 = input file to feed the separator, $2 = base name for skip/output
    local input="$1" base="$2"
    if [[ -f "$OUT_DIR/${base}_0_drums.wav" ]]; then
        echo "skip (already done): $base"
        return 0
    fi
    echo "=== processing: $base ==="
    "$BIN" "$MODEL" "$input" "$OUT_DIR"
}

shopt -s nullglob nocaseglob

# Natively supported formats: hand straight to the separator.
for ext in "${NATIVE_EXTS[@]}"; do
    for f in "$AUDIO_DIR"/*."$ext"; do
        base="$(basename "$f")"
        base="${base%.*}"
        separate "$f" "$base"
    done
done

# AIFF: transcode to a temporary wav, then separate.
for ext in "${CONVERT_EXTS[@]}"; do
    for f in "$AUDIO_DIR"/*."$ext"; do
        base="$(basename "$f")"
        base="${base%.*}"
        if [[ -f "$OUT_DIR/${base}_0_drums.wav" ]]; then
            echo "skip (already done): $base"
            continue
        fi
        tmp_wav="$(mktemp -t demucs-XXXXXX).wav"
        echo "=== converting AIFF -> wav: $base ==="
        afconvert -f WAVE -d LEI16 "$f" "$tmp_wav"
        separate "$tmp_wav" "$base"
        rm -f "$tmp_wav"
    done
done

echo "all done"
