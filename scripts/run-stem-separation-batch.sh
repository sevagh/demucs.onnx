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
    if [[ -f "$OUT_DIR/${base}_0_drums.mp3" ]]; then
        echo "skip (already done): $base"
        return 0
    fi
    echo "=== processing: $base ==="
    "$BIN" "$MODEL" "$input" "$OUT_DIR"
    # The CLI writes .wav stems; encode them to 320 kbps mp3 and drop the wavs
    # to keep the shipped stem set small.
    for stem in "$OUT_DIR/${base}"_*.wav; do
        [[ -f "$stem" ]] || continue
        if ffmpeg -v error -y -i "$stem" -b:a 320k "${stem%.wav}.mp3" </dev/null; then
            rm -f "$stem"
        fi
    done
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
        if [[ -f "$OUT_DIR/${base}_0_drums.mp3" ]]; then
            echo "skip (already done): $base"
            continue
        fi
        # Convert into a temp dir under the real track name so the CLI (which
        # names stems after the input file) produces "<base>_N_name.wav".
        tmp_dir="$(mktemp -d)"
        tmp_wav="$tmp_dir/$base.wav"
        echo "=== converting AIFF -> wav: $base ==="
        afconvert -f WAVE -d LEI16 "$f" "$tmp_wav"
        separate "$tmp_wav" "$base"
        rm -rf "$tmp_dir"
    done
done

echo "all done"
