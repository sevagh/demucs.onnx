#!/usr/bin/env python
"""Prepare an audio file for demucs.onnx (which only accepts 44.1 kHz stereo).

demucs.onnx does its own STFT at a fixed 44.1 kHz (demucsonnx::SUPPORTED_SAMPLE_RATE)
and libnyquist only decodes - it never resamples. So anything that is not
44.1 kHz / 16-bit / stereo has to be converted first; there is no ffmpeg
dependency here.

usage:
  prepare-input.py <in.wav> <out.wav> [--rate 44100] [--bits 16]

notes:
  * high-rate sources (96/192 kHz) are downsampled with a band-limited
    rational resampler; the full-scale peak is normalised only if the
    resampler pushes a sample past 1.0
  * mono is duplicated to stereo, >2 channels are rejected
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wavio import read_wav, resample, write_pcm16  # noqa: E402

TARGET_RATE = 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--rate", type=int, default=TARGET_RATE,
                    help="target sample rate (demucs.onnx requires 44100)")
    ap.add_argument("--no-resample", action="store_true",
                    help="keep the source rate (only valid if it is already 44100)")
    args = ap.parse_args()

    audio, rate, bits, tag = read_wav(args.src)
    print(f"in : {os.path.basename(args.src)}")
    print(f"     {rate} Hz, {audio.shape[0]} ch, {bits}-bit (fmt {tag}), "
          f"{audio.shape[1]} frames ({audio.shape[1] / rate:.2f} s)")

    if audio.shape[0] > 2:
        raise SystemExit(f"{audio.shape[0]} channels: demucs.onnx supports mono/stereo only")
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
        print("     mono -> duplicated to stereo")

    if args.no_resample and rate != args.rate:
        print(f"     keeping {rate} Hz as requested", file=sys.stderr)

    out = audio if args.no_resample else resample(audio, rate, args.rate)

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
        print(f"     peak {peak:.3f} > 1.0, scaled by 1/{peak:.3f}")

    write_pcm16(args.dst, out, args.rate)
    print(f"out: {args.dst}")
    print(f"     {args.rate} Hz, 2 ch, 16-bit, {out.shape[1]} frames "
          f"({out.shape[1] / args.rate:.2f} s)")


if __name__ == "__main__":
    main()
