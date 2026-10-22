#!/usr/bin/env python
"""Render demucs.onnx stems (libnyquist float32 WAV) into deliverable PCM WAVs.

Two things this handles that the C++ CLI does not:

1. Format. libnyquist writes 32-bit IEEE-float WAVs (fmt tag 3). DAWs read
   them, but plenty of players and Python's own `wave` module do not, so stems
   are written as 24-bit PCM here.

2. Headroom. Demucs is not energy preserving: individual stems routinely peak
   above full scale (drums/vocals usually do). Writing PCM directly would clip
   those transients, so one common gain is applied to *all* stems. The mix
   balance is unchanged, and the gain is printed so it can be undone.

Also optionally resamples the stems back to the source sample rate
(`--match-rate`), so a 192 kHz session keeps sample-aligned parts. This does
not restore content above 22.05 kHz - the network ran at 44.1 kHz, so the
bandwidth is gone for good; only the sample grid is matched.

usage:
  render-stems.py <stem_dir> <out_dir> <name_prefix> \
      [--out-rate 44100] [--match-rate <original.wav>] [--peak 0.995]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wavio import probe, read_wav, resample, write_pcm24  # noqa: E402

STEMS = [(0, "drums"), (1, "bass"), (2, "other"), (3, "vocals"),
         (4, "guitar"), (5, "piano")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir")
    ap.add_argument("out_dir")
    ap.add_argument("prefix")
    ap.add_argument("--out-rate", type=int, default=None,
                    help="resample the stems to this rate before writing")
    ap.add_argument("--match-rate", default=None,
                    help="resample the stems back to this file's sample rate")
    ap.add_argument("--peak", type=float, default=0.995,
                    help="target peak after the common gain (default 0.995)")
    args = ap.parse_args()

    stems = {}
    for idx, name in STEMS:
        path = os.path.join(args.stem_dir, f"target_{idx}_{name}.wav")
        if os.path.exists(path):
            frames, rate, _, _ = read_wav(path)
            stems[name] = (frames, rate)

    if not stems:
        raise SystemExit(f"no target_*_*.wav found in {args.stem_dir}")

    out_rate = args.out_rate
    if args.match_rate:
        out_rate = probe(args.match_rate)["sample_rate"]
        print(f"matching source rate of {os.path.basename(args.match_rate)}: {out_rate} Hz")

    src_rate = next(iter(stems.values()))[1]
    need_resample = bool(out_rate and out_rate != src_rate)

    def materialize(name):
        """Final samples for one stem: gain is applied later, rate fixed here."""
        frames, rate = stems[name]
        return resample(frames, rate, out_rate) if need_resample else frames

    # Peak must be measured *after* resampling: band-limited interpolation
    # overshoots (Gibbs), so a gain derived from the 44.1 kHz peak can still
    # clip the 192 kHz render. Costs one extra resample pass.
    peak = max(float(np.max(np.abs(materialize(name)))) for name in stems)
    gain = min(1.0, args.peak / peak) if peak > 0 else 1.0
    db = 20 * np.log10(gain) if gain > 0 else -999
    print(f"common peak {peak:.4f}{' (post-resample)' if need_resample else ''} "
          f"-> gain {gain:.6f} ({db:+.2f} dB)")
    if gain < 1.0:
        print(f"(undo with x{1/gain:.6f} = {peak:.4f} peak)")

    os.makedirs(args.out_dir, exist_ok=True)
    for name in stems:
        out = materialize(name) * gain
        rate_out = out_rate or src_rate
        path = os.path.join(args.out_dir, f"{args.prefix} - {name}.wav")
        write_pcm24(path, out, rate_out)
        print(f"  {name:6s} {out.shape[1] / rate_out:7.2f}s  "
              f"{rate_out} Hz  peak {np.max(np.abs(out)):.3f}  "
              f"{os.path.getsize(path) / 1e6:6.1f} MB  -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
