#!/usr/bin/env python
"""Compare execution providers on one htdemucs.onnx segment.

Times one segment per provider and, when two providers are given, reports how
well their outputs agree. Always check agreement before trusting a provider:
the DirectML build is ~17x faster than CPU on this model but computes garbage,
with no error from ONNX Runtime (see WINDOWS.md).

usage:
  bench-ep.py <model.onnx> cpu                 # timing only
  bench-ep.py <model.onnx> cpu cuda            # timing + agreement
  bench-ep.py <model.onnx> cuda --runs 10 --threads 16
"""
import argparse
import time

import numpy as np
import onnxruntime as ort

# shapes the C++ CLI feeds (src/demucs.hpp)
MIX_SHAPE = (1, 2, 343980)
SPEC_SHAPE = (1, 4, 2048, 336)

PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}


def run(model, ep, feeds, runs, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model, so, providers=PROVIDERS[ep])
    active = sess.get_providers()
    expected = PROVIDERS[ep][0]
    expected = expected[0] if isinstance(expected, tuple) else expected
    if expected != "CPUExecutionProvider" and active[0] != expected:
        print(f"  [warn] requested {ep} but the session is running {active[0]}")

    sess.run(None, feeds)  # warm up (also pays cuDNN/cuBLAS init on GPU)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = sess.run(None, feeds)
        times.append(time.perf_counter() - t0)
    return out, min(times), active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("eps", nargs="+", choices=sorted(PROVIDERS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    probe = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    names = [i.name for i in probe.get_inputs()]
    feeds = {names[0]: rng.standard_normal(MIX_SHAPE, dtype=np.float32),
             names[1]: rng.standard_normal(SPEC_SHAPE, dtype=np.float32)}

    results = {}
    for ep in args.eps:
        out, best, active = run(args.model, ep, feeds, args.runs, args.threads)
        results[ep] = out
        per_song = best * 34  # ~34 segments for a 4 minute song
        print(f"{ep:5s} providers={active}  min {best*1000:7.0f} ms/segment"
              f"  (~{per_song:5.1f} s for a 238 s song)")

    if len(args.eps) == 2:
        a, b = (results[e] for e in args.eps)
        print(f"\nagreement {args.eps[0]} vs {args.eps[1]}:")
        for k in range(len(a)):
            if a[k].shape != b[k].shape:
                print(f"  out{k}: shape mismatch {a[k].shape} vs {b[k].shape}")
                continue
            rel = np.sqrt(((a[k] - b[k]) ** 2).mean()) / np.sqrt((a[k] ** 2).mean())
            corr = np.corrcoef(a[k].ravel(), b[k].ravel())[0, 1]
            verdict = "ok" if corr > 0.999 else "MISMATCH - do not trust this provider"
            print(f"  out{k} absmean {np.abs(a[k]).mean():10.4f} vs {np.abs(b[k]).mean():10.4f}"
                  f"  rel {rel*100:9.3f}%  corr {corr:8.6f}  {verdict}")


if __name__ == "__main__":
    main()
