#!/usr/bin/env python
"""Small WAV helpers shared by prepare-input.py / render-stems.py.

Only what demucs.onnx needs: read PCM16/PCM24/IEEE-float WAV, write plain PCM,
and a band-limited rational resampler that works on a 4-minute 192 kHz track
without blowing up memory.
"""
import os
import struct

import numpy as np


# --------------------------------------------------------------------- reading
def read_wav(path):
    """Return (frames[channels, samples] float32, sample_rate, bits, fmt_tag)."""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    pos, fmt, data = 12, None, None
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = blob[pos + 8:pos + 8 + size]
        elif cid == b"data":
            data = blob[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)

    if fmt is None or data is None:
        raise ValueError(f"{path}: missing fmt/data chunk")

    tag, channels, rate, _, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 1 and bits == 16:
        samples = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and bits == 24:
        raw = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        val = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        val = (val ^ 0x800000) - 0x800000          # sign-extend 24 -> 32 bit
        samples = val.astype(np.float32) / 8388608.0
    elif tag == 1 and bits == 32:
        samples = np.frombuffer(data, "<i4").astype(np.float32) / 2147483648.0
    elif tag == 3 and bits == 32:
        samples = np.frombuffer(data, "<f4").astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported PCM tag={tag} bits={bits}")

    return samples.reshape(-1, channels).T, rate, bits, tag


def probe(path):
    """Cheap header-only probe (no sample decode)."""
    with open(path, "rb") as f:
        blob = f.read(4096)
    pos = 12
    while pos + 8 <= len(blob):
        cid = blob[pos:pos + 4]
        size = struct.unpack("<I", blob[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            tag, ch, sr, byte_rate, _, bits = struct.unpack("<HHIIHH", blob[pos + 8:pos + 24])
            if byte_rate:
                frames = size // byte_rate if size else 0
            return {"sample_rate": sr, "channels": ch, "bits": bits, "fmt_tag": tag}
        pos += 8 + size + (size & 1)
    raise ValueError(f"{path}: missing fmt chunk")


# --------------------------------------------------------------------- writing
def _write_riff(path, data_bytes, channels, rate, bits, fmt_tag):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    block = channels * (bits // 8)
    fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt \
        + b"data" + struct.pack("<I", len(data_bytes)) + data_bytes
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_pcm16(path, frames, rate):
    """16-bit PCM: what libnyquist reads most reliably."""
    pcm = np.clip(frames.T, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    _write_riff(path, pcm.tobytes(), frames.shape[0], rate, 16, 1)


def write_pcm24(path, frames, rate):
    """24-bit PCM: what the stems are delivered as."""
    pcm = np.clip(frames.T, -1.0, 1.0) * 8388607.0
    pcm = pcm.round().astype(np.int32).reshape(-1)
    u = pcm.astype(np.int64) & 0xFFFFFF               # two's complement, 24 bit
    raw = np.empty(u.size * 3, dtype=np.uint8)
    raw[0::3] = (u & 0xFF).astype(np.uint8)
    raw[1::3] = ((u >> 8) & 0xFF).astype(np.uint8)
    raw[2::3] = ((u >> 16) & 0xFF).astype(np.uint8)
    _write_riff(path, raw.tobytes(), frames.shape[0], rate, 24, 1)


# ------------------------------------------------------------------ resampling
def resample(waveform, sr_in, sr_out, chunk_blocks=1500, margin_blocks=20):
    """Band-limited rational resample in phase-aligned chunks.

    torchaudio's resampler uses a conv1d over the whole signal, which for
    192 kHz -> 44.1 kHz allocates an im2col buffer proportional to length x
    filter taps (tens of GB for a 4-minute track). Splitting is safe because
    sr_in/sr_out reduce to `up`/`down`: chunk starts and margins are kept at
    multiples of `down`, so every slice begins on the same conv grid phase as
    the whole signal and the trimmed pieces splice together sample-exactly
    (verified: identical to a single-call resample, max |diff| = 0).

    Input/output are [channels, samples] float32. Works in both directions, so
    it also handles writing stems back at the source sample rate.
    """
    if sr_in == sr_out:
        return np.asarray(waveform, dtype=np.float32)

    import torch
    import torchaudio

    gcd = int(np.gcd(sr_in, sr_out))
    up, down = sr_out // gcd, sr_in // gcd

    chunk_in = down * chunk_blocks
    margin_in = margin_blocks * down

    x = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
    total_in = x.shape[-1]

    outs = []
    for start in range(0, total_in, chunk_in):
        lo = max(0, start - margin_in)
        hi = min(total_in, start + chunk_in + margin_in)
        res = torchaudio.functional.resample(x[:, lo:hi], sr_in, sr_out)

        block = min(chunk_in, total_in - start) // down
        head = (start - lo) // down * up
        end = min(head + block * up, res.shape[-1])
        outs.append(res[:, head:end])

    return torch.cat(outs, dim=-1).numpy()
