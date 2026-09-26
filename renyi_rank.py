#!/usr/bin/env python3
"""
Rank transients by Renyi spectral entropy drop (stage 1 only, no gates).

Entropy: De la Cruz & Seto (2026) eq. 2, normalised to [0, 1].
Defaults: sperm_whale_site1 preset.

in:  audio files or folders (wav, flac, aif)
out: CSV, one row per transient, most negative z first:
     rank, file, time_s, z, entropy, drop, alpha

    python renyi_rank.py data/ -o ranked.csv --top 500
"""
import argparse
import bisect
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, spectrogram

try:
    import soundfile as sf
except ImportError:
    sf = None

EXT = (".wav", ".flac", ".aif", ".aiff")


@dataclass
class Params:
    band: tuple = (3000.0, 15000.0)   # Hz
    alphas: tuple = (2.0, 2.5, 3.0)   # candidates, one picked per file
    alpha: float = None               # fixed alpha, skips the pick
    nperseg: int = 512
    noverlap: int = 448
    w_floor: float = 0.15             # floor of sqrt(f) weights
    smooth: float = 0.25              # baseline window (s)
    z_thr: float = -4.0
    refract: float = 0.05             # s
    chunk: float = 300.0              # s


def wav(path):
    """in: path. out: fs, samples (memory-mapped unless 24-bit)."""
    try:
        return wavfile.read(path, mmap=True)
    except ValueError:
        return wavfile.read(path)


def info(path):
    """in: path. out: fs, n samples."""
    if sf:
        i = sf.info(path)
        return i.samplerate, i.frames
    fs, x = wav(path)
    return fs, x.shape[0]


def read(path, a, b):
    """in: path, samples a:b. out: channel 0 as float."""
    if sf:
        with sf.SoundFile(path) as f:
            f.seek(a)
            return f.read(b - a, dtype="float64", always_2d=True)[:, 0]
    _, x = wav(path)
    x = np.asarray(x[a:b] if x.ndim == 1 else x[a:b, 0])
    if np.issubdtype(x.dtype, np.integer):
        return x / np.iinfo(x.dtype).max
    return x.astype(float)


def band_psd(x, fs, p):
    """in: x, fs, p. out: f (Hz), t (s), P (bins x frames) inside the band."""
    lo, hi = p.band[0], min(p.band[1], 0.99 * fs / 2)
    if lo >= hi:
        raise ValueError(f"band {p.band} unusable at fs={fs}")
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    f, t, P = spectrogram(sosfiltfilt(sos, x), fs, window="hann", nperseg=p.nperseg,
                          noverlap=p.noverlap, detrend=False)
    k = (f >= lo) & (f <= hi)
    return f[k], t, P[k]


def entropy(f, P, alpha, floor):
    """in: f, P, alpha, weight floor. out: H per frame, 0 to 1."""
    w = floor + (1 - floor) * np.sqrt(f / f.max())
    q = (P + 1e-12) * w[:, None]
    q /= q.sum(0)
    if alpha == 1:
        H = -(q * np.log2(q)).sum(0)
    else:
        H = np.log2((q ** alpha).sum(0)) / (1 - alpha)
    return H / np.log2(len(f))


def drop_z(H, dt, smooth):
    """in: H, frame step dt (s), baseline window (s). out: D = H - median baseline, robust z of D."""
    n = max(3, int(round(smooth / dt))) | 1
    D = H - median_filter(H, n, mode="nearest")
    m = np.median(D)
    mad = np.median(np.abs(D - m)) + 1e-12
    return D, (D - m) / (1.4826 * mad)


def pick_alpha(f, P, dt, p):
    """in: f, P, dt, p. out: alpha whose z has the lowest 1% quantile."""
    q = [np.quantile(drop_z(entropy(f, P, a, p.w_floor), dt, p.smooth)[1], 0.01)
         for a in p.alphas]
    return p.alphas[int(np.argmin(q))]


def minima(t, z, p):
    """in: t, z, p. out: indices of local minima below z_thr, deepest kept per refract window."""
    zi = z[1:-1]
    i = np.flatnonzero((zi < p.z_thr) & (zi <= z[:-2]) & (zi < z[2:])) + 1
    keep, taken = [], []
    for j in i[np.argsort(z[i])]:
        k = bisect.bisect(taken, t[j])
        if (k == 0 or t[j] - taken[k - 1] >= p.refract) and \
           (k == len(taken) or taken[k] - t[j] >= p.refract):
            taken.insert(k, t[j])
            keep.append(j)
    return np.array(keep, dtype=int)


def rank_file(path, p):
    """in: path, p. out: list of transient rows, alpha used."""
    fs, n = info(path)
    step, pad = int(p.chunk * fs), int(max(p.smooth, 0.5) * fs)
    alpha, rows = p.alpha, []
    for s in range(0, n, step):
        a, b = max(0, s - pad), min(n, s + step + pad)
        x = read(path, a, b)
        if len(x) < 4 * p.nperseg:
            continue
        f, t, P = band_psd(x, fs, p)
        dt = np.median(np.diff(t))
        if alpha is None:
            alpha = pick_alpha(f, P, dt, p)   # first chunk sets it for the file
        H = entropy(f, P, alpha, p.w_floor)
        D, z = drop_z(H, dt, p.smooth)
        for j in minima(t, z, p):
            ts = a / fs + t[j]
            if s / fs <= ts < (s + step) / fs:   # skip hits in the padding
                rows.append(dict(file=os.path.basename(path), time_s=round(ts, 4),
                                 z=round(z[j], 3), entropy=round(H[j], 4),
                                 drop=round(D[j], 4), alpha=alpha))
    return rows, alpha


def audio_files(paths):
    """in: files and/or folders. out: audio files (folders searched recursively)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for d, _, names in os.walk(p):
                out += [os.path.join(d, x) for x in sorted(names)
                        if x.lower().endswith(EXT) and not x.startswith(".")]
        elif os.path.isfile(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rank transients by Renyi spectral entropy drop")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="renyi_ranked.csv")
    ap.add_argument("--top", type=int, default=0, help="keep the N best (0 = all)")
    ap.add_argument("--band", type=float, nargs=2, default=Params.band)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--alphas", type=float, nargs="+", default=Params.alphas)
    ap.add_argument("--nperseg", type=int, default=Params.nperseg)
    ap.add_argument("--noverlap", type=int, default=Params.noverlap)
    ap.add_argument("--smooth", type=float, default=Params.smooth)
    ap.add_argument("--z", type=float, default=Params.z_thr)
    ap.add_argument("--refract", type=float, default=Params.refract)
    ap.add_argument("--chunk", type=float, default=Params.chunk)
    a = ap.parse_args()
    p = Params(band=tuple(a.band), alphas=tuple(a.alphas), alpha=a.alpha, nperseg=a.nperseg,
               noverlap=a.noverlap, smooth=a.smooth, z_thr=a.z, refract=a.refract, chunk=a.chunk)

    paths = audio_files(a.paths)
    if not paths:
        sys.exit("no audio files found")
    rows = []
    for i, path in enumerate(paths, 1):
        r, alpha = rank_file(path, p)
        rows += r
        print(f"[{i}/{len(paths)}] {os.path.basename(path)}: {len(r)} transients, alpha={alpha}")

    rows.sort(key=lambda r: r["z"])
    if a.top:
        rows = rows[:a.top]
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "file", "time_s", "z", "entropy", "drop", "alpha"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["file"], r["time_s"], r["z"], r["entropy"], r["drop"], r["alpha"]])
    print(f"{len(rows)} transients -> {a.out}")


if __name__ == "__main__":
    main()
