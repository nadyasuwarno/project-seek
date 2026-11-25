"""Synthetic microscope clip with known answers.

Dots swim in straight lines at a set speed. Modes:
  aligned — all share one heading           → alignment ≈ 1
  random  — independent random headings     → alignment ≈ 0, signal drift ≈ 0
  biased  — headings lean toward the signal → signal drift > 0

Usage:
  python tools/make_test_video.py --mode aligned --out data/synthetic_aligned.mp4
writes the MP4 plus <out>_truth.csv (frame, dot_id, x_px, y_px).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SIGNAL_ANGLE = {"right": 0.0, "up": -np.pi / 2, "left": np.pi, "down": np.pi / 2}  # image axes


def make_clip(n_dots=50, n_frames=60, size=512, fps=20.0, um_per_px=0.5, speed_um_s=20.0,
              mode="aligned", signal="right", min_spacing_px=20, dot_sigma=1.6, noise=3.0, seed=0):
    rng = np.random.default_rng(seed)
    step = speed_um_s / fps / um_per_px                      # px per frame
    travel = step * (n_frames - 1)
    margin = 12

    if mode == "aligned":
        headings = np.full(n_dots, np.deg2rad(30))
    elif mode == "random":
        headings = rng.uniform(-np.pi, np.pi, n_dots)
    elif mode == "biased":
        headings = SIGNAL_ANGLE[signal] + rng.vonmises(0, 2.0, n_dots)
    else:
        raise ValueError(mode)

    # place dots so start and end stay in view and starts are spaced apart
    starts = []
    for h in headings:
        d = np.array([np.cos(h), np.sin(h)]) * travel
        for _ in range(5000):
            p = rng.uniform(margin, size - margin, 2)
            if np.all((p + d > margin) & (p + d < size - margin)) and \
               all(np.hypot(*(p - q)) >= min_spacing_px for q in starts):
                starts.append(p)
                break
        else:
            raise RuntimeError("Could not place dots; lower n_dots or spacing.")
    starts = np.array(starts)
    vel = np.c_[np.cos(headings), np.sin(headings)] * step

    yy, xx = np.mgrid[0:size, 0:size]
    light = 25 + 15 * xx / size                               # uneven illumination
    debris = np.zeros((size, size))
    for cx, cy in rng.uniform(0, size, (8, 2)):              # still debris (removed by background step)
        debris += 60 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))

    frames, rows, r = [], [], int(4 * dot_sigma) + 1
    for t in range(n_frames):
        img = light + debris + rng.normal(0, noise, (size, size))
        pos = starts + vel * t
        for i, (x, y) in enumerate(pos):
            x0, y0 = int(x), int(y)
            ys, xs = slice(max(y0 - r, 0), y0 + r + 1), slice(max(x0 - r, 0), x0 + r + 1)
            img[ys, xs] += 140 * np.exp(-((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2 * dot_sigma ** 2))
            rows.append((t, i, x, y))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    truth = pd.DataFrame(rows, columns=["frame", "dot_id", "x_px", "y_px"])
    meta = {"fps": fps, "um_per_px": um_per_px, "speed_um_s": speed_um_s, "mode": mode,
            "signal": signal if mode == "biased" else None, "cell_um": 4 * dot_sigma * um_per_px}
    return np.stack(frames), truth, meta



def simulate_agents(n=30, n_frames=120, fps=20.0, size=512, um_per_px=0.5, speed_um_s=20.0, k_align=0.0,
                    k_pref=0.0, pref=0.0, sigma=0.3, R_um=40.0, seed=0, substeps=4):
    """Cells that follow Seek's rule model, with known coefficients (math axes, periodic field).

    turn rate = k_pref·sin(pref−θ) + k_align·sin(φ_neighbors−θ) + σ·noise.  Returns positions in px,
    image axes (T × n × 2) and headings (T × n). Cells that cross an edge re-enter on the other side,
    so in the video they leave the field and new tracks start, like real cells."""
    rng = np.random.default_rng(seed)
    L = size * um_per_px
    p = rng.uniform(0, L, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    dt = 1 / fps / substeps
    P, TH = [], []
    for t in range(n_frames * substeps):
        if t % substeps == 0:
            P.append(np.c_[p[:, 0], L - p[:, 1]] / um_per_px)
            TH.append(th.copy())
        d = p[None] - p[:, None]
        d -= L * np.round(d / L)
        dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(n) * 1e9
        nb = dist < R_um
        phi = np.arctan2((nb * np.sin(th)[None]).sum(1), (nb * np.cos(th)[None]).sum(1))
        turn = k_pref * np.sin(pref - th) + np.where(nb.any(1), k_align * np.sin(phi - th), 0.0)
        th = th + turn * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
        p = (p + speed_um_s * dt * np.c_[np.cos(th), np.sin(th)]) % L
    return np.array(P), np.array(TH)


def make_flock_clip(n_dots=30, n_frames=120, size=512, fps=20.0, um_per_px=0.5, speed_um_s=20.0,
                    k_align=3.0, k_pref=0.0, pref=0.0, sigma=0.4, R_um=40.0, dot_sigma=1.6, noise=3.0, seed=0):
    """Video of rule-following cells (e.g. random start → flock). Same outputs as make_clip."""
    pos, _ = simulate_agents(n_dots, n_frames, fps, size, um_per_px, speed_um_s, k_align, k_pref, pref,
                             sigma, R_um, seed)
    rng = np.random.default_rng(seed + 1)
    yy, xx = np.mgrid[0:size, 0:size]
    light = 25 + 15 * xx / size
    frames, rows, r = [], [], int(4 * dot_sigma) + 1
    for t in range(n_frames):
        img = light + rng.normal(0, noise, (size, size))
        for i, (x, y) in enumerate(pos[t]):
            x0, y0 = int(x), int(y)
            ys, xs = slice(max(y0 - r, 0), y0 + r + 1), slice(max(x0 - r, 0), x0 + r + 1)
            img[ys, xs] += 140 * np.exp(-((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2 * dot_sigma ** 2))
            rows.append((t, i, x, y))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    truth = pd.DataFrame(rows, columns=["frame", "dot_id", "x_px", "y_px"])
    meta = {"fps": fps, "um_per_px": um_per_px, "speed_um_s": speed_um_s, "mode": "flock", "signal": None,
            "cell_um": 4 * dot_sigma * um_per_px, "k_align": k_align, "k_pref": k_pref, "sigma": sigma}
    return np.stack(frames), truth, meta

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from seek.io import write_mp4

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="aligned", choices=["aligned", "random", "biased"])
    ap.add_argument("--dots", type=int, default=50)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--out", default="data/synthetic.mp4")
    a = ap.parse_args()
    frames, truth, meta = make_clip(n_dots=a.dots, n_frames=a.frames, mode=a.mode)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(write_mp4([np.repeat(fr[..., None], 3, -1) for fr in frames], meta["fps"]))
    truth.to_csv(a.out.rsplit(".", 1)[0] + "_truth.csv", index=False)
    print(f"Wrote {a.out}  ({meta})")
