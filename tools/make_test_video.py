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
