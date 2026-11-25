"""
swarm.fake_cells — synthetic "cells" with KNOWN rules, written in Seek's tracks.csv format.

Use it to check the bridge (does it recover the rules we put in?) and to try the
whole pipeline before you have real microbe records.

    python -m swarm.fake_cells --mode couzin       --out fake_couzin_tracks.csv
    python -m swarm.fake_cells --mode independent  --out fake_independent_tracks.csv
    python -m swarm.bridge --tracks fake_couzin_tracks.csv --cell-um 2.5

couzin       cells follow repulsion 2 BL / orientation 6 BL / attraction 14 BL
independent  cells wander with the same speed and noise but ignore each other
"""
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TRUTH = {"cell_um": 2.5, "speed_bl_s": 12.0, "zor": 2.0, "zoo": 6.0, "zoa": 14.0,
         "noise_rad_per_step": 0.12, "max_turn_rad_per_step": 0.35, "fps": 20.0}


def simulate(mode, n=40, frames=600, seed=1):
    T = TRUTH
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-12, 12, (n, 2))                     # body lengths, y-up
    th = rng.uniform(-np.pi, np.pi, n)
    step = T["speed_bl_s"] / T["fps"]
    rows = []
    for f in range(frames):
        u = np.c_[np.cos(th), np.sin(th)]
        rel = pos[None, :, :] - pos[:, None, :]            # i → j
        dist = np.linalg.norm(rel, axis=2); np.fill_diagonal(dist, np.inf)
        want = th.copy()
        if mode == "couzin":
            unit = rel / np.maximum(dist, 1e-9)[..., None]
            for i in range(n):
                rep = dist[i] < T["zor"]
                if rep.any():
                    d = -unit[i, rep].sum(0)
                else:
                    ori = (dist[i] >= T["zor"]) & (dist[i] < T["zoo"])
                    att = (dist[i] >= T["zoo"]) & (dist[i] < T["zoa"])
                    do = u[ori].sum(0) + u[i] if ori.any() else None
                    da = unit[i, att].sum(0) if att.any() else None
                    if do is not None and da is not None:
                        d = do / np.linalg.norm(do) + da / max(np.linalg.norm(da), 1e-9)
                    elif do is not None:
                        d = do
                    elif da is not None:
                        d = da
                    else:
                        d = u[i]
                    # gentle pull home so the group stays in the field of view
                    if np.linalg.norm(pos[i]) > 40:
                        d = d / max(np.linalg.norm(d), 1e-9) - 0.5 * pos[i] / np.linalg.norm(pos[i])
                want[i] = np.arctan2(d[1], d[0])
        else:  # independent: soft walls only
            far = np.linalg.norm(pos, axis=1) > 25
            want[far] = np.arctan2(-pos[far, 1], -pos[far, 0])
        dth = (want - th + np.pi) % (2 * np.pi) - np.pi
        dth = np.clip(dth, -T["max_turn_rad_per_step"], T["max_turn_rad_per_step"])
        th = th + dth + rng.normal(0, T["noise_rad_per_step"], n)
        pos = pos + step * np.c_[np.cos(th), np.sin(th)]
        nn = cKDTree(pos).query(pos, k=2)[0][:, 1]
        for i in range(n):
            rows.append((f, f / T["fps"], i, pos[i, 0] * T["cell_um"], -pos[i, 1] * T["cell_um"],
                         T["speed_bl_s"] * T["cell_um"], np.degrees(th[i] % (2 * np.pi) - np.pi * (th[i] % (2 * np.pi) > np.pi) * 2),
                         nn[i] * T["cell_um"]))
    cols = ["frame", "time_s", "cell_id", "x_um", "y_um", "speed_um_s", "heading_deg", "nn_dist_um"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["couzin", "independent"], default="couzin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    out = a.out or f"fake_{a.mode}_tracks.csv"
    simulate(a.mode, seed=a.seed).round(4).to_csv(out, index=False)
    print(f"Wrote {out}  (cell length {TRUTH['cell_um']} µm, truth: {TRUTH})")
