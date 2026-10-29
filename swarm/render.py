"""
swarm.render — turn a recorded run (trajectories.csv + metrics.csv) into:
    swarm_run.mp4   top-down replay of the arena (needs ffmpeg)
    metrics.png     alignment / rotation / spacing over time
Run after swarm.run:   python -m swarm.render data/swarm/runs/<name> <profile.json>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/swarm/runs/behavior_profile"
PROFILE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "profiles", "behavior_profile.json")
ARENA = [-1.6, 1.6, -1.0, 1.0]
R = 0.055

traj = pd.read_csv(os.path.join(OUT, "trajectories.csv"))
met = pd.read_csv(os.path.join(OUT, "metrics.csv"))
ov = json.load(open(PROFILE))["experiment"].get("override", {})
summ = json.load(open(os.path.join(OUT, "summary.json")))
if ov.get("enabled") and summ.get("override_target_m"):
    ov["target_m"] = summ["override_target_m"]
times = np.sort(traj.t_s.unique())
ids = np.sort(traj.track_id.unique())
P = traj.pivot(index="t_s", columns="track_id")
X, Y, TH = P.x_m.values, P.y_m.values, P.theta_rad.values

INK, MUTED, SWARM, OVR = "#2C2C2A", "#888780", "#1D9E75", "#D85A30"

# ---------------- metrics chart ----------------
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
series = [("polarization", "Alignment (polarization)", (0, 1.05)),
          ("milling", "Rotation (milling)", (0, 1.05)),
          ("nnd_bl", "Spacing (nearest neighbour, body lengths)", None)]
for ax, (col, title, ylim) in zip(axes, series):
    if ov.get("enabled"):
        ax.axvspan(ov["start_s"], ov["end_s"], color=OVR, alpha=0.12, lw=0)
    ax.plot(met.t_s, met[col], color=SWARM, lw=1.6)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    if ylim:
        ax.set_ylim(*ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#E5E4DF", lw=0.6)
axes[-1].set_xlabel("Time (s)", color=MUTED)
if ov.get("enabled"):
    axes[0].text(ov["start_s"] + 0.3, 0.08, "human override", color=OVR, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "metrics.png"), dpi=150)
plt.close(fig)

# ---------------- arena replay video ----------------
fig, ax = plt.subplots(figsize=(8, 5.4))
fig.subplots_adjust(0.02, 0.02, 0.98, 0.93)
ax.set_xlim(ARENA[0] - 0.05, ARENA[1] + 0.05)
ax.set_ylim(ARENA[2] - 0.05, ARENA[3] + 0.05)
ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((ARENA[0], ARENA[2]), 3.2, 2.0, fill=False, ec=MUTED, lw=1))
target = None
if ov.get("enabled"):
    target = ax.plot(*ov["target_m"], marker="*", ms=16, color=OVR, alpha=0)[0]

trails = [ax.plot([], [], color=SWARM, lw=0.8, alpha=0.35)[0] for _ in ids]
bodies = [ax.add_patch(Circle((0, 0), R, fc=SWARM, ec=INK, lw=0.6)) for _ in ids]
noses = [ax.plot([], [], color=INK, lw=1.4)[0] for _ in ids]
title = ax.set_title("", loc="left", fontsize=11, color=INK)

TRAIL = 45
writer = FFMpegWriter(fps=30, codec="libx264",
                      extra_args=["-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"])
with writer.saving(fig, os.path.join(OUT, "swarm_run.mp4"), dpi=120):
    for k in range(0, len(times), 1):
        t = times[k]
        m = met.iloc[k]
        overriding = m.phase == "override"
        col = OVR if overriding else SWARM
        for i in range(len(ids)):
            a = max(0, k - TRAIL)
            trails[i].set_data(X[a:k + 1, i], Y[a:k + 1, i]); trails[i].set_color(col)
            bodies[i].center = (X[k, i], Y[k, i]); bodies[i].set_facecolor(col)
            noses[i].set_data([X[k, i], X[k, i] + 0.09 * np.cos(TH[k, i])],
                              [Y[k, i], Y[k, i] + 0.09 * np.sin(TH[k, i])])
        if target is not None:
            target.set_alpha(1 if overriding else 0)
        label = "HUMAN OVERRIDE" if overriding else "Swarm rules (Couzin profile)"
        title.set_text(f"t = {t:5.1f} s   ·   {label}   ·   alignment {m.polarization:.2f}")
        title.set_color(col if overriding else INK)
        writer.grab_frame()
plt.close(fig)
print("Wrote", os.path.join(OUT, "swarm_run.mp4"), "and metrics.png")
