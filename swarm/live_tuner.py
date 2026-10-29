"""
swarm.live_tuner — run the swarm in real time and change its rules while it moves.

Opened by Seek's "Open live swarm" button, or from a terminal inside project-seek:
    python -m swarm.live_tuner --profile data/swarm/SK-2610-001.json
    python -m swarm.live_tuner                                   # default placeholder rules

Controls
    sliders            change a rule; the swarm reacts on the next tick
    click the arena    HUMAN OVERRIDE: everyone goes to where you clicked
    Release            hand control back to the swarm rules
    Scatter            throw robots into random headings (perturbation test)
    Reset              restart with the slider values (and robot count)
    Save profile       write the current sliders to <profile>_tuned.json

Charts: green = robots right now. Dashed = what the cells did in Seek (if the
profile came from seek_bridge.py). Match the dashed lines and you've reproduced
the cells' collective behaviour with robots.
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rps.robotarium as robotarium
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

from swarm.sim import (ARENA, BODY_LENGTH_M, DT, couzin_heading, initial_poses, metrics,
                       steer, unit, wall_push)

INK, MUTED, SWARM, OVR, CELL = "#2C2C2A", "#888780", "#1D9E75", "#D85A30", "#534AB7"
HISTORY_S = 30
TRAIL = 25

SLIDERS = [  # key, label, min, max, step
    ("zone_repulsion_bl", "Repulsion (BL)", 1.1, 6, 0.1),
    ("zone_orientation_bl", "Orientation (BL)", 0, 15, 0.25),
    ("zone_attraction_bl", "Attraction (BL)", 0, 25, 0.5),
    ("speed_bl_per_s", "Speed (BL/s)", 0.2, 1.5, 0.05),
    ("heading_noise_deg", "Noise (°)", 0, 60, 1),
    ("max_turn_rate_deg_per_s", "Max turn (°/s)", 20, 200, 5),
    ("blind_angle_deg", "Blind angle (°)", 0, 180, 5),
    ("num_robots", "Robots (on Reset)", 3, 30, 1),
]


class Tuner:
    def __init__(self, profile_path):
        self.profile_path = Path(profile_path)
        self.raw = json.loads(self.profile_path.read_text())
        self.targets = self.raw.get("targets") or {}
        self.n = int(self.raw.get("experiment", {}).get("num_robots", 12))
        self.rng = np.random.default_rng(self.raw.get("experiment", {}).get("seed", 7))
        self.safety = create_uni_barrier_certificate_with_boundary(safety_radius=0.12)
        self._build_ui()
        self.reset()

    # ── simulation ───────────────────────────────────────────────────────
    def reset(self, _=None):
        self.n = int(self.sl["num_robots"].val)
        self.r = robotarium.Robotarium(number_of_robots=self.n, show_figure=False, sim_in_real_time=False,
                                       initial_conditions=initial_poses(self.n, self.rng),
                                       skip_initialization=True)
        self.t = 0.0
        self.hist = {k: deque(maxlen=int(HISTORY_S / DT)) for k in ("t", "pol", "mill", "nnd")}
        self.trail = deque(maxlen=TRAIL)
        self.target = None
        self.events = []                 # (t, text)
        self.pre_override_pol = None
        self.release_t = None
        self._clear_status()
        self.wall_start = time.perf_counter()
        self.sim_at_wall_start = 0.0

    def prof(self):
        v = {k: s.val for k, s in self.sl.items()}
        rep = v["zone_repulsion_bl"]
        ori = max(v["zone_orientation_bl"], rep)          # keep zones ordered: rep ≤ ori ≤ att
        att = max(v["zone_attraction_bl"], ori)
        bl = BODY_LENGTH_M
        return {"r_rep": rep * bl, "r_ori": ori * bl, "r_att": att * bl,
                "blind": np.deg2rad(v["blind_angle_deg"]), "speed": v["speed_bl_per_s"] * bl,
                "max_turn": np.deg2rad(v["max_turn_rate_deg_per_s"]),
                "noise": np.deg2rad(v["heading_noise_deg"])}

    def tick(self):
        p = self.prof()
        x = self.r.get_poses()
        pos, head = x[:2], x[2]
        dxu = np.zeros((2, self.n))
        for i in range(self.n):
            desired = unit(self.target - pos[:, i]) if self.target is not None else couzin_heading(i, pos, head, p)
            desired = unit(desired + 1.5 * wall_push(pos[:, i]))
            a = np.arctan2(desired[1], desired[0]) + self.rng.normal(0, p["noise"])
            dxu[:, i] = steer(head[i], np.array([np.cos(a), np.sin(a)]), p["speed"], p["max_turn"])
        self.r.set_velocities(np.arange(self.n), self.safety(dxu, x))
        self.r.step()
        self.t += DT
        pol, mill, nnd, c = metrics(pos, head)
        for k, val in zip(("t", "pol", "mill", "nnd"), (self.t, pol, mill, nnd)):
            self.hist[k].append(val)
        self.trail.append(pos.copy())
        self._track_events(pol, c)
        return pos, head

    def _track_events(self, pol, c):
        if self.target is not None and not getattr(self, "_reached", True):
            if np.linalg.norm(c - self.target) < 0.4:
                self._reached = True
                self.events.append((self.t, f"reached target in {self.t - self.override_t:.1f} s"))
        if self.release_t is not None and self.pre_override_pol is not None:
            if pol >= 0.9 * self.pre_override_pol and self.t - self.release_t > 1.0:
                self.events.append((self.t, f"swarm re-formed {self.t - self.release_t:.1f} s after release"))
                self.release_t = None

    # ── interaction ──────────────────────────────────────────────────────
    def on_click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        if self.target is None:
            pol = list(self.hist["pol"])[-90:]
            self.pre_override_pol = float(np.mean(pol)) if pol else None
            self.override_t = self.t
            self.events.append((self.t, "human override ON"))
        self.target = np.array([ev.xdata, ev.ydata])
        self._reached = False
        self.release_t = None

    def release(self, _=None):
        if self.target is not None:
            self.target = None
            self.release_t = self.t
            self.events.append((self.t, "control released"))

    def scatter(self, _=None):
        # Simulator-only shortcut: spin every robot to a random heading instantly.
        # (On real robots this would be a "spin in place" command.)
        self.r._poses[2, :] = self.rng.uniform(-np.pi, np.pi, self.n)
        self.events.append((self.t, "scattered headings"))

    def save(self, _=None):
        out = dict(self.raw)
        for k, s in self.sl.items():
            if k == "num_robots":
                out.setdefault("experiment", {})["num_robots"] = int(s.val)
            else:
                out[k] = round(float(s.val), 3)
        out["source"] = f"tuned live from {self.profile_path.name}"
        path = self.profile_path.with_name(self.profile_path.stem + "_tuned.json")
        path.write_text(json.dumps(out, indent=2))
        self.events.append((self.t, f"saved {path.name}"))

    # ── drawing ──────────────────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8.2))
        try:
            self.fig.canvas.manager.set_window_title(f"Seek swarm · {self.raw.get('source') or 'live tuner'}")
        except AttributeError:          # headless backends have no window
            pass
        self.ax = self.fig.add_axes([0.02, 0.24, 0.56, 0.70])
        self.ax.set_xlim(ARENA[0] - 0.05, ARENA[1] + 0.05); self.ax.set_ylim(ARENA[2] - 0.05, ARENA[3] + 0.05)
        self.ax.set_aspect("equal"); self.ax.set_xticks([]); self.ax.set_yticks([])
        for s in self.ax.spines.values():
            s.set_visible(False)
        self.ax.add_patch(Rectangle((ARENA[0], ARENA[2]), 3.2, 2.0, fill=False, ec=MUTED, lw=1))
        self.trails = LineCollection([], colors=SWARM, linewidths=0.8, alpha=0.35)
        self.ax.add_collection(self.trails)
        self.bodies = self.ax.scatter([], [], s=150, c=SWARM, edgecolors=INK, linewidths=0.6, zorder=3)
        self.noses = LineCollection([], colors=INK, linewidths=1.4, zorder=4)
        self.ax.add_collection(self.noses)
        self.star, = self.ax.plot([], [], marker="*", ms=18, color=OVR, zorder=5)
        self.title = self.fig.text(0.02, 0.955, "", fontsize=12, color=INK)
        self.status = self.fig.text(0.02, 0.215, "", fontsize=10, color=MUTED, va="top", family="monospace")

        self.max = {}
        specs = [("pol", "Alignment (polarization)", "polarization", (0, 1.05)),
                 ("mill", "Rotation (milling)", "milling", (0, 1.05)),
                 ("nnd", "Spacing (nearest neighbour, BL)", "nnd_bl", None)]
        self.lines = {}
        for k, (key, title, tkey, ylim) in enumerate(specs):
            a = self.fig.add_axes([0.64, 0.80 - k * 0.18, 0.29, 0.11])
            a.set_title(title, loc="left", fontsize=10, color=INK)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=8, colors=MUTED)
            if ylim:
                a.set_ylim(*ylim)
            self.lines[key] = (a, a.plot([], [], color=SWARM, lw=1.5)[0], ylim)
            tv = self.targets.get(tkey)
            if tv is not None:
                a.axhline(tv, color=CELL, ls="--", lw=1.2)
                a.text(1.01, tv, f"cells\n{tv:.2f}", color=CELL, fontsize=8, va="center",
                       transform=a.get_yaxis_transform(), clip_on=False)

        self.sl = {}
        exp = self.raw.get("experiment", {})
        for k, (key, label, lo, hi, step) in enumerate(SLIDERS):
            a = self.fig.add_axes([0.75, 0.335 - k * 0.032, 0.20, 0.02])
            init = exp.get("num_robots", 12) if key == "num_robots" else self.raw.get(key, (lo + hi) / 2)
            s = Slider(a, label, lo, hi, valinit=float(np.clip(init, lo, hi)), valstep=step, color=SWARM)
            s.label.set_fontsize(9); s.valtext.set_fontsize(9)
            self.sl[key] = s

        self.btns = []
        for k, (label, fn) in enumerate([("Release", self.release), ("Scatter", self.scatter),
                                         ("Reset", self.reset), ("Save profile", self.save)]):
            b = Button(self.fig.add_axes([0.64 + k * 0.083, 0.035, 0.075, 0.045]), label)
            b.on_clicked(fn)
            self.btns.append(b)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def _clear_status(self):
        if hasattr(self, "status"):
            self.status.set_text("")

    def draw(self, _frame=None, steps=1):
        for _ in range(steps):
            pos, head = self.tick()
        col = OVR if self.target is not None else SWARM
        self.bodies.set_offsets(pos.T); self.bodies.set_facecolor(col)
        self.noses.set_segments([[(pos[0, i], pos[1, i]),
                                  (pos[0, i] + 0.09 * np.cos(head[i]), pos[1, i] + 0.09 * np.sin(head[i]))]
                                 for i in range(self.n)])
        tr = np.array(self.trail)                       # T x 2 x N
        self.trails.set_segments([tr[:, :, i] for i in range(self.n)] if len(tr) > 1 else [])
        self.trails.set_color(col)
        self.star.set_data(([self.target[0]], [self.target[1]]) if self.target is not None else ([], []))

        t = np.array(self.hist["t"])
        for key, (a, line, ylim) in self.lines.items():
            y = np.array(self.hist[key])
            line.set_data(t, y)
            a.set_xlim(max(0, self.t - HISTORY_S), max(HISTORY_S, self.t))
            if ylim is None and len(y):
                tv = self.targets.get("nnd_bl") or 0
                a.set_ylim(0, max(y.max(), tv) * 1.2 + 0.1)

        mode = "HUMAN OVERRIDE — click to move target, Release to hand back" if self.target is not None \
            else "Swarm rules — click the arena to take control"
        speedup = self.t / max(time.perf_counter() - self.wall_start, 1e-6)
        self.title.set_text(f"t = {self.t:5.1f} s · {mode}")
        self.title.set_color(OVR if self.target is not None else INK)
        src = self.raw.get("source", "")
        ts = (self.raw.get("fit") or {}).get("time_scale")
        lines = [f"profile: {self.profile_path.name}  ·  {src}",
                 f"sim speed: {speedup:.2f}× real time" + (f"  ·  1 s robot ≈ {ts:g} s cells" if ts else "")]
        lines += [f"{et:6.1f} s  {txt}" for et, txt in self.events[-4:]]
        self.status.set_text("\n".join(lines))
        return ()

    def run(self):
        self.anim = FuncAnimation(self.fig, self.draw, interval=int(DT * 1000), cache_frame_data=False)
        plt.show()


def use_window_backend():
    """The tuner needs an on-screen window. If something (e.g. Streamlit) left matplotlib drawing
    off-screen ('Agg'), switch to this computer's window backend before any figure is made."""
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        return
    for name in (["macosx"] if sys.platform == "darwin" else []) + ["QtAgg", "TkAgg"]:
        try:
            plt.switch_backend(name)
            return
        except Exception:
            continue
    raise SystemExit("No on-screen window is available for matplotlib on this computer. "
                     "On a Mac this should not happen — run it from a normal Terminal session.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(Path(__file__).parent / "profiles" / "behavior_profile.json"))
    a = ap.parse_args()
    use_window_backend()
    Tuner(a.profile).run()
