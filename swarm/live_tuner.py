"""
swarm.live_tuner — the Seek Simulator window.

Opened by "Open in Simulator" on a record in Seek's Archives, or from a terminal inside project-seek:
    python -m swarm.live_tuner --profile data/swarm/SK-2611-001.json

Every number comes from the profile (fitted from a Seek record) — there are no sliders.
    Blue Light / Red Light   arm a light, then click the arena to place it (click again to move it)
    Release                  turn every light off: the swarm is on its own again
    Reset                    new robots, no lights

Each light pulls the robots by: instinct (measured by Seek from cells filmed under that light)
+ training (a set value until the learning step exists). The log under the arena shows both.
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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

from swarm.sim import (ARENA, DT, LIGHT_COLORS, at_lights, decide, initial_poses, metrics,
                       profile_to_params)

INK, MUTED, FAINT, PANEL, EDGE = "#111111", "#8A8A8A", "#CFCFCF", "#F3F3F3", "#DADADA"
ROBOT, CELLS = "#3A3A3A", "#8A8A8A"
LIGHT = {"blue": {"fill": "#A7C7FA", "text": "#0B5FFF", "glow": "#2F7BFF", "label": "Blue Light"},
         "red": {"fill": "#FCA5A5", "text": "#E10600", "glow": "#FF2A1F", "label": "Red Light"}}
HISTORY_S, TRAIL = 30, 22


def panel(fig, x, y, w, h, r=0.02):
    """A rounded grey card in figure coordinates (drawn on the background layer)."""
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                      transform=fig.transFigure, fc=PANEL, ec=EDGE, lw=1, zorder=-10,
                                      mutation_aspect=fig.get_figwidth() / fig.get_figheight()))


class Pill:
    """A rounded button: its own axes, so a click inside it is easy to detect."""

    def __init__(self, fig, rect, label, fill, color, caption=None, caption_color=None):
        self.ax = fig.add_axes(rect)
        w_in, h_in = rect[2] * fig.get_figwidth(), rect[3] * fig.get_figheight()
        self.ax.set_xlim(0, w_in); self.ax.set_ylim(0, h_in); self.ax.axis("off")
        self.box = FancyBboxPatch((0.02, 0.02), w_in - 0.04, h_in - 0.04,
                                  boxstyle=f"round,pad=0,rounding_size={h_in / 2 - 0.02}", fc=fill, ec="none")
        self.ax.add_patch(self.box)
        self.text = self.ax.text(w_in / 2, h_in / 2, label, ha="center", va="center", fontsize=13,
                                 color=color, weight="bold")
        if caption:
            fig.text(rect[0] + 0.012, rect[1] + rect[3] + 0.01, caption, fontsize=9, color=caption_color)

    def highlight(self, on):
        self.box.set_edgecolor(INK if on else "none"); self.box.set_linewidth(2 if on else 0)


class Simulator:
    def __init__(self, profile_path):
        self.path = Path(profile_path)
        self.raw = json.loads(self.path.read_text())
        self.p = profile_to_params(self.raw)
        self.targets = self.raw.get("targets") or {}
        self.code = self.raw.get("record_code") or self.raw.get("source") or self.path.stem
        self.n = int((self.raw.get("experiment") or {}).get("num_robots", 12))
        self.rng = np.random.default_rng((self.raw.get("experiment") or {}).get("seed", 7))
        self.safety = create_uni_barrier_certificate_with_boundary(safety_radius=0.12)
        self.build()
        self.reset()

    # ── simulation ─────────────────────────────────────────────────────────
    def reset(self):
        self.r = robotarium.Robotarium(number_of_robots=self.n, show_figure=False, sim_in_real_time=False,
                                       initial_conditions=initial_poses(self.n, self.rng), skip_initialization=True)
        self.t, self.state, self.lights, self.armed = 0.0, {}, {}, None
        self.placed, self.reached = {}, set()
        self.hist = {k: deque(maxlen=int(HISTORY_S / DT)) for k in ("t", "pol", "nnd", "blue", "red")}
        self.trail = deque(maxlen=TRAIL)
        self.events = []
        self.wall0, self.sim0 = time.perf_counter(), 0.0
        for pill in self.pills.values():
            pill.highlight(False)

    def tick(self):
        x = self.r.get_poses()
        pos, head = x[:2], x[2]
        dxu = decide(pos, head, self.p, self.rng, state=self.state, lights=self.lights)
        self.r.set_velocities(np.arange(self.n), self.safety(dxu, x))
        self.r.step()
        self.t += DT
        pol, _, nnd, _ = metrics(pos, head)
        share = at_lights(pos, self.lights)
        for k, v in (("t", self.t), ("pol", pol), ("nnd", nnd), ("blue", share.get("blue", np.nan)),
                     ("red", share.get("red", np.nan))):
            self.hist[k].append(v)
        self.trail.append(pos.copy())
        for c, s in share.items():
            if s >= 0.5 and c not in self.reached:
                self.reached.add(c)
                self.log(f"half the swarm reached the {c} light in {self.t - self.placed[c]:.1f} s")
        return pos, head

    def log(self, text):
        self.events.append((self.t, text))

    # ── interaction ────────────────────────────────────────────────────────
    def click(self, ev):
        for name, pill in self.pills.items():
            if ev.inaxes is pill.ax:
                return self.press(name)
        if ev.inaxes is self.ax and self.armed and ev.xdata is not None:
            c = self.armed
            xy = (float(np.clip(ev.xdata, *ARENA[:2])), float(np.clip(ev.ydata, *ARENA[2:])))
            self.log(f"{c} light {'moved' if c in self.lights else 'on'}")
            self.lights[c], self.placed[c] = xy, self.t
            self.reached.discard(c)
            self.armed = None
            self.pills[c].highlight(False)

    def press(self, name):
        if name in LIGHT_COLORS:
            self.armed = None if self.armed == name else name
            for c in LIGHT_COLORS:
                self.pills[c].highlight(c == self.armed)
        elif name == "release":
            if self.lights:
                self.log("lights off · swarm released")
            self.lights, self.armed = {}, None
            for c in LIGHT_COLORS:
                self.pills[c].highlight(False)
        elif name == "reset":
            self.reset()

    # ── layout ─────────────────────────────────────────────────────────────
    def build(self):
        self.fig = fig = plt.figure(figsize=(15, 9.6), facecolor="white")
        try:
            fig.canvas.manager.set_window_title(f"Seek Simulator · {self.code}")
        except AttributeError:
            pass
        fig.text(0.035, 0.935, "Seek", fontsize=26, weight="bold", color=INK)
        fig.text(0.035, 0.895, "Simulator", fontsize=26, weight="bold", color=MUTED)

        panel(fig, 0.035, 0.775, 0.60, 0.085)
        self.pills = {
            "blue": Pill(fig, [0.055, 0.795, 0.11, 0.045], LIGHT["blue"]["label"], LIGHT["blue"]["fill"],
                         LIGHT["blue"]["text"], self.p["lights"]["blue"]["role"], "#7FA7E8"),
            "red": Pill(fig, [0.20, 0.795, 0.11, 0.045], LIGHT["red"]["label"], LIGHT["red"]["fill"],
                        LIGHT["red"]["text"], self.p["lights"]["red"]["role"], "#EE8A8A"),
            "release": Pill(fig, [0.345, 0.795, 0.11, 0.045], "Release", "#A3A3A3", "white"),
            "reset": Pill(fig, [0.49, 0.795, 0.11, 0.045], "Reset", "#A3A3A3", "white"),
        }

        panel(fig, 0.035, 0.235, 0.60, 0.515)
        self.ax = ax = fig.add_axes([0.05, 0.25, 0.57, 0.485])
        ax.set_xlim(ARENA[0] - 0.05, ARENA[1] + 0.05); ax.set_ylim(ARENA[2] - 0.05, ARENA[3] + 0.05)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((ARENA[0], ARENA[2]), 3.2, 2.0, fill=False, ec=FAINT, lw=1))
        self.glows = {c: [ax.add_patch(Circle((0, 0), r, fc=LIGHT[c]["glow"], ec="none", alpha=0, zorder=1))
                          for r in (0.42, 0.3, 0.2, 0.11)] for c in LIGHT_COLORS}
        self.cores = {c: ax.add_patch(Circle((0, 0), 0.045, fc=LIGHT[c]["glow"], ec="white", lw=1.2,
                                             alpha=0, zorder=2)) for c in LIGHT_COLORS}
        self.trails = LineCollection([], colors=ROBOT, linewidths=0.8, alpha=0.25, zorder=3)
        ax.add_collection(self.trails)
        self.bodies = ax.scatter([], [], s=120, c=ROBOT, edgecolors="white", linewidths=0.8, zorder=4)
        self.noses = LineCollection([], colors="white", linewidths=1.3, zorder=5)
        ax.add_collection(self.noses)
        self.hint = ax.text(0, ARENA[3] - 0.12, "", ha="center", fontsize=11, color=MUTED, zorder=6)

        self.status = fig.text(0.045, 0.205, "", fontsize=10, color=MUTED, va="top", family="monospace",
                               linespacing=1.6)

        charts = [("Alignment (polarization)", "pol", "polarization"),
                  ("Spacing (body lengths)", "nnd", "nnd_bl"),
                  ("Robots at each light", "light", None)]
        self.lines = {}
        for k, (title, key, tkey) in enumerate(charts):
            y0 = 0.635 - k * 0.285
            fig.text(0.665, y0 + 0.235, title, fontsize=12, weight="bold", color=MUTED)
            panel(fig, 0.665, y0, 0.30, 0.22)
            a = fig.add_axes([0.685, y0 + 0.03, 0.26, 0.17])
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(FAINT)
            a.set_facecolor(PANEL); a.tick_params(labelsize=8, colors=MUTED)
            if key == "light":
                a.set_ylim(0, 1.05)
                self.lines["blue"] = a.plot([], [], color=LIGHT["blue"]["glow"], lw=1.8)[0]
                self.lines["red"] = a.plot([], [], color=LIGHT["red"]["glow"], lw=1.8)[0]
                self.lines["light_ax"] = a
            else:
                if key == "pol":
                    a.set_ylim(0, 1.05)
                self.lines[key] = a.plot([], [], color=ROBOT, lw=1.6)[0]
                self.lines[key + "_ax"] = a
                tv = self.targets.get(tkey)
                if tv is not None:
                    a.axhline(tv, color=CELLS, ls="--", lw=1.2)
                    a.text(1.0, tv, f" cells {tv:.2f}", color=CELLS, fontsize=8, va="bottom", ha="right",
                           transform=a.get_yaxis_transform())
        fig.canvas.mpl_connect("button_press_event", self.click)

    def sources(self):
        """Where each light's pull comes from — the data trail from Seek to this window."""
        out = []
        for c in LIGHT_COLORS:
            L = self.p["lights"][c]
            inst = (f"instinct {L['instinct']:+.2f} ({L['instinct_source']})" if L["instinct"] is not None
                    else f"instinct — (no cells filmed under {c} light yet)")
            out.append(f"{c:<4} light  pull {L['pull']:+.2f} = {inst} + training {L['trained']:+.2f} (set)")
        return out

    def draw(self, _frame=None):
        pos, head = self.tick()
        self.bodies.set_offsets(pos.T)
        self.noses.set_segments([[(pos[0, i], pos[1, i]),
                                  (pos[0, i] + 0.075 * np.cos(head[i]), pos[1, i] + 0.075 * np.sin(head[i]))]
                                 for i in range(self.n)])
        tr = np.array(self.trail)
        self.trails.set_segments([tr[:, :, i] for i in range(self.n)] if len(tr) > 1 else [])
        for c in LIGHT_COLORS:
            on = c in self.lights
            bright = self.p["lights"][c]["brightness"]
            for k, g in enumerate(self.glows[c]):
                g.set_alpha((0.07 + 0.05 * k) * bright if on else 0)
                if on:
                    g.center = self.lights[c]
            self.cores[c].set_alpha(1 if on else 0)
            if on:
                self.cores[c].center = self.lights[c]
        self.hint.set_text(f"Click in the arena to place the {self.armed} light" if self.armed else "")

        t = np.array(self.hist["t"])
        lo, hi = max(0, self.t - HISTORY_S), max(HISTORY_S, self.t)
        for key in ("pol", "nnd", "blue", "red"):
            self.lines[key].set_data(t, np.array(self.hist[key]))
        for key in ("pol_ax", "nnd_ax", "light_ax"):
            self.lines[key].set_xlim(lo, hi)
        nnd = np.array(self.hist["nnd"])
        if len(nnd):
            self.lines["nnd_ax"].set_ylim(0, max(nnd.max(), self.targets.get("nnd_bl") or 0) * 1.25 + 0.1)

        speed = self.t / max(time.perf_counter() - self.wall0, 1e-6)
        ts = (self.raw.get("fit") or {}).get("time_scale")
        lines = [f"Profile: {self.code}",
                 f"Simulator speed: {speed:.2f} x real time" + (f"  (1 s robot ≈ {ts:g} s cells)" if ts else ""),
                 *self.sources(), ""]
        lines += [f"{et:6.1f} s  {txt}" for et, txt in self.events[-4:]]
        self.status.set_text("\n".join(lines))
        return ()

    def run(self):
        self.anim = FuncAnimation(self.fig, self.draw, interval=int(DT * 1000), cache_frame_data=False)
        plt.show()


def use_window_backend():
    """The simulator needs an on-screen window. If something (e.g. Streamlit) left matplotlib drawing
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


Tuner = Simulator          # older name, kept for scripts that import it

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(Path(__file__).parent / "profiles" / "behavior_profile.json"))
    a = ap.parse_args()
    use_window_backend()
    Simulator(a.profile).run()
