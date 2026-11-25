"""Side-by-side preview: the cells (left) and the Robotarium swarm running their rules (right)."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .embody import _math_frame

PAPER, INK, MUTED, LINE = (230, 229, 225), (27, 27, 27), (93, 95, 98), (150, 151, 150)
BLUE, YELLOW, RED = (31, 134, 176), (227, 207, 28), (180, 60, 40)
W, H = 1280, 720
FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/Library/Fonts/Courier New.ttf",
              "/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/cour.ttf"]


def _font(size):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fmt(v, spec="{:.2f}"):
    return "—" if v is None else spec.format(v)


def render(tracks, rules, plan, hist, cmp, title="", fps_out=15, max_seconds=None):
    """Returns a list of RGB frames (H×W×3 uint8)."""
    f_big, f, f_small = _font(22), _font(15), _font(12)
    d = _math_frame(tracks)
    cell_frames = {k: g for k, g in d.groupby("frame")}
    frame_ids = sorted(cell_frames)
    t_cells = d.groupby("frame")["time_s"].first()
    clip_len = float(t_cells.max() - t_cells.min()) or 1.0
    t0 = float(t_cells.min())
    cx0, cx1, cy0, cy1 = d["X"].min(), d["X"].max(), d["Y"].min(), d["Y"].max()
    Wf, Hf = rules["field_um"]
    mx, my = (cx0 + cx1) / 2, (cy0 + cy1) / 2

    # panels
    L = (40, 120, 470, 550)                         # cells field box (x0, y0, x1, y1)
    A = (520, 140, 1240, 140 + 720 * 2.0 / 3.2)     # arena box (3.2 × 2.0 m)
    lw, lh = L[2] - L[0], L[3] - L[1]
    k_cell = min(lw / Wf, lh / Hf) * 0.98
    k_ar = (A[2] - A[0]) / 3.2

    def cell_px(X, Y):
        return L[0] + lw / 2 + (X - mx) * k_cell, L[1] + lh / 2 - (Y - my) * k_cell

    def arena_px(x, y):
        return A[0] + (x + 1.6) * k_ar, A[1] + (1.0 - y) * k_ar

    P, dt, D = hist["poses"], hist["dt"], plan["_scale"]["time_dilation"]
    total = len(P) * dt if max_seconds is None else min(len(P) * dt, max_seconds)
    step = max(1, int(round(1 / fps_out / dt)))
    trail_r = int(2.0 / dt)
    cells_by_id = {cid: g.sort_values("time_s") for cid, g in d.groupby("cell_id")}

    c, r = cmp["cells"], cmp["robots"]
    rows = [("", "cells", "robots"),
            ("group order", _fmt(c["group_order"]), _fmt(r["group_order"])),
            ("local alignment", _fmt(c["local_alignment"]), _fmt(r["local_alignment"])),
            ("toward signal", _fmt(c["heading_toward_pref"]), _fmt(r["heading_toward_pref"])),
            ("spacing / field", _fmt(c["nn_field"]), _fmt(r["nn_field"]))]
    rule_lines = [f"turn = {plan['k_pref']:.2f}·pull + {plan['k_align']:.2f}·align + {plan['k_sep']:.1f}·sep + noise {plan['sigma']:.2f}",
                  f"{plan['N']} robots · {plan['v']:.2f} m/s · R {plan['R']:.2f} m · "
                  f"1 s of cells = {D:.1f} s of robots",
                  f"pull: {rules['k_pref_source']}"]

    frames = []
    for k in range(0, int(total / dt), step):
        t_r = k * dt
        img = Image.new("RGB", (W, H), PAPER)
        g = ImageDraw.Draw(img)
        g.text((40, 26), title, font=f_big, fill=INK)
        g.text((40, 58), "SEEK → ROBOTARIUM : rules learned from the cells, run by the robots", font=f, fill=MUTED)
        g.text((L[0], L[1] - 26), "CELLS (microscope field)", font=f, fill=INK)
        g.text((A[0], A[1] - 26), "ROBOTS (Robotarium arena 3.2 × 2 m, simulator)", font=f, fill=INK)

        # cells, looped and slowed to robot time
        tc = t0 + (t_r / D) % clip_len
        g.rectangle(L, outline=LINE, width=1)
        for cid, tr in cells_by_id.items():
            seg = tr[(tr["time_s"] <= tc) & (tr["time_s"] >= tc - 2.0 / D)]
            if len(seg) < 1 or tr["time_s"].max() < tc:
                continue
            pts = [cell_px(x, y) for x, y in zip(seg["X"], seg["Y"])]
            if len(pts) > 1:
                g.line(pts, fill=LINE, width=2)
            x, y = pts[-1]
            g.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK)
        g.text((L[0], L[3] + 8), f"t = {tc - t0:5.2f} s (cell time, looped)", font=f_small, fill=MUTED)

        # robots
        g.rectangle(A, outline=INK, width=2)
        if plan["k_pref"] > 0:
            ang = hist["pref"][min(k, len(P) - 1)]
            ax, ay = arena_px(1.25, -0.82)
            g.ellipse((ax - 5, ay - 5, ax + 5, ay + 5), fill=YELLOW)
            ex, ey = ax + 38 * np.cos(ang), ay - 38 * np.sin(ang)
            g.line([(ax, ay), (ex, ey)], fill=INK, width=2)
            g.text((ax - 100, ay - 7), "pull", font=f_small, fill=MUTED)
        lo = max(0, k - trail_r)
        for i in range(P.shape[2]):
            pts = [arena_px(x, y) for x, y in P[lo:k + 1:3, :2, i]]
            if len(pts) > 1:
                g.line(pts, fill=(160, 190, 205), width=2)
        rad = 0.055 * k_ar
        for i in range(P.shape[2]):
            x, y, th = P[k, :, i]
            px, py = arena_px(x, y)
            g.ellipse((px - rad, py - rad, px + rad, py + rad), fill=BLUE, outline=INK)
            g.line([(px, py), (px + rad * 1.3 * np.cos(th), py - rad * 1.3 * np.sin(th))], fill=PAPER, width=2)
        g.text((A[0], A[3] + 8), f"t = {t_r:5.1f} s (robot time)   ·   {hist['verdict']}", font=f_small, fill=MUTED)

        # rules + comparison
        y0 = 632
        for j, line in enumerate(rule_lines):
            g.text((40, y0 + j * 20), line, font=f_small if j else f, fill=INK if j == 0 else MUTED)
        for j, (a, b, cc) in enumerate(rows):
            col = INK if j else MUTED
            g.text((880, 620 + j * 19), f"{a:<16}{b:>8}{cc:>9}", font=f_small, fill=col)
        frames.append(np.asarray(img))
    return frames


def make_package(tracks, fps, R_um, signal_dir, field_um, name, title, seconds=45.0):
    """Everything for one recording: rules, robot plan, simulator check, comparison, preview MP4, upload script."""
    from . import embody, io
    rules = embody.extract_rules(tracks, fps=fps, R_um=R_um, signal_dir=signal_dir, field_um=field_um)
    plan = embody.to_robotarium(rules, tracks, record=name, seconds=seconds)
    hist = embody.preview(plan)
    cmp = embody.compare(rules, tracks, plan, hist)
    mp4 = io.write_mp4(render(tracks, rules, plan, hist, cmp, title=title), 15)
    report = {"rules_cells": rules, "plan_robots": plan, "comparison": cmp,
              "simulator": {k: hist[k] for k in ("errors", "verdict", "flips")}}
    return {"rules": rules, "plan": plan, "cmp": cmp, "verdict": hist["verdict"], "mp4": mp4,
            "script": embody.submission_source(plan),
            "report_json": json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))}


def record_inputs(rec, tracks):
    """Field size, rates and signal for a repository record (as stored by seek.archive)."""
    from .embody import SIGNAL_ANGLE
    s = rec["settings"]
    upp = s.get("um_per_px") or rec["um_per_px"]
    w = rec["analyzed_width_px"] or rec["width_px"]
    h = round(w * rec["height_px"] / rec["width_px"])
    sig = rec["signal_dir"] if rec["signal_dir"] in SIGNAL_ANGLE else None
    return {"tracks": tracks, "fps": rec["fps"], "R_um": s.get("R_um", 10.0), "signal_dir": sig,
            "field_um": (w * upp, h * upp)}
