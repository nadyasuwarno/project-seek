"""Embody: turn a Seek recording into rules for a robot swarm on the Robotarium.

    rules = extract_rules(tracks, fps=…, R_um=…, signal_dir=…, field_um=(W, H))   # what the cells do
    plan  = to_robotarium(rules, tracks, field_um)                                # same rules, robot units
    hist  = preview(plan)                                                         # Robotarium simulator
    cmp   = compare(rules, tracks, plan, hist)                                    # did the swarm keep the pattern?
    write_submission(plan, "robotarium_SK-2610-001.py")                          # the file you upload

The rule model (per cell, every moment):  turn rate = k_pref·sin(θ*−θ) + k_align·sin(φ_neighbors−θ) + noise,
cruise at speed v. The coefficients are fitted from how the cells actually turned (least squares), so the
robots follow rules learned from the video, not a hand-tuned flocking demo.

Mapping: the Robotarium arena stands in for the microscope field (shapes and distances keep their proportions),
and time is slowed by a dilation factor D so the robots cruise at ~0.10 m/s. Every rate (1/s) is divided by D;
heading noise by √D. Anything clamped for the hardware is listed in plan["notes"].
"""
import json
import pprint
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import robotarium_swarm as swarm

SIGNAL_ANGLE = {"right": 0.0, "up": np.pi / 2, "left": np.pi, "down": -np.pi / 2}   # math convention
ROBOT_SPEED = 0.10        # m/s cruise (Robotarium max 0.2; headroom for turning)
MAX_ROBOTS = 20           # Robotarium hardware limit


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _math_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Seek stores image axes (y down). Rules are in math axes (y up)."""
    d = df[["frame", "time_s", "cell_id", "x_um", "y_um", "vx_um_s", "vy_um_s"]].copy()
    d["X"], d["Y"] = d["x_um"], -d["y_um"]
    d["VX"], d["VY"] = d["vx_um_s"], -d["vy_um_s"]
    d["speed"] = np.hypot(d["VX"], d["VY"])
    d["theta"] = np.arctan2(d["VY"], d["VX"])
    d = d[np.isfinite(d[["X", "Y", "VX", "VY"]]).all(axis=1)]           # 1-frame tracks have no velocity
    return d.sort_values(["frame", "cell_id"]).reset_index(drop=True)


def _a1inv(r):
    """Inverse of the mean resultant length → von Mises concentration κ (Fisher's approximation)."""
    r = float(np.clip(r, 0, 0.9999))
    if r < 0.53:
        return 2 * r + r ** 3 + 5 * r ** 5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1 / (r ** 3 - 4 * r ** 2 + 3 * r)


def _neighbor_heading(d: pd.DataFrame, R: float) -> np.ndarray:
    """Mean heading of other cells within R, per row (NaN when alone)."""
    out = np.full(len(d), np.nan)
    pts, th = d[["X", "Y"]].to_numpy(), d["theta"].to_numpy()
    for _, idx in d.groupby("frame").indices.items():
        if len(idx) < 2:
            continue
        for i, nb in zip(idx, cKDTree(pts[idx]).query_ball_point(pts[idx], R)):
            nb = [idx[j] for j in nb if idx[j] != i]
            if nb:
                out[i] = np.arctan2(np.sin(th[nb]).sum(), np.cos(th[nb]).sum())
    return out


# ── 1. what the cells do ───────────────────────────────────────────────────────────────────────────
def extract_rules(tracks: pd.DataFrame, fps: float, R_um: float, signal_dir: str | None,
                  field_um: tuple[float, float]) -> dict:
    d = _math_frame(tracks)
    d["phi_nb"] = _neighbor_heading(d, R_um)
    lag = max(1, int(round(0.25 * fps)))                       # compare headings 0.25 s apart (beats jitter)
    tau = lag / fps
    nxt = d[["cell_id", "frame", "theta"]].copy()
    nxt["frame"] -= lag
    s = d.merge(nxt, on=["cell_id", "frame"], suffixes=("", "_next"))
    s = s[(s["speed"] > 0.25 * d["speed"].median()) & np.isfinite(s["theta"]) & np.isfinite(s["theta_next"])]   # heading is meaningless when nearly still

    per_track = d.groupby("cell_id").agg(cx=("VX", "mean"), cy=("VY", "mean"), v=("speed", "mean"))
    unit = per_track[["cx", "cy"]].to_numpy() / np.maximum(np.hypot(per_track.cx, per_track.cy), 1e-9).to_numpy()[:, None]
    mean_dir = unit.mean(0)
    rbar = float(np.hypot(*mean_dir))                           # 0 = all directions, 1 = all the same way
    if signal_dir in SIGNAL_ANGLE:
        pref, pref_src = SIGNAL_ANGLE[signal_dir], f"signal ({signal_dir})"
        rbar = max(0.0, float(np.mean(unit @ np.array([np.cos(pref), np.sin(pref)]))))   # order toward the signal
    else:
        pref, pref_src = float(np.arctan2(mean_dir[1], mean_dir[0])), "shared drift"

    y = wrap(s["theta_next"] - s["theta"]).to_numpy() / tau     # turn rate, rad/s
    th = s["theta"].to_numpy()
    x_pref = np.sin(pref - th)
    x_align = np.nan_to_num(np.sin(s["phi_nb"].to_numpy() - th))
    def fit(X):
        if len(y) <= 10:
            return np.zeros(X.shape[1]), np.zeros(X.shape[1]), 0.0
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ c
        cov = np.linalg.pinv(X.T @ X) * (res @ res / max(len(y) - X.shape[1], 1))
        return c, c / np.sqrt(np.maximum(np.diag(cov), 1e-12)), float(np.sqrt(np.var(res) * tau))  # Var(Δθ)=σ²τ

    coef, tstat, sigma = fit(np.c_[x_pref, x_align])
    reason = None
    if signal_dir not in SIGNAL_ANGLE:
        # Without a signal the direction is taken from the data itself, so chance drift in a short clip can
        # look like a pull. Only keep a pull when headings are clearly shared and alignment doesn't explain it.
        if rbar < 0.6:
            reason = "no clear shared direction"
        elif tstat[1] > 3:
            reason = "shared direction emerges from alignment (no external pull)"
        if reason:
            c1, t1, sigma = fit(x_align[:, None])
            coef, tstat = np.r_[0.0, c1], np.r_[0.0, t1]

    def relax(k):
        """A heading gap closes like e^(−kτ) within the lag, so the per-lag fit underestimates k. Undo that."""
        return float(-np.log(1 - min(k * tau, 0.95)) / tau) if k > 0 else 0.0

    kappa = min(_a1inv(rbar), 50.0)
    k_pref_eq = kappa * sigma ** 2 / 2                          # stationary von Mises: κ = 2k/σ²
    if reason:
        k_pref, k_pref_src = 0.0, reason
    elif coef[0] > 0 and tstat[0] > 3:
        k_pref, k_pref_src = relax(coef[0]), "fitted from turning"
    elif rbar > 0.25 and len(unit) * rbar ** 2 > 7:          # Rayleigh test, p < 0.001: not just chance
        k_pref, k_pref_src = float(k_pref_eq), "from how concentrated headings are"
    else:
        k_pref, k_pref_src = 0.0, "no shared direction"
    k_align = relax(coef[1]) if coef[1] > 0 and tstat[1] > 3 else 0.0

    per_frame = d.groupby("frame")
    nn = []
    for _, g in per_frame:
        if len(g) > 1:
            nn += list(cKDTree(g[["X", "Y"]]).query(g[["X", "Y"]], k=2)[0][:, 1])
    W, H = field_um
    return {
        "speed_um_s": float(per_track["v"].mean()),
        "speed_cv": float(per_track["v"].std(ddof=0) / max(per_track["v"].mean(), 1e-9)),
        "sigma": sigma, "k_pref": k_pref, "k_pref_source": k_pref_src, "pref": pref, "pref_source": pref_src,
        "k_align": k_align, "R_um": float(R_um), "nn_um": float(np.median(nn)) if nn else float(min(W, H) / 4),
        "cells_per_frame": float(per_frame.size().mean()), "field_um": [float(W), float(H)],
        "fit": {"samples": int(len(y)), "lag_s": tau, "coef_pref": float(coef[0]), "t_pref": float(tstat[0]),
                "coef_align": float(coef[1]), "t_align": float(tstat[1]), "heading_order": rbar,
                "kappa": kappa},
    }


# ── 2. same rules in robot units ───────────────────────────────────────────────────────────────────
def to_robotarium(rules: dict, tracks: pd.DataFrame, record: str = "record", seconds: float = 60.0,
                  seed: int = 0) -> dict:
    W, H = rules["field_um"]
    s = min(2.8 / W, 1.7 / H)                                   # metres of arena per µm of field
    D = max(rules["speed_um_s"] * s / ROBOT_SPEED, 1e-6)        # robot seconds per microbe second
    notes = []

    def clamp(name, val, lo, hi, why):
        c = float(np.clip(val, lo, hi))
        if abs(c - val) > 1e-9:
            notes.append(f"{name}: {val:.3g} → {c:.3g} ({why})")
        return c

    n_cells = rules["cells_per_frame"]
    N = int(np.clip(round(n_cells), 4, MAX_ROBOTS))
    if N != round(n_cells):
        notes.append(f"robots: {n_cells:.0f} cells per frame → {N} robots (Robotarium allows 1–{MAX_ROBOTS})")
    k_pref = rules["k_pref"] / D
    if k_pref > 0:
        k_pref = clamp("k_pref", k_pref, 0.6, 2.0, "so the swarm can re-orient within seconds after a sweep")
    plan = {
        "record": record, "N": N, "v": ROBOT_SPEED,
        "v_spread": clamp("speed spread", rules["speed_cv"], 0.0, 0.35, "robots can't vary speed that much"),
        "sigma": clamp("heading noise", rules["sigma"] / np.sqrt(D), 0.05, 0.8, "motor-safe range"),
        "tau_noise": 0.6, "k_pref": k_pref, "pref": rules["pref"], "sweep": True,
        "k_align": clamp("k_align", rules["k_align"] / D, 0.0, 2.0, "motor-safe range"),
        "d_sep": clamp("spacing", rules["nn_um"] * s * 0.7, 0.35, 0.50,
                       "robots keep their distance early so the safety barrier rarely has to step in"),
        "k_sep": 4.0, "seconds": float(seconds), "seed": int(seed),
    }
    # neighbours must be seen beyond the spacing robots keep, or alignment has no one to align with
    plan["R"] = clamp("neighbor radius", rules["R_um"] * s, max(0.30, 1.7 * plan["d_sep"]), 0.90,
                      "must reach past the robots' spacing")
    plan["initial"] = _initial_poses(tracks, N, s, seed)
    plan["_scale"] = {"m_per_um": s, "time_dilation": D, "field_m": [W * s, H * s]}
    plan["_notes"] = notes
    return plan


def _initial_poses(tracks, N, s, seed):
    """Start the robots where the cells were (busiest frame), mapped into the arena and spaced ≥ 0.35 m."""
    d = _math_frame(tracks)
    f = d.groupby("frame").size().idxmax()
    g = d[d["frame"] == f]
    cx, cy = (d["X"].min() + d["X"].max()) / 2, (d["Y"].min() + d["Y"].max()) / 2
    cand = np.c_[(g["X"] - cx) * s, (g["Y"] - cy) * s, g["theta"]]
    rng = np.random.default_rng(seed)
    cand = cand[rng.permutation(len(cand))]
    chosen = []
    for c in cand:
        c = c.copy()
        c[0], c[1] = np.clip(c[0], -1.35, 1.35), np.clip(c[1], -0.75, 0.75)
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= 0.35 for q in chosen):
            chosen.append(c)
        if len(chosen) == N:
            break
    while len(chosen) < N:                                        # fewer well-spaced cells than robots
        c = np.r_[rng.uniform(-1.35, 1.35), rng.uniform(-0.75, 0.75), rng.uniform(-np.pi, np.pi)]
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) >= 0.35 for q in chosen):
            chosen.append(c)
    return np.round(np.array(chosen).T, 3).tolist()


# ── 3. run it on the Robotarium simulator ──────────────────────────────────────────────────────────
def preview(plan: dict) -> dict:
    rules = {k: v for k, v in plan.items() if not k.startswith("_")}
    poses, prefs = [], []

    def rec(t, x, st):
        poses.append(x.copy())
        prefs.append(st["pref"])

    r, st = swarm.run(rules, show_figure=False, sim_in_real_time=False, skip_initialization=True, record=rec)
    errors = dict(r._errors)
    return {"poses": np.array(poses), "pref": np.array(prefs), "dt": swarm.DT, "flips": st["flips"],
            "errors": errors,
            "verdict": "would run on the Robotarium" if not errors.get("robots_too_close")
            and not errors.get("robots_outside_boundaries") else "would be rejected (collisions/boundary)"}


# ── 4. does the swarm keep the pattern? ────────────────────────────────────────────────────────────
def _pattern(frames, R):
    """frames: list of (pos N×2, vel N×2, pref_angle). Same measures for cells and robots."""
    pol, ali, nn, drift = [], [], [], []
    for p, v, pref in frames:
        sp = np.linalg.norm(v, axis=1)
        ok = sp > 1e-9
        if ok.sum() < 2:
            continue
        u = v[ok] / sp[ok, None]
        pol.append(np.hypot(*u.mean(0)))
        tree = cKDTree(p[ok])
        nn += list(tree.query(p[ok], k=2)[0][:, 1])
        for i, nb in enumerate(tree.query_ball_point(p[ok], R)):
            nb = [j for j in nb if j != i]
            if nb:
                ali.append(float(np.mean(u[nb] @ u[i])))
        if pref is not None:
            drift.append(float(np.mean(u @ np.array([np.cos(pref), np.sin(pref)]))))
    m = lambda a: float(np.mean(a)) if len(a) else None   # noqa: E731
    return {"group_order": m(pol), "local_alignment": m(ali), "nn": m(nn), "heading_toward_pref": m(drift)}


def compare(rules, tracks, plan, hist) -> dict:
    d = _math_frame(tracks)
    pref = rules["pref"] if rules["k_pref"] > 0 else None
    cells = _pattern([(g[["X", "Y"]].to_numpy(), g[["VX", "VY"]].to_numpy(), pref) for _, g in d.groupby("frame")],
                     rules["R_um"])
    P, dt = hist["poses"], hist["dt"]
    k = max(1, int(round(0.25 / dt)))                           # velocity from poses 0.25 s apart
    frames = []
    for t in range(0, len(P) - k, 3):
        v = (P[t + k, :2] - P[t, :2]).T / (k * dt)
        frames.append((P[t, :2].T, v, hist["pref"][t] if plan["k_pref"] > 0 else None))
    robots = _pattern(frames, plan["R"])
    s = plan["_scale"]["m_per_um"]
    cells["nn_field"] = cells["nn"] / rules["field_um"][0] if cells["nn"] else None
    robots["nn_field"] = robots["nn"] / (rules["field_um"][0] * s) if robots["nn"] else None
    return {"cells": cells, "robots": robots}


# ── 5. the file you upload ─────────────────────────────────────────────────────────────────────────
def submission_source(plan: dict) -> str:
    src = Path(swarm.__file__).read_text()
    rules = {k: v for k, v in plan.items() if not k.startswith("_")}
    rules = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rules.items()}
    block = "RULES = " + pprint.pformat(rules, width=100, sort_dicts=False) + "\n"
    sc = plan["_scale"]
    head = (f'# Seek → Robotarium · record {plan["record"]}\n'
            f'# Rules extracted by Seek from the recording. 1 m of arena = {1 / sc["m_per_um"]:.0f} µm of '
            f'microscope field; 1 s of cell time = {sc["time_dilation"]:.1f} s of robot time.\n')
    for n in plan["_notes"]:
        head += f"# adjusted: {n}\n"
    out = re.sub(r"RULES = \{.*?\n\}\n", lambda _: block, src, count=1, flags=re.S)
    out = out.replace("python robotarium_swarm.py", f"python robotarium_{plan['record']}.py")
    return head + out


def write_submission(plan: dict, path) -> Path:
    path = Path(path)
    path.write_text(submission_source(plan))
    return path


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
