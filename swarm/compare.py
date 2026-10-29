"""
swarm.compare — is the Seek data accurately depicted in the swarm?

Runs the robots with a record's rules, then measures the robots EXACTLY the way the bridge measures the
cells, and puts the two side by side: paths, speed, turning, spacing, alignment-vs-distance, group alignment.

To make the two comparable:
  * robots are put on cell time (robot seconds × time_scale), so speeds read in the cells' body lengths/s;
  * robot paths are sampled at the video's frame rate and smoothed like Seek's tracks;
  * the arena gets as many robots as gives the same crowding the cells had (3–22 robots).

    python -m swarm.compare SK-2610-001                     # saves the check with the record
    python -m swarm.compare --tracks t.csv --cell-um 2.5 --profile p.json --out check.png

A mismatch is not a bug: the rules are a simple model (Couzin zones) and attraction is assumed.
The check shows what the model captures and what it misses.
"""
import argparse
import io as _io
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from swarm import bridge, sim

RULE_KEYS = ("zone_repulsion_bl", "zone_orientation_bl", "zone_attraction_bl", "speed_bl_per_s", "max_turn_rate_deg_per_s", "heading_noise_deg")
CELL, ROBOT, INK, MUTED = "#534AB7", "#0F6E56", "#2C2C2A", "#6B6A64"
VERDICT_COLOR = {"close": "#2f6e3a", "partial": "#9a6a12", "off": "#9b2c22"}


# ──────────────────────────────────────────────────────────────────────────
# 1. Simulate the robots and express them as Seek tracks on cell time
# ──────────────────────────────────────────────────────────────────────────
def simulate_robots(profile: dict, n: int, seconds: float, seed: int = 7, warmup_s: float = 15.0):
    """Robot poses (T, 3, n) every simulator tick after a warm-up. Uses the Robotarium (with its collision
    filter) when installed, otherwise the same rules with plain unicycle motion."""
    p = sim.profile_to_params(profile)
    rng = np.random.default_rng(seed)
    x0 = sim.initial_poses(n, rng)
    steps, warm = int(seconds / sim.DT), int(warmup_s / sim.DT)
    try:
        import rps.robotarium as robotarium
        from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
        r = robotarium.Robotarium(number_of_robots=n, show_figure=False, sim_in_real_time=False,
                                  initial_conditions=x0, skip_initialization=True)
        safety = create_uni_barrier_certificate_with_boundary(safety_radius=0.12)
        engine = "Robotarium simulator"
    except ImportError:
        r, safety, engine = None, None, "rules only (Robotarium not installed)"
        x = x0.copy()
    out, wander = [], {}
    for k in range(warm + steps):
        if r is not None:
            x = r.get_poses()
        pos, head = x[:2], x[2]
        dxu = sim.decide(pos, head, p, rng, state=wander)
        if r is not None:
            r.set_velocities(np.arange(n), safety(dxu, x))
            r.step()
        else:
            x = x.copy()
            x[0] += sim.DT * dxu[0] * np.cos(x[2]); x[1] += sim.DT * dxu[0] * np.sin(x[2])
            x[2] = sim.wrap(x[2] + sim.DT * dxu[1])
            x[0] = np.clip(x[0], *sim.ARENA[:2]); x[1] = np.clip(x[1], *sim.ARENA[2:])
        if k >= warm:
            out.append(x.copy())
    return np.array(out), engine


def robots_as_tracks(poses, cell_um, time_scale, fps):
    """Robot poses → Seek-style tracks: cell time, sampled at the video's fps, positions in µm (y down)."""
    dt_cell_tick = sim.DT * time_scale
    k = max(1, round((1.0 / fps) / dt_cell_tick))
    sub = poses[::k]
    rows = []
    for f, x in enumerate(sub):
        for i in range(x.shape[1]):
            rows.append((f, f * k * dt_cell_tick, i, x[0, i] / sim.BODY_LENGTH_M * cell_um,
                         -x[1, i] / sim.BODY_LENGTH_M * cell_um))
    df = pd.DataFrame(rows, columns=["frame", "time_s", "cell_id", "x_um", "y_um"])
    for c in ("x_um", "y_um"):           # 3-frame smoothing, like Seek's tracks
        df[c] = df.groupby("cell_id")[c].transform(lambda s: s.rolling(3, center=True).mean()).fillna(df[c])
    return df


# ──────────────────────────────────────────────────────────────────────────
# 2. Measure both the same way
# ──────────────────────────────────────────────────────────────────────────
def measures(df, cell_um):
    """Distributions in body lengths. Headings and speeds come from positions for BOTH datasets."""
    d = bridge.prepare(df.drop(columns=[c for c in ("heading_deg", "speed_um_s") if c in df]), cell_um)
    mv = d[(d["v"] > 0.2) & d["ds"].gt(0)]
    nn, pol = [], []
    for _, f in d.groupby("frame"):
        if len(f) >= 2:
            p = f[["x", "y"]].to_numpy()
            nn.extend(cKDTree(p).query(p, k=2)[0][:, 1])
    for _, f in mv.groupby("frame"):
        if len(f) >= 3:
            pol.append(np.hypot(np.cos(f["th"]).mean(), np.sin(f["th"]).mean()))
    centers, C, counts = bridge.alignment_vs_distance(d, max_bl=12.0, bin_bl=1.0)
    def finite(a):
        a = np.asarray(a, float)
        return a[np.isfinite(a)]
    return {"d": d, "speed": finite(mv["v"]), "turn": finite(np.degrees(np.abs(mv["dth"]) / mv["ds"])),
            "nn": finite(nn), "pol": finite(pol), "align": (centers, C, counts)}


def overlap(a, b, bins=30):
    """Share of two distributions that coincide (0 = no overlap, 1 = identical)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    lo, hi = np.percentile(np.r_[a, b], [1, 99])
    if hi <= lo:
        return 1.0
    pa, _ = np.histogram(np.clip(a, lo, hi), bins, (lo, hi))
    pb, _ = np.histogram(np.clip(b, lo, hi), bins, (lo, hi))
    return float(np.minimum(pa / pa.sum(), pb / pb.sum()).sum())


def verdict_overlap(x):
    return "off" if not np.isfinite(x) or x < 0.4 else ("partial" if x < 0.7 else "close")


def range_verdict(cell, robot_vals, rel_tol=0.10, abs_tol=0.0, partial=0.35, absolute=False):
    """Is the cells' value inside the range the robots produce across runs (chance included)?
    close = inside the range (± a small tolerance); partial = near it; off = far outside."""
    vals = [v for v in robot_vals if v is not None and np.isfinite(v)]
    if cell is None or not np.isfinite(cell) or not vals:
        return "off", None
    lo, hi = min(vals), max(vals)
    scale = 1.0 if absolute else max(abs(np.median(vals)), 1e-9)
    tol = abs_tol if absolute else rel_tol * scale
    if lo - tol <= cell <= hi + tol:
        return "close", 0.0
    miss = (lo - tol - cell if cell < lo else cell - hi - tol) / scale
    return ("partial" if miss <= partial else "off"), round(float(miss), 2)


def summarize_runs(cells, runs):
    """cells: measures of the cells. runs: measures of several robot runs (different chance)."""
    out = {}
    names = {"speed": ("Speed", "BL/s"), "turn": ("Turning", "° per BL travelled"), "nn": ("Spacing", "BL")}
    for k, (label, unit) in names.items():
        cm = float(np.median(cells[k])) if len(cells[k]) else None
        rv = [float(np.median(r[k])) for r in runs if len(r[k])]
        verdict, miss = range_verdict(cm, rv)
        out[k] = {"label": label, "unit": unit, "verdict": verdict,
                  "cells": None if cm is None else round(cm, 2),
                  "robots_range": [round(min(rv), 2), round(max(rv), 2)] if rv else None,
                  "miss": miss, "miss_meaning": "how far outside the robots' range, as a share of their typical value"}
    cp = float(np.mean(cells["pol"])) if len(cells["pol"]) else None
    rp = [float(np.mean(r["pol"])) for r in runs if len(r["pol"])]
    verdict, miss = range_verdict(cp, rp, abs_tol=0.05, partial=0.2, absolute=True)
    out["pol"] = {"label": "Group alignment", "unit": "polarization 0–1", "verdict": verdict,
                  "cells": None if cp is None else round(cp, 2),
                  "robots_range": [round(min(rp), 2), round(max(rp), 2)] if rp else None,
                  "miss": miss, "miss_meaning": "distance outside the robots' range (polarization units)"}
    c, Cc, nc = cells["align"]
    curves = [r["align"][1] for r in runs]
    counts = np.min([r["align"][2] for r in runs], axis=0)
    ok = (nc >= 20) & (counts >= 20)
    if ok.any():
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            env_lo, env_hi = np.nanmin(curves, axis=0) - 0.1, np.nanmax(curves, axis=0) + 0.1
        inside = float(np.mean((Cc[ok] >= env_lo[ok]) & (Cc[ok] <= env_hi[ok])))
    else:
        inside = np.nan
    out["align"] = {"label": "Alignment vs distance", "unit": "mean cos Δheading",
                    "verdict": "off" if not np.isfinite(inside) or inside < 0.5 else ("close" if inside >= 0.8 else "partial"),
                    "inside_share": None if not np.isfinite(inside) else round(inside, 2),
                    "inside_meaning": "share of distances where the cells' curve sits inside the robots' band"}
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3. One picture
# ──────────────────────────────────────────────────────────────────────────
def figure(cells, runs, metrics, title):
    robots = {k: np.concatenate([r[k] for r in runs]) for k in ("speed", "turn", "nn", "pol")}
    robots["d"] = runs[0]["d"]
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 9.6))
    fig.text(0.03, 0.965, title, fontsize=15, weight="bold", color=INK)
    fig.text(0.03, 0.935, f"Purple = cells measured by Seek · green = {len(runs)} robot runs with the fitted rules "
             "(chance differs per run), measured the same way on the cells' clock · 1 BL = one body length · "
             "CLOSE = cells inside the robots' range", fontsize=10.5, color=MUTED)

    def window(d):
        last = d["time_s"].max()
        return d[d["time_s"] >= last - 1.0]                  # last second of cell time

    wc, wr = window(cells["d"]), window(robots["d"])
    span = max(np.ptp(wc["x"]), np.ptp(wc["y"]), np.ptp(wr["x"]), np.ptp(wr["y"])) * 1.08 + 2

    def paths(ax, w, color, label):
        for _, t in w.groupby("cell_id"):
            ax.plot(t["x"], t["y"], color=color, lw=1.0, alpha=0.75)
        end = w.sort_values("time_s").groupby("cell_id").tail(1)
        ax.scatter(end["x"], end["y"], s=14, color=color, zorder=3)
        cx, cy = (w["x"].min() + w["x"].max()) / 2, (w["y"].min() + w["y"].max()) / 2
        ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2 * 0.62, cy + span / 2 * 0.62)
        ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#D3D1C7")
        x0, _ = ax.get_xlim(); y0, _ = ax.get_ylim()
        ax.plot([x0 + 1, x0 + 6], [y0 + 1, y0 + 1], color=INK, lw=2)
        ax.text(x0 + 3.5, y0 + 1.5, "5 BL", ha="center", va="bottom", fontsize=9, color=INK,
                bbox=dict(fc="white", ec="none", pad=1))
        ax.set_title(label, loc="left", fontsize=12, color=color, weight="bold")

    paths(fig.add_axes([0.03, 0.47, 0.45, 0.42]), wc, CELL, "Cells (Seek) · last 1 s of paths")
    paths(fig.add_axes([0.52, 0.47, 0.45, 0.42]), wr, ROBOT, "Robots (same rules, run 1) · last 1 s, cell time")
    fig.text(0.5, 0.445, "Both panels share one scale", ha="center", fontsize=9, color=MUTED)

    panels = [("speed", "hist"), ("turn", "hist"), ("nn", "hist"), ("align", "curve"), ("pol", "hist")]
    for k, (key, kind) in enumerate(panels):
        ax = fig.add_axes([0.03 + k * 0.194, 0.08, 0.165, 0.27])
        r = metrics[key]
        if kind == "hist":
            both = np.r_[cells[key], robots[key]]
            both = both[np.isfinite(both)]
            hi = np.percentile(both, 98) if len(both) else 1
            lo = 0 if key != "pol" else 0
            bins = np.linspace(lo, max(hi, 1e-6) if key != "pol" else 1, 25)
            ax.hist(np.clip(cells[key], lo, bins[-1]), bins, density=True, color=CELL, alpha=0.45, label="cells")
            ax.hist(np.clip(robots[key], lo, bins[-1]), bins, density=True, color=ROBOT, alpha=0.45, label="robots")
            ax.set_yticks([])
        else:
            c, Cc, nc = cells["align"]
            for run in runs:
                _, Cr, nr = run["align"]
                ax.plot(c[nr >= 20], Cr[nr >= 20], color=ROBOT, lw=1.2, alpha=0.6)
            ax.plot(c[nc >= 20], Cc[nc >= 20], color=CELL, lw=2.4, label="cells")
            ax.set_ylim(-0.3, 1.05); ax.axhline(0, color="#D3D1C7", lw=0.8)
            ax.set_xlabel("distance between pair (BL)", fontsize=8.5, color=MUTED)
        for s in ("top", "right", "left") if kind == "hist" else ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=MUTED)
        if kind == "hist":
            ax.set_xlabel(r["unit"], fontsize=8.5, color=MUTED)
        ax.set_title(r["label"], loc="left", fontsize=11, color=INK, weight="bold", pad=22)
        if r.get("robots_range"):
            lo, hi = r["robots_range"]
            score = f"cells {r['cells']:g} · robots {lo:g}–{hi:g}" if r.get("cells") is not None else "no data"
        else:
            score = "inside band " + ("—" if r.get("inside_share") is None else f"{r['inside_share']:.0%}")
        ax.text(0, 1.025, f"{r['verdict'].upper()}  ·  {score}", transform=ax.transAxes, fontsize=9.2,
                color=VERDICT_COLOR[r["verdict"]], weight="bold", va="bottom")
        if k == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────────────
ARENA_AREA_BL2 = float(np.ptp(sim.ARENA[:2]) * np.ptp(sim.ARENA[2:]) / sim.BODY_LENGTH_M ** 2)


def robots_for_density(tracks, cell_um):
    """As many robots as gives the arena the same crowding the cells had (cells per body length²),
    using each frame's occupied area. The Robotarium allows 3–22 robots."""
    areas, counts = [], []
    for _, f in tracks.groupby("frame"):
        if len(f) >= 3:
            areas.append(np.ptp(f["x_um"] / cell_um) * np.ptp(f["y_um"] / cell_um))
            counts.append(len(f))
    if not areas or np.median(areas) <= 0:
        return 12, None
    density = float(np.median(counts) / np.median(areas))
    return int(np.clip(round(density * ARENA_AREA_BL2), 3, sim.MAX_ROBOTS)), density


def check(tracks: pd.DataFrame, profile: dict, title: str, runs: int = 4, seed: int = 7, progress=None):
    """Full fidelity check: robots run `runs` times with different chance. Returns (result dict, PNG bytes)."""
    f = profile["fit"]
    cell_um, ts = f["cell_length_um"], f["time_scale"]
    dt = tracks.sort_values(["cell_id", "frame"]).groupby("cell_id")["time_s"].diff().median()
    fps = 1.0 / dt if dt and dt > 0 else 20.0
    n, density = robots_for_density(tracks, cell_um)
    cell_seconds = tracks["time_s"].max() - tracks["time_s"].min()
    robot_seconds = float(np.clip(cell_seconds / ts, 20, 45))
    cells = measures(tracks, cell_um)
    robot_runs, engine = [], None
    for k in range(runs):
        if progress:
            progress(k, runs)
        poses, engine = simulate_robots(profile, n, robot_seconds, seed + k, warmup_s=10)
        robot_runs.append(measures(robots_as_tracks(poses, cell_um, ts, fps), cell_um))
    metrics = summarize_runs(cells, robot_runs)
    n_close = sum(m["verdict"] == "close" for m in metrics.values())
    result = {"source": profile.get("source"), "rules": {k: profile.get(k) for k in RULE_KEYS},
              "robots": n, "runs": runs,
              "cells_per_bl2": None if density is None else round(density, 4),
              "cells_per_frame": profile["targets"].get("cells_per_frame"),
              "robot_seconds_per_run": round(robot_seconds, 1), "engine": engine, "metrics": metrics,
              "summary": f"{n_close} of {len(metrics)} measures match",
              "notes": ["CLOSE means the cells' value falls inside the range the robots produce across runs.",
                        "The robot count is chosen so the arena is as crowded as the cells were "
                        "(between 3 and 22 robots, the Robotarium's limits).",
                        "Robots move inside a walled 29 × 18 BL arena; cells were filmed in an open field of view.",
                        "Attraction range is assumed, so spacing and group alignment can drift from the cells.",
                        "Rules are a simple model (Couzin zones); a mismatch shows what the model misses."]}
    png = figure(cells, robot_runs, metrics, f"Fidelity check · {title} · {result['summary']}")
    return result, png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", nargs="?")
    ap.add_argument("--tracks"); ap.add_argument("--cell-um", type=float); ap.add_argument("--profile")
    ap.add_argument("--out", default="fidelity_check.png")
    a = ap.parse_args()
    if a.record:
        from seek.archive import Archive
        arc = Archive(); me = arc.owner()
        rec = arc.get_record(a.record, me)
        prof = arc.swarm_profile(a.record) or bridge.profile_for_record(arc, rec)
        result, png = check(arc.tracks(rec), prof, a.record)
        arc.attach_swarm_check(a.record, me["id"], result, png)
        print(f"Saved with the record {a.record}")
    else:
        df = pd.read_parquet(a.tracks) if a.tracks.endswith(".parquet") else pd.read_csv(a.tracks)
        prof = json.load(open(a.profile)) if a.profile else bridge.fit(df, a.cell_um, {"source": a.tracks})
        result, png = check(df, prof, a.tracks)
    open(a.out, "wb").write(png)
    print(result["summary"])
    for m in result["metrics"].values():
        detail = (f"cells {m['cells']} · robots {m['robots_range']}" if m.get("robots_range")
                  else f"cells inside robots' band {m.get('inside_share')}")
        print(f"  {m['label']:24} {m['verdict']:8} {detail}")
    print(f"Figure: {a.out}")


if __name__ == "__main__":
    main()
