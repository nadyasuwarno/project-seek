"""
swarm.bridge — turn a Seek record into a robot behavior profile.

Seek (Perceive) measures cells: speed, heading, nearest neighbour, alignment, density.
The bridge turns that into swarm rules in BODY LENGTHS that the robots (Embody) can run, plus
"targets" — what the cells' group looked like — so the live tuner can show whether robots reproduce it.

From Seek's record page: the "Send to swarm" button calls `profile_for_record`.
From a terminal (inside project-seek):
    python -m swarm.bridge --list
    python -m swarm.bridge SK-2610-001                       # saves it with the record
    python -m swarm.bridge --tracks tracks.csv --cell-um 2.5 --out my_profile.json

Honest limits (also written into each profile's "fit" section)
  * Speed, turning and spacing are measured directly.
  * Orientation range comes from how heading-similarity fades with distance (approximate).
  * Attraction range is NOT measurable from one video: set to a default, flagged "assumed".
  * Cells cover many body lengths per second; robots can't. The bridge keeps the SHAPE of the
    paths (turning per body length travelled) and slows time down (fit.time_scale).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROBOT_MAX_BL_S = 1.2          # comfortable top speed for an 11 cm robot (0.13 m/s)
ROBOT_MIN_REPULSION_BL = 1.3  # robots physically can't sit closer (safety filter radius)
MAX_TURN_CAP_DEG_S = 180
DEFAULT_EXPERIMENT = {"num_robots": 12, "duration_s": 80, "seed": 7,
                      "override": {"enabled": False, "start_s": 30, "end_s": 55,
                                   "target_m": "farthest_corner"}}


def cell_length_um(settings: dict) -> float | None:
    """Seek's detection diameter (px) × µm per pixel = one body length."""
    if settings.get("diameter") and settings.get("um_per_px"):
        return float(settings["diameter"]) * float(settings["um_per_px"])
    return None


def profile_for_record(arc, rec: dict, cell_um: float | None = None) -> dict:
    """Fit a profile from a saved Seek record (as returned by Archive.get_record)."""
    how = "given by the researcher" if cell_um else "Seek settings: diameter × µm/px"
    cell_um = cell_um or cell_length_um(rec.get("settings") or {})
    if not cell_um:
        raise ValueError("This record has no cell diameter in its settings — pass the cell length in µm.")
    meta = {"source": f"Seek record {rec['record_code']}", "record_code": rec["record_code"],
            "organism": rec.get("organism"), "cell_um_from": how}
    prof = fit(arc.tracks(rec), cell_um, meta)
    prof["experiment"] = DEFAULT_EXPERIMENT
    return prof


def prepare(df, cell_um):
    """Seek columns → y-up coordinates in body lengths, per-step turning."""
    d = df.sort_values(["cell_id", "frame"]).copy()
    d["x"] = d["x_um"] / cell_um
    d["y"] = -d["y_um"] / cell_um                     # Seek images are y-down; flip to y-up
    g = d.groupby("cell_id")
    if "heading_deg" in d and "speed_um_s" in d:
        d["th"] = np.deg2rad(d["heading_deg"])         # Seek headings are already y-up (90° = up)
        d["v"] = d["speed_um_s"] / cell_um
    else:
        dt = g["time_s"].diff()
        d["th"] = np.arctan2(g["y"].diff(), g["x"].diff())
        d["v"] = np.hypot(g["x"].diff(), g["y"].diff()) / dt
    d["dt"] = g["time_s"].diff()
    d["dth"] = (g["th"].diff() + np.pi) % (2 * np.pi) - np.pi
    d["ds"] = d["v"] * d["dt"]                         # body lengths travelled this step
    return d


def alignment_vs_distance(d, max_bl=15.0, bin_bl=0.5, max_frames=300):
    """Mean cos(heading difference) for pairs of cells at each distance (in body lengths)."""
    frames = d["frame"].unique()
    if len(frames) > max_frames:
        frames = np.random.default_rng(0).choice(frames, max_frames, replace=False)
    sums = np.zeros(int(max_bl / bin_bl)); counts = np.zeros_like(sums)
    moving = d[d["v"] > 0.2]
    for _, f in moving[moving["frame"].isin(frames)].groupby("frame"):
        if len(f) < 3:
            continue
        p = f[["x", "y"]].to_numpy(); th = f["th"].to_numpy()
        for i, j in cKDTree(p).query_pairs(max_bl - 1e-9, output_type="ndarray"):
            b = int(np.linalg.norm(p[i] - p[j]) / bin_bl)
            sums[b] += np.cos(th[i] - th[j]); counts[b] += 1
    centers = (np.arange(len(sums)) + 0.5) * bin_bl
    with np.errstate(invalid="ignore"):
        return centers, sums / counts, counts


def personal_space(d, max_bl=10.0, bin_bl=0.25, max_frames=300):
    """Repulsion range from the pair-distance distribution g(r): how often two cells sit
    at distance r compared with chance. Personal space = where g first reaches half its peak."""
    frames = d["frame"].unique()
    if len(frames) > max_frames:
        frames = np.random.default_rng(0).choice(frames, max_frames, replace=False)
    edges = np.arange(0, max_bl + bin_bl, bin_bl)
    g = np.zeros(len(edges) - 1)
    for _, f in d[d["frame"].isin(frames)].groupby("frame"):
        n = len(f)
        if n < 3:
            continue
        p = f[["x", "y"]].to_numpy()
        area = np.ptp(p[:, 0]) * np.ptp(p[:, 1]) or 1.0
        pairs = cKDTree(p).query_pairs(max_bl - 1e-9, output_type="ndarray")
        dist = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1) if len(pairs) else np.array([])
        h, _ = np.histogram(dist, edges)
        shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        g += h / (shell * (n * (n - 1) / 2) / area)
    if not g.any():
        return 0.0, "not enough pairs — default"
    g /= max(g.max(), 1e-9)
    first = np.argmax(g >= 0.5)
    return float(edges[first]), "where cell pairs stop being rarer than chance (pair-distance distribution)"


def group_targets(d):
    """What the group looked like — the numbers the robots should reproduce."""
    pol, mill = [], []
    for _, f in d[d["v"] > 0.2].groupby("frame"):
        if len(f) < 5:
            continue
        u = np.c_[np.cos(f["th"]), np.sin(f["th"])]
        pol.append(np.linalg.norm(u.mean(0)))
        r = f[["x", "y"]].to_numpy() - f[["x", "y"]].to_numpy().mean(0)
        rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        mill.append(abs(np.mean(rn[:, 0] * u[:, 1] - rn[:, 1] * u[:, 0])))
    nn = []
    for _, f in d.groupby("frame"):
        if len(f) >= 2:
            p = f[["x", "y"]].to_numpy()
            nn.extend(cKDTree(p).query(p, k=2)[0][:, 1])
    return {"polarization": round(float(np.mean(pol)), 3) if pol else None,
            "milling": round(float(np.mean(mill)), 3) if mill else None,
            "nnd_bl": round(float(np.mean(nn)), 3) if nn else None,
            "cells_per_frame": round(float(d.groupby("frame").size().mean()), 1)}


def fit(df, cell_um, meta):
    d = prepare(df, cell_um)
    mv = d[(d["v"] > 0.2) & d["dt"].gt(0) & d["ds"].gt(0)]
    if len(mv) < 50:
        raise ValueError("Too few moving cells to fit a profile — this record needs tracks with motion.")

    # speed and time scaling
    cell_speed = float(np.median(mv["v"]))
    robot_speed = min(cell_speed, ROBOT_MAX_BL_S)
    time_scale = robot_speed / cell_speed               # <1 means robots replay slower

    # turning measured per body length travelled (path shape), then replayed at robot speed
    turn_per_bl = np.abs(mv["dth"]) / mv["ds"]            # rad per BL
    max_turn = float(np.degrees(np.nanpercentile(turn_per_bl, 95)) * robot_speed)
    var_per_bl = float(np.nanvar(mv["dth"]) / np.nanmean(mv["ds"]))
    noise_tick = float(np.degrees(np.sqrt(var_per_bl * robot_speed * 0.033)))

    # spacing → repulsion
    nn = []
    for _, f in d.groupby("frame"):
        if len(f) >= 2:
            p = f[["x", "y"]].to_numpy(); nn.extend(cKDTree(p).query(p, k=2)[0][:, 1])
    nn = np.array(nn)
    rep_cell, rep_how = personal_space(d)
    rep = max(rep_cell, ROBOT_MIN_REPULSION_BL)

    # alignment fading with distance → orientation range
    centers, C, counts = alignment_vs_distance(d)
    ok = counts >= 20
    near = ok & (centers >= rep_cell) & (centers <= rep_cell + 3)
    far = ok & (centers >= centers[ok].max() - 4) if ok.any() else ok
    c_near = float(np.nanmean(C[near])) if near.any() else np.nan
    c_far = float(np.nanmean(C[far])) if far.any() else np.nan
    targets = group_targets(d)
    if np.isnan(c_near):
        ori, ori_note = rep + 2, "not enough close pairs — default"
    elif c_near - c_far >= 0.1:
        half = c_far + 0.5 * (c_near - c_far)
        beyond = centers[ok & (centers > rep_cell) & (C < half)]
        ori = float(beyond.min()) if len(beyond) else float(centers[ok].max())
        ori_note = "distance where heading similarity falls halfway to background"
    elif (targets["polarization"] or 0) > 0.8:
        ori, ori_note = 10.0, "aligned at every observed distance — set wide (10 BL)"
    else:
        ori, ori_note = rep, "no local alignment detected — orientation zone switched off"
    ori = max(ori, rep)
    att = ori + max(ori - rep, 3.0)

    profile = {
        "_about": "Behavior profile fitted by swarm.bridge from a Seek record. Distances in BODY LENGTHS, speeds in BODY "
                  "LENGTHS PER SECOND. 'targets' = what the cells' group looked like.",
        "source": meta.get("source"),
        "model": "couzin_zones",
        "zone_repulsion_bl": round(rep, 2),
        "zone_orientation_bl": round(ori, 2),
        "zone_attraction_bl": round(att, 2),
        "blind_angle_deg": 60,
        "speed_bl_per_s": round(robot_speed, 3),
        "max_turn_rate_deg_per_s": round(float(np.clip(max_turn, 20, MAX_TURN_CAP_DEG_S)), 1),
        "heading_noise_deg": round(float(np.clip(noise_tick, 0, 60)), 1),
        "targets": targets,
        "fit": {
            "organism": meta.get("organism"),
            "cell_length_um": round(cell_um, 3),
            "cell_length_from": meta.get("cell_um_from"),
            "cell_speed_bl_per_s": round(cell_speed, 3),
            "time_scale": round(time_scale, 4),
            "time_scale_meaning": f"1 s of robot motion ≈ {time_scale:.3f} s of cell motion",
            "repulsion_measured_bl": round(rep_cell, 2),
            "repulsion_note": rep_how + (" (raised to robot minimum)" if rep > rep_cell else ""),
            "orientation_note": ori_note,
            "alignment_near_vs_far": [None if np.isnan(c_near) else round(c_near, 3),
                                      None if np.isnan(c_far) else round(c_far, 3)],
            "attraction_note": "ASSUMED (not measurable from one video) — tune live against targets",
            "turning_note": "turn rate and noise measured per body length travelled, replayed at robot speed",
            "frames_used": int(d["frame"].nunique()), "tracks_used": int(d["cell_id"].nunique()),
        },
    }
    return profile


def describe(prof: dict) -> str:
    f, t = prof["fit"], prof["targets"]
    return "\n".join([
        f"Fitted from {prof['source']}  ({f['tracks_used']} tracks, {f['frames_used']} frames)",
        f"  body length       {f['cell_length_um']} µm",
        f"  cell speed        {f['cell_speed_bl_per_s']} BL/s  → robot {prof['speed_bl_per_s']} BL/s (time ×{f['time_scale']})",
        f"  repulsion         {prof['zone_repulsion_bl']} BL   ({f['repulsion_note']})",
        f"  orientation       {prof['zone_orientation_bl']} BL   ({f['orientation_note']})",
        f"  attraction        {prof['zone_attraction_bl']} BL   (assumed — tune live)",
        f"  max turn / noise  {prof['max_turn_rate_deg_per_s']} °/s / {prof['heading_noise_deg']}°",
        f"  cell targets      alignment {t['polarization']}, rotation {t['milling']}, spacing {t['nnd_bl']} BL"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", nargs="?", help="Seek record code, e.g. SK-2610-001")
    ap.add_argument("--list", action="store_true", help="list saved Seek records")
    ap.add_argument("--tracks", help="a Seek tracks.csv / .parquet instead of a record")
    ap.add_argument("--cell-um", type=float, help="cell length in µm (overrides the record's setting)")
    ap.add_argument("--out", help="also write the profile to this file")
    a = ap.parse_args()

    if a.tracks:
        if not a.cell_um:
            raise SystemExit("With --tracks, pass the cell length: --cell-um 2.5")
        p = Path(a.tracks)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        prof = fit(df, a.cell_um, {"source": f"tracks file {p.name}", "cell_um_from": "--cell-um"})
        prof["experiment"] = DEFAULT_EXPERIMENT
        out = Path(a.out or "behavior_profile_from_seek.json")
        out.write_text(json.dumps(prof, indent=2))
        print(describe(prof)); print(f"\nSaved {out}")
        return

    from seek.archive import Archive          # only needed for records
    arc = Archive(); me = arc.owner()
    if a.list:
        for r in arc.list_records(viewer=me):
            print(f"{r['record_code']}  {(r.get('title') or r.get('original_name') or '')[:32]:32}  "
                  f"{(r.get('organism') or '')[:20]:20}  {r.get('n_tracks') or '?'} tracks")
        return
    if not a.record:
        raise SystemExit("Give a record code, --tracks FILE, or --list.")
    rec = arc.get_record(a.record, me)
    if rec is None:
        raise SystemExit(f"No record {a.record}. Try --list.")
    try:
        prof = profile_for_record(arc, rec, a.cell_um)
    except ValueError as e:
        raise SystemExit(str(e))
    key = arc.attach_swarm_profile(a.record, me["id"], prof)
    print(describe(prof)); print(f"\nSaved with the record: {key}")
    if a.out:
        Path(a.out).write_text(json.dumps(prof, indent=2)); print(f"Also wrote {a.out}")


if __name__ == "__main__":
    main()
