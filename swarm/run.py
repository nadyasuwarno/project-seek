"""
swarm.run — one recorded swarm run on the Robotarium simulator (scripted override optional).

    python -m swarm.run --profile data/swarm/SK-2610-001.json          # headless, saves outputs
    python -m swarm.run --profile swarm/profiles/behavior_profile.json --live
    python -m swarm.render data/swarm/runs/behavior_profile             # video + chart afterwards

The loop every 0.033 s: read every robot's pose → apply the profile's rules → send (v, ω) → log.
trajectories.csv uses Seek's track columns, so robot runs can be compared with cells.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import rps.robotarium as robotarium
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

from swarm.sim import DT, couzin_heading, initial_poses, load_profile, metrics, steer, unit, wall_push


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run(profile_path, out_dir, live):
    prof = load_profile(profile_path)
    exp = prof["exp"]
    rng = np.random.default_rng(exp["seed"])
    N = exp["num_robots"]
    steps = int(exp["duration_s"] / DT)
    ov = exp.get("override", {"enabled": False})

    r = robotarium.Robotarium(
        number_of_robots=N,
        show_figure=live,
        sim_in_real_time=live,
        initial_conditions=initial_poses(N, rng),
        skip_initialization=True,
    )
    safety = create_uni_barrier_certificate_with_boundary(safety_radius=0.12)

    os.makedirs(out_dir, exist_ok=True)
    traj_rows, metric_rows = [], []
    reached_at = None
    target = None            # chosen when the override starts
    corners = np.array([[-1.2, -0.65], [-1.2, 0.65], [1.2, -0.65], [1.2, 0.65]])

    for k in range(steps):
        t = k * DT
        x = r.get_poses()                          # PERCEIVE: 3 x N
        pos, head = x[:2], x[2]

        overriding = ov.get("enabled") and ov["start_s"] <= t < ov["end_s"]
        phase = "override" if overriding else "swarm"
        if overriding and target is None:
            if ov.get("target_m") == "farthest_corner":
                c0 = pos.mean(axis=1)
                target = corners[np.argmax(np.linalg.norm(corners - c0, axis=1))]
            else:
                target = np.array(ov["target_m"])
            start_dist = float(np.linalg.norm(pos.mean(axis=1) - target))

        dxu = np.zeros((2, N))
        for i in range(N):                         # DECIDE
            if overriding:
                desired = unit(target - pos[:, i])
            else:
                desired = couzin_heading(i, pos, head, prof)
            desired = unit(desired + 1.5 * wall_push(pos[:, i]))
            ang = np.arctan2(desired[1], desired[0]) + rng.normal(0, prof["noise"])
            desired = np.array([np.cos(ang), np.sin(ang)])
            dxu[:, i] = steer(head[i], desired, prof["speed"], prof["max_turn"])

        dxu = safety(dxu, x)                       # invisible bumper
        r.set_velocities(np.arange(N), dxu)        # ACT
        r.step()

        pol, mill, nnd, c = metrics(pos, head)     # LOG
        if overriding and reached_at is None:
            if np.linalg.norm(c - target) < 0.4:
                reached_at = t - ov["start_s"]
        metric_rows.append([round(t, 3), phase, pol, mill, nnd, c[0], c[1]])
        for i in range(N):
            traj_rows.append([i, round(t, 3), pos[0, i], pos[1, i], head[i], phase])

    with open(os.path.join(out_dir, "trajectories.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "t_s", "x_m", "y_m", "theta_rad", "phase"])
        w.writerows(traj_rows)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "phase", "polarization", "milling", "nnd_bl", "centroid_x", "centroid_y"])
        w.writerows(metric_rows)

    summary = {"robots": N, "duration_s": exp["duration_s"],
               "override_target_m": None if target is None else target.round(3).tolist(),
               "override_start_distance_m": None if target is None else round(start_dist, 3),
               "override_time_to_target_s": None if reached_at is None else round(reached_at, 2),
               "profile_source": prof["raw"].get("source")}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    r.debug()                                      # Robotarium's own error check
    print("Summary:", summary)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(Path(__file__).parent / "profiles" / "behavior_profile.json"))
    ap.add_argument("--out", help="output folder (default: data/swarm/runs/<profile name>)")
    ap.add_argument("--live", action="store_true", help="show the arena window while running")
    a = ap.parse_args()
    run(a.profile, a.out or str(Path("data/swarm/runs") / Path(a.profile).stem), a.live)
