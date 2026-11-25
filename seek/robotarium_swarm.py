"""Seek → Robotarium swarm controller.  Self-contained: needs only numpy + the Robotarium's `rps`.

This file is BOTH the controller Seek uses for its local preview AND the template for the script you
upload to the Robotarium (seek/embody.py fills in RULES and writes a copy per record).

Each robot follows the rules Seek extracted from a microscope recording. Every step, robot i turns by

    ω_i = k_pref  · sin(θ* − θ_i)          pull toward the recording's preferred direction (signal / drift)
        + k_align · sin(φ_neighbors − θ_i)  turn toward the mean heading of neighbors within R  (alignment)
        + k_sep   · (push away from neighbors closer than d_sep)                            (separation)
        + wall avoidance                     the arena has walls; the microscope field did not
        + ω_noise                            heading noise measured from the cells (smoothed for motors)

and drives at its cruise speed v (each robot keeps its own speed, spread like the cells' speeds).
Robotarium barrier certificates sit on top as the hard safety layer (no collisions, stay in the arena).

Run it yourself:  python robotarium_swarm.py      (opens the Robotarium simulator window)
"""
import numpy as np

# ── rules (seek/embody.py replaces this block for each record) ─────────────────────────────────────
RULES = {
    "record": "demo",
    "N": 10,                 # robots (Robotarium hardware: 1-20)
    "v": 0.10,               # cruise speed, m/s
    "v_spread": 0.15,        # robot-to-robot speed spread (fraction of v)
    "sigma": 0.25,           # heading noise, rad/√s
    "tau_noise": 0.6,        # noise smoothing, s (keeps wheel commands within limits)
    "k_pref": 0.0,           # pull toward preferred direction, 1/s
    "pref": 0.0,             # preferred direction, rad (math convention: 0 = +x, π/2 = +y)
    "sweep": True,           # reverse the preferred direction when the swarm nears the far wall
    "k_align": 0.8,          # alignment with neighbors, 1/s
    "R": 0.45,               # neighbor radius, m
    "d_sep": 0.35,           # preferred minimum spacing, m
    "k_sep": 4.0,            # separation strength, 1/s
    "seconds": 60.0,         # experiment length
    "seed": 0,
    "initial": None,         # 3×N initial poses [x; y; θ], or None for random well-spaced poses
}
# ───────────────────────────────────────────────────────────────────────────────────────────────────

ARENA = np.array([-1.6, 1.6, -1.0, 1.0])     # x_min, x_max, y_min, y_max (m)
DT = 0.033                                    # Robotarium time step (s)
HALF_AXLE = 0.055                             # wheel limit:  |v| + |ω|·0.055 ≤ 0.2 m/s
WHEEL_LIMIT = 0.19                            # stay a little under 0.2 m/s per wheel


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def init_state(rules, n):
    rng = np.random.default_rng(rules["seed"])
    return {"rng": rng,
            "speed": rules["v"] * np.clip(1 + rules["v_spread"] * rng.standard_normal(n), 0.5, 1.5),
            "noise": np.zeros(n),
            "pref": float(rules["pref"]),
            "flips": [], "t": 0.0}


def swarm_step(x, rules, st):
    """x: 3×N poses. Returns 2×N unicycle commands [v; ω] before the safety barrier."""
    n = x.shape[1]
    p, th = x[:2].T, x[2]
    rng = st["rng"]

    # sweep: flip the preferred direction when the swarm's centre gets close to the far wall
    if rules["k_pref"] > 0 and rules["sweep"]:
        u = np.array([np.cos(st["pref"]), np.sin(st["pref"])])
        reach = abs(u[0]) * ARENA[1] + abs(u[1]) * ARENA[3]           # half-extent along u
        if p.mean(0) @ u > 0.45 * reach:
            st["pref"] = wrap(st["pref"] + np.pi)
            st["flips"].append(round(st["t"], 2))

    d = p[None, :, :] - p[:, None, :]                                 # d[i, j] = p_j − p_i
    dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(n) * 1e9
    omega = rules["k_pref"] * np.sin(st["pref"] - th)

    nb = dist < rules["R"]                                            # alignment
    hx, hy = (nb * np.cos(th)[None]).sum(1), (nb * np.sin(th)[None]).sum(1)
    has = nb.any(1)
    omega[has] += rules["k_align"] * np.sin(np.arctan2(hy, hx)[has] - th[has])

    close = dist < rules["d_sep"]                                     # separation
    w = np.where(close, (rules["d_sep"] - dist) / rules["d_sep"], 0.0)
    push = -(w[..., None] * d / np.maximum(dist, 1e-6)[..., None]).sum(1)
    mag = np.hypot(push[:, 0], push[:, 1])
    omega += rules["k_sep"] * np.minimum(mag, 1.0) * np.sin(np.arctan2(push[:, 1], push[:, 0]) - th)

    look = p + 0.35 * np.c_[np.cos(th), np.sin(th)]                   # walls: steer back toward the middle
    over = np.maximum.reduce([ARENA[0] + 0.3 - look[:, 0], look[:, 0] - ARENA[1] + 0.3,
                              ARENA[2] + 0.3 - look[:, 1], look[:, 1] - ARENA[3] + 0.3, np.zeros(n)])
    to_mid = np.arctan2(-p[:, 1], -p[:, 0])
    omega += 6.0 * np.minimum(over / 0.3, 1.0) * np.sin(to_mid - th)

    tau = rules["tau_noise"]                                          # smoothed heading noise (OU process):
    st["noise"] += -st["noise"] * DT / tau \
        + rules["sigma"] / tau * np.sqrt(DT) * rng.standard_normal(n)  # long-run heading diffusion = sigma²
    omega += st["noise"]

    v = st["speed"] * (1 - 0.6 * np.minimum(over / 0.3, 1.0))         # slow down near walls
    heading = np.c_[np.cos(th), np.sin(th)]                           # someone too close in front → slow down
    blocked = (close & (np.einsum("ijk,ik->ij", d, heading) > 0)).any(1)
    v = np.where(blocked, v * 0.5, v)
    omega = np.clip(omega, -(WHEEL_LIMIT - v) / HALF_AXLE, (WHEEL_LIMIT - v) / HALF_AXLE)
    st["t"] += DT
    return np.vstack([v, omega])


def limit_wheels(dxu):
    """Scale commands so neither wheel exceeds the Robotarium's limit (after the barrier)."""
    need = np.abs(dxu[0]) + np.abs(dxu[1]) * HALF_AXLE
    return dxu * np.minimum(1.0, WHEEL_LIMIT / np.maximum(need, 1e-9))


def run(rules=RULES, show_figure=True, sim_in_real_time=True, skip_initialization=False, record=None):
    """Run on the Robotarium (or its simulator). `record(t, poses)` is called every step if given."""
    import rps.robotarium as robotarium
    from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
    from rps.utilities.misc import generate_random_poses

    n = int(rules["N"])
    init = np.array(rules["initial"], float) if rules["initial"] is not None else \
        generate_random_poses(n, spacing=0.35, width=2.8, height=1.6)
    r = robotarium.Robotarium(number_of_robots=n, show_figure=show_figure, initial_conditions=init,
                              sim_in_real_time=sim_in_real_time, skip_initialization=skip_initialization)
    barrier = create_uni_barrier_certificate_with_boundary(safety_radius=0.12, barrier_gain=150.0,
                                                           projection_distance=0.03, velocity_magnitude_limit=0.2)
    st = init_state(rules, n)
    for k in range(int(rules["seconds"] / DT)):
        x = r.get_poses()
        if record:
            record(k * DT, x, st)
        dxu = limit_wheels(barrier(swarm_step(x, rules, st), x))
        r.set_velocities(np.arange(n), dxu)
        r.step()
    r.debug()
    return r, st


if __name__ == "__main__":
    run()
