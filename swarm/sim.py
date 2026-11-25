"""The swarm's rules engine — shared by the live tuner and recorded runs.

No Robotarium import here, so Seek (and the tests) can use it without the simulator installed.

Units: robots are 0.11 m across = 1 body length (BL). Profiles store rules in body lengths, so the same
profile can describe a 2.5 µm cell or an 11 cm robot.
"""
import json

import numpy as np

BODY_LENGTH_M = 0.11                          # Robotarium robot diameter = 1 body length
DT = 0.033                                    # Robotarium time step (s)
ARENA = np.array([-1.6, 1.6, -1.0, 1.0])      # x_min, x_max, y_min, y_max (m)
WALL_MARGIN_M = 0.30                          # start steering away from walls this close
MAX_ROBOTS = 22                               # the Robotarium can place at most 22 robots
WHEEL_MAX_M_S = 0.19                          # keep the outer wheel under the 0.2 m/s motor limit


def profile_to_params(p: dict) -> dict:
    """Behavior profile (body lengths) → simulator parameters (metres, radians)."""
    bl = BODY_LENGTH_M
    rep = p["zone_repulsion_bl"]
    ori = max(p["zone_orientation_bl"], rep)          # keep zones ordered: repulsion ≤ orientation ≤ attraction
    att = max(p["zone_attraction_bl"], ori)
    return {"r_rep": rep * bl, "r_ori": ori * bl, "r_att": att * bl,
            "blind": np.deg2rad(p.get("blind_angle_deg", 60)),
            "speed": p["speed_bl_per_s"] * bl,
            "max_turn": np.deg2rad(p["max_turn_rate_deg_per_s"]),
            "noise": np.deg2rad(p["heading_noise_deg"])}


def load_profile(path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {"raw": raw, **profile_to_params(raw), "exp": raw.get("experiment", {})}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def couzin_heading(i, pos, head, prof):
    """Desired direction for robot i (Couzin et al. 2002): repulsion first, else orientation + attraction."""
    my_dir = np.array([np.cos(head[i]), np.sin(head[i])])
    rep, ori, att = np.zeros(2), np.zeros(2), np.zeros(2)
    n_rep = n_ori = n_att = 0
    for j in range(pos.shape[1]):
        if j == i:
            continue
        rij = pos[:, j] - pos[:, i]
        d = np.linalg.norm(rij)
        if d < 1e-9:
            continue
        if np.arccos(np.clip(np.dot(my_dir, rij / d), -1, 1)) > np.pi - prof["blind"] / 2:
            continue                                  # blind zone behind
        if d < prof["r_rep"]:
            rep -= rij / d; n_rep += 1
        elif d < prof["r_ori"]:
            ori += np.array([np.cos(head[j]), np.sin(head[j])]); n_ori += 1
        elif d < prof["r_att"]:
            att += rij / d; n_att += 1
    if n_rep:
        return unit(rep)
    if n_ori and n_att:
        return unit(0.5 * (unit(ori + my_dir) + unit(att)))
    if n_ori:
        return unit(ori + my_dir)
    if n_att:
        return unit(att)
    return my_dir


def couzin_all(pos, head, prof):
    """couzin_heading for every robot at once (same rules, vectorized). pos 2×N, head N → 2×N unit vectors."""
    n = pos.shape[1]
    u = np.vstack([np.cos(head), np.sin(head)])                     # 2×N headings
    rel = pos[:, None, :] - pos[:, :, None]                         # 2×N×N, i → j
    d = np.linalg.norm(rel, axis=0); np.fill_diagonal(d, np.inf)
    rhat = rel / np.maximum(d, 1e-9)
    cosang = np.einsum("ki,kij->ij", u, rhat)
    seen = cosang >= np.cos(np.pi - prof["blind"] / 2)               # outside the blind zone behind
    zr = seen & (d < prof["r_rep"])
    zo = seen & (d >= prof["r_rep"]) & (d < prof["r_ori"])
    za = seen & (d >= prof["r_ori"]) & (d < prof["r_att"])
    rep = -(rhat * zr).sum(axis=2)
    ori = (u[:, None, :] * zo).sum(axis=2) + u
    att = (rhat * za).sum(axis=2)
    def norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)
    has_r, has_o, has_a = zr.any(1), zo.any(1), za.any(1)
    out = u.copy()
    out[:, has_a] = norm(att)[:, has_a]
    out[:, has_o] = norm(ori)[:, has_o]
    both = has_o & has_a
    out[:, both] = norm(0.5 * (norm(ori) + norm(att)))[:, both]
    out[:, has_r] = norm(rep)[:, has_r]
    return out


def wall_push(p):
    """Steer back toward the middle when near the arena edge."""
    push = np.zeros(2)
    if p[0] < ARENA[0] + WALL_MARGIN_M: push[0] += 1
    if p[0] > ARENA[1] - WALL_MARGIN_M: push[0] -= 1
    if p[1] < ARENA[2] + WALL_MARGIN_M: push[1] += 1
    if p[1] > ARENA[3] - WALL_MARGIN_M: push[1] -= 1
    return push


def steer(theta, desired_dir, speed, max_turn, gain=3.0, wander=0.0):
    """Desired direction → unicycle command (v m/s, ω rad/s), within real wheel limits.
    `wander` (rad) is this tick's random heading change: the heading drifts like a cell's does."""
    err = wrap(np.arctan2(desired_dir[1], desired_dir[0]) - theta)
    omega = np.clip(gain * err, -max_turn, max_turn) + wander / DT
    v = speed                                       # keep cruising while turning, like swimming cells;
                                                    # only the wheel limit below slows a robot down
    outer = v + abs(omega) * BODY_LENGTH_M / 2
    if outer > WHEEL_MAX_M_S:
        v, omega = v * WHEEL_MAX_M_S / outer, omega * WHEEL_MAX_M_S / outer
    return v, omega


WANDER_TAU_S = 0.3   # smoothness of the random wander (seconds); same total drift, without wheel-breaking spikes


def decide(pos, head, prof, rng, target=None, state=None):
    """One tick for the whole swarm: rules (or a human target) + wall avoidance + heading wander → (v, ω) per robot.
    Used by the live window, recorded runs and the fidelity check, so all three behave identically.

    heading_noise_deg = how much a robot's heading wanders at random per 0.033 s tick (a random walk),
    the same quantity the bridge measures on cells (heading change per body length travelled).
    The wander is smooth (an Ornstein–Uhlenbeck turn rate, correlation time WANDER_TAU_S) with the same
    long-run drift as a per-tick random walk; pass the same `state` dict every tick to keep it continuous."""
    n = pos.shape[1]
    state = {} if state is None else state
    w = state.get("wander")
    if w is None or len(w) != n:
        w = np.zeros(n)
    drift = prof["noise"] ** 2 / (2 * DT)                    # rad²/s of a per-tick random walk with that step
    sd = np.sqrt(drift / WANDER_TAU_S)                       # stationary sd of the smooth turn rate
    a = np.exp(-DT / WANDER_TAU_S)
    w = a * w + sd * np.sqrt(1 - a * a) * rng.normal(size=n)
    state["wander"] = w
    want = couzin_all(pos, head, prof) if target is None else None
    dxu = np.zeros((2, n))
    for i in range(n):
        d = unit(np.asarray(target) - pos[:, i]) if target is not None else want[:, i]
        d = unit(d + 1.5 * wall_push(pos[:, i]))
        dxu[:, i] = steer(head[i], d, prof["speed"], prof["max_turn"], wander=w[i] * DT)
    return dxu


def metrics(pos, head):
    """Group metrics — the same ones Seek's bridge computes on cells."""
    u = np.vstack([np.cos(head), np.sin(head)])
    polarization = np.linalg.norm(u.mean(axis=1))
    c = pos.mean(axis=1, keepdims=True)
    r = pos - c
    rn = r / np.maximum(np.linalg.norm(r, axis=0), 1e-9)
    milling = abs(np.mean(rn[0] * u[1] - rn[1] * u[0]))
    d = np.linalg.norm(pos[:, :, None] - pos[:, None, :], axis=0)
    np.fill_diagonal(d, np.inf)
    return polarization, milling, d.min(axis=1).mean() / BODY_LENGTH_M, c.ravel()


def initial_poses(n, rng):
    pts = []
    while len(pts) < n:
        p = rng.uniform([-1.2, -0.7], [1.2, 0.7])
        if all(np.linalg.norm(p - q) > 0.25 for q in pts):
            pts.append(p)
    pts = np.array(pts).T
    return np.vstack([pts, rng.uniform(-np.pi, np.pi, n)])
