"""Tier 1 metrics. Input: tracks in px/frames. Output: everything in µm and seconds.

Headings use the math convention on screen: 0° = right, 90° = up.
vx_um_s / vy_um_s are kept in image axes (y down) for drawing.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# unit vectors in image axes (y down)
SIGNAL_VEC = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}


def kinematics(tracks: pd.DataFrame, um_per_px: float, fps: float) -> pd.DataFrame:
    """Rank 1 (trajectory) + rank 2 (velocity)."""
    df = tracks.sort_values(["particle", "frame"]).copy()
    df["cell_id"] = df["particle"]
    df["time_s"] = df["frame"] / fps
    g = df.groupby("particle")
    for c in "xy":  # 3-frame rolling mean (track ends keep raw position, avoiding edge bias) → µm
        sm = g[c].transform(lambda s: s.rolling(3, center=True).mean())
        df[f"{c}_um"] = sm.fillna(df[c]) * um_per_px
    g = df.groupby("particle")
    t_next, t_prev = g["time_s"].shift(-1), g["time_s"].shift(1)
    dt = t_next.fillna(df["time_s"]) - t_prev.fillna(df["time_s"])     # central difference, one-sided at ends
    for c in "xy":
        nxt, prv = g[f"{c}_um"].shift(-1), g[f"{c}_um"].shift(1)
        df[f"v{c}_um_s"] = (nxt.fillna(df[f"{c}_um"]) - prv.fillna(df[f"{c}_um"])) / dt
    df["speed_um_s"] = np.hypot(df["vx_um_s"], df["vy_um_s"])
    df["heading_deg"] = np.degrees(np.arctan2(-df["vy_um_s"], df["vx_um_s"]))
    return df.sort_values(["frame", "cell_id"]).reset_index(drop=True)


def _per_frame(df: pd.DataFrame, fn) -> np.ndarray:
    """Run fn(points_um, rows) per frame; fn returns one value per row (NaN if <2 cells)."""
    out = np.full(len(df), np.nan)
    pts = df[["x_um", "y_um"]].to_numpy()
    for _, idx in df.groupby("frame").indices.items():
        if len(idx) >= 2:
            out[idx] = fn(pts[idx], idx)
    return out


def nearest_neighbor(df: pd.DataFrame) -> np.ndarray:
    """Rank 3 — distance to the closest other cell (µm). Separation."""
    return _per_frame(df, lambda p, _: cKDTree(p).query(p, k=2)[0][:, 1])


def alignment(df: pd.DataFrame, R: float) -> np.ndarray:
    """Rank 4 — mean cos(heading difference) with neighbors within R. Alignment."""
    v = df[["vx_um_s", "vy_um_s"]].to_numpy()
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def fn(p, idx):
        uu = u[idx]
        res = np.full(len(idx), np.nan)
        for i, nb in enumerate(cKDTree(p).query_ball_point(p, R)):
            nb = [j for j in nb if j != i]
            if nb:
                res[i] = float(np.mean(uu[nb] @ uu[i]))
        return res
    return _per_frame(df, fn)


def signal_drift(df: pd.DataFrame, direction: str | None) -> np.ndarray:
    """Rank 5 — velocity component along the signal (µm/s); + = toward it. Signal-following."""
    if direction not in SIGNAL_VEC:
        return np.full(len(df), np.nan)
    ux, uy = SIGNAL_VEC[direction]
    return (df["vx_um_s"] * ux + df["vy_um_s"] * uy).to_numpy()


def local_density(df: pd.DataFrame, R: float) -> np.ndarray:
    """Rank 6 — number of other cells within R. Cohesion."""
    return _per_frame(df, lambda p, _: np.array([len(n) - 1 for n in cKDTree(p).query_ball_point(p, R)], float))


def _mean(s) -> float | None:
    s = pd.Series(s).dropna()
    return None if s.empty else round(float(s.mean()), 4)


def summary(df: pd.DataFrame) -> dict:
    """One number per metric (None when it can't be computed)."""
    life = df.groupby("cell_id")["time_s"].agg(lambda t: t.max() - t.min())
    return {
        "1_trajectory": {"tracks": int(df["cell_id"].nunique()), "mean_track_duration_s": _mean(life)},
        "2_velocity_mean_speed_um_s": _mean(df["speed_um_s"]),
        "3_nearest_neighbor_mean_um": _mean(df.get("nn_dist_um")),
        "4_alignment_mean": _mean(df.get("alignment")),
        "5_signal_drift_mean_um_s": _mean(df.get("signal_drift_um_s")),
        "6_local_density_mean": _mean(df.get("neighbors_in_R")),
    }


EXPORT_COLS = ["frame", "time_s", "cell_id", "x_um", "y_um", "speed_um_s", "heading_deg",
               "nn_dist_um", "alignment", "signal_drift_um_s", "neighbors_in_R"]
