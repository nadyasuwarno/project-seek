"""Draw analysis layers on a frame. Pure drawing — reads stored data, never re-analyzes.
Palette (BGR): cyan tracks/markers, yellow arrows, gray links — max 3 colors on footage."""
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .metrics import SIGNAL_VEC

CYAN, YELLOW, GRAY, WHITE = (255, 230, 0), (0, 215, 255), (150, 150, 150), (235, 235, 235)
LAYERS = ["Markers", "Tracks", "Velocity", "Neighbors", "Alignment", "Signal", "Density", "IDs"]
TAIL = 20
AA = cv2.LINE_AA


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _scale_bar(img, um_per_px, s):
    """10 µm bar, bottom right; switches to a nearby round length if 10 µm would be tiny or huge."""
    h, w = img.shape[:2]
    um = 10
    if not 0.03 * w <= 10 / um_per_px * s <= 0.35 * w:
        target = 0.15 * w / s * um_per_px
        um = min([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], key=lambda n: abs(np.log(n / target)))
    L = int(round(um / um_per_px * s))
    x1, y = w - 20, h - 22
    cv2.line(img, (x1 - L, y), (x1, y), WHITE, 3, AA)
    cv2.putText(img, f"{um} um", (x1 - L, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, AA)


def draw(frame: np.ndarray, f: int, df: pd.DataFrame | None, features: pd.DataFrame | None,
         layers, um_per_px: float, R_um: float = 10.0, signal_dir=None,
         speed_colors=False, vmax=None, min_width=720) -> np.ndarray:
    """Return an RGB image of frame f with the requested layers drawn on it."""
    s = max(1.0, min_width / frame.shape[1])                  # upscale small videos so lines stay crisp
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if s > 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    layers = set(layers)
    P = lambda x, y: (int(round(x * s)), int(round(y * s)))   # noqa: E731
    R_px = R_um / um_per_px * s if um_per_px else 0

    cur = df[df["frame"] == f] if df is not None else None
    if cur is not None and len(cur):
        xy = cur[["x", "y"]].to_numpy()
        if "Density" in layers and "neighbors_in_R" in cur:
            halo = img.copy()
            for x, y in xy:
                cv2.circle(halo, P(x, y), int(R_px), GRAY, 1, AA)
            img = cv2.addWeighted(halo, 0.5, img, 0.5, 0)
        if "Neighbors" in layers and "nn_dist_um" in cur and len(xy) > 1:
            for i, j in cKDTree(xy).query_pairs(R_um / um_per_px):
                cv2.line(img, P(*xy[i]), P(*xy[j]), GRAY, 1, AA)
        if "Tracks" in layers:
            vmax = vmax or 1
            win = df[(df["frame"] > f - TAIL) & (df["frame"] <= f) & df["cell_id"].isin(cur["cell_id"])]
            for _, t in win.groupby("cell_id"):
                pts = t[["x", "y"]].to_numpy()
                sp = t["speed_um_s"].to_numpy() if speed_colors and "speed_um_s" in t else None
                for k in range(1, len(pts)):
                    fade = 0.25 + 0.75 * k / len(pts)
                    col = _lerp(CYAN, YELLOW, min(sp[k] / vmax, 1)) if sp is not None else CYAN
                    cv2.line(img, P(*pts[k - 1]), P(*pts[k]), tuple(int(c * fade) for c in col), 1, AA)
        if "Alignment" in layers and "alignment" in cur:
            for (x, y), a in zip(xy, cur["alignment"]):
                if not np.isnan(a):
                    cv2.circle(img, P(x, y), int(4 * s), _lerp(GRAY, CYAN, (a + 1) / 2), -1, AA)
        if "Velocity" in layers and "vx_um_s" in cur:
            vmax = vmax or max(cur["speed_um_s"].max(), 1e-9)
            for (x, y), vx, vy, sp in zip(xy, cur["vx_um_s"], cur["vy_um_s"], cur["speed_um_s"]):
                if sp > 0:
                    L = 28 * min(sp / vmax, 1.5)
                    end = (x * s + vx / sp * L, y * s + vy / sp * L)
                    cv2.arrowedLine(img, P(x, y), (int(end[0]), int(end[1])), YELLOW, 1, AA, tipLength=0.3)
        if "Markers" in layers:
            for x, y in xy:
                cv2.circle(img, P(x, y), int(6 * s), CYAN, 1, AA)
        if "IDs" in layers:
            for (x, y), cid in zip(xy, cur["cell_id"]):
                cv2.putText(img, str(cid), P(x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, AA)
    elif features is not None and "Markers" in layers:          # detection done, tracks not yet
        for x, y in features.loc[features["frame"] == f, ["x", "y"]].to_numpy():
            cv2.circle(img, P(x, y), int(6 * s), CYAN, 1, AA)

    if "Signal" in layers and signal_dir in SIGNAL_VEC:
        ux, uy = SIGNAL_VEC[signal_dir]
        c = (60, 60)
        cv2.arrowedLine(img, (c[0] - 25 * ux, c[1] - 25 * uy), (c[0] + 25 * ux, c[1] + 25 * uy), YELLOW, 2, AA, tipLength=0.35)
        if cur is not None and "signal_drift_um_s" in cur and len(cur):
            cv2.putText(img, f"drift {cur['signal_drift_um_s'].mean():+.1f} um/s", (20, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, AA)

    if um_per_px:
        _scale_bar(img, um_per_px, s)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
