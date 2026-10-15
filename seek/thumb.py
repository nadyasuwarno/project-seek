"""Specimen thumbnail: every track of a recording drawn as one ink silhouette (RGBA PNG).

Used on the archive sheet, so each record is recognizable by the shape its cells traced.
"""
import cv2
import numpy as np
import pandas as pd

INK = (17, 17, 17)


def trajectory_png(df: pd.DataFrame, size=(320, 240), pad=0.12, weight=None) -> bytes:
    """Fit all tracks (x, y in px) into `size`, keeping aspect; return a transparent PNG."""
    W, H = size
    alpha = np.zeros((H, W), np.uint8)
    if df is not None and len(df) and {"x", "y", "cell_id"} <= set(df.columns):
        x0, x1 = df["x"].min(), df["x"].max()
        y0, y1 = df["y"].min(), df["y"].max()
        span = max(x1 - x0, y1 - y0, 1e-6)
        k = min(W, H) * (1 - 2 * pad) / span
        ox = (W - (x1 - x0) * k) / 2 - x0 * k
        oy = (H - (y1 - y0) * k) / 2 - y0 * k
        n = df["cell_id"].nunique()
        w = weight or int(np.clip(9 - np.log2(max(n, 1)) * 1.2, 2, 7))   # fewer tracks → bolder strokes
        for _, t in df.sort_values("frame").groupby("cell_id"):
            pts = np.c_[t["x"] * k + ox, t["y"] * k + oy].round().astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(alpha, [pts], False, 255, w, cv2.LINE_AA)
            cv2.circle(alpha, tuple(pts[0]), w, 255, -1, cv2.LINE_AA)
        # soften then re-threshold → slightly organic, printed-ink edges
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.6)
        alpha = np.clip((alpha.astype(np.float32) - 70) * 2.2, 0, 255).astype(np.uint8)
    rgba = np.dstack([np.full((H, W), c, np.uint8) for c in INK[::-1]] + [alpha])
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes()
