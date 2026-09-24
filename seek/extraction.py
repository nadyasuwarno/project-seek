"""Layers 02 Normalize + 04 Measure: video -> trajectories, in physical units.

Tracking uses trackpy (Crocker-Grier particle tracking, widely used in
microscopy): locate bright or dark blobs of a given size in every frame, then
link detections across frames into trajectories. Nothing here is trained; the
parameters are recorded with every run so a measurement can be reproduced.

Normalization turns pixels into micrometres (um_per_px) and frame numbers into
seconds (frame_interval_s), both supplied by the contributor, so signatures
from different microscopes and cameras sit on the same units (R2).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import trackpy as tp

tp.quiet()
logging.getLogger("trackpy").setLevel(logging.WARNING)

LIME = (0, 255, 200)  # BGR: close to #C8FF00


@dataclass
class ExtractionParams:
    diameter: int = 9               # approx. blob size in pixels (made odd)
    minmass: float | None = None    # brightness threshold; None = estimate automatically
    invert: bool = False            # True if units are dark on a light background
    search_range: float | None = None  # max movement per frame in px; None = diameter
    memory: int = 2                 # frames a unit may vanish and still be linked
    min_track_length: int = 8       # drop tracks shorter than this (frames)
    max_frames: int = 400           # cap for long videos
    frame_step: int = 1             # analyse every n-th frame

    def resolved(self) -> "ExtractionParams":
        p = ExtractionParams(**asdict(self))
        p.diameter = int(p.diameter) | 1  # trackpy requires an odd diameter
        if p.search_range is None:
            p.search_range = float(p.diameter)
        return p


@dataclass
class ExtractionResult:
    tracks: pd.DataFrame            # particle, frame, t_s, x_px, y_px, x_um, y_um, mass
    params: ExtractionParams
    n_frames: int
    n_detections: int
    n_tracks: int
    notes: list[str] = field(default_factory=list)


def read_frames(path: Path | str, max_frames: int = 400, frame_step: int = 1) -> list[np.ndarray]:
    """Read frames as float32 grayscale images."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    i = 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % frame_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                frames.append(gray.astype(np.float32))
            i += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("The video contains no readable frames.")
    return frames


def estimate_minmass(frames: list[np.ndarray], diameter: int, invert: bool) -> float:
    """Pick a brightness threshold that separates real units from background noise.

    Locate every candidate blob in a few frames, then keep those brighter than
    30% of the bright end of the distribution.
    """
    sample = frames[:: max(1, len(frames) // 5)][:5]
    masses = []
    for img in sample:
        f = tp.locate(img, diameter, minmass=0, invert=invert)
        if len(f):
            masses.append(f["mass"].to_numpy())
    if not masses:
        return 0.0
    m = np.concatenate(masses)
    return float(0.3 * np.percentile(m, 95))


def locate_first_frame(path: Path | str, params: ExtractionParams) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Detection preview: first frame, detected blobs, and the threshold used."""
    p = params.resolved()
    frames = read_frames(path, max_frames=min(p.max_frames, 25), frame_step=p.frame_step)
    minmass = p.minmass if p.minmass is not None else estimate_minmass(frames, p.diameter, p.invert)
    found = tp.locate(frames[0], p.diameter, minmass=minmass, invert=p.invert)
    return frames[0], found, minmass


def extract_trajectories(path: Path | str, params: ExtractionParams, frame_interval_s: float,
                         um_per_px: float) -> ExtractionResult:
    p = params.resolved()
    frames = read_frames(path, max_frames=p.max_frames, frame_step=p.frame_step)
    notes: list[str] = []
    if p.minmass is None:
        p.minmass = estimate_minmass(frames, p.diameter, p.invert)
        notes.append(f"minmass estimated automatically: {p.minmass:.1f}")

    detections = tp.batch(frames, p.diameter, minmass=p.minmass, invert=p.invert, processes=1)
    empty = pd.DataFrame(columns=["particle", "frame", "t_s", "x_px", "y_px", "x_um", "y_um", "mass"])
    if detections.empty:
        notes.append("No units detected. Try a different diameter, invert, or minmass.")
        return ExtractionResult(empty, p, len(frames), 0, 0, notes)

    linked = tp.link(detections, search_range=p.search_range, memory=p.memory)
    linked = tp.filter_stubs(linked, threshold=p.min_track_length)
    if linked.empty:
        notes.append("Units were detected but no track lasted long enough. "
                     "Try a larger search range or a smaller minimum track length.")
        return ExtractionResult(empty, p, len(frames), len(detections), 0, notes)

    linked = linked.reset_index(drop=True)
    dt = frame_interval_s * p.frame_step
    tracks = pd.DataFrame({
        "particle": linked["particle"].astype(int),
        "frame": linked["frame"].astype(int),
        "t_s": linked["frame"].astype(float) * dt,
        "x_px": linked["x"].astype(float),
        "y_px": linked["y"].astype(float),
        "x_um": linked["x"].astype(float) * um_per_px,
        "y_um": linked["y"].astype(float) * um_per_px,
        "mass": linked["mass"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)

    return ExtractionResult(tracks, p, len(frames), len(detections), int(tracks["particle"].nunique()), notes)


# ---------------------------------------------------------------- rendering

def _first_frame_bgr(path: Path | str) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read the first frame.")
    return frame


def render_detections(path: Path | str, found: pd.DataFrame, diameter: int) -> np.ndarray:
    """First frame (dimmed) with detected units circled. Returns RGB."""
    img = (_first_frame_bgr(path) * 0.6).astype(np.uint8)
    r = max(3, int(diameter))
    for _, row in found.iterrows():
        cv2.circle(img, (int(round(row["x"])), int(round(row["y"]))), r, LIME, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def render_tracks(path: Path | str, tracks: pd.DataFrame, max_tracks: int = 300) -> np.ndarray:
    """First frame (dimmed) with every trajectory drawn as a lime polyline. Returns RGB."""
    img = (_first_frame_bgr(path) * 0.35).astype(np.uint8)
    if not tracks.empty:
        for i, (_, tr) in enumerate(tracks.groupby("particle")):
            if i >= max_tracks:
                break
            pts = np.stack([tr["x_px"].to_numpy(), tr["y_px"].to_numpy()], axis=1).round().astype(np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, LIME, 1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in pts[-1]), 2, LIME, -1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
