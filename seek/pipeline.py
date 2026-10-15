"""Runs the stages in order. Each finished stage is published on the Job so the UI
can show its layer while later stages are still running (progressive layers)."""
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import detect, metrics, track


@dataclass(frozen=True)
class Settings:
    um_per_px: float            # of the frames being analyzed (after any resize)
    fps: float
    diameter: int               # px, odd
    minmass: float
    polarity: str = "auto"      # auto | bright | dark
    max_speed_um_s: float = 50.0
    R_um: float = 10.0
    signal_dir: str | None = None
    min_len: int = 10
    memory: int = 3

    @property
    def search_range(self) -> float:
        return track.search_range_px(self.max_speed_um_s, self.fps, self.um_per_px)

    def as_dict(self):
        return {**asdict(self), "search_range_px": round(self.search_range, 2)}


# stage key → (overlay layer, metric card it unlocks, status label)
STAGES = [
    ("detect", "Markers", "Detecting"),
    ("link", "Tracks", "Linking"),
    ("velocity", "Velocity", "Velocity"),
    ("neighbors", "Neighbors", "Neighbors"),
    ("alignment", "Alignment", "Alignment"),
    ("signal", "Signal", "Signal"),
    ("density", "Density", "Density"),
]


_POOL = None


def _pool():
    """Detection runs in worker processes: uses every core and keeps the UI thread responsive."""
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 2)),
                                    mp_context=mp.get_context("spawn"))
    return _POOL


def _detect_chunk(frames, start, bg, polarity, diameter, minmass):
    return detect.detect(frames, bg, polarity, diameter, minmass).assign(frame=lambda d: d["frame"] + start)


class Job:
    def __init__(self, frames: np.ndarray, s: Settings):
        self.frames, self.s = frames, s
        self.done: list[str] = []       # finished stage keys, in order
        self.features = None            # detections (px)
        self.df = None                  # tracks + metrics; replaced (never mutated) per stage
        self.progress, self.error, self.elapsed_s = 0.0, None, None
        self.polarity = s.polarity

    @property
    def finished(self):
        return self.error is not None or len(self.done) == len(STAGES)

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def run(self):
        s = self.s
        t0 = time.perf_counter()
        try:
            bg = detect.background(self.frames)
            if self.polarity == "auto":
                self.polarity = detect.guess_polarity(self.frames[0], bg)
            step = 25
            futs = [_pool().submit(_detect_chunk, self.frames[i:i + step], i, bg, self.polarity, s.diameter, s.minmass)
                    for i in range(0, len(self.frames), step)]
            parts = []
            for k, fut in enumerate(as_completed(futs), 1):
                parts.append(fut.result())
                self.progress = k / len(futs)
            self.features = pd.concat(parts).sort_values("frame").reset_index(drop=True)
            self.done.append("detect")

            tracks = track.link(self.features, s.search_range, s.memory, s.min_len)
            df = metrics.kinematics(tracks, s.um_per_px, s.fps)
            self.df = df.drop(columns=["speed_um_s", "heading_deg", "vx_um_s", "vy_um_s"])
            self.done.append("link")
            self.df = df
            self.done.append("velocity")

            steps = [("neighbors", "nn_dist_um", lambda d: metrics.nearest_neighbor(d)),
                     ("alignment", "alignment", lambda d: metrics.alignment(d, s.R_um)),
                     ("signal", "signal_drift_um_s", lambda d: metrics.signal_drift(d, s.signal_dir)),
                     ("density", "neighbors_in_R", lambda d: metrics.local_density(d, s.R_um))]
            for key, col, fn in steps:
                df = df.assign(**{col: fn(df)})
                self.df = df
                self.done.append(key)
        except Exception as e:  # surfaced in the UI
            self.error = f"{type(e).__name__}: {e}"
        self.elapsed_s = round(time.perf_counter() - t0, 2)
        return self


def run_all(frames, s: Settings):
    """Synchronous run (tests, scripts). Returns the final per-cell per-frame table."""
    job = Job(frames, s).run()
    if job.error:
        raise RuntimeError(job.error)
    return job.df
