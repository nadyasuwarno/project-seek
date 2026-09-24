"""Layer 05 Represent: trajectories -> behavioral signature (with uncertainty).

Every feature in schema.py is computed here from the raw trajectory table.
Uncertainty comes from bootstrapping: resample the tracked units (or frames,
for per-frame features) with replacement, recompute, and report the spread.
A signature built from 6 wobbly tracks therefore carries wide error bars, and
the comparison step treats it more cautiously than one built from 200 tracks.

The perturbation features follow the PRD's behavioral episode grammar:
baseline -> perturbation -> response -> adaptation -> stabilization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .schema import FEATURES, SCHEMA_VERSION

MIN_TRACK_POINTS = 8
STRAIGHTNESS_WINDOW = 16    # frames
MAX_MSD_LAG = 20
SPECTRUM_SEGMENT = 32       # frames per spectral segment
BOOTSTRAP_SAMPLES = 40
MAX_FRAMES_FOR_SPATIAL = 80


@dataclass
class SignatureResult:
    features: dict[str, float | None]
    uncertainty: dict[str, float | None]
    quality: dict
    schema_version: str = SCHEMA_VERSION
    population_speed: pd.DataFrame = field(default_factory=pd.DataFrame)  # t_s, speed (for charts)


@dataclass
class _Track:
    frames: np.ndarray
    xy: np.ndarray
    step_frames: np.ndarray       # frame at which each 1-frame step starts
    speeds: np.ndarray            # um/s for each 1-frame step
    mean_speed: float | None
    speed_cv: float | None
    straightness: float | None
    turning: float | None
    msd_sum: np.ndarray
    msd_count: np.ndarray
    spec_sum: np.ndarray
    spec_count: int


def _per_track(tr: pd.DataFrame, dt: float) -> _Track | None:
    frames = tr["frame"].to_numpy(int)
    xy = tr[["x_um", "y_um"]].to_numpy(float)
    if len(frames) < MIN_TRACK_POINTS:
        return None

    consecutive = np.diff(frames) == 1
    d = np.diff(xy, axis=0)
    step_len = np.hypot(d[:, 0], d[:, 1])
    speeds = step_len[consecutive] / dt
    step_frames = frames[:-1][consecutive]

    mean_speed = float(speeds.mean()) if len(speeds) else None
    speed_cv = float(speeds.std() / speeds.mean()) if len(speeds) > 2 and speeds.mean() > 0 else None

    # Straightness over fixed-length windows, so it does not depend on how long
    # the tracker managed to follow each unit (longer paths always look less straight).
    ratios = []
    pos_by_frame = {int(f): xy[i] for i, f in enumerate(frames)}
    for i in range(0, len(frames) - STRAIGHTNESS_WINDOW, STRAIGHTNESS_WINDOW // 2):
        f0 = int(frames[i])
        if frames[i + STRAIGHTNESS_WINDOW] - f0 != STRAIGHTNESS_WINDOW:
            continue  # window contains a gap
        seg = np.array([pos_by_frame[f0 + k] for k in range(STRAIGHTNESS_WINDOW + 1)])
        path = np.hypot(*np.diff(seg, axis=0).T).sum()
        if path > 0:
            ratios.append(float(np.hypot(*(seg[-1] - seg[0])) / path))
    straightness = float(np.mean(ratios)) if ratios else None

    # turning angles between consecutive 1-frame steps
    turning = None
    if consecutive.sum() >= 3:
        idx = np.where(consecutive)[0]
        pairs = idx[:-1][np.diff(idx) == 1]
        if len(pairs):
            a1 = np.arctan2(d[pairs, 1], d[pairs, 0])
            a2 = np.arctan2(d[pairs + 1, 1], d[pairs + 1, 0])
            dtheta = np.angle(np.exp(1j * (a2 - a1)))
            turning = float(np.mean(np.abs(dtheta)) / np.pi)

    # mean squared displacement by lag (in frames), gaps handled via frame lookup
    msd_sum = np.zeros(MAX_MSD_LAG)
    msd_count = np.zeros(MAX_MSD_LAG)
    pos = {int(f): xy[i] for i, f in enumerate(frames)}
    max_lag = min(MAX_MSD_LAG, max(1, (frames[-1] - frames[0]) // 4))
    for lag in range(1, max_lag + 1):
        disp = [pos[f + lag] - pos[f] for f in pos if (f + lag) in pos]
        if disp:
            arr = np.asarray(disp)
            msd_sum[lag - 1] = float((arr ** 2).sum())
            msd_count[lag - 1] = len(arr)

    # power spectrum of speed fluctuations, from runs of consecutive steps
    n_bins = SPECTRUM_SEGMENT // 2 + 1
    spec_sum = np.zeros(n_bins)
    spec_count = 0
    window = np.hanning(SPECTRUM_SEGMENT)
    runs = np.split(speeds, np.where(np.diff(step_frames) != 1)[0] + 1) if len(speeds) else []
    for run in runs:
        for start in range(0, len(run) - SPECTRUM_SEGMENT + 1, SPECTRUM_SEGMENT):
            seg = run[start:start + SPECTRUM_SEGMENT]
            seg = (seg - seg.mean()) * window
            spec_sum += np.abs(np.fft.rfft(seg)) ** 2
            spec_count += 1

    return _Track(frames, xy, step_frames, speeds, mean_speed, speed_cv, straightness, turning,
                  msd_sum, msd_count, spec_sum, spec_count)


def _nanmean(values) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _msd_alpha(tracks: list[_Track], dt: float) -> float | None:
    s = np.sum([t.msd_sum for t in tracks], axis=0)
    c = np.sum([t.msd_count for t in tracks], axis=0)
    ok = (c > 0) & (s > 0)
    if ok.sum() < 4:
        return None
    lags = (np.arange(1, MAX_MSD_LAG + 1) * dt)[ok]
    msd = s[ok] / c[ok]
    slope, _ = np.polyfit(np.log(lags), np.log(msd), 1)
    return float(slope)


def _periodicity(tracks: list[_Track], dt: float) -> tuple[float | None, float | None]:
    count = sum(t.spec_count for t in tracks)
    if count == 0:
        return None, None
    spec = np.sum([t.spec_sum for t in tracks], axis=0) / count
    p = spec[1:]                      # drop the DC bin
    if p.sum() <= 0:
        return None, None
    k = int(np.argmax(p))
    band = p[max(0, k - 1): k + 2].sum() / p.sum()   # Hann window spreads a peak over ~3 bins
    chance = min(3, len(p)) / len(p)
    strength = float(np.clip((band - chance) / (1 - chance), 0, 1))
    period = SPECTRUM_SEGMENT * dt / (k + 1)
    return strength, float(period)


def _population_speed(tracks: list[_Track], n_frames: int) -> np.ndarray:
    total = np.zeros(n_frames)
    count = np.zeros(n_frames)
    for t in tracks:
        np.add.at(total, t.step_frames, t.speeds)
        np.add.at(count, t.step_frames, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _perturbation(speed: np.ndarray, dt: float, p_time: float | None) -> tuple[float | None, float | None, str | None]:
    if p_time is None:
        return None, None, None
    p_frame = int(round(p_time / dt))
    valid = np.isfinite(speed)
    pre = speed[:p_frame][valid[:p_frame]]
    post = speed[p_frame:]
    if len(pre) < 5 or np.isfinite(post).sum() < 5:
        return None, None, "Perturbation time leaves too few frames before or after it."
    baseline = float(pre.mean())
    if baseline <= 0:
        return None, None, "Baseline speed is zero; response cannot be measured."
    w = int(max(3, min(10, len(post) // 4)))
    response = float(np.nanmean(post[:w]))
    ratio = response / baseline
    if 0.8 <= ratio <= 1.25:
        return ratio, None, "No clear response to the perturbation (within 20-25% of baseline)."
    smooth = pd.Series(post).rolling(5, min_periods=3, center=True).mean().to_numpy()
    close = np.abs(smooth - baseline) <= 0.2 * baseline
    for i in range(w, len(close) - 2):
        if close[i] and close[i + 1] and close[i + 2]:
            return ratio, float(i * dt), None
    return ratio, None, "Did not return to baseline within the observation window."


def _spatial_per_frame(df: pd.DataFrame, field_area_um2: float) -> tuple[list[float], list[float]]:
    """Per-frame alignment and Clark-Evans ratio (sampled frames)."""
    frames = np.sort(df["frame"].unique())
    if len(frames) > MAX_FRAMES_FOR_SPATIAL:
        frames = frames[np.linspace(0, len(frames) - 1, MAX_FRAMES_FOR_SPATIAL).astype(int)]
    by_frame = {f: g for f, g in df.groupby("frame")}
    align, clust = [], []
    for f in frames:
        cur = by_frame.get(f)
        nxt = by_frame.get(f + 1)
        if cur is None:
            continue
        pts = cur[["x_um", "y_um"]].to_numpy()
        n = len(pts)
        if n >= 5 and field_area_um2 > 0:
            dist, _ = cKDTree(pts).query(pts, k=2)
            expected = 0.5 * np.sqrt(field_area_um2 / n)
            clust.append(float(dist[:, 1].mean() / expected))
        if nxt is not None:
            m = cur.merge(nxt, on="particle", suffixes=("", "_n"))
            v = m[["x_um_n", "y_um_n"]].to_numpy() - m[["x_um", "y_um"]].to_numpy()
            norms = np.hypot(v[:, 0], v[:, 1])
            v = v[norms > 0] / norms[norms > 0, None]
            k = len(v)
            if k >= 3:
                polar = float(np.hypot(*v.mean(axis=0)))
                chance = np.sqrt(np.pi / (4 * k))       # expected polarization of k random directions
                align.append(float(np.clip((polar - chance) / (1 - chance), 0, 1)))
    return align, clust


def _aggregate(tracks: list[_Track], dt: float, n_frames: int, p_time: float | None) -> dict[str, float | None]:
    strength, period = _periodicity(tracks, dt)
    speed = _population_speed(tracks, n_frames)
    ratio, recovery, _ = _perturbation(speed, dt, p_time)
    return {
        "mean_speed": _nanmean(t.mean_speed for t in tracks),
        "speed_cv": _nanmean(t.speed_cv for t in tracks),
        "straightness": _nanmean(t.straightness for t in tracks),
        "turning": _nanmean(t.turning for t in tracks),
        "msd_alpha": _msd_alpha(tracks, dt),
        "periodicity_strength": strength,
        "dominant_period": period if strength and strength > 0.15 else None,
        "response_ratio": ratio,
        "recovery_time": recovery,
    }


def compute_signature(tracks_df: pd.DataFrame, frame_interval_s: float, field_w_um: float,
                      field_h_um: float, perturbation_time_s: float | None = None,
                      n_frames: int | None = None, bootstrap: int = BOOTSTRAP_SAMPLES,
                      seed: int = 0) -> SignatureResult:
    dt = frame_interval_s
    keys = [f.key for f in FEATURES]
    quality: dict = {"warnings": []}

    if tracks_df.empty:
        quality.update(n_tracks=0, warnings=["No trajectories were extracted."])
        return SignatureResult({k: None for k in keys}, {k: None for k in keys}, quality)

    n_frames = n_frames or int(tracks_df["frame"].max()) + 2
    tracks = [t for _, g in tracks_df.groupby("particle") if (t := _per_track(g, dt)) is not None]
    lengths = tracks_df.groupby("particle").size()
    quality.update(
        n_tracks=len(tracks),
        median_track_length=int(lengths.median()),
        frames_analyzed=int(n_frames),
        bootstrap_samples=bootstrap,
    )
    if not tracks:
        quality["warnings"].append(f"No track has at least {MIN_TRACK_POINTS} points.")
        return SignatureResult({k: None for k in keys}, {k: None for k in keys}, quality)
    if len(tracks) < 10:
        quality["warnings"].append(f"Only {len(tracks)} tracks: treat this signature as low-confidence.")

    features = _aggregate(tracks, dt, n_frames, perturbation_time_s)
    align, clust = _spatial_per_frame(tracks_df, field_w_um * field_h_um)
    features["alignment"] = float(np.mean(align)) if align else None
    features["clustering"] = float(np.mean(clust)) if clust else None

    speed = _population_speed(tracks, n_frames)
    _, _, p_note = _perturbation(speed, dt, perturbation_time_s)
    if perturbation_time_s is None:
        quality["warnings"].append("No perturbation time given: perturbation response not measured.")
    elif p_note:
        quality["warnings"].append(p_note)
    if features["periodicity_strength"] is None:
        quality["warnings"].append(f"Tracks shorter than {SPECTRUM_SEGMENT} steps: periodicity not measured.")

    # bootstrap uncertainty
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(bootstrap):
        pick = [tracks[i] for i in rng.integers(0, len(tracks), len(tracks))]
        for k, v in _aggregate(pick, dt, n_frames, perturbation_time_s).items():
            if v is not None and np.isfinite(v):
                samples[k].append(v)
        if align:
            samples["alignment"].append(float(np.mean(rng.choice(align, len(align)))))
        if clust:
            samples["clustering"].append(float(np.mean(rng.choice(clust, len(clust)))))
    uncertainty = {k: (float(np.std(samples[k])) if features.get(k) is not None and len(samples[k]) > 2 else None)
                   for k in keys}

    pop = pd.DataFrame({"t_s": np.arange(n_frames) * dt, "speed_um_s": speed}).dropna()
    return SignatureResult({k: features.get(k) for k in keys}, uncertainty, quality, population_speed=pop)
