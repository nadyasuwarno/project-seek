"""Extraction + signature tests on synthetic systems whose dynamics we control."""

import numpy as np
import pandas as pd
import pytest

from seek.demo import SIZE, render_video, simulate
from seek.extraction import ExtractionParams, extract_trajectories
from seek.features import compute_signature


def _tracks_from_positions(pos, um_per_px=1.0):
    frames, n, _ = pos.shape
    return pd.DataFrame({
        "particle": np.repeat(np.arange(n)[None], frames, 0).ravel(),
        "frame": np.repeat(np.arange(frames), n),
        "x_um": pos[:, :, 0].ravel() * um_per_px,
        "y_um": pos[:, :, 1].ravel() * um_per_px,
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


def test_straight_constant_velocity_lines():
    frames, n = 60, 12
    t = np.arange(frames)[:, None]
    start = np.random.default_rng(0).uniform(20, 60, (n, 2))
    pos = start[None] + np.stack([2.0 * t, 1.0 * t], axis=2) * np.ones((1, n, 1))
    sig = compute_signature(_tracks_from_positions(pos), 0.1, SIZE, SIZE, n_frames=frames, bootstrap=5)
    f = sig.features
    assert f["mean_speed"] == pytest.approx(np.hypot(2, 1) / 0.1, rel=1e-6)
    assert f["straightness"] == pytest.approx(1.0, abs=1e-6)
    assert f["msd_alpha"] == pytest.approx(2.0, abs=0.01)
    assert f["alignment"] > 0.95          # everyone moves the same way
    assert f["speed_cv"] == pytest.approx(0.0, abs=1e-9)


def test_perturbation_response_measured_from_ground_truth():
    pos = simulate("perturbed", seed=3)
    sig = compute_signature(_tracks_from_positions(pos, 0.5), 0.05, SIZE * 0.5, SIZE * 0.5,
                            perturbation_time_s=60 * 0.05, n_frames=len(pos), bootstrap=5)
    assert sig.features["response_ratio"] > 1.5
    assert sig.features["recovery_time"] is not None


def test_missing_perturbation_is_reported_not_invented():
    pos = simulate("brownian", seed=1)
    sig = compute_signature(_tracks_from_positions(pos), 0.05, SIZE, SIZE, n_frames=len(pos), bootstrap=5)
    assert sig.features["response_ratio"] is None
    assert any("perturbation" in w.lower() for w in sig.quality["warnings"])


@pytest.fixture(scope="module")
def tracked(tmp_path_factory):
    """Render three systems to video, then track them with trackpy."""
    out = {}
    d = tmp_path_factory.mktemp("videos")
    for i, key in enumerate(["brownian", "run_tumble", "oscillators", "clusters"]):
        video = render_video(simulate(key, n=24, frames=96, seed=i), d / key, seed=i)
        ex = extract_trajectories(video, ExtractionParams(), frame_interval_s=0.05, um_per_px=0.5)
        sig = compute_signature(ex.tracks, 0.05, SIZE * 0.5, SIZE * 0.5, n_frames=ex.n_frames, bootstrap=10)
        out[key] = (ex, sig)
    return out


def test_trackpy_finds_tracks_in_physical_units(tracked):
    ex, _ = tracked["brownian"]
    assert ex.n_tracks >= 10
    assert {"particle", "frame", "t_s", "x_um", "y_um"} <= set(ex.tracks.columns)
    assert np.allclose(ex.tracks["x_um"], ex.tracks["x_px"] * 0.5)
    assert ex.params.diameter % 2 == 1


def test_signatures_separate_distinct_dynamics(tracked):
    brown = tracked["brownian"][1].features
    swim = tracked["run_tumble"][1].features
    osc = tracked["oscillators"][1].features
    clus = tracked["clusters"][1].features

    assert 0.7 < brown["msd_alpha"] < 1.3          # diffusive
    assert swim["msd_alpha"] > 1.5                 # directed
    assert swim["straightness"] > brown["straightness"] + 0.3
    assert osc["periodicity_strength"] > 0.5       # rhythmic
    assert brown["periodicity_strength"] < 0.3
    assert clus["clustering"] < 0.75               # spatially clustered
    assert clus["msd_alpha"] < 0.8                 # confined


def test_uncertainty_reported_for_measured_features(tracked):
    sig = tracked["run_tumble"][1]
    for key in ("mean_speed", "straightness", "msd_alpha", "alignment"):
        assert sig.uncertainty[key] is not None and sig.uncertainty[key] >= 0
    assert sig.uncertainty["response_ratio"] is None   # not measured, so no fake error bar
