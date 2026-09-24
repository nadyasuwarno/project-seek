import numpy as np
import pandas as pd
import pytest

from seek.compare import MATCH_THRESHOLD, compare, search
from seek.demo import DEMO_SYSTEMS, SIZE, simulate
from seek.features import compute_signature


def _record(key, obs_id, seed=0):
    s = next(d for d in DEMO_SYSTEMS if d.key == key)
    pos = simulate(key, seed=seed)
    frames, n, _ = pos.shape
    tracks = pd.DataFrame({
        "particle": np.repeat(np.arange(n)[None], frames, 0).ravel(),
        "frame": np.repeat(np.arange(frames), n),
        "x_um": pos[:, :, 0].ravel() * s.um_per_px,
        "y_um": pos[:, :, 1].ravel() * s.um_per_px,
    }).sort_values(["particle", "frame"])
    p = s.perturbation_frame * s.frame_interval_s if s.perturbation_frame else None
    sig = compute_signature(tracks, s.frame_interval_s, SIZE * s.um_per_px, SIZE * s.um_per_px, p,
                            n_frames=frames, bootstrap=10)
    return {"observation_id": obs_id, "system_name": key, "features": sig.features, "uncertainty": sig.uncertainty}


@pytest.fixture(scope="module")
def archive():
    keys = ["brownian", "run_tumble", "walkers_macro", "flock", "oscillators", "clusters"]
    return {k: _record(k, i, seed=i) for i, k in enumerate(keys)}


def test_same_dynamics_at_different_scales_match(archive):
    c = compare(archive["run_tumble"], archive["walkers_macro"])
    assert c.kind == "match" and c.score > 0.8
    assert c.scale_ratio > 10                 # ~50x apart in absolute speed
    assert c.causal_status == "unknown"       # R8: similarity is not causality


def test_absolute_scale_mode_penalizes_the_scale_gap(archive):
    free = compare(archive["run_tumble"], archive["walkers_macro"], scale_free=True)
    absolute = compare(archive["run_tumble"], archive["walkers_macro"], scale_free=False)
    assert absolute.dimension("movement").score < free.dimension("movement").score


def test_similarity_is_decomposed_by_dimension(archive):
    c = compare(archive["run_tumble"], archive["walkers_macro"])
    labels = {d.family for d in c.dimensions}
    assert labels == {"movement", "trajectory", "collective", "temporal", "perturbation"}
    assert c.dimension("perturbation").status == "unmeasured"
    for d in c.dimensions:
        if d.comparable:
            assert d.low <= d.score <= d.high


def test_shared_absence_of_rhythm_is_not_scored(archive):
    c = compare(archive["brownian"], archive["run_tumble"])
    assert c.dimension("temporal").status == "shared_absence"
    assert not c.dimension("temporal").comparable


def test_no_meaningful_match(archive):
    result = search(archive["flock"], [archive["oscillators"], archive["clusters"]])
    assert result.no_match
    assert all((c.score or 0) < MATCH_THRESHOLD for c in result.weak + result.counterexamples)


def test_query_never_matches_itself(archive):
    result = search(archive["run_tumble"], list(archive.values()))
    ids = [c.target["observation_id"] for c in result.matches + result.weak + result.counterexamples]
    assert archive["run_tumble"]["observation_id"] not in ids
    assert result.matches[0].target["system_name"] == "walkers_macro"
    assert result.discovery_prompt and "?" in result.discovery_prompt


def test_counterexample_is_similar_on_one_dimension_and_different_on_another():
    base = {"mean_speed": 10, "speed_cv": 0.3, "straightness": 0.8, "turning": 0.1, "msd_alpha": 1.8,
            "alignment": 0.9, "clustering": 1.0, "periodicity_strength": 0.05}
    lookalike = {**base, "alignment": 0.02, "clustering": 0.3, "straightness": 0.2, "msd_alpha": 0.9, "turning": 0.5}
    q = {"observation_id": 1, "system_name": "q", "features": base, "uncertainty": {}}
    t = {"observation_id": 2, "system_name": "t", "features": lookalike, "uncertainty": {}}
    c = compare(q, t)
    assert c.kind == "counterexample"
    assert "movement" in c.reason.lower()


def test_too_few_measured_dimensions_is_insufficient():
    q = {"observation_id": 1, "features": {"speed_cv": 0.3}, "uncertainty": {}}
    t = {"observation_id": 2, "features": {"speed_cv": 0.31}, "uncertainty": {}}
    assert compare(q, t).kind == "insufficient"
