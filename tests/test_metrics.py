"""Accuracy checks against the synthetic clip (known answers). Run: pytest -q"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import detect                           # noqa: E402
from seek.pipeline import Settings, run_all       # noqa: E402
from tools.make_test_video import make_clip       # noqa: E402

R_TEST = 40.0  # µm — wide enough that most dots have neighbors


def analyze(mode, **kw):
    frames, truth, meta = make_clip(mode=mode, **kw)
    s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"],
                 diameter=detect.odd_diameter(meta["cell_um"], meta["um_per_px"]),
                 minmass=0, polarity="auto", R_um=R_TEST, signal_dir=meta["signal"] or "right")
    bg = detect.background(frames)
    probe = detect.locate(detect.preprocess(frames[0], bg, "bright"), s.diameter)
    s = Settings(**{**s.__dict__, "minmass": detect.suggest_minmass(probe["mass"])})
    return run_all(frames, s), truth, meta, s


@pytest.fixture(scope="module")
def aligned():
    return analyze("aligned")


@pytest.fixture(scope="module")
def random_():
    return analyze("random")


@pytest.fixture(scope="module")
def biased():
    return analyze("biased")


def test_detection_rate(aligned):
    df, truth, *_ = aligned
    found = df.groupby("frame").size().reindex(range(truth.frame.max() + 1), fill_value=0)
    assert (found / truth.groupby("frame").size()).min() >= 0.95


def test_track_identity():
    """Each true dot keeps one track ID for ≥90% of its life (few ID swaps).
    Uses a sparse clip (30 dots, random headings): cells mostly separate, as the prototype assumes.
    Denser clips with dots passing through each other break tracks at the crossing."""
    df, truth, meta, _ = analyze("random", n_dots=30)
    upp = meta["um_per_px"]
    shares = []
    for f, tf in truth.groupby("frame"):
        d = df[df.frame == f]
        dist, j = cKDTree(d[["x_um", "y_um"]].to_numpy() / upp).query(tf[["x_px", "y_px"]].to_numpy())
        shares.append(pd.DataFrame({"dot": tf.dot_id.to_numpy(), "cell": np.where(dist < 3, d.cell_id.to_numpy()[j], -1)}))
    m = pd.concat(shares)
    keep = m.groupby("dot")["cell"].agg(lambda c: (c == c.mode()[0]).mean())
    assert (keep >= 0.9).mean() >= 0.95


def test_speed_within_5pct(aligned):
    df, _, meta, _ = aligned
    assert df.speed_um_s.mean() == pytest.approx(meta["speed_um_s"], rel=0.05)


def test_nearest_neighbor_within_5pct(aligned):
    df, truth, meta, _ = aligned
    true_nn = np.concatenate([cKDTree(t[["x_px", "y_px"]]).query(t[["x_px", "y_px"]], k=2)[0][:, 1]
                              for _, t in truth.groupby("frame")]) * meta["um_per_px"]
    assert df.nn_dist_um.mean() == pytest.approx(true_nn.mean(), rel=0.05)


def test_alignment(aligned, random_):
    assert aligned[0].alignment.mean() > 0.95
    assert abs(random_[0].alignment.mean()) < 0.15


def test_signal_drift(biased, random_):
    v = biased[2]["speed_um_s"]
    assert biased[0].signal_drift_um_s.mean() > 0.3 * v
    assert abs(random_[0].signal_drift_um_s.mean()) < 0.15 * v


def test_single_cell_neighbor_metrics_are_empty():
    """One cell → neighbor metrics are NaN ("needs 2+ cells"), never a wrong number."""
    frames, _, meta = make_clip(n_dots=1, mode="random", seed=3)
    s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"], diameter=7, minmass=50, R_um=10)
    df = run_all(frames, s)
    assert df.cell_id.nunique() == 1
    assert df.nn_dist_um.isna().all() and df.alignment.isna().all() and df.neighbors_in_R.isna().all()
    assert df.signal_drift_um_s.isna().all()   # no signal set
