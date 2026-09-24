from pathlib import Path

import pandas as pd
import pytest

from seek import db
from seek.demo import render_video, simulate
from seek.pipeline import ContributorInput, DuplicateObservation, ObservationInput, analyze
from seek.storage import store_video_file


@pytest.fixture
def video(tmp_path):
    return render_video(simulate("run_tumble", n=20, frames=80, seed=5), tmp_path / "swim", seed=5)


def _obs(**kw):
    base = dict(title="Swimmers", system_name="Test swimmers", um_per_px=0.5, frame_interval_s=0.05,
                license="CC-BY-4.0", visibility="archive", no_human_subjects=True)
    base.update(kw)
    return ObservationInput(**base)


def test_upload_to_signature_with_full_provenance(video):
    path, sha = store_video_file(video)
    out = analyze(path, sha, ContributorInput("J. Carberry", "0000-0002-1825-0097", "Brown University"), _obs())

    rec = db.get_signature(out.signature_id)
    assert rec["orcid"] == "0000-0002-1825-0097"
    assert rec["contributor_name"] == "J. Carberry"
    assert rec["video_sha256"] == sha
    assert rec["tool"] == "trackpy" and rec["params"]["diameter"] == 9
    assert rec["n_tracks"] == out.extraction.n_tracks > 0
    raw = pd.read_csv(rec["tracks_path"])          # raw measurements kept apart from the signature
    assert len(raw) > 0 and "x_um" in raw.columns
    assert rec["features"]["mean_speed"] > 0


def test_invalid_orcid_rejected_before_anything_is_stored(video):
    path, sha = store_video_file(video)
    with pytest.raises(ValueError, match="checksum"):
        analyze(path, sha, ContributorInput("Typo", "0000-0002-1825-0098"), _obs())
    assert db.counts()["observations"] == 0


def test_same_video_twice_is_flagged(video):
    path, sha = store_video_file(video)
    analyze(path, sha, ContributorInput("J. Carberry", "0000-0002-1825-0097"), _obs())
    with pytest.raises(DuplicateObservation):
        analyze(path, sha, ContributorInput("J. Carberry", "0000-0002-1825-0097"), _obs())
