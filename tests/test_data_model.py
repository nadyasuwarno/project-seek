import sqlite3

import pytest

from seek import db
from seek.identity import normalize_orcid

EXAMPLE_ORCID = "0000-0002-1825-0097"  # ORCID's own documented example iD


def _observation(contributor_id, sha="a" * 64, **overrides):
    fields = dict(
        contributor_id=contributor_id, title="Test obs", system_name="E. coli swimming",
        domain="microbiology", conditions="37C", video_path="/tmp/x.mp4", video_sha256=sha,
        frame_count=100, width_px=256, height_px=256, frame_interval_s=0.1, um_per_px=0.5,
        license="CC-BY-4.0", visibility="archive", no_human_subjects=True,
    )
    fields.update(overrides)
    return db.create_observation(**fields)


def test_orcid_normalization_and_checksum():
    assert normalize_orcid("https://orcid.org/0000-0002-1825-0097") == EXAMPLE_ORCID
    assert normalize_orcid("0000000218250097") == EXAMPLE_ORCID
    with pytest.raises(ValueError, match="checksum"):
        normalize_orcid("0000-0002-1825-0098")
    with pytest.raises(ValueError):
        normalize_orcid("not-an-orcid")


def test_real_contributor_requires_orcid():
    with pytest.raises(ValueError):
        db.upsert_contributor("No ID", orcid=None)


def test_contributor_upsert_is_keyed_by_orcid():
    a = db.upsert_contributor("J. Carberry", EXAMPLE_ORCID, "Brown")
    b = db.upsert_contributor("Josiah Carberry", EXAMPLE_ORCID, "Brown University")
    assert a == b


def test_observation_requires_no_human_attestation():
    cid = db.upsert_contributor("J. Carberry", EXAMPLE_ORCID)
    with pytest.raises(ValueError, match="human"):
        _observation(cid, no_human_subjects=False)


def test_duplicate_video_hash_rejected():
    cid = db.upsert_contributor("J. Carberry", EXAMPLE_ORCID)
    _observation(cid)
    with pytest.raises(sqlite3.IntegrityError):
        _observation(cid)


def test_signature_traces_back_to_observation_and_contributor():
    cid = db.upsert_contributor("J. Carberry", EXAMPLE_ORCID, "Brown University")
    oid = _observation(cid)
    rid = db.create_extraction_run(oid, "trackpy", "0.7", {"diameter": 9}, "/tmp/t.csv", 12, 100)
    sid = db.create_signature(oid, rid, "0.1", {"mean_speed": 3.2}, {"mean_speed": 0.1}, {"n_tracks": 12})

    rec = db.get_signature(sid)
    assert rec["observation_id"] == oid
    assert rec["contributor_name"] == "J. Carberry"
    assert rec["orcid"] == EXAMPLE_ORCID
    assert rec["affiliation"] == "Brown University"
    assert rec["tool"] == "trackpy" and rec["params"] == {"diameter": 9}
    assert rec["features"]["mean_speed"] == 3.2


def test_private_observations_stay_out_of_archive_search():
    cid = db.upsert_contributor("J. Carberry", EXAMPLE_ORCID)
    public = _observation(cid, sha="b" * 64)
    private = _observation(cid, sha="c" * 64, visibility="private")
    for oid in (public, private):
        rid = db.create_extraction_run(oid, "trackpy", "0.7", {}, "/tmp/t.csv", 5, 50)
        db.create_signature(oid, rid, "0.1", {}, {}, {})
    archive_ids = {r["observation_id"] for r in db.latest_signatures("archive")}
    assert archive_ids == {public}
