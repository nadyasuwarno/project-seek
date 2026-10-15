"""Archive round-trip on a throwaway data folder. Run: pytest -q"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import detect, io                              # noqa: E402
from seek.archive import Archive                         # noqa: E402
from seek.metrics import summary                         # noqa: E402
from seek.pipeline import Settings, run_all              # noqa: E402
from seek.storage import LocalStorage                    # noqa: E402
from tools.make_test_video import make_clip              # noqa: E402


@pytest.fixture(scope="module")
def analyzed():
    frames, _, meta = make_clip(mode="aligned", n_dots=20, n_frames=40, seed=3)
    s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"],
                 diameter=detect.odd_diameter(meta["cell_um"], meta["um_per_px"]), minmass=0, R_um=30.0)
    df = run_all(frames, s)
    video = io.write_mp4([np.repeat(f[..., None], 3, -1) for f in frames], meta["fps"])
    h, w = frames.shape[1:]
    tier0 = {"0a_scale_um_per_px": meta["um_per_px"],
             "0b_file": {"fps": meta["fps"], "frames": len(frames), "duration_s": len(frames) / meta["fps"],
                         "resolution_px": f"{w}×{h}", "analyzed_width_px": w},
             "0c_conditions": {"organism": "Synthetic dots", "strain": "", "medium": "", "temperature_c": 25.0,
                               "signal": "", "signal_dir": "none", "source": "test", "license": "Ours"}}
    return dict(video=video, tier0=tier0, settings=s.as_dict(), summary=summary(df), tracks=df)


def submit(arc, who, a, **kw):
    return arc.submit(who["id"], video=a["video"], video_name="clip.mp4", tier0=a["tier0"],
                      settings=a["settings"], summary=a["summary"], tracks=a["tracks"], **kw)


def test_submit_and_read_back(tmp_path, analyzed):
    arc = Archive(tmp_path)
    me = arc.owner()
    assert me["researcher_code"] == "R-0001" and me["role"] == "admin"

    code = submit(arc, me, analyzed, title="first", note="looks aligned")
    assert code.startswith("SK-") and code.endswith("-001")
    assert submit(arc, me, analyzed).endswith("-002")

    rec = arc.get_record(code, me)
    assert rec["status"] == "submitted" and rec["researcher_code"] == "R-0001"
    assert rec["organism"] == "Synthetic dots" and rec["n_tracks"] == analyzed["summary"]["1_trajectory"]["tracks"]
    assert rec["notes"][0]["body"] == "looks aligned"
    assert arc.file(rec["video_key"]) == analyzed["video"]
    assert arc.file(rec["thumb_key"])[:4] == b"\x89PNG"
    t = arc.tracks(rec)
    assert len(t) == len(analyzed["tracks"]) and {"x", "y", "speed_um_s"} <= set(t.columns)
    assert json.loads(arc.file(rec["summary_key"]))["record_code"] == code

    assert [r["record_code"] for r in arc.list_records(me, q="first")] == [code]
    assert len(arc.list_records(me, organism="Synthetic dots")) == 2


def test_visibility_and_review(tmp_path, analyzed):
    arc = Archive(tmp_path)
    me, other = arc.owner(), arc.ensure_researcher("Ana", "ana@lab.org", status="approved")
    mine = submit(arc, me, analyzed)
    theirs = submit(arc, other, analyzed)

    assert {r["record_code"] for r in arc.list_records(other)} == {theirs}      # not my unapproved record
    arc.review(mine, me["id"], "approved", "good clip")
    assert {r["record_code"] for r in arc.list_records(other)} == {mine, theirs}
    assert arc.get_record(mine, other)["reviews"][0]["decision"] == "approved"
    assert len(arc.list_records(me)) == 2                                      # admin sees everything


def test_failed_save_leaves_no_files(tmp_path, analyzed):
    arc = Archive(tmp_path)
    with pytest.raises(ValueError):
        submit(arc, {"id": 999}, analyzed)
    assert not any((tmp_path / "files").rglob("*.*"))


def test_storage_keys_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        LocalStorage(tmp_path).put("../outside.txt", b"x")
