"""Swarm bridge (prototype 3). Run: pytest -q

1. Recovers known rules from synthetic cells (speed exact; alignment detected only when present).
2. A profile fitted from a saved Seek record is stored with that record and can be exported for the simulator.
3. The rules engine keeps robots within real wheel limits.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import detect, io                              # noqa: E402
from seek.archive import Archive                         # noqa: E402
from seek.metrics import summary                         # noqa: E402
from seek.pipeline import Settings, run_all              # noqa: E402
from swarm import bridge, fake_cells, sim                # noqa: E402
from tools.make_test_video import make_clip              # noqa: E402


@pytest.fixture(scope="module")
def profiles():
    out = {}
    for mode in ("couzin", "independent"):
        df = fake_cells.simulate(mode, n=30, frames=300, seed=2)
        out[mode] = bridge.fit(df, fake_cells.TRUTH["cell_um"], {"source": mode})
    return out


def test_speed_recovered_exactly(profiles):
    for p in profiles.values():
        assert p["fit"]["cell_speed_bl_per_s"] == pytest.approx(fake_cells.TRUTH["speed_bl_s"], rel=0.02)
        assert p["speed_bl_per_s"] <= bridge.ROBOT_MAX_BL_S


def test_alignment_detected_only_when_cells_align(profiles):
    assert profiles["couzin"]["zone_orientation_bl"] > profiles["couzin"]["zone_repulsion_bl"] + 1
    assert "no local alignment" in profiles["independent"]["fit"]["orientation_note"]


def test_personal_space_only_when_cells_repel(profiles):
    assert profiles["couzin"]["fit"]["repulsion_measured_bl"] > 0.5
    assert profiles["independent"]["fit"]["repulsion_measured_bl"] < 0.5


def test_profile_lives_with_the_record(tmp_path):
    frames, _, meta = make_clip(mode="aligned", n_dots=20, n_frames=60, seed=4)
    s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"],
                 diameter=detect.odd_diameter(meta["cell_um"], meta["um_per_px"]), minmass=0, R_um=30.0)
    df = run_all(frames, s)
    h, w = frames.shape[1:]
    tier0 = {"0a_scale_um_per_px": meta["um_per_px"],
             "0b_file": {"fps": meta["fps"], "frames": len(frames), "duration_s": len(frames) / meta["fps"],
                         "resolution_px": f"{w}×{h}", "analyzed_width_px": w},
             "0c_conditions": {"organism": "Synthetic dots"}}
    arc = Archive(tmp_path / "data")
    me = arc.owner()
    code = arc.submit(me["id"], video=io.write_mp4([np.repeat(f[..., None], 3, -1) for f in frames], meta["fps"]),
                      video_name="clip.mp4", tier0=tier0, settings=s.as_dict(), summary=summary(df), tracks=df)
    rec = arc.get_record(code, me)
    assert arc.swarm_profile(code) is None

    prof = bridge.profile_for_record(arc, rec)
    arc.attach_swarm_profile(code, me["id"], prof)
    stored = arc.swarm_profile(code)
    assert stored["source"] == f"Seek record {code}"
    assert stored["fit"]["cell_length_um"] == pytest.approx(s.diameter * meta["um_per_px"])
    assert stored["fit"]["cell_speed_bl_per_s"] == pytest.approx(meta["speed_um_s"] / stored["fit"]["cell_length_um"],
                                                               rel=0.05)
    assert stored["targets"]["polarization"] > 0.9                    # aligned clip
    path = arc.swarm_profile_file(code)
    assert path.exists() and sim.load_profile(path)["r_rep"] > 0


def test_commands_respect_wheel_limits():
    for err in np.linspace(-np.pi, np.pi, 25):
        v, w = sim.steer(0.0, np.array([np.cos(err), np.sin(err)]), speed=0.18, max_turn=np.deg2rad(200))
        assert v + abs(w) * sim.BODY_LENGTH_M / 2 <= sim.WHEEL_MAX_M_S + 1e-9


def test_fidelity_check_trusts_identical_rules():
    """Sanity of the check itself: 'cells' that are robots running a profile must match robots
    running the same profile with different chance."""
    import json
    from swarm import compare
    prof = json.load(open(os.path.join(os.path.dirname(__file__), "..", "swarm", "profiles", "behavior_profile.json")))
    prof["fit"], prof["targets"] = {"cell_length_um": 2.5, "time_scale": 0.1}, {"cells_per_frame": 10}
    poses, _ = compare.simulate_robots(prof, 10, 25, seed=101, warmup_s=5)
    pseudo_cells = compare.robots_as_tracks(poses, 2.5, 0.1, 20)
    result, png = compare.check(pseudo_cells, prof, "sanity", runs=3)
    verdicts = {k: m["verdict"] for k, m in result["metrics"].items()}
    assert verdicts["speed"] == "close" and verdicts["nn"] == "close"
    assert sum(v == "close" for v in verdicts.values()) >= 4
    assert png[:4] == b"\x89PNG"
