"""Seek → Robotarium: rule fitting recovers known rules; the robot script is valid and collision-free.

Cells here follow Seek's rule model with known coefficients (tools/make_test_video.simulate_agents)."""
import ast
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import embody                                    # noqa: E402
from seek.metrics import kinematics                        # noqa: E402
from tools.make_test_video import simulate_agents          # noqa: E402

FIELD = (256.0, 256.0)


def tracks_from(pos, fps=20.0, um_per_px=0.5):
    """Positions → Seek tracks. A cell that wraps across the edge becomes a new track (it left the field)."""
    T, n, _ = pos.shape
    ids, nxt, rows = list(range(n)), n, []
    for t in range(T):
        if t:
            for i in np.where(np.hypot(*(pos[t] - pos[t - 1]).T) > 50)[0]:
                ids[i], nxt = nxt, nxt + 1
        rows += [(t, ids[i], *pos[t, i]) for i in range(n)]
    return kinematics(pd.DataFrame(rows, columns=["frame", "particle", "x", "y"]), um_per_px, fps)


def fit(signal_dir=None, **kw):
    df = tracks_from(simulate_agents(n=30, n_frames=200, seed=1, **kw)[0])
    return embody.extract_rules(df, fps=20, R_um=40, signal_dir=signal_dir, field_um=FIELD), df


def test_recovers_signal_pull():
    r, _ = fit("right", k_pref=1.0, pref=0.0, sigma=0.5)
    assert r["k_pref"] == pytest.approx(1.0, rel=0.3)
    assert r["k_align"] == 0.0


def test_recovers_alignment_and_calls_flock_emergent():
    r, _ = fit(k_align=3.0, sigma=0.4)
    assert r["k_align"] == pytest.approx(3.0, rel=0.35)
    assert r["k_pref"] == 0.0 and "alignment" in r["k_pref_source"]


def test_random_cells_get_no_social_rules():
    r, _ = fit(sigma=0.5)
    assert r["k_pref"] == 0.0 and r["k_align"] < 0.3


def test_robot_plan_and_script():
    r, df = fit(k_align=3.0, sigma=0.4)
    plan = embody.to_robotarium(r, df, record="t", seconds=8)
    assert 1 <= plan["N"] <= 20 and plan["R"] > plan["d_sep"] and plan["v"] <= 0.2
    hist = embody.preview(plan)
    assert not hist["errors"].get("robots_too_close") and not hist["errors"].get("robots_outside_boundaries")
    cmp = embody.compare(r, df, plan, hist)
    assert cmp["robots"]["local_alignment"] > 0.5                      # aligning robots, not a random walk

    src = embody.submission_source(plan)
    tree = ast.parse(src)                                              # valid Python, no Seek imports
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and "seek" in ast.unparse(n) for n in ast.walk(tree))
    rules = ast.literal_eval(next(ast.unparse(n.value) for n in tree.body
                                  if isinstance(n, ast.Assign) and n.targets[0].id == "RULES"))
    assert rules["N"] == plan["N"] and rules["k_align"] == pytest.approx(plan["k_align"], abs=1e-3)
