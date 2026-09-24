"""Load the synthetic demo archive (see demo.py) through the real pipeline."""

from __future__ import annotations

from typing import Callable

from . import config, db
from .demo import DEMO_CONTRIBUTORS, DEMO_SYSTEMS, render_video, simulate
from .pipeline import ContributorInput, DuplicateObservation, ObservationInput, analyze
from .storage import store_video_file


def seed_demo(progress: Callable[[float, str], None] | None = None) -> int:
    """Generate, track, and sign every demo system. Returns how many were added."""
    db.init_db()
    src_dir = config.data_dir() / "demo_src"
    added = 0
    for i, s in enumerate(DEMO_SYSTEMS):
        if progress:
            progress(i / len(DEMO_SYSTEMS), f"Tracking {s.title.lower()}")
        video = render_video(simulate(s.key, seed=i), src_dir / s.key, seed=i)
        path, sha = store_video_file(video)
        name, affiliation = DEMO_CONTRIBUTORS[s.contributor]
        try:
            analyze(
                path, sha,
                ContributorInput(name=name, orcid=None, affiliation=affiliation, is_demo=True),
                ObservationInput(
                    title=s.title, system_name=s.system_name, domain=s.domain, conditions=s.conditions,
                    um_per_px=s.um_per_px, frame_interval_s=s.frame_interval_s,
                    perturbation_time_s=(s.perturbation_frame * s.frame_interval_s
                                         if s.perturbation_frame is not None else None),
                    license="CC0-1.0 (synthetic)", visibility="archive", no_human_subjects=True,
                ),
            )
            added += 1
        except DuplicateObservation:
            pass
    if progress:
        progress(1.0, "Demo archive ready")
    return added
