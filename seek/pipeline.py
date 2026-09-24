"""End-to-end ingest: video + contributor + metadata -> stored, tracked, signed.

    store video (hash) -> contributor (ORCID) -> observation -> trackpy run
    -> raw trajectories CSV -> signature (+ uncertainty) -> archive

Each step writes its own row, so any signature can be traced back through the
run that measured it, to the observation, to the person who contributed it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import trackpy as tp

from . import config, db
from .extraction import ExtractionParams, ExtractionResult, extract_trajectories
from .features import SignatureResult, compute_signature
from .identity import normalize_orcid
from .storage import probe_video


class DuplicateObservation(Exception):
    def __init__(self, observation_id: int):
        super().__init__(f"This exact video is already in the archive (observation #{observation_id}).")
        self.observation_id = observation_id


@dataclass
class ContributorInput:
    name: str
    orcid: str | None
    affiliation: str | None = None
    is_demo: bool = False


@dataclass
class ObservationInput:
    title: str
    system_name: str
    um_per_px: float
    frame_interval_s: float
    license: str
    visibility: str = "private"
    domain: str | None = None
    conditions: str | None = None
    doi: str | None = None
    perturbation_time_s: float | None = None
    no_human_subjects: bool = False


@dataclass
class AnalysisOutput:
    observation_id: int
    signature_id: int
    extraction: ExtractionResult
    signature: SignatureResult


def analyze(video_path: Path, video_sha256: str, contributor: ContributorInput,
            observation: ObservationInput, params: ExtractionParams | None = None) -> AnalysisOutput:
    db.init_db()
    params = params or ExtractionParams()

    existing = db.find_observation_by_hash(video_sha256)
    if existing is not None:
        raise DuplicateObservation(int(existing["id"]))

    orcid = None if contributor.is_demo and not contributor.orcid else normalize_orcid(contributor.orcid or "")
    contributor_id = db.upsert_contributor(contributor.name, orcid, contributor.affiliation, contributor.is_demo)

    info = probe_video(video_path)
    observation_id = db.create_observation(
        contributor_id=contributor_id,
        video_path=str(video_path),
        video_sha256=video_sha256,
        frame_count=info.frame_count,
        width_px=info.width,
        height_px=info.height,
        **{k: v for k, v in asdict(observation).items()},
    )

    extraction = extract_trajectories(video_path, params, observation.frame_interval_s, observation.um_per_px)
    tracks_path = config.tracks_dir() / f"observation_{observation_id}.csv"
    extraction.tracks.to_csv(tracks_path, index=False)
    run_id = db.create_extraction_run(
        observation_id, tool="trackpy", tool_version=tp.__version__, params=asdict(extraction.params),
        tracks_path=str(tracks_path), n_tracks=extraction.n_tracks, n_frames=extraction.n_frames,
    )

    step = extraction.params.frame_step
    signature = compute_signature(
        extraction.tracks,
        frame_interval_s=observation.frame_interval_s * step,
        field_w_um=info.width * observation.um_per_px,
        field_h_um=info.height * observation.um_per_px,
        perturbation_time_s=observation.perturbation_time_s,
        n_frames=extraction.n_frames,
    )
    signature.quality["extraction_notes"] = extraction.notes
    signature.quality["n_detections"] = extraction.n_detections
    signature_id = db.create_signature(
        observation_id, run_id, signature.schema_version,
        signature.features, signature.uncertainty, signature.quality,
    )
    return AnalysisOutput(observation_id, signature_id, extraction, signature)
