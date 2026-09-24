"""The data model: every signature traces back to its observation and contributor.

    contributors 1───* observations 1───* extraction_runs 1───1 signatures
                                    \\
                                     *── decisions (researcher actions on a comparison)

contributors     who uploaded it (ORCID iD, name, affiliation). Real uploads
                 must carry a valid ORCID; synthetic demo rows are flagged.
observations     the original video + everything needed to interpret it
                 responsibly (PRD R2): system, domain, conditions, scale
                 (micrometres per pixel), time (frame interval), license, DOI,
                 visibility, and the no-human-subjects attestation (R10).
extraction_runs  how measurements were made (R3: raw measurement kept apart
                 from derived interpretation): tool, version, parameters, and
                 the path to the raw trajectory table.
signatures       the derived behavioral signature: feature values, bootstrap
                 uncertainty, quality flags, and the schema version.
decisions        what a researcher did with a comparison (R12): save, reject,
                 or turn it into a hypothesis.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS contributors (
    id            INTEGER PRIMARY KEY,
    orcid         TEXT UNIQUE,
    name          TEXT NOT NULL,
    affiliation   TEXT,
    is_demo       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    CHECK (is_demo = 1 OR orcid IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS observations (
    id                    INTEGER PRIMARY KEY,
    contributor_id        INTEGER NOT NULL REFERENCES contributors(id),
    title                 TEXT NOT NULL,
    system_name           TEXT NOT NULL,
    domain                TEXT,
    conditions            TEXT,
    video_path            TEXT NOT NULL,
    video_sha256          TEXT NOT NULL UNIQUE,
    frame_count           INTEGER NOT NULL,
    width_px              INTEGER NOT NULL,
    height_px             INTEGER NOT NULL,
    frame_interval_s      REAL NOT NULL CHECK (frame_interval_s > 0),
    um_per_px             REAL NOT NULL CHECK (um_per_px > 0),
    perturbation_time_s   REAL,
    license               TEXT NOT NULL,
    doi                   TEXT,
    visibility            TEXT NOT NULL CHECK (visibility IN ('private', 'archive')),
    no_human_subjects     INTEGER NOT NULL CHECK (no_human_subjects = 1),
    uploaded_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id              INTEGER PRIMARY KEY,
    observation_id  INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    tool            TEXT NOT NULL,
    tool_version    TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    tracks_path     TEXT NOT NULL,
    n_tracks        INTEGER NOT NULL,
    n_frames        INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signatures (
    id                 INTEGER PRIMARY KEY,
    observation_id     INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    extraction_run_id  INTEGER NOT NULL UNIQUE REFERENCES extraction_runs(id) ON DELETE CASCADE,
    schema_version     TEXT NOT NULL,
    features_json      TEXT NOT NULL,
    uncertainty_json   TEXT NOT NULL,
    quality_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id                     INTEGER PRIMARY KEY,
    query_observation_id   INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    target_observation_id  INTEGER NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
    decision               TEXT NOT NULL CHECK (decision IN ('save', 'reject', 'hypothesis')),
    note                   TEXT,
    comparison_json        TEXT NOT NULL,
    created_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_obs_contributor ON observations(contributor_id);
CREATE INDEX IF NOT EXISTS idx_sig_observation ON signatures(observation_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- contributors

def upsert_contributor(name: str, orcid: str | None, affiliation: str | None = None,
                       is_demo: bool = False) -> int:
    """Find a contributor by ORCID (or by name for demo rows), else create one.

    The name and affiliation are refreshed on every upload so the credit shown
    in results stays current.
    """
    if not name or not name.strip():
        raise ValueError("Contributor name is required.")
    if orcid is None and not is_demo:
        raise ValueError("Real contributions need an ORCID iD.")
    with connect() as conn:
        if orcid:
            row = conn.execute("SELECT id FROM contributors WHERE orcid = ?", (orcid,)).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM contributors WHERE orcid IS NULL AND name = ? AND is_demo = 1", (name,)
            ).fetchone()
        if row:
            conn.execute("UPDATE contributors SET name = ?, affiliation = ? WHERE id = ?",
                         (name.strip(), affiliation, row["id"]))
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO contributors (orcid, name, affiliation, is_demo, created_at) VALUES (?,?,?,?,?)",
            (orcid, name.strip(), affiliation, int(is_demo), now()),
        )
        return int(cur.lastrowid)


# ---------------------------------------------------------------- observations

def find_observation_by_hash(sha256: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM observations WHERE video_sha256 = ?", (sha256,)).fetchone()


def create_observation(**fields: Any) -> int:
    required = ["contributor_id", "title", "system_name", "video_path", "video_sha256", "frame_count",
                "width_px", "height_px", "frame_interval_s", "um_per_px", "license", "visibility"]
    missing = [k for k in required if fields.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing observation fields: {', '.join(missing)}")
    if not fields.get("no_human_subjects"):
        raise ValueError("Uploads must attest that no identifiable human subjects appear (PRD R10).")
    fields = {**fields, "no_human_subjects": 1, "uploaded_at": fields.get("uploaded_at") or now()}
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    with connect() as conn:
        cur = conn.execute(f"INSERT INTO observations ({cols}) VALUES ({marks})", tuple(fields.values()))
        return int(cur.lastrowid)


def set_visibility(observation_id: int, visibility: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE observations SET visibility = ? WHERE id = ?", (visibility, observation_id))


# ---------------------------------------------------------------- runs + signatures

def create_extraction_run(observation_id: int, tool: str, tool_version: str, params: dict,
                          tracks_path: str, n_tracks: int, n_frames: int) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO extraction_runs
               (observation_id, tool, tool_version, params_json, tracks_path, n_tracks, n_frames, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (observation_id, tool, tool_version, json.dumps(params), tracks_path, n_tracks, n_frames, now()),
        )
        return int(cur.lastrowid)


def create_signature(observation_id: int, extraction_run_id: int, schema_version: str,
                     features: dict, uncertainty: dict, quality: dict) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO signatures
               (observation_id, extraction_run_id, schema_version, features_json, uncertainty_json,
                quality_json, created_at) VALUES (?,?,?,?,?,?,?)""",
            (observation_id, extraction_run_id, schema_version, json.dumps(features),
             json.dumps(uncertainty), json.dumps(quality), now()),
        )
        return int(cur.lastrowid)


_PROVENANCE_SELECT = """
SELECT s.id AS signature_id, s.schema_version, s.features_json, s.uncertainty_json, s.quality_json,
       s.created_at AS signature_created_at,
       r.id AS run_id, r.tool, r.tool_version, r.params_json, r.tracks_path, r.n_tracks, r.n_frames,
       o.id AS observation_id, o.title, o.system_name, o.domain, o.conditions, o.video_path,
       o.video_sha256, o.frame_count, o.width_px, o.height_px, o.frame_interval_s, o.um_per_px,
       o.perturbation_time_s, o.license, o.doi, o.visibility, o.uploaded_at,
       c.id AS contributor_id, c.orcid, c.name AS contributor_name, c.affiliation, c.is_demo
FROM signatures s
JOIN extraction_runs r ON r.id = s.extraction_run_id
JOIN observations o ON o.id = s.observation_id
JOIN contributors c ON c.id = o.contributor_id
"""


def _row_to_record(row: sqlite3.Row) -> dict:
    rec = dict(row)
    rec["features"] = json.loads(rec.pop("features_json"))
    rec["uncertainty"] = json.loads(rec.pop("uncertainty_json"))
    rec["quality"] = json.loads(rec.pop("quality_json"))
    rec["params"] = json.loads(rec.pop("params_json"))
    rec["is_demo"] = bool(rec["is_demo"])
    return rec


def get_signature(signature_id: int) -> dict | None:
    """A signature with its full provenance chain: run -> observation -> contributor."""
    with connect() as conn:
        row = conn.execute(_PROVENANCE_SELECT + " WHERE s.id = ?", (signature_id,)).fetchone()
    return _row_to_record(row) if row else None


def latest_signatures(visibility: str | None = "archive") -> list[dict]:
    """The most recent signature for each observation (optionally archive-only)."""
    where = "WHERE s.id IN (SELECT MAX(id) FROM signatures GROUP BY observation_id)"
    args: tuple = ()
    if visibility:
        where += " AND o.visibility = ?"
        args = (visibility,)
    with connect() as conn:
        rows = conn.execute(_PROVENANCE_SELECT + where + " ORDER BY o.uploaded_at DESC", args).fetchall()
    return [_row_to_record(r) for r in rows]


def latest_signature_for_observation(observation_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            _PROVENANCE_SELECT + " WHERE s.observation_id = ? ORDER BY s.id DESC LIMIT 1", (observation_id,)
        ).fetchone()
    return _row_to_record(row) if row else None


# ---------------------------------------------------------------- decisions

def record_decision(query_observation_id: int, target_observation_id: int, decision: str,
                    comparison: dict, note: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO decisions (query_observation_id, target_observation_id, decision, note,
               comparison_json, created_at) VALUES (?,?,?,?,?,?)""",
            (query_observation_id, target_observation_id, decision, note, json.dumps(comparison), now()),
        )
        return int(cur.lastrowid)


def list_decisions(query_observation_id: int | None = None) -> list[dict]:
    sql = """SELECT d.*, q.system_name AS query_system, t.system_name AS target_system
             FROM decisions d
             JOIN observations q ON q.id = d.query_observation_id
             JOIN observations t ON t.id = d.target_observation_id"""
    args: tuple = ()
    if query_observation_id is not None:
        sql += " WHERE d.query_observation_id = ?"
        args = (query_observation_id,)
    with connect() as conn:
        rows = conn.execute(sql + " ORDER BY d.id DESC", args).fetchall()
    return [dict(r) for r in rows]


def counts() -> dict:
    with connect() as conn:
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("contributors", "observations", "signatures", "decisions")
        }
