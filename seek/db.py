"""Searchable metadata + summary metrics (SQLite for now; Oracle Autonomous Database in step 3).

Files never go in the database — only their storage keys. Types are kept plain (INTEGER, REAL,
TEXT) and timestamps are ISO-8601 UTC strings so the schema ports to Oracle without surprises.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("submitted", "in_review", "approved", "needs_changes", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS researchers (
    id              INTEGER PRIMARY KEY,
    researcher_code TEXT UNIQUE,                 -- R-0001, shown on every record
    name            TEXT NOT NULL,
    email           TEXT NOT NULL UNIQUE,
    affiliation     TEXT,
    role            TEXT NOT NULL DEFAULT 'researcher' CHECK (role IN ('researcher', 'admin')),
    status          TEXT NOT NULL DEFAULT 'pending'    CHECK (status IN ('pending', 'approved', 'suspended')),
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY,
    record_code   TEXT NOT NULL UNIQUE,          -- SK-2610-001
    researcher_id INTEGER NOT NULL REFERENCES researchers(id),
    title         TEXT,
    organism      TEXT, strain TEXT, medium TEXT, temperature_c REAL,
    signal        TEXT, signal_dir TEXT, source TEXT, license TEXT,
    um_per_px     REAL, fps REAL, n_frames INTEGER, duration_s REAL,
    width_px      INTEGER, height_px INTEGER, analyzed_width_px INTEGER,
    original_name TEXT, file_hash TEXT,
    video_key     TEXT NOT NULL, video_bytes INTEGER,
    status        TEXT NOT NULL DEFAULT 'submitted' CHECK (status IN
                  ('submitted', 'in_review', 'approved', 'needs_changes', 'rejected')),
    removed_at    TEXT,                          -- taken down (kept for the audit trail)
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rec_researcher ON recordings(researcher_id);
CREATE INDEX IF NOT EXISTS ix_rec_status     ON recordings(status);
CREATE INDEX IF NOT EXISTS ix_rec_organism   ON recordings(organism);

CREATE TABLE IF NOT EXISTS analyses (
    id                 INTEGER PRIMARY KEY,
    recording_id       INTEGER NOT NULL REFERENCES recordings(id),
    n_tracks           INTEGER,
    mean_track_s       REAL,
    mean_speed_um_s    REAL,                     -- Tier 1 summary, one column per metric
    nn_mean_um         REAL,
    alignment_mean     REAL,
    signal_drift_um_s  REAL,
    density_mean       REAL,
    R_um               REAL,
    settings_json      TEXT,
    tracks_key         TEXT NOT NULL,
    summary_key        TEXT NOT NULL,
    thumb_key          TEXT,
    annotated_key      TEXT,
    elapsed_s          REAL,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_an_recording ON analyses(recording_id);

CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    reviewer_id  INTEGER NOT NULL REFERENCES researchers(id),
    decision     TEXT NOT NULL CHECK (decision IN ('in_review', 'approved', 'needs_changes', 'rejected')),
    comment      TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (                -- researcher notes + discussion on a record
    id           INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    author_id    INTEGER NOT NULL REFERENCES researchers(id),
    body         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY,
    actor_id    INTEGER REFERENCES researchers(id),
    action      TEXT NOT NULL,                    -- submit, note, review, edit, take_down, export…
    target      TEXT,                             -- record or researcher code
    detail_json TEXT,
    created_at  TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con
