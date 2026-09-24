"""Where Seeker keeps its local data.

Everything lives under one data directory (default: ./data) so the archive is
easy to back up, reset, or point elsewhere with the SEEK_DATA_DIR environment
variable (the tests use this to work in a temporary folder).
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    root = Path(os.environ.get("SEEK_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_dir() / "seek.sqlite3"


def videos_dir() -> Path:
    d = data_dir() / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tracks_dir() -> Path:
    d = data_dir() / "tracks"
    d.mkdir(parents=True, exist_ok=True)
    return d
