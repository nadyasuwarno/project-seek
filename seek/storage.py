"""Layer 01 Acquire: keep the original video untouched and fingerprint it.

The original file is stored as-is (R1: preserve the original source). Its
SHA-256 hash is the observation's fingerprint: it detects duplicate uploads and
lets anyone verify later that a signature was computed from exactly this file.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from . import config

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


@dataclass
class VideoInfo:
    frame_count: int
    width: int
    height: int
    container_fps: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def store_video_bytes(data: bytes, original_name: str) -> tuple[Path, str]:
    """Save uploaded bytes under data/videos/<hash><ext>. Returns (path, sha256)."""
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    digest = sha256_bytes(data)
    dest = config.videos_dir() / f"{digest[:16]}{ext}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest, digest


def store_video_file(src: Path) -> tuple[Path, str]:
    """Copy a video already on disk into the archive (used by the demo seeder)."""
    digest = sha256_file(src)
    dest = config.videos_dir() / f"{digest[:16]}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copyfile(src, dest)
    return dest, digest


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open the video. Try re-exporting it as MP4 (H.264).")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:  # some containers do not report a count; walk the file
            count = 0
            while cap.read()[0]:
                count += 1
        return VideoInfo(
            frame_count=count,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            container_fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        cap.release()
