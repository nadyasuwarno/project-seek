"""Where record files live (video, tracks.parquet, summary.json, thumbnail, annotated MP4).

The rest of Seek only talks to the `Storage` interface with plain string keys such as
"records/SK-2610-001/tracks.parquet". Today that is a folder on disk (`LocalStorage`, under
data/files/). In step 3 an Oracle Object Storage backend implements the same five methods and
is picked with SEEK_STORAGE=oci — nothing else in the app changes.
"""
import os
import shutil
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def size(self, key: str) -> int: ...
    def delete_prefix(self, prefix: str) -> None: ...


class LocalStorage:
    """Files under one root folder. Writes are atomic (temp file, then rename)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if p != self.root and self.root not in p.parents:     # keys can never escape the root
            raise ValueError(f"Bad storage key: {key!r}")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()


def from_env(data_dir: str | os.PathLike) -> Storage:
    backend = os.environ.get("SEEK_STORAGE", "local").lower()
    if backend == "local":
        return LocalStorage(Path(data_dir) / "files")
    raise NotImplementedError(f"SEEK_STORAGE={backend!r} isn't available yet (Oracle Object Storage comes in step 3).")
