"""The archive: analyzed recordings kept like project files.

Each record = one video + its analysis (tracks, summary, thumbnail, optional annotated MP4),
tied to the researcher who submitted it, with a status, notes thread, reviews and an audit log.
Files go through `storage`, metadata through `db`; the UI only calls this module.

    arc = Archive()                         # data/seek.sqlite3 + data/files/
    me = arc.owner()                        # step 1: the local owner (admin) is the only researcher
    code = arc.submit(me["id"], video=..., ...)
    arc.list_records(viewer=me)
"""
import io as _io
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from . import db, storage as storage_mod, thumb

# columns kept in tracks.parquet: the export columns + what the record viewer needs to redraw layers
TRACK_COLS = ["frame", "time_s", "cell_id", "x", "y", "x_um", "y_um", "vx_um_s", "vy_um_s",
              "speed_um_s", "heading_deg", "nn_dist_um", "alignment", "signal_drift_um_s", "neighbors_in_R"]
REVIEW_DECISIONS = ("in_review", "approved", "needs_changes", "rejected")


class Archive:
    def __init__(self, data_dir: str | os.PathLike | None = None, files: storage_mod.Storage | None = None):
        self.dir = Path(data_dir or os.environ.get("SEEK_DATA_DIR", "data"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "seek.sqlite3"
        self.files = files or storage_mod.from_env(self.dir)
        with self._db() as con:
            con.executescript(db.SCHEMA)

    @contextmanager
    def _db(self):
        con = db.connect(self.db_path)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _audit(con, actor_id, action, target, detail=None):
        con.execute("INSERT INTO audit_log (actor_id, action, target, detail_json, created_at) VALUES (?,?,?,?,?)",
                    (actor_id, action, target, json.dumps(detail, default=str) if detail else None, db.now()))

    # ── researchers ────────────────────────────────────────────────────────────────
    def ensure_researcher(self, name: str, email: str, role="researcher", status="pending", affiliation=None) -> dict:
        email = email.strip().lower()
        with self._db() as con:
            row = con.execute("SELECT * FROM researchers WHERE email = ?", (email,)).fetchone()
            if row is None:
                cur = con.execute("INSERT INTO researchers (name, email, affiliation, role, status, created_at) "
                                  "VALUES (?,?,?,?,?,?)", (name, email, affiliation, role, status, db.now()))
                code = f"R-{cur.lastrowid:04d}"
                con.execute("UPDATE researchers SET researcher_code = ? WHERE id = ?", (code, cur.lastrowid))
                self._audit(con, cur.lastrowid, "register", code, {"name": name, "role": role})
                row = con.execute("SELECT * FROM researchers WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def owner(self) -> dict:
        """Step 1 has no login yet: everything is submitted by the local owner (admin, approved)."""
        return self.ensure_researcher(os.environ.get("SEEK_OWNER_NAME", "Ryan"),
                                      os.environ.get("SEEK_OWNER_EMAIL", "owner@seek.local"),
                                      role="admin", status="approved")

    # ── submit ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _next_code(con) -> str:
        yymm = db.now()[2:7].replace("-", "")                     # 2026-10 → 2610
        last = con.execute("SELECT record_code FROM recordings WHERE record_code LIKE ? "
                           "ORDER BY record_code DESC LIMIT 1", (f"SK-{yymm}-%",)).fetchone()
        n = int(last[0].rsplit("-", 1)[1]) + 1 if last else 1
        return f"SK-{yymm}-{n:03d}"

    def submit(self, researcher_id: int, *, video: bytes, video_name: str, tier0: dict, settings: dict,
               summary: dict, tracks: pd.DataFrame, title: str = "", note: str = "",
               annotated_mp4: bytes | None = None, elapsed_s: float | None = None,
               file_hash: str | None = None, extra: dict | None = None) -> str:
        """Store one analyzed recording. Returns its record code. Files and rows land together or not at all."""
        f, c = tier0.get("0b_file", {}), tier0.get("0c_conditions", {})
        w, h = (int(v) for v in str(f.get("resolution_px", "0×0")).split("×"))
        ext = (os.path.splitext(video_name)[1] or ".mp4").lower()
        t1 = summary or {}
        tr = (t1.get("1_trajectory") or {})
        cols = [col for col in TRACK_COLS if col in tracks.columns]

        with self._db() as con:
            con.execute("BEGIN IMMEDIATE")                         # lock so two saves can't take the same code
            code = self._next_code(con)
            pre = f"records/{code}/"
            keys = {"video": pre + "video" + ext, "tracks": pre + "tracks.parquet",
                    "summary": pre + "summary.json", "thumb": pre + "thumb.png",
                    "annotated": pre + "annotated.mp4" if annotated_mp4 else None}
            researcher = con.execute("SELECT researcher_code FROM researchers WHERE id = ?", (researcher_id,)).fetchone()
            if researcher is None:
                raise ValueError(f"Unknown researcher id {researcher_id}")
            record_json = {"record_code": code, "researcher_code": researcher[0], "file": video_name,
                           "file_hash": file_hash, **tier0, "tier1_summary": summary, "settings": settings,
                           "analysis_time_s": elapsed_s, **(extra or {})}
            try:
                buf = _io.BytesIO()
                tracks[cols].to_parquet(buf, index=False)
                self.files.put(keys["video"], video)
                self.files.put(keys["tracks"], buf.getvalue())
                self.files.put(keys["summary"], json.dumps(record_json, indent=2, default=str).encode())
                self.files.put(keys["thumb"], thumb.trajectory_png(tracks))
                if annotated_mp4:
                    self.files.put(keys["annotated"], annotated_mp4)

                ts = db.now()
                cur = con.execute(
                    """INSERT INTO recordings (record_code, researcher_id, title, organism, strain, medium,
                       temperature_c, signal, signal_dir, source, license, um_per_px, fps, n_frames, duration_s,
                       width_px, height_px, analyzed_width_px, original_name, file_hash, video_key, video_bytes,
                       status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (code, researcher_id, title.strip() or None, c.get("organism") or None, c.get("strain") or None,
                     c.get("medium") or None, c.get("temperature_c"), c.get("signal") or None,
                     c.get("signal_dir") or None, c.get("source") or None, c.get("license") or None,
                     tier0.get("0a_scale_um_per_px"), f.get("fps"), f.get("frames"), f.get("duration_s"),
                     w, h, f.get("analyzed_width_px"), video_name, file_hash, keys["video"], len(video),
                     "submitted", ts, ts))
                rec_id = cur.lastrowid
                con.execute(
                    """INSERT INTO analyses (recording_id, n_tracks, mean_track_s, mean_speed_um_s, nn_mean_um,
                       alignment_mean, signal_drift_um_s, density_mean, R_um, settings_json, tracks_key,
                       summary_key, thumb_key, annotated_key, elapsed_s, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rec_id, tr.get("tracks"), tr.get("mean_track_duration_s"),
                     t1.get("2_velocity_mean_speed_um_s"), t1.get("3_nearest_neighbor_mean_um"),
                     t1.get("4_alignment_mean"), t1.get("5_signal_drift_mean_um_s"), t1.get("6_local_density_mean"),
                     settings.get("R_um"), json.dumps(settings, default=str), keys["tracks"], keys["summary"],
                     keys["thumb"], keys["annotated"], elapsed_s, ts))
                if note.strip():
                    con.execute("INSERT INTO notes (recording_id, author_id, body, created_at) VALUES (?,?,?,?)",
                                (rec_id, researcher_id, note.strip(), ts))
                self._audit(con, researcher_id, "submit", code, {"file": video_name, "bytes": len(video)})
            except Exception:
                self.files.delete_prefix(pre)                      # no orphan files if the rows fail
                raise
        return code

    # ── read ───────────────────────────────────────────────────────────────────────
    _SELECT = """SELECT r.*, a.n_tracks, a.mean_track_s, a.mean_speed_um_s, a.nn_mean_um, a.alignment_mean,
                        a.signal_drift_um_s, a.density_mean, a.R_um, a.settings_json, a.tracks_key, a.summary_key,
                        a.thumb_key, a.annotated_key, a.elapsed_s, a.id AS analysis_id,
                        p.name AS researcher_name, p.researcher_code
                 FROM recordings r
                 JOIN researchers p ON p.id = r.researcher_id
                 LEFT JOIN analyses a ON a.id = (SELECT MAX(id) FROM analyses WHERE recording_id = r.id)"""

    @staticmethod
    def _visibility(viewer: dict | None):
        """Admins see everything. Researchers see their own records + everyone's approved ones."""
        if viewer is None or viewer.get("role") == "admin":
            return "", ()
        return " AND (r.researcher_id = ? OR r.status = 'approved')", (viewer["id"],)

    def list_records(self, viewer: dict | None = None, q: str = "", organism: str | None = None,
                     status: str | None = None, mine_only: bool = False, include_removed: bool = False) -> list[dict]:
        sql, args = self._SELECT + " WHERE 1=1", []
        vis, vargs = self._visibility(viewer)
        sql += vis
        args += vargs
        if not include_removed:
            sql += " AND r.removed_at IS NULL"
        if organism:
            sql += " AND r.organism = ?"
            args.append(organism)
        if status:
            sql += " AND r.status = ?"
            args.append(status)
        if mine_only and viewer:
            sql += " AND r.researcher_id = ?"
            args.append(viewer["id"])
        for word in re.findall(r"\S+", q or ""):
            like = f"%{word.lower()}%"
            sql += (" AND (lower(r.record_code) LIKE ? OR lower(coalesce(r.title,'')) LIKE ? OR "
                    "lower(coalesce(r.organism,'')) LIKE ? OR lower(coalesce(r.strain,'')) LIKE ? OR "
                    "lower(coalesce(r.medium,'')) LIKE ? OR lower(coalesce(r.signal,'')) LIKE ? OR "
                    "lower(p.name) LIKE ? OR lower(coalesce(r.original_name,'')) LIKE ?)")
            args += [like] * 8
        sql += " ORDER BY r.created_at DESC, r.id DESC"
        with self._db() as con:
            return [dict(r) for r in con.execute(sql, args).fetchall()]

    def get_record(self, code: str, viewer: dict | None = None) -> dict | None:
        vis, vargs = self._visibility(viewer)
        with self._db() as con:
            row = con.execute(self._SELECT + " WHERE r.record_code = ?" + vis, (code, *vargs)).fetchone()
            if row is None:
                return None
            rec = dict(row)
            rec["settings"] = json.loads(rec.pop("settings_json") or "{}")
            rec["notes"] = [dict(n) for n in con.execute(
                "SELECT n.*, p.name AS author_name, p.researcher_code FROM notes n "
                "JOIN researchers p ON p.id = n.author_id WHERE n.recording_id = ? ORDER BY n.id", (rec["id"],))]
            rec["reviews"] = [dict(v) for v in con.execute(
                "SELECT v.*, p.name AS reviewer_name FROM reviews v JOIN researchers p ON p.id = v.reviewer_id "
                "WHERE v.recording_id = ? ORDER BY v.id", (rec["id"],))]
        return rec

    def organisms(self, viewer: dict | None = None) -> list[str]:
        return sorted({r["organism"] for r in self.list_records(viewer) if r["organism"]})

    def file(self, key: str) -> bytes:
        return self.files.get(key)

    def tracks(self, rec: dict) -> pd.DataFrame:
        return pd.read_parquet(_io.BytesIO(self.files.get(rec["tracks_key"])))

    def summary_json(self, rec: dict) -> dict:
        return json.loads(self.files.get(rec["summary_key"]))

    # ── write ──────────────────────────────────────────────────────────────────────
    def add_note(self, code: str, author_id: int, body: str) -> None:
        if not body.strip():
            return
        with self._db() as con:
            rec = con.execute("SELECT id FROM recordings WHERE record_code = ?", (code,)).fetchone()
            con.execute("INSERT INTO notes (recording_id, author_id, body, created_at) VALUES (?,?,?,?)",
                        (rec["id"], author_id, body.strip(), db.now()))
            self._audit(con, author_id, "note", code)

    def attach_annotated(self, code: str, actor_id: int, mp4: bytes) -> None:
        """Store an annotated MP4 rendered later from the record page."""
        key = f"records/{code}/annotated.mp4"
        self.files.put(key, mp4)
        with self._db() as con:
            con.execute("UPDATE analyses SET annotated_key = ? WHERE recording_id = "
                        "(SELECT id FROM recordings WHERE record_code = ?)", (key, code))
            self._audit(con, actor_id, "render_mp4", code, {"bytes": len(mp4)})

    def review(self, code: str, reviewer_id: int, decision: str, comment: str = "") -> None:
        """Admin decision: in_review / approved / needs_changes / rejected (admin page arrives in step 2)."""
        if decision not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of {REVIEW_DECISIONS}")
        with self._db() as con:
            rec = con.execute("SELECT id FROM recordings WHERE record_code = ?", (code,)).fetchone()
            con.execute("INSERT INTO reviews (recording_id, reviewer_id, decision, comment, created_at) "
                        "VALUES (?,?,?,?,?)", (rec["id"], reviewer_id, decision, comment.strip() or None, db.now()))
            con.execute("UPDATE recordings SET status = ?, updated_at = ? WHERE id = ?", (decision, db.now(), rec["id"]))
            self._audit(con, reviewer_id, "review", code, {"decision": decision})

    def storage_used(self) -> int:
        with self._db() as con:
            return int(con.execute("SELECT coalesce(sum(video_bytes), 0) FROM recordings WHERE removed_at IS NULL")
                       .fetchone()[0])
