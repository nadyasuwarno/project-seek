"""Seek → Robotarium from the command line.

    python tools/embody.py SK-2610-001            # a record from your repository (data/)
    python tools/embody.py --demo                 # four synthetic recordings, analyzed by Seek first
    python tools/embody.py SK-2610-001 --seconds 90

For each recording, writes out/robotarium/<name>/:
    robotarium_<name>.py   the script to upload to the Robotarium (or run locally: python robotarium_<name>.py)
    rules.json             what Seek measured, the robot rules, and how well the robots kept the pattern
    preview.mp4            cells (left) next to the Robotarium simulator (right)
"""
import argparse
import os
import sys
import time
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import detect                                                       # noqa: E402
from seek.embody_video import make_package, record_inputs                    # noqa: E402
from seek.pipeline import Settings, run_all                                  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out" / "robotarium"


def analyze(frames, meta, R_um=40.0):
    """Run Seek's real pipeline (detect → track → metrics) on a clip."""
    s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"],
                 diameter=detect.odd_diameter(meta["cell_um"], meta["um_per_px"]), minmass=0,
                 R_um=R_um, signal_dir=meta["signal"])
    bg = detect.background(frames)
    probe = detect.locate(detect.preprocess(frames[0], bg, "bright"), s.diameter)
    s = Settings(**{**s.__dict__, "minmass": detect.suggest_minmass(probe["mass"])})
    return run_all(frames, s), s


def embody_one(name, title, tracks, fps, R_um, signal_dir, field_um, seconds, out=OUT):
    t = time.time()
    pkg = make_package(tracks, fps, R_um, signal_dir, field_um, name, title, seconds)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"robotarium_{name}.py").write_text(pkg["script"])
    (d / "rules.json").write_text(pkg["report_json"])
    (d / "preview.mp4").write_bytes(pkg["mp4"])
    p = pkg["plan"]
    print(f"{name}: {pkg['verdict']} · pull {p['k_pref']:.2f} · align {p['k_align']:.2f} · "
          f"{p['N']} robots · {time.time() - t:.0f}s → {d}")
    return d


def demo(seconds):
    from tools.make_test_video import make_clip, make_flock_clip
    clips = [
        ("demo-flock", "Flock · cells align with neighbours", make_flock_clip(n_dots=18, n_frames=120, seed=4)),
        ("demo-signal", "Signal · cells biased toward a cue (→)", make_clip(mode="biased", n_dots=16, n_frames=90, seed=11)),
        ("demo-aligned", "Aligned · cells swim the same way", make_clip(mode="aligned", n_dots=14, n_frames=90, seed=12)),
        ("demo-random", "Random · independent swimmers", make_clip(mode="random", n_dots=16, n_frames=90, seed=13)),
    ]
    dirs = []
    for name, title, (frames, _, meta) in clips:
        tracks, s = analyze(frames, meta)
        h, w = frames.shape[1:]
        dirs.append(embody_one(name, title, tracks, meta["fps"], s.R_um, meta["signal"],
                               (w * meta["um_per_px"], h * meta["um_per_px"]), seconds))
    return dirs


def from_record(code, seconds):
    from seek.archive import Archive
    arc = Archive()
    rec = arc.get_record(code)
    if rec is None:
        sys.exit(f"No record {code} in {arc.dir}")
    return embody_one(code, f"{code} · {rec['title'] or rec['original_name']}",
                      seconds=seconds, **record_inputs(rec, arc.tracks(rec)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", help="record codes, e.g. SK-2610-001")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seconds", type=float, default=45.0)
    a = ap.parse_args()
    if a.demo:
        demo(a.seconds)
    for c in a.codes:
        from_record(c, a.seconds)
    if not a.demo and not a.codes:
        ap.print_help()
