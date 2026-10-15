"""Fill the archive with synthetic records so the Archive page has something to show.

    python tools/seed_archive.py            # 6 records into data/
    python tools/seed_archive.py --n 12     # more
    SEEK_DATA_DIR=/tmp/seek-demo python tools/seed_archive.py   # somewhere else

Every record is a real analysis of a synthetic clip with known answers (tools/make_test_video.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seek import detect, io                     # noqa: E402
from seek.archive import Archive                # noqa: E402
from seek.metrics import summary                # noqa: E402
from seek.pipeline import Settings, run_all     # noqa: E402
from tools.make_test_video import make_clip     # noqa: E402

MODES = [("aligned", "Aligned swimmers"), ("random", "Random walkers"), ("biased", "Biased toward signal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    arc = Archive()
    me = arc.owner()
    for i in range(a.n):
        mode, title = MODES[i % len(MODES)]
        n_dots = int(np.random.default_rng(i).integers(4, 30))
        frames, _, meta = make_clip(mode=mode, n_dots=n_dots, n_frames=90, seed=i + 10)
        s = Settings(um_per_px=meta["um_per_px"], fps=meta["fps"],
                     diameter=detect.odd_diameter(meta["cell_um"], meta["um_per_px"]), minmass=0,
                     R_um=30.0, signal_dir=meta["signal"])
        bg = detect.background(frames)
        probe = detect.locate(detect.preprocess(frames[0], bg, "bright"), s.diameter)
        s = Settings(**{**s.__dict__, "minmass": detect.suggest_minmass(probe["mass"])})
        df = run_all(frames, s)
        h, w = frames.shape[1:]
        tier0 = {"0a_scale_um_per_px": meta["um_per_px"],
                 "0b_file": {"fps": meta["fps"], "frames": len(frames), "duration_s": round(len(frames) / meta["fps"], 2),
                             "resolution_px": f"{w}×{h}", "analyzed_width_px": w},
                 "0c_conditions": {"organism": "Synthetic dots", "strain": mode, "medium": "—", "temperature_c": 25.0,
                                   "signal": "Synthetic bias" if meta["signal"] else "", "signal_dir": meta["signal"] or "none",
                                   "source": "make_test_video.py", "license": "Ours", "uploader": me["name"]}}
        video = io.write_mp4([np.repeat(f[..., None], 3, -1) for f in frames], meta["fps"])
        code = arc.submit(me["id"], video=video, video_name=f"synthetic_{mode}.mp4", tier0=tier0,
                          settings={**s.as_dict(), "max_width": 800}, summary=summary(df), tracks=df,
                          title=f"{title} · {n_dots} dots", note="Seeded synthetic record.")
        print(code, mode, df["cell_id"].nunique(), "tracks")


if __name__ == "__main__":
    main()
