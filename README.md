# Project Seek

A multiscale discovery instrument for searching behavior across systems.

> Google lets us search what we can name; Seeker explores whether we can search what we can observe.

## Prototype 1 · Pattern Analysis

Upload one microscope video of swimming microbes → Seek tracks every cell, draws the analysis on the
video, and returns Tier 0 (calibration & conditions) and Tier 1 (six rule inputs) as cards, charts and files.

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
pytest -q                     # accuracy checks against synthetic clips with known answers
```

No clip handy? Pick **Synthetic demo** in the app (aligned / random / biased dots with known answers).

### How it works

| Step | File | What happens |
| --- | --- | --- |
| Read | `seek/io.py` | Video or .tif stack → grayscale frames; reads fps/size; downsizes big videos (µm/px corrected) |
| Detect | `seek/detect.py` | Median background subtract → invert if dark cells → blur → `trackpy` finds cells |
| Link | `seek/track.py` | `trackpy.link` (memory 3), drops tracks < 10 frames |
| Measure | `seek/metrics.py` | Tier 1 metrics, all in µm and seconds |
| Run | `seek/pipeline.py` | Runs stages in order in the background; each finished stage unlocks its layer + card |
| Draw | `seek/overlay.py` | Draws layers on frames from stored data (toggling never re-analyzes) |
| UI | `app.py` | Upload → analyze → results |

### Tier 1 outputs

| # | Metric | Unit | Swarm rule |
| --- | --- | --- | --- |
| 1 | Trajectory (tracks) | µm | Foundation |
| 2 | Velocity (speed, heading) | µm/s, ° | Movement |
| 3 | Nearest-neighbor distance | µm | Separation |
| 4 | Neighbor alignment (mean cos Δheading within R) | −1…1 | Alignment |
| 5 | Signal response (velocity along signal) | µm/s | Signal-following |
| 6 | Local density (neighbors within R) | count | Cohesion |

Headings: 0° = right, 90° = up. Exports: `tracks.csv` (per cell per frame), `summary.json`
(Tier 0 + six summary numbers + settings + file hash), annotated MP4 (H.264).

### Repo layout

```
app.py                 Streamlit page
seek/                  io · detect · track · metrics · pipeline · overlay
tools/make_test_video.py   synthetic clip with known answers (also a CLI)
tests/test_metrics.py      accuracy checks
.streamlit/config.toml     dark theme
```

### Known limits

- Built for sparse footage. When two cells pass through each other the track usually breaks at the crossing
  (the identity test uses a 30-dot clip; a 50-dot random clip keeps ~75% of IDs).
- Still cells vanish with the median background step (by design: it removes debris).
- Real test clip: Illinois Data Bank, DOI 10.13012/B2IDB-4912922_V2, CC BY — trim to 10–20 s before upload;
  its fps and µm/px aren't published, so speeds are relative until calibrated.
