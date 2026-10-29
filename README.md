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

## Prototype 2 · Repository (step 1: local)

When an analysis finishes, Seek asks **"Save this recording to the repository?"** (with an optional title and
note). **Save** keeps the video and its analysis as one record; **Not now** keeps nothing (a *Save to
repository…* button stays on the results if you change your mind). Saved records live on the **Repository**
page: open one to replay the video with its analysis layers, read the conditions and the six measures,
download the files, and keep notes on it.

```bash
streamlit run app.py               # Analyze → (save?) → Repository tab
python tools/seed_archive.py       # optional: 6 synthetic records so the Repository isn't empty
```

| Piece | File | What it does |
| --- | --- | --- |
| Storage | `seek/storage.py` | One interface (`put/get/exists/size/delete_prefix`). Today: `data/files/`. Step 3: Oracle Object Storage via `SEEK_STORAGE=oci` |
| Database | `seek/db.py` | SQLite at `data/seek.sqlite3`: `researchers`, `recordings`, `analyses`, `reviews`, plus `notes` and `audit_log` |
| Repository | `seek/archive.py` | `submit`, `list_records` (search + filters), `get_record`, `add_note`, `review`; files and rows are saved together or not at all |
| Specimen | `seek/thumb.py` | Each record's thumbnail: all of its tracks drawn as one ink silhouette |
| Pages | `app.py` | `main()` = Analyze (with the save pop-up); `archive_page()` = Repository sheet + record view (`?id=SK-2610-001`) |

A record (`SK-yymm-nnn`) stores: the original video, `tracks.parquet` (per cell per frame, incl. pixel
positions so layers can be redrawn), `summary.json`, `thumb.png`, and the annotated MP4 if rendered.
Every record is tied to a researcher (`R-0001`) and starts with status **submitted** (for the review flow in
step 2). Visibility: researchers see their own records and everyone's approved ones; admins see everything.

Until login lands (step 2) everything is saved under the local owner, set with `SEEK_OWNER_NAME` /
`SEEK_OWNER_EMAIL`. `SEEK_DATA_DIR` moves the whole repository. `data/` is git-ignored.

### Repo layout

```
app.py                 Streamlit app: Analyze page (main) + Repository page (archive_page)
seek/                  io · detect · track · metrics · pipeline · overlay · storage · db · archive · thumb
tools/make_test_video.py   synthetic clip with known answers (also a CLI)
tools/seed_archive.py      fill the repository with synthetic records
tests/test_metrics.py      accuracy checks
tests/test_archive.py      archive round-trip, visibility, failed-save cleanup
.streamlit/config.toml     dark theme
```

### Known limits

- Built for sparse footage. When two cells pass through each other the track usually breaks at the crossing
  (the identity test uses a 30-dot clip; a 50-dot random clip keeps ~75% of IDs).
- Still cells vanish with the median background step (by design: it removes debris).
- Real test clip: Illinois Data Bank, DOI 10.13012/B2IDB-4912922_V2, CC BY — trim to 10–20 s before upload;
  its fps and µm/px aren't published, so speeds are relative until calibrated.

## Prototype 3 · Seek → Swarm

On any saved record, **Send to swarm** turns the cells' movement into rules for a robot swarm (in body
lengths), shows them next to the cells' own alignment, rotation and spacing, and **Open live swarm** runs
them on the Robotarium simulator in a separate window, with sliders and click-to-override.

```bash
pip install -r requirements-swarm.txt     # once: simulator packages
streamlit run app.py                      # Repository → a record → Send to swarm
```

Details, terminal commands and where files are saved: [`swarm/README.md`](swarm/README.md).
