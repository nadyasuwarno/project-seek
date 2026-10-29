# Prototype 3 · Seek → Swarm

Seek measures how cells move. The swarm package turns one saved Seek record into rules a group of
robots can follow, and runs those robots on Georgia Tech's Robotarium simulator. Live, with sliders,
against the cells' own numbers.

```
Seek record ──► swarm.bridge ──► swarm_profile.json ──► swarm.live_tuner / swarm.run
 tracks + settings   measures rules     rules + cell targets       robots in the simulator
```

The thesis question it serves: **can we take back control of what we set free?**
See `docs/outlier_experiment.jpg` for the full experiment (instinct → training → set free → intervention).

## Set up (once)

Inside `project-seek`, with your environment active:

```bash
pip install -r requirements.txt          # Seek
pip install -r requirements-swarm.txt    # simulator: matplotlib, cvxopt, Robotarium
```

## Use it from Seek

1. `streamlit run app.py` → **Repository** → open a record.
2. **Send to swarm** measures the cells' rules and saves them with the record.
3. The page shows the rules (body lengths), the cells' targets, and how each rule was measured.
4. **Open live swarm** opens the simulator in its own window on your computer.
   Drag sliders, click the arena to take control, **Release** to hand it back, **Save profile** to keep your tuned rules.
   Seek then offers your tuned rules as a download on the record.

## Use it from a terminal

```bash
python -m swarm.bridge --list                              # saved records
python -m swarm.bridge SK-2610-001                         # fit + save with the record
python -m swarm.live_tuner --profile data/swarm/SK-2610-001.json
python -m swarm.run --profile data/swarm/SK-2610-001.json  # recorded 80 s run → data/swarm/runs/
python -m swarm.render data/swarm/runs/SK-2610-001 data/swarm/SK-2610-001.json   # video + chart (needs ffmpeg)
python -m swarm.fake_cells --mode couzin                   # test cells with known rules
```

## Where things are saved

| File | Made by | Contents |
|---|---|---|
| `data/files/records/<code>/swarm_profile.json` | Send to swarm / `swarm.bridge` | rules, cell targets, `fit` notes |
| `data/swarm/<code>.json` | Open live swarm | a copy the simulator opens |
| `data/swarm/<code>_tuned.json` | live tuner's Save profile | your slider values |
| `data/swarm/runs/<name>/` | `swarm.run`, `swarm.render` | trajectories (Seek's track format), metrics, summary, video |

`data/` stays out of git, like the rest of Seek's local data.

## What the bridge measures

| Rule | How | Confidence |
|---|---|---|
| Body length | Seek's detection diameter × µm/px | direct |
| Speed | median cell speed | direct |
| Turning + noise | heading change per body length travelled | direct |
| Repulsion | distance where cell pairs stop being rarer than chance | good; reads low for very fast cells |
| Orientation | distance where heading similarity fades | approximate |
| Attraction | assumed (one video can't show it) | tune live |

Cells cover far more body lengths per second than robots can, so the bridge keeps the path shape and slows
time down (`fit.time_scale`, e.g. 1 s robot ≈ 0.1 s cells).

## Files

| File | Role |
|---|---|
| `bridge.py` | Seek record or tracks → behavior profile |
| `sim.py` | rules engine (Couzin zones, steering, wheel limits, group metrics), no simulator needed |
| `live_tuner.py` | real-time window: arena, sliders, live charts with dashed cell targets, click-to-override |
| `run.py`, `render.py` | recorded runs, video and chart |
| `fake_cells.py` | synthetic cells with known rules |
| `profiles/` | placeholder rules and a fake-cell profile |

Tested in `tests/test_swarm.py`: known rules recovered from synthetic cells, a profile fitted from a real Seek
record stored with it, robot commands within wheel limits.

## Next

- **Light conditions in Seek**: detect the light's color, position and on/off times from each video.
- **Learning**: imitation model from cell tracks (instinct), reward training (the outlier goal), continued learning after release.
- **Intervention**: human-controlled lights and robots in the simulator; robot runs saved back to Seek.
