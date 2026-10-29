"""Seek — run:  streamlit run app.py

Two pages in this file: Analyze (`main`: upload → analyze → results → asked to save to the repository)
and Repository (`archive_page`: every saved recording, kept like a project file).
Analysis runs in a background thread; each finished stage adds its layer and its card.
"""
import base64
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from seek import detect, io, overlay
from seek.overlay import LAYERS
from seek.archive import Archive
from seek.db import STATUSES
from seek.metrics import EXPORT_COLS, summary
from seek.pipeline import STAGES, Job, Settings
from tools.make_test_video import make_clip

ss = st.session_state
STAGE_LAYER = {k: layer for k, layer, _ in STAGES}
SIGNAL_DIRS = ["none", "left", "right", "up", "down"]
ILLINOIS = "Illinois Data Bank, DOI 10.13012/B2IDB-4912922_V2, CC BY."


# ── cached loaders (keyed on file hash, so re-renders never re-read or re-analyze) ──
@st.cache_resource(max_entries=3, show_spinner="Reading video…")
def load_video(h, _data, name, max_width):
    return io.read_video(_data, name, max_width)


@st.cache_resource(max_entries=3, show_spinner="Generating synthetic clip…")
def load_synthetic(mode):
    frames, _, meta = make_clip(mode=mode, n_dots=40, n_frames=120, seed=1)
    h, w = frames.shape[1:]
    info = {"fps": meta["fps"], "n_frames": len(frames), "n_frames_in_file": len(frames),
            "width": w, "height": h, "scale": 1.0}
    return frames, info, meta


@st.cache_resource(max_entries=3)
def background(h, _frames):
    return detect.background(_frames)


@st.cache_resource
def archive():
    return Archive()        # data/seek.sqlite3 + data/files/ (SEEK_DATA_DIR / SEEK_STORAGE to change)


@st.cache_resource
def job_registry():
    return {}   # (file hash, Settings) → Job


def main():
    st.markdown("""<style>
    .block-container {padding-top: 2rem; max-width: 1500px;}
    .lbl {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: #8b96a3;}
    .big {font-size: 1.9rem; font-weight: 600; line-height: 1.15; color: #e8edf2;}
    .big span {font-size: .95rem; font-weight: 400; color: #8b96a3; margin-left: .25rem;}
    .muted {color: #6b7480; font-size: .95rem; padding: .35rem 0;}
    .stages {font-size: .8rem; color: #59626d; letter-spacing: .02em;}
    .stages b {color: #00e6ff; font-weight: 600;} .stages i {color: #c9d1d9; font-style: normal;}
    </style>""", unsafe_allow_html=True)

    # defaults live in session_state (not widget args) so code can prefill them safely
    DEFAULTS = dict(um_in=0.0, fps_in=0.0, cell_um=3.0, R=10.0, max_speed=50.0, polarity="auto", min_len=10,
                    max_width=800, speed_colors=False, temperature_c=25.0, signal_dir="none", organism="", strain="",
                    medium="", signal="", uploader="", license="", source="")
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
        ss[k] = ss[k]          # keep form values alive while their widgets are hidden (results view)
    if ss.get("toast"):
        st.toast(ss.pop("toast"))

    # ── 1 · Upload ──────────────────────────────────────────────────────────────────
    st.title("Seek")
    st.caption("Pattern analysis · prototype 1 — track every cell, measure the rule inputs.")

    src = st.radio("Source", ["Upload video", "Synthetic demo"], horizontal=True, label_visibility="collapsed")
    meta = None
    if src == "Upload video":
        up = st.file_uploader("Drop one microscope video", type=["mp4", "avi", "mov", "tif", "tiff"])
        if up is None:
            st.info("Drop an .mp4, .avi or .tif stack (sparse footage, cells mostly separate). "
                    "No clip handy? Switch to **Synthetic demo**.")
            st.stop()
        data = video_bytes = up.getvalue()
        vid_hash, name = io.file_hash(data), up.name
        try:
            frames, info = load_video(vid_hash, data, name, ss.get("max_width", 800))
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            st.stop()
    else:
        mode = st.segmented_control("Pattern", ["aligned", "random", "biased"], default="aligned",
                                    help="Dots with known speed, spacing and alignment — answers are known.")
        frames, info, meta = load_synthetic(mode or "aligned")
        video_bytes = None      # encoded to MP4 only if it's saved to the repository
        vid_hash, name = f"synthetic-{mode}", f"synthetic_{mode}.mp4"

    if ss.get("vid_hash") != vid_hash:          # new video → reset state, prefill form from file
        ss.update(vid_hash=vid_hash, phase="setup", job=None, seen=0, layers=[], frame=0,
                  playing=True, play_mode=True, mp4=None,
                  fps_in=float(info["fps"] or 0.0), um_in=0.0)
        if meta:
            ss.update(um_in=meta["um_per_px"], cell_um=meta["cell_um"], R=30.0, organism="Synthetic dots",
                      source="make_test_video.py", license="Ours", uploader="Seek",
                      signal_dir=meta["signal"] or "none", signal="Synthetic bias" if meta["signal"] else "")

    n_frames = len(frames)
    um_eff = (ss.um_in / info["scale"]) if ss.um_in else None        # µm/px of the analyzed frames
    fps = ss.fps_in or None

    # ── sidebar settings ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("Analysis settings")
        st.number_input("Expected cell size (µm)", 0.5, 100.0, step=0.5, key="cell_um",
                        help="Sets trackpy's feature diameter (rounded up to an odd number of px).")
        st.number_input("Neighbor radius R (µm)", 1.0, 500.0, step=1.0, key="R",
                        help="Used by alignment, density and neighbor links.")
        st.number_input("Max expected speed (µm/s)", 1.0, 1000.0, step=5.0, key="max_speed",
                        help="Sets how far a cell may move between frames when linking.")
        st.selectbox("Cell contrast", ["auto", "bright", "dark"], key="polarity",
                     format_func={"auto": "Auto-detect", "bright": "Bright cells on dark (fluorescence)",
                                  "dark": "Dark cells on bright (phase / brightfield)"}.get)
        st.number_input("Min track length (frames)", 3, 200, key="min_len")
        st.select_slider("Analyze at max width (px)", [480, 640, 800, 1024, 1600], key="max_width",
                         help="Large videos are downscaled for speed; µm/px is corrected automatically.")
        st.toggle("Color tails by speed", key="speed_colors", help="Cyan = slow → yellow = fast")
        st.divider()
        st.caption(f"Real test clip: {ILLINOIS}")

    diameter = detect.odd_diameter(ss.cell_um, um_eff) if um_eff else None


    def current_settings(minmass):
        return Settings(um_per_px=um_eff, fps=fps, diameter=diameter, minmass=float(minmass),
                        polarity=ss.polarity, max_speed_um_s=ss.max_speed, R_um=ss.R,
                        signal_dir=None if ss.get("signal_dir", "none") == "none" else ss.signal_dir,
                        min_len=int(ss.min_len))


    # ── video viewer (fragment: slider / toggles / playback refresh only this block) ──
    def solo(layer):
        ss.layers = [layer]


    def viewer():
        job: Job | None = ss.job if ss.phase == "results" else None
        if job and len(job.done) > ss.seen:                  # a stage just finished → layer emerges
            for k in job.done[ss.seen:]:
                ss.layers = [*ss.layers, STAGE_LAYER[k]]
            ss.seen = len(job.done)
            ss.toast = f"{STAGE_LAYER[job.done[-1]]} ready"
            st.rerun()                                        # full rerun so its metric card appears

        if ss.playing:
            ss.frame = (ss.frame + max(1, round((fps or 10) * 0.12))) % n_frames

        ready = [STAGE_LAYER[k] for k in (job.done if job else [])] + (["IDs"] if job and "link" in job.done else [])
        if job:
            st.pills("Layers", ready, selection_mode="multi", key="layers", label_visibility="collapsed")

        img = overlay.draw(frames[ss.frame], ss.frame, job.df if job else None, job.features if job else None,
                           ss.layers if job else [], um_eff, ss.R, job.s.signal_dir if job else None,
                           speed_colors=ss.speed_colors, vmax=ss.get("vmax"))
        st.image(img, width="stretch", output_format="JPEG")

        c1, c2, c3 = st.columns([1, 5, 1], vertical_alignment="center")
        c1.toggle("Play", key="playing")
        c2.slider("Frame", 0, n_frames - 1, key="frame", label_visibility="collapsed")
        if job:
            with c3.popover("Solo", disabled=len(ready) < 2, width="stretch"):
                for layer in ready:     # show just this layer, all others off
                    st.button(layer, key=f"solo-{layer}", on_click=solo, args=(layer,), width="stretch")
        t = f"{ss.frame / fps:.2f} s · " if fps else ""
        st.caption(f"{t}frame {ss.frame + 1} / {n_frames}")
        if ss.playing != ss.play_mode:                        # play state changed → reschedule the timer
            st.rerun()

        if job:
            parts = []
            for i, (k, _, label) in enumerate(STAGES):
                if k in job.done:
                    parts.append(f"<i>{label} ✓</i>")
                elif i == len(job.done) and not job.error:
                    pct = f" {job.progress:.0%}" if k == "detect" else ""
                    parts.append(f"<b>{label}{pct}…</b>")
                else:
                    parts.append(label)
            st.markdown(f"<div class='stages'>{' → '.join(parts)}</div>", unsafe_allow_html=True)


    def run_viewer():
        job = ss.job if ss.phase == "results" else None
        busy = job is not None and not job.finished
        ss.play_mode = ss.playing
        every = 0.12 if ss.playing else (0.4 if busy else None)
        st.fragment(viewer, run_every=every)()


    # ── helpers for the results panel ───────────────────────────────────────────────
    def spark(x, y, unit, signed=False):
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#00e6ff", width=1.5),
                                   hovertemplate=f"%{{x:.2f}} s · %{{y:.2f}} {unit}<extra></extra>"))
        fig.update_layout(height=70, margin=dict(l=0, r=0, t=4, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                          xaxis=dict(visible=False),
                          yaxis=dict(visible=False, rangemode="normal" if signed else "tozero"))
        if signed:
            fig.add_hline(0, line=dict(color="#3a424c", width=1, dash="dot"))
        return fig


    def card(title, value=None, unit="", note="", series=None, empty=None, pending=False, signed=False):
        with st.container(border=True):
            st.markdown(f"<div class='lbl'>{title}</div>", unsafe_allow_html=True)
            if pending:
                st.markdown("<div class='muted'>computing…</div>", unsafe_allow_html=True)
                return
            if empty:
                st.markdown(f"<div class='muted'>{empty}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='big'>{value}<span>{unit}</span></div>", unsafe_allow_html=True)
            st.caption(note)
            if series is not None and not empty and series.notna().any():
                st.plotly_chart(spark(series.index / fps, series.values, unit, signed), config={"displayModeBar": False},
                                key=f"spark-{title}")


    def tier0():
        return {
            "0a_scale_um_per_px": ss.um_in,
            "0b_file": {"fps": fps, "frames": n_frames, "duration_s": round(n_frames / fps, 2) if fps else None,
                        "resolution_px": f"{info['width']}×{info['height']}",
                        "analyzed_width_px": frames.shape[2]},
            "0c_conditions": {k: ss.get(k) for k in
                              ["organism", "strain", "medium", "temperature_c", "signal", "signal_dir",
                               "source", "license", "uploader"]},
        }


    # ── 2 · Setup (metadata + detection preview) ────────────────────────────────────
    if ss.phase == "setup":
        left, right = st.columns([3, 2], gap="large")
        with left:
            run_viewer()
        with right:
            st.markdown("**Tier 0 · Calibration & conditions**")
            st.caption(f"From file: {info['fps'] or '?'} fps · {info['n_frames_in_file']} frames · "
                       f"{info['width']}×{info['height']} px"
                       + (f" · analyzing at {frames.shape[2]} px wide" if info["scale"] < 1 else "")
                       + (f" · first {n_frames} frames" if n_frames < info["n_frames_in_file"] else ""))
            a, b = st.columns(2)
            a.number_input("µm per pixel *", 0.0, 100.0, step=0.01, format="%.4f", key="um_in")
            b.number_input("Frame rate (fps) *", 0.0, 10000.0, step=1.0, key="fps_in")
            a.text_input("Organism", key="organism", placeholder="E. coli")
            b.text_input("Strain", key="strain", placeholder="RP437")
            a.text_input("Medium", key="medium", placeholder="Motility buffer")
            b.number_input("Temperature (°C)", -10.0, 100.0, step=0.5, key="temperature_c")
            a.text_input("Signal applied", key="signal", placeholder="Serine gradient")
            b.selectbox("Signal direction", SIGNAL_DIRS, key="signal_dir")
            a.text_input("Uploader name", key="uploader")
            b.text_input("License", key="license", placeholder="CC BY")
            st.text_input("Source", key="source", placeholder="Lab / dataset / DOI")

        st.divider()
        if not (um_eff and fps):
            st.button("Analyze", type="primary", disabled=True)
            st.caption("Fill in **µm per pixel** and **frame rate** to preview detection and analyze.")
            st.stop()

        # detection preview on frame 1
        bg = background(vid_hash, frames)
        pol = ss.polarity if ss.polarity != "auto" else detect.guess_polarity(frames[0], bg)
        feats = detect.locate(detect.preprocess(frames[0], bg, pol), diameter)
        hi = float(feats["mass"].max()) if len(feats) else 1.0
        if ss.get("minmass_for") != (vid_hash, diameter, pol):
            ss.minmass = min(detect.suggest_minmass(feats["mass"]), hi)
            ss.minmass_for = (vid_hash, diameter, pol)
        p1, p2 = st.columns([3, 2], gap="large")
        with p2:
            st.markdown("**Detection preview · frame 1**")
            st.slider("Brightness threshold (minmass)", 0.0, max(hi, 1.0), key="minmass",
                      help="Lower finds fainter cells; raise it if noise is being marked.")
            kept = feats[feats["mass"] >= ss.minmass].assign(frame=0)
            st.markdown(f"<div class='big'>{len(kept)}<span>cells found</span></div>", unsafe_allow_html=True)
            st.caption(f"Diameter {diameter} px · contrast: {pol} cells · "
                       f"search range {current_settings(0).search_range:.1f} px")
            def start_analysis():
                s = current_settings(ss.minmass)
                reg = job_registry()
                if (vid_hash, s) not in reg or reg[(vid_hash, s)].error:
                    reg[(vid_hash, s)] = Job(frames, s).start()
                ss.update(job=reg[(vid_hash, s)], phase="results", seen=0, layers=[], mp4=None, playing=True)
                ss.pop("vmax", None)

            st.button("Analyze", type="primary", width="stretch", on_click=start_analysis)
        with p1:
            st.image(overlay.draw(frames[0], 0, None, kept, ["Markers"], um_eff), width="stretch")
        st.stop()

    # ── 3 · Results ─────────────────────────────────────────────────────────────────
    job: Job = ss.job
    if st.button("← Back to setup"):
        ss.update(phase="setup", job=None)
        st.rerun()
    if job.error:
        st.error(f"Analysis failed: {job.error}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        run_viewer()

    done = set(job.done)   # read stages first: df is always published before its stage is marked done
    df = job.df
    if df is not None and "speed_um_s" in df and "vmax" not in ss:
        ss.vmax = float(np.nanpercentile(df["speed_um_s"], 95)) or 1.0
    by_frame = df.groupby("frame") if df is not None else None


    def frame_mean(col):
        return by_frame[col].mean().reindex(range(n_frames)) if df is not None and col in df else None


    with right:
        t0 = tier0()
        with st.container(border=True):
            st.markdown("<div class='lbl'>Tier 0 · Calibration & conditions</div>", unsafe_allow_html=True)
            c = t0["0c_conditions"]
            st.markdown(
                f"**{c['organism'] or 'Unknown organism'}** {c['strain'] or ''}  \n"
                f"{ss.um_in:.4f} µm/px · {fps:g} fps · {n_frames} frames · {t0['0b_file']['duration_s']} s · "
                f"{t0['0b_file']['resolution_px']} px  \n"
                f"{c['medium'] or '—'} · {c['temperature_c']} °C · signal: {c['signal'] or 'none'} "
                f"({c['signal_dir']})  \n"
                f"<span style='color:#6b7480'>{c['uploader'] or '—'} · {c['source'] or '—'} · "
                f"{c['license'] or '—'}</span>", unsafe_allow_html=True)

        S = summary(df) if df is not None and "speed_um_s" in df else None
        has_pairs = df is not None and (df.groupby("frame").size() >= 2).any()

        def neighbor_card(title, key, stage, col, unit, fmt, note, signed=False):
            if stage not in done:
                return card(title, pending=True)
            empty = "needs 2+ cells" if not has_pairs else "no neighbors within R — try a larger R" \
                if S[key] is None else None
            card(title, None if empty else fmt(S[key]), unit, note, frame_mean(col), empty=empty, signed=signed)

        g1, g2 = st.columns(2)
        with g1:
            if "detect" in done:
                counts = job.features.groupby("frame").size().reindex(range(n_frames), fill_value=0)
                card("Cells per frame", f"{counts.median():.0f}", "median", "Detected cells, before linking.", counts)
            else:
                card("Cells per frame", pending=True)
            if "velocity" in done:
                card("2 · Velocity", f"{S['2_velocity_mean_speed_um_s']:.1f}", "µm/s",
                     "Mean swimming speed → movement rule.", frame_mean("speed_um_s"))
            else:
                card("2 · Velocity", pending=True)
            neighbor_card("4 · Alignment", "4_alignment_mean", "alignment", "alignment", "", lambda v: f"{v:+.2f}",
                          "1 same direction · 0 unrelated · −1 opposite.", signed=True)
            neighbor_card("6 · Local density", "6_local_density_mean", "density", "neighbors_in_R", f"in {ss.R:g} µm",
                          lambda v: f"{v:.1f}", "Neighbors within R → cohesion rule.")
        with g2:
            if "link" in done:
                tr = df.groupby("cell_id")["time_s"].agg(lambda t: t.max() - t.min())
                active = df.groupby("frame").size().reindex(range(n_frames))
                card("1 · Trajectory", f"{df['cell_id'].nunique()}", "tracks",
                     f"Mean track {tr.mean():.1f} s. Chart: active tracks over time.", active)
            else:
                card("1 · Trajectory", pending=True)
            neighbor_card("3 · Nearest neighbor", "3_nearest_neighbor_mean_um", "neighbors", "nn_dist_um", "µm",
                          lambda v: f"{v:.1f}", "Distance to the closest cell → separation rule.")
            if "signal" not in done:
                card("5 · Signal response", pending=True)
            elif not job.s.signal_dir:
                card("5 · Signal response", empty="no signal set", note="Set a signal direction in setup.")
            else:
                v = S["5_signal_drift_mean_um_s"]
                card("5 · Signal response", f"{v:+.1f}", "µm/s", f"Mean drift toward the signal ({job.s.signal_dir}).",
                     frame_mean("signal_drift_um_s"), signed=True)

    # ── save to the repository (prototype 2) ─────────────────────────────────────────
    def save_record(job, df, S, out, title, note):
        """Store the video + its analysis as one repository record, tied to the researcher."""
        arc = archive()
        me = arc.owner()                          # step 2 replaces this with the signed-in researcher
        video = video_bytes
        if video is None:                         # synthetic demo → store the frames as an MP4
            video = io.write_mp4([np.repeat(f[..., None], 3, -1) for f in frames], fps)
        ss.saved[(vid_hash, job.s)] = arc.submit(
            me["id"], video=video, video_name=name, tier0=tier0(), settings=out["settings"], summary=S,
            tracks=df, title=title, note=note, annotated_mp4=ss.mp4, elapsed_s=job.elapsed_s, file_hash=vid_hash)

    def close_ask():
        ss.ask_open = None

    @st.dialog("Save this recording to the repository?", width="medium", on_dismiss=close_ask)
    def ask_to_save(job, df, S, out):
        me = archive().owner()
        st.write("Analysis finished. **Save** keeps this video with its tracks, summary and thumbnail as one "
                 "record in the repository. **Not now** saves nothing.")
        title = st.text_input("Title", key="save_title", placeholder="E. coli · serine gradient · run 3")
        note = st.text_area("Researcher note", key="save_note", height=80,
                            placeholder="What you saw, what to check, anything odd about the clip…")
        st.caption(f"Saved under **{me['name']}** · {me['researcher_code']}")
        a, b = st.columns(2)
        if a.button("Save", type="primary", width="stretch"):
            with st.spinner("Saving…"):
                save_record(job, df, S, out, title, note)
            close_ask()
            st.rerun()
        if b.button("Not now", width="stretch"):
            close_ask()
            st.rerun()

    def repository_panel(job, df, S, out):
        """Asks once, automatically, when analysis finishes; afterwards shows the record or a Save button."""
        key = (vid_hash, job.s)
        saved, asked = ss.setdefault("saved", {}), ss.setdefault("asked", set())
        if key not in saved and key not in asked:     # first time this analysis is finished → ask
            asked.add(key)
            ss.ask_open = key
        if ss.get("ask_open") == key:                  # stays open across reruns until answered or closed
            ask_to_save(job, df, S, out)
        st.divider()
        with st.container(border=True):
            st.markdown("<div class='lbl'>Repository</div>", unsafe_allow_html=True)
            if key in saved:
                st.success(f"Saved to the repository as **{saved[key]}**")
                st.page_link(REPOSITORY, label=f"Open {saved[key]}", icon=":material/arrow_forward:",
                             query_params={"id": saved[key]})
            else:
                st.caption("Not saved to the repository.")
                if st.button("Save to repository…", type="primary"):
                    ss.ask_open = key
                    st.rerun()

    # ── exports ─────────────────────────────────────────────────────────────────────
    if len(done) == len(STAGES) and df is not None:
        st.divider()
        st.caption(f"Analyzed {n_frames} frames in {job.elapsed_s} s · {df['cell_id'].nunique()} tracks · "
                   f"contrast: {job.polarity} · diameter {job.s.diameter} px · minmass {job.s.minmass:.0f} · "
                   f"search range {job.s.search_range:.1f} px · R {job.s.R_um:g} µm")
        out = {"file": name, "file_hash": vid_hash, **tier0(), "tier1_summary": S,
               "settings": {**job.s.as_dict(), "polarity_used": job.polarity, "resize_scale": info["scale"],
                            "max_width": ss.max_width},
               "analysis_time_s": job.elapsed_s}
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download tracks.csv", df[EXPORT_COLS].round(4).to_csv(index=False),
                           "tracks.csv", "text/csv", width="stretch")
        d2.download_button("Download summary.json", json.dumps(out, indent=2, default=str),
                           "summary.json", "application/json", width="stretch")
        if ss.mp4:
            d3.download_button("Download annotated MP4", ss.mp4, f"seek_{vid_hash}.mp4", "video/mp4",
                               width="stretch")
        elif d3.button("Render annotated MP4 (current layers)", width="stretch"):
            with st.spinner("Rendering…"):
                ss.mp4 = io.write_mp4([overlay.draw(frames[i], i, df, job.features, ss.layers, um_eff, ss.R,
                                                    job.s.signal_dir, ss.speed_colors, ss.get("vmax"))
                                       for i in range(n_frames)], fps)
            st.rerun()
        repository_panel(job, df, S, out)



# ══ Repository page (prototype 2) ═════════════════════════════════════════════════════
# The sheet: featured sample, filters, ALL SAMPLES grid of trajectory silhouettes.
# One record: stored video replayed with analysis layers, conditions, measures, files, notes.
INK, PAPER, BAR, LINE = "#1b1b1b", "#e6e5e1", "#cfcfca", "#55575a"


INK, PAPER, BAR, LINE = "#1b1b1b", "#e6e5e1", "#cfcfca", "#55575a"


@st.cache_resource(max_entries=2, show_spinner="Opening the recording…")
def record_frames(code, _arc, key, name, max_width):
    return io.read_video(_arc.file(key), name, max_width)


@st.cache_data(max_entries=200, show_spinner=False)
def thumb_b64(_arc, key):
    return base64.b64encode(_arc.file(key)).decode() if key else ""


def archive_style():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap');
    .stApp, [data-testid="stHeader"] {{background: {PAPER};}}
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stCaptionContainer"], .stApp input, .stApp textarea {{
        font-family: 'Courier Prime', 'Courier New', monospace !important; color: {INK};}}
    [data-testid="stHeader"] a, [data-testid="stHeader"] span, [data-testid="stHeader"] p {{color: {INK} !important;}}
    .block-container {{max-width: 1180px; padding-top: 3.2rem;}}
    .stApp [data-testid="stCaptionContainer"] {{color: #5d5f62;}}
    /* inputs + buttons: pencil-on-paper */
    .stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"], .stApp [data-baseweb="select"] > div {{
        background: #f2f1ed !important; border: 1px dashed {LINE} !important; border-radius: 0 !important;}}
    .stApp input, .stApp textarea {{-webkit-text-fill-color: {INK}; background: transparent !important;}}
    .stApp [data-baseweb="select"] svg {{fill: {INK};}}
    .stApp .stButton > button, .stApp .stDownloadButton > button, .stApp [data-testid="stPopover"] button {{
        background: transparent; color: {INK}; border: 1px dashed {LINE}; border-radius: 0;
        font-family: 'Courier Prime', monospace;}}
    .stApp .stButton > button:hover, .stApp .stDownloadButton > button:hover {{background: {BAR}; color: {INK};}}
    .stApp [data-testid="stToggle"] p {{color: {INK};}}
    hr {{border-color: {LINE} !important; border-style: dashed !important;}}
    /* sheet pieces */
    .box {{border: 1px dashed {LINE}; position: relative; padding: 10px;}}
    .dot {{width: 9px; height: 9px; border-radius: 50%; position: absolute; top: 9px; left: 9px;}}
    .dot.blue {{background: #1f86b0;}} .dot.yellow {{background: #e3cf1c;}}
    .bar {{background: {BAR}; border: 1px dashed {LINE}; border-top: 0; font-size: .74rem; text-align: center;
           letter-spacing: .06em; padding: 2px 4px; text-transform: uppercase; white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis;}}
    .bar.top {{border-top: 1px dashed {LINE};}}
    .num {{border: 1px dashed {LINE}; background: {BAR}; width: 52px; height: 52px; display: flex;
           align-items: center; justify-content: center; font-size: 1.1rem;}}
    .head {{font-size: 1.15rem; letter-spacing: .05em; text-align: right; margin: 6px 0 14px;}}
    .strip {{background: {BAR}; border: 1px dashed {LINE}; padding: 4px 14px; letter-spacing: .08em;
             font-size: .95rem; margin-top: 18px;}}
    .specimen {{display: block; margin: 0 auto; max-width: 100%;}}
    .foot {{text-align: center; font-size: .9rem; letter-spacing: .1em; margin: 28px 0 8px;}}
    .stamp {{display: inline-block; border: 2px solid currentColor; padding: 1px 8px; font-weight: 700;
             letter-spacing: .12em; transform: rotate(-2deg); font-size: .8rem;}}
    .st-approved {{color: #2f6e3a;}} .st-rejected {{color: #9b2c22;}} .st-needs_changes {{color: #9a6a12;}}
    .st-submitted, .st-in_review {{color: #1f5f8a;}}
    .kv {{width: 100%; border-collapse: collapse; font-size: .86rem;}}
    .kv td {{border-bottom: 1px dashed #b9b9b4; padding: 3px 2px; vertical-align: top;}}
    .kv td:first-child {{color: #5d5f62; text-transform: uppercase; letter-spacing: .05em; font-size: .74rem;
                        width: 42%;}}
    .kv td.v {{text-align: right; font-weight: 700;}}
    .lbl {{font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: #5d5f62; margin: 6px 0 4px;}}
    .note {{border-left: 2px solid {INK}; padding: 2px 0 2px 10px; margin: 0 0 10px;}}
    .note small {{color: #5d5f62;}}
    [class*="st-key-cell-"] {{border: 1px dashed {LINE}; gap: 0 !important; padding-top: 4px;}}
    [class*="st-key-cell-"] .stButton > button {{background: {BAR}; border: 0; border-top: 1px dashed {LINE};
        width: 100%; font-size: .74rem; letter-spacing: .06em; min-height: 0; padding: 2px; text-transform: lowercase;}}
    </style>""", unsafe_allow_html=True)


def spaced(code):        # SK-2610-001 → SK 2610 001
    return code.replace("-", " ")


def seq(code):
    return int(code.rsplit("-", 1)[1])


def fmt(v, spec="{:.1f}", none="—"):
    return none if v is None else spec.format(v)


def kv(rows):
    body = "".join(f"<tr><td>{k}</td><td class='v'>{html.escape(str(v))}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{body}</table>"


def stamp(status):
    return f"<span class='stamp st-{status}'>{status.replace('_', ' ')}</span>"


def when(ts):
    return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")


def esc(v):
    return html.escape(str(v)) if v is not None else ""


def open_record(code):
    st.query_params["id"] = code


# ── the sheet ──────────────────────────────────────────────────────────────────────
def archive_sheet(arc, me):
    all_recs = arc.list_records(me)
    if not all_recs:
        st.markdown("<div class='head'>SEEK REPOSITORY : microorganisms 「—」</div>", unsafe_allow_html=True)
        st.markdown("<div class='strip'>ALL SAMPLES :</div>", unsafe_allow_html=True)
        st.markdown("<div class='box' style='border-top:0;padding:48px;text-align:center'>No samples yet. "
                    "Analyze a video and choose <b>Save</b> when it asks.</div>",
                    unsafe_allow_html=True)
        return

    lo, hi = min(all_recs, key=lambda r: r["id"]), max(all_recs, key=lambda r: r["id"])
    feat = next((r for r in all_recs if r["record_code"] == ss.get("archive_focus")), hi)

    # featured sample (left) · title + ratio diagram (right)
    c1, c2, c3 = st.columns([2.2, .6, 4.2])
    with c1:
        img = thumb_b64(arc, feat["thumb_key"])
        st.markdown(f"""<div class='box' style='height:170px;display:flex;align-items:center'>
            <span class='dot blue'></span><img class='specimen' style='max-height:150px'
            src='data:image/png;base64,{img}'></div>
            <div class='bar'>{spaced(feat['record_code'])}</div>
            <div class='bar'>TRACKS : {fmt(feat['n_tracks'], '{:d}')}</div>""", unsafe_allow_html=True)
        st.button("Open record →", key="open-feat", on_click=open_record, args=(feat["record_code"],), width="stretch")
    with c2:
        st.markdown(f"<div class='num'>{seq(feat['record_code']):02d}</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='head'>SEEK REPOSITORY : {esc((feat['organism'] or 'microorganisms').lower())} "
                    f"「{seq(lo['record_code']):02d}–{seq(hi['record_code']):02d}」</div>", unsafe_allow_html=True)
        st.markdown(ratio_box(feat), unsafe_allow_html=True)
        st.caption(f"{feat['title'] or feat['original_name']} · {feat['researcher_name']} "
                   f"{feat['researcher_code']} · {when(feat['created_at'])}")

    # filters
    st.markdown("<div class='strip'>FIND :</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([3, 2, 2, 1.3], vertical_alignment="bottom")
    q = f1.text_input("Search", placeholder="code, title, organism, medium, signal, researcher…",
                      label_visibility="collapsed")
    org = f2.selectbox("Organism", ["all organisms", *arc.organisms(me)], label_visibility="collapsed")
    status = f3.selectbox("Status", ["any status", *STATUSES], label_visibility="collapsed",
                          format_func=lambda s: s.replace("_", " "))
    mine = f4.toggle("Mine", value=False)
    recs = arc.list_records(me, q=q, organism=None if org == "all organisms" else org,
                            status=None if status == "any status" else status, mine_only=mine)

    # all samples grid
    st.markdown(f"<div class='strip'>ALL SAMPLES : <span style='float:right'>{len(recs)}</span></div>",
                unsafe_allow_html=True)
    if not recs:
        st.markdown("<div class='box' style='border-top:0;padding:30px;text-align:center'>Nothing matches.</div>",
                    unsafe_allow_html=True)
    for i in range(0, len(recs), 5):
        cols = st.columns(5, gap=None)
        for col, r in zip(cols, recs[i:i + 5]):
            with col, st.container(key=f"cell-{r['record_code']}"):
                dot = "<span class='dot blue'></span>" if r["record_code"] == feat["record_code"] else ""
                st.markdown(f"""<div style='height:132px;display:flex;align-items:center;position:relative'>{dot}
                    <img class='specimen' style='max-height:118px'
                    src='data:image/png;base64,{thumb_b64(arc, r["thumb_key"])}'></div>""", unsafe_allow_html=True)
                st.button(r["record_code"].lower(), key=f"b-{r['record_code']}", on_click=open_record,
                          args=(r["record_code"],), help=f"{r['title'] or r['original_name']} · {r['status']}")

    st.markdown(f"<div class='foot'>{spaced(feat['record_code'])}</div>", unsafe_allow_html=True)
    st.caption(f"{len(all_recs)} records · {arc.storage_used() / 1e6:.1f} MB of video stored locally")


def ratio_box(r):
    """Yellow-dot box: speed vs spacing. NN / V = seconds a cell needs to reach its nearest neighbor."""
    v, nn = r["mean_speed_um_s"], r["nn_mean_um"]
    if not v or not nn:
        inner, lab = "<text x='20' y='60' font-size='13'>needs speed + nearest neighbor</text>", "V : NN = —"
    else:
        k = 330 / max(v, nn)
        lv, ln = max(v * k, 6), max(nn * k, 6)
        inner = (f"<text x='92' y='34' font-size='11'>v</text>"
                 f"<line x1='104' y1='30' x2='{104 + lv * .55:.0f}' y2='22' stroke='{INK}'/>"
                 f"<text x='20' y='92' font-size='11'>nn</text>"
                 f"<line x1='38' y1='88' x2='{38 + ln:.0f}' y2='54' stroke='{INK}'/>")
        lab = f"V : NN = 1 : {nn / v:.1f}"
    return (f"<div class='box' style='height:118px'><span class='dot yellow'></span>"
            f"<svg viewBox='0 0 420 100' width='100%' height='100' style='font-family:Courier Prime,monospace'>"
            f"<g fill='{INK}' stroke-width='1.2'>{inner}</g></svg>"
            f"<div class='bar top' style='position:absolute;right:10px;bottom:10px;width:auto'>{lab}</div></div>")


# ── one record ─────────────────────────────────────────────────────────────────────
def record_page(arc, me, code):
    rec = arc.get_record(code, me)
    if st.button("← ALL SAMPLES"):
        st.query_params.clear()
        st.rerun()
    if rec is None:
        st.markdown(f"<div class='box'>No record <b>{esc(code)}</b> (or it isn't visible to you).</div>",
                    unsafe_allow_html=True)
        return
    ss.archive_focus = code
    s = rec["settings"]

    h1, h2 = st.columns([4, 1.4], vertical_alignment="bottom")
    h1.markdown(f"<div style='font-size:2rem;letter-spacing:.06em'>{spaced(code)}</div>"
                f"<div>{esc(rec['title'] or rec['original_name'])}</div>", unsafe_allow_html=True)
    h2.markdown(f"<div style='text-align:right'>{stamp(rec['status'])}<br><small>{esc(rec['researcher_name'])} · "
                f"{rec['researcher_code']}<br>{when(rec['created_at'])}</small></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:.6rem 0 1rem'>", unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        record_viewer(arc, rec, s)
    with right:
        st.markdown("<div class='lbl'>Conditions · Tier 0</div>", unsafe_allow_html=True)
        st.markdown(kv([
            ("organism", f"{rec['organism'] or '—'} {rec['strain'] or ''}"),
            ("medium", rec["medium"] or "—"),
            ("temperature", fmt(rec["temperature_c"], "{:g} °C")),
            ("signal", f"{rec['signal'] or 'none'} ({rec['signal_dir'] or 'none'})"),
            ("scale", fmt(rec["um_per_px"], "{:.4f} µm/px")),
            ("file", f"{fmt(rec['fps'], '{:g} fps')} · {rec['n_frames']} fr · {fmt(rec['duration_s'], '{:g} s')}"),
            ("resolution", f"{rec['width_px']}×{rec['height_px']} px"),
            ("source", f"{rec['source'] or '—'} · {rec['license'] or '—'}"),
        ]), unsafe_allow_html=True)
        st.markdown("<div class='lbl' style='margin-top:14px'>Measures · Tier 1</div>", unsafe_allow_html=True)
        st.markdown(kv([
            ("1 trajectory", f"{fmt(rec['n_tracks'], '{:d}')} tracks · {fmt(rec['mean_track_s'])} s"),
            ("2 velocity", fmt(rec["mean_speed_um_s"], "{:.1f} µm/s")),
            ("3 nearest neighbor", fmt(rec["nn_mean_um"], "{:.1f} µm")),
            ("4 alignment", fmt(rec["alignment_mean"], "{:+.2f}")),
            ("5 signal drift", fmt(rec["signal_drift_um_s"], "{:+.1f} µm/s")),
            ("6 local density", f"{fmt(rec['density_mean'])} in {fmt(rec['R_um'], '{:g}')} µm"),
        ]), unsafe_allow_html=True)
        speed_chart(arc, rec)

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)
    record_swarm(arc, rec, me)

    st.markdown("<hr style='margin:1.2rem 0'>", unsafe_allow_html=True)
    files_col, notes_col = st.columns([2, 3], gap="large")
    with files_col:
        record_files(arc, rec, me, s)
    with notes_col:
        record_notes(arc, rec, me)


def record_viewer(arc, rec, s):
    max_w = s.get("max_width") or rec["analyzed_width_px"] or 800
    try:
        frames, _ = record_frames(rec["record_code"], arc, rec["video_key"], rec["original_name"], max_w)
    except Exception as e:
        st.error(f"Couldn't open the stored video: {e}")
        return
    df = load_tracks(arc, rec)
    n = len(frames)
    vmax = float(np.nanpercentile(df["speed_um_s"], 95)) if len(df) else 1.0
    ss.setdefault("rec_layers", ["Tracks", "Markers"])
    ss.setdefault("rec_frame", 0)
    ss.setdefault("rec_play", False)
    ss._rec_play_mode = ss.rec_play       # the fragment's timer is fixed when it's defined → rerun on change
    st.pills("Layers", LAYERS, selection_mode="multi", key="rec_layers", label_visibility="collapsed")

    @st.fragment(run_every=0.12 if ss.rec_play else None)
    def player():
        if ss.rec_play:
            ss.rec_frame = (ss.rec_frame + max(1, round((rec["fps"] or 10) * 0.12))) % n
        ss.rec_frame = min(ss.rec_frame, n - 1)
        st.image(overlay.draw(frames[ss.rec_frame], ss.rec_frame, df, None, ss.rec_layers, s.get("um_per_px"),
                              s.get("R_um", 10.0), s.get("signal_dir"), vmax=vmax), width="stretch",
                 output_format="JPEG")
        c1, c2 = st.columns([1, 6], vertical_alignment="center")
        c1.toggle("Play", key="rec_play")
        c2.slider("Frame", 0, n - 1, key="rec_frame", label_visibility="collapsed")
        t = f"{ss.rec_frame / rec['fps']:.2f} s · " if rec["fps"] else ""
        st.caption(f"{t}frame {ss.rec_frame + 1} / {n} · redrawn from the stored tracks")
        if ss.rec_play != ss._rec_play_mode:
            st.rerun()

    player()


@st.cache_data(max_entries=4, show_spinner=False)
def _tracks(code, _arc, key):
    return _arc.tracks({"tracks_key": key})


def load_tracks(arc, rec):
    return _tracks(rec["record_code"], arc, rec["tracks_key"])


def speed_chart(arc, rec):
    df = load_tracks(arc, rec)
    if df.empty or not rec["fps"]:
        return
    m = df.groupby("time_s")["speed_um_s"].mean()
    fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines", line=dict(color=INK, width=1.3),
                               hovertemplate="%{x:.2f} s · %{y:.1f} µm/s<extra></extra>"))
    fig.update_layout(height=120, margin=dict(l=0, r=0, t=6, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Courier Prime, monospace", color=INK, size=10),
                      xaxis=dict(showgrid=False, ticksuffix=" s", color=INK),
                      yaxis=dict(showgrid=False, rangemode="tozero", color=INK))
    st.markdown("<div class='lbl' style='margin-top:14px'>Mean speed over time · µm/s</div>", unsafe_allow_html=True)
    st.plotly_chart(fig, config={"displayModeBar": False}, key="rec-speed")


# ── Embody (prototype 3): send this record to the robot swarm ─────────────────────────
def record_swarm(arc, rec, me):
    """Fit swarm rules from this record (swarm.bridge), show them, and open the live simulator."""
    from swarm import bridge                       # numpy/scipy only; the simulator is imported by the tuner
    code = rec["record_code"]
    st.markdown("<div class='lbl'>Embody · send to swarm</div>", unsafe_allow_html=True)

    def send():
        try:
            with st.spinner("Measuring the cells' rules…"):
                arc.attach_swarm_profile(code, me["id"], bridge.profile_for_record(arc, rec))
        except ValueError as e:
            ss.swarm_error = str(e)

    prof = arc.swarm_profile(code)
    if ss.get("swarm_error"):
        st.error(ss.pop("swarm_error"))
    if prof is None:
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        c1.button("Send to swarm", on_click=send, width="stretch")
        c2.caption("Turns this record's cell behaviour into rules a robot swarm can run, measured in body "
                   "lengths, plus the cells' group alignment, rotation and spacing as targets for the robots.")
        return

    f, t = prof["fit"], prof["targets"]
    rules, targets, actions = st.columns([1.15, 1, 1], gap="large")
    with rules:
        st.markdown("<div class='lbl'>Rules for the robots · body lengths</div>", unsafe_allow_html=True)
        st.markdown(kv([
            ("repulsion", f"{prof['zone_repulsion_bl']:g} BL"),
            ("orientation", f"{prof['zone_orientation_bl']:g} BL"),
            ("attraction (assumed)", f"{prof['zone_attraction_bl']:g} BL"),
            ("speed", f"{prof['speed_bl_per_s']:g} BL/s"),
            ("max turn", f"{prof['max_turn_rate_deg_per_s']:g} °/s"),
            ("heading noise", f"{prof['heading_noise_deg']:g} °"),
        ]), unsafe_allow_html=True)
    with targets:
        st.markdown("<div class='lbl'>Cell targets · what the group did</div>", unsafe_allow_html=True)
        st.markdown(kv([
            ("alignment", fmt(t.get("polarization"), "{:.2f}")),
            ("rotation", fmt(t.get("milling"), "{:.2f}")),
            ("spacing", fmt(t.get("nnd_bl"), "{:.2f} BL")),
            ("cells per frame", fmt(t.get("cells_per_frame"), "{:g}")),
            ("body length", f"{f['cell_length_um']:g} µm"),
            ("time scale", f"1 s robot ≈ {f['time_scale']:g} s cells"),
        ]), unsafe_allow_html=True)
    with actions:
        st.markdown("<div class='lbl'>Run it</div>", unsafe_allow_html=True)
        if st.button("Open live swarm", width="stretch", type="primary"):
            open_live_swarm(arc, code)
        st.download_button("swarm_profile.json", json.dumps(prof, indent=2), f"{code}_swarm_profile.json",
                           "application/json", width="stretch")
        st.button("Re-fit from this record", on_click=send, width="stretch")
        tuned = arc.dir / "swarm" / f"{code}_tuned.json"
        if tuned.exists():
            st.download_button("Your tuned rules (from the live swarm)", tuned.read_bytes(),
                               f"{code}_tuned.json", "application/json", width="stretch")
    with st.expander("How each rule was measured"):
        st.markdown(kv([
            ("body length", f["cell_length_from"]),
            ("repulsion", f["repulsion_note"]),
            ("orientation", f["orientation_note"]),
            ("attraction", f["attraction_note"]),
            ("turning + noise", f["turning_note"]),
            ("data used", f"{f['tracks_used']} tracks · {f['frames_used']} frames"),
        ]), unsafe_allow_html=True)


def open_live_swarm(arc, code):
    """The live swarm runs in its own window on this computer (Seek must be running locally)."""
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec("rps") is None:
        st.warning("The robot simulator isn't installed in this Python environment. In a terminal inside "
                   "project-seek, run:  pip install -r requirements-swarm.txt  — then press the button again.")
        return
    import os
    path = arc.swarm_profile_file(code)
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}   # Streamlit sets Agg (off-screen) for itself
    subprocess.Popen([sys.executable, "-m", "swarm.live_tuner", "--profile", str(path)],
                     cwd=str(Path(__file__).resolve().parent), env=env)
    st.toast(f"Opening the live swarm for {code} in its own window…")


def record_files(arc, rec, me, s):
    code = rec["record_code"]
    st.markdown("<div class='lbl'>Files</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='box' style='text-align:center'><img class='specimen' style='max-height:120px' "
                f"src='data:image/png;base64,{thumb_b64(arc, rec['thumb_key'])}'></div>"
                f"<div class='bar'>specimen · all tracks</div>", unsafe_allow_html=True)
    st.write("")
    ext = rec["video_key"].rsplit(".", 1)[-1]
    st.download_button(f"Video · {rec['video_bytes'] / 1e6:.1f} MB", arc.file(rec["video_key"]),
                       f"{code}.{ext}", width="stretch")
    st.download_button("tracks.parquet", arc.file(rec["tracks_key"]), f"{code}_tracks.parquet", width="stretch")
    st.download_button("tracks.csv", load_tracks(arc, rec).round(4).to_csv(index=False), f"{code}_tracks.csv",
                       "text/csv", width="stretch")
    st.download_button("summary.json", arc.file(rec["summary_key"]), f"{code}_summary.json", "application/json",
                       width="stretch")
    if rec["annotated_key"]:
        st.download_button("Annotated MP4", arc.file(rec["annotated_key"]), f"{code}_annotated.mp4", "video/mp4",
                           width="stretch")
    elif st.button("Render annotated MP4 (current layers)", width="stretch"):
        frames, _ = record_frames(code, arc, rec["video_key"], rec["original_name"],
                                  s.get("max_width") or rec["analyzed_width_px"] or 800)
        df = load_tracks(arc, rec)
        with st.spinner("Rendering…"):
            mp4 = io.write_mp4([overlay.draw(frames[i], i, df, None, ss.rec_layers, s.get("um_per_px"),
                                             s.get("R_um", 10.0), s.get("signal_dir")) for i in range(len(frames))],
                               rec["fps"] or 10)
            arc.attach_annotated(code, me["id"], mp4)
        st.rerun()
    with st.expander("Analysis settings"):
        st.json(s, expanded=False)


def record_notes(arc, rec, me):
    st.markdown("<div class='lbl'>Notes &amp; discussion</div>", unsafe_allow_html=True)
    if not rec["notes"]:
        st.caption("No notes yet.")
    for n in rec["notes"]:
        st.markdown(f"<div class='note'>{esc(n['body'])}<br><small>{esc(n['author_name'])} · {n['researcher_code']} · "
                    f"{when(n['created_at'])}</small></div>", unsafe_allow_html=True)

    def add():
        arc.add_note(rec["record_code"], me["id"], ss.get("new_note", ""))
        ss.new_note = ""

    st.text_area("Add a note", key="new_note", height=80, label_visibility="collapsed",
                 placeholder="Add a note: what you see, what to compare it with, what to re-run…")
    st.button("Add note", on_click=add)
    if rec["reviews"]:
        st.markdown("<div class='lbl' style='margin-top:14px'>Review history</div>", unsafe_allow_html=True)
        for v in rec["reviews"]:
            st.markdown(f"<div class='note'>{stamp(v['decision'])} {esc(v['comment'])}<br><small>"
                        f"{esc(v['reviewer_name'])} · {when(v['created_at'])}</small></div>", unsafe_allow_html=True)

def archive_page():
    """Repository (prototype 2): /repository is the specimen sheet, /repository?id=SK-… opens one record."""
    archive_style()
    arc = archive()
    me = arc.owner()          # step 2: the signed-in researcher (st.user) instead
    code = st.query_params.get("id")
    if code:
        record_page(arc, me, code)
    else:
        archive_sheet(arc, me)


if __name__ == "__main__":   # streamlit runs the page as __main__; detection workers import it as __mp_main__ and skip this
    st.set_page_config(page_title="Seek", page_icon="◎", layout="wide")
    REPOSITORY = st.Page(archive_page, title="Repository", icon=":material/inventory_2:", url_path="repository")
    st.navigation([st.Page(main, title="Analyze", icon=":material/biotech:", url_path="analyze", default=True),
                   REPOSITORY], position="top").run()
