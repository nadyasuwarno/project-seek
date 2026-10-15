"""Seek · Pattern Analysis — prototype 1.   Run:  streamlit run app.py

One page, three steps: upload → analyze → results. Analysis runs in a background
thread; each finished stage adds its layer to the video and its card to the panel.
"""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from seek import detect, io, overlay
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
def job_registry():
    return {}   # (file hash, Settings) → Job


def main():
    st.set_page_config(page_title="Seek · Pattern Analysis", page_icon="◎", layout="wide")
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
        data = up.getvalue()
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

    # ── exports ─────────────────────────────────────────────────────────────────────
    if len(done) == len(STAGES) and df is not None:
        st.divider()
        st.caption(f"Analyzed {n_frames} frames in {job.elapsed_s} s · {df['cell_id'].nunique()} tracks · "
                   f"contrast: {job.polarity} · diameter {job.s.diameter} px · minmass {job.s.minmass:.0f} · "
                   f"search range {job.s.search_range:.1f} px · R {job.s.R_um:g} µm")
        out = {"file": name, "file_hash": vid_hash, **tier0(), "tier1_summary": S,
               "settings": {**job.s.as_dict(), "polarity_used": job.polarity, "resize_scale": info["scale"]},
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



if __name__ == "__main__":   # streamlit runs the page as __main__; detection workers import it as __mp_main__ and skip this
    main()
