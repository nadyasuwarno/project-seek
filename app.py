"""Seek — run:  streamlit run app.py

Seek Terminal  drop a microscope video → set its conditions → analyze → save to the Archives
Archives       every saved recording → open one → see what the robots take from it → Open in Simulator

Every screen says what data it hands to the next one, so the path from a cell video to the robots'
rules and light instinct stays visible.
"""
import base64
import html
import json
from pathlib import Path

import numpy as np
import streamlit as st

from seek import detect, io, overlay
from seek.archive import Archive
from seek.metrics import EXPORT_COLS, summary
from seek.pipeline import STAGES, Job, Settings
from tools.make_test_video import make_clip

ss = st.session_state
ROOT = Path(__file__).resolve().parent
LIGHT_FROM = {"none": "No light", "left": "Light from the left", "right": "Light from the right",
              "up": "Light from the top", "down": "Light from the bottom"}
LIGHT_COLORS = ["Blue", "Red", "White", "None"]
INK, MUTED, PANEL, EDGE = "#111111", "#8A8A8A", "#F3F3F3", "#DADADA"


# ── cached loaders ─────────────────────────────────────────────────────────────────
@st.cache_resource(max_entries=3, show_spinner="Reading video…")
def load_video(h, _data, name, max_width=800):
    return io.read_video(_data, name, max_width)


@st.cache_resource(max_entries=2, show_spinner="Making the demo clip…")
def load_demo():
    frames, _, meta = make_clip(mode="biased", n_dots=40, n_frames=120, seed=1, signal="right")
    h, w = frames.shape[1:]
    return frames, {"fps": meta["fps"], "n_frames": len(frames), "n_frames_in_file": len(frames),
                    "width": w, "height": h, "scale": 1.0}, meta


@st.cache_resource(max_entries=3)
def background(h, _frames):
    return detect.background(_frames)


@st.cache_resource
def archive():
    return Archive()


@st.cache_resource
def job_registry():
    return {}


@st.cache_resource(max_entries=2, show_spinner="Opening the recording…")
def record_frames(code, _arc, key, name, max_width):
    return io.read_video(_arc.file(key), name, max_width)


@st.cache_data(max_entries=200, show_spinner=False)
def thumb_b64(_arc, key):
    return base64.b64encode(_arc.file(key)).decode() if key else ""


@st.cache_data(max_entries=4, show_spinner=False)
def _tracks(code, _arc, key):
    return _arc.tracks({"tracks_key": key})


def esc(v):
    return html.escape(str(v)) if v is not None else ""


def fmt(v, spec="{:.1f}", none="—"):
    return none if v is None else spec.format(v)


# ── look ───────────────────────────────────────────────────────────────────────────
def style():
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea, .stApp li, .stApp td,
    .stApp th, [data-testid="stMarkdownContainer"] div, [data-testid="stMarkdownContainer"] span {{
        font-family: 'Inter', -apple-system, sans-serif;}}
    [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
        display: none;}}
    table.flow, table.kv {{border: 0 !important; table-layout: auto;}}
    table.flow th, table.flow td, table.kv td {{border-left: 0 !important; border-right: 0 !important;
        border-bottom: 0 !important; background: none !important;}}
    table.flow tr:first-child th {{border-top: 0 !important;}}
    .stApp {{background: #fff;}}
    [data-testid="stHeader"] {{background: transparent;}}
    .block-container {{max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem;}}
    .brand {{font-weight: 700; font-size: 1.6rem; line-height: 1.08; color: {INK}; letter-spacing: -.01em;}}
    /* navigation pills */
    [class*="st-key-nav-"] a {{border-radius: 999px; justify-content: center; padding: .5rem 1rem;}}
    [class*="st-key-nav-"] a p {{color: #fff !important; font-size: .9rem;}}
    .st-key-nav-on a {{background: {INK};}} .st-key-nav-off a {{background: #A3A3A3;}}
    .st-key-open-saved a {{background: {INK}; border-radius: 999px; justify-content: center;}}
    .st-key-open-saved a p {{color: #fff !important;}}
    /* layers: a small extra under the video */
    [class*="st-key-layers-"] button {{border-radius: 999px; font-size: .8rem; min-height: 0; padding: .2rem .8rem;}}
    .st-key-nav-off a:hover {{background: #8f8f8f;}}
    /* cards */
    .st-key-card {{background: {PANEL}; border: 1px solid {EDGE}; border-radius: 36px; padding: 2.2rem 2.4rem;}}
    .st-key-chip {{background: {PANEL}; border: 1px solid {EDGE}; border-radius: 999px; padding: .45rem 1.2rem;}}
    .chip {{display: inline-flex; align-items: center; gap: .6rem; background: #A3A3A3; color: #fff;
            border-radius: 999px; padding: .35rem 1rem; font-size: .85rem;}}
    .chip small {{opacity: .8; font-size: .7rem;}}
    /* drop zone */
    .st-key-drop [data-testid="stFileUploaderDropzone"] {{background: {PANEL}; border: 1px solid {EDGE};
        border-radius: 36px; min-height: 260px; display: flex; flex-direction: column; justify-content: center;
        align-items: center; gap: .8rem; position: relative;}}
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"] span,
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"] small {{display: none;}}
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"] svg {{width: 52px; height: 52px; color: #BDBDBD;}}
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"]::after {{content: "Drop what you see under your microscope";
        display: block; color: #BDBDBD; font-size: 1.5rem; margin-top: 3.2rem;}}
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"] > div {{display: flex; flex-direction: column;
        align-items: center;}}
    .st-key-drop [data-testid="stFileUploaderDropzoneInstructions"] svg {{display: none;}}
    /* the Upload button sits just above the text, in the middle of the box (whatever the Streamlit version) */
    .st-key-drop [data-testid="stFileUploaderDropzone"] button {{position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -110%); border-radius: 999px; border: 1px solid {EDGE}; background: #fff;
        color: {MUTED};}}
    /* type */
    .hero {{font-size: 6.4rem; font-weight: 700; letter-spacing: -.03em; line-height: 1; margin: 3rem 0 2rem;}}
    .q {{font-size: 1.5rem; margin-bottom: .5rem;}}
    .lede {{font-size: 1.3rem; line-height: 1.45; max-width: 1050px;}}
    .lbl {{font-size: .78rem; color: {MUTED}; font-weight: 600; margin: .2rem 0 .4rem;}}
    .note {{font-size: .78rem; color: {MUTED}; line-height: 1.45;}}
    .count {{font-size: 1.5rem; color: {MUTED};}} .count span {{font-size: .75rem; margin-left: .35rem;}}
    .stTextInput label p, .stNumberInput label p, .stSelectbox label p {{font-size: .75rem; color: {MUTED};}}
    .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div {{border-radius: 12px !important;
        background: #fff !important;}}
    .stButton button[kind="primary"] {{background: {INK}; border: 0; border-radius: 12px; min-height: 46px;}}
    .stButton button[kind="secondary"] {{border-radius: 999px;}}
    .stButton button[kind="primary"]:disabled {{background: #CFCFCF; color: #fff; opacity: 1;}}
    .live {{font-size: .95rem; margin-bottom: .5rem; display: flex; gap: .6rem; align-items: baseline;}}
    .live b {{font-size: 1.6rem; font-weight: 700;}} .live span {{color: {MUTED}; font-size: .8rem;
        margin-left: auto; margin-right: 2.6rem;}}
    /* steps strip on the landing page */
    .steps {{display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.4rem; margin-top: 3.2rem;
             border-top: 1px solid {EDGE}; padding-top: 1.4rem;}}
    .steps b {{display: block; font-size: .95rem; margin-bottom: .3rem;}}
    .steps span {{color: {MUTED}; font-size: .85rem; line-height: 1.45;}}
    .steps i {{font-style: normal; color: {MUTED}; font-size: .8rem;}}
    /* rule table (record page) */
    .flow {{width: 100%; border-collapse: collapse; font-size: .9rem;}}
    .flow th {{text-align: left; color: {MUTED}; font-weight: 600; font-size: .75rem; padding: 0 .6rem .5rem 0;}}
    .flow td {{border-top: 1px solid {EDGE}; padding: .55rem .6rem .55rem 0; vertical-align: top;}}
    .flow td.v {{font-weight: 600; white-space: nowrap;}} .flow td.arrow {{color: {MUTED}; width: 1.5rem;}}
    .flow tr.gap td {{color: {MUTED};}}
    .kv {{width: 100%; border-collapse: collapse; font-size: .9rem;}}
    .kv td {{border-top: 1px solid {EDGE}; padding: .45rem 0;}} .kv td:first-child {{color: {MUTED}; width: 44%;}}
    /* archive cards */
    [class*="st-key-sample-"] {{background: {PANEL}; border: 1px solid {EDGE}; border-radius: 28px;
        padding: 1.1rem 1.1rem 1.2rem;}}
    [class*="st-key-sample-"] img {{width: 100%; height: 150px; object-fit: contain; background: #CFCFCF;
        border-radius: 16px; display: block;}}
    [class*="st-key-sample-"] .stButton button {{background: none; border: 0; padding: 0; min-height: 0;
        color: {MUTED}; font-weight: 700; font-size: .95rem;}}
    [class*="st-key-sample-"] .stButton button:hover {{color: {INK};}}
    .pill {{display: inline-block; background: #CFCFCF; color: #fff; font-weight: 600; border-radius: 999px;
            padding: .45rem 1.6rem; font-size: .9rem;}}
    </style>""", unsafe_allow_html=True)


def header(active):
    left, a, b = st.columns([6, 1.3, 1.3], vertical_alignment="center")
    left.markdown("<div class='brand'>project<br>seek</div>", unsafe_allow_html=True)
    with a, st.container(key="nav-on" if active == "terminal" else "nav-off"):
        st.page_link(TERMINAL, label="Seek Terminal")
    with b, st.container(key="nav-on" if active == "archives" else "nav-off"):
        st.page_link(ARCHIVES, label="Archives")
    st.write("")


# ══ Seek Terminal ═════════════════════════════════════════════════════════════════════
def terminal():
    style()
    header("terminal")
    ss.setdefault("up_key", 0)
    for k, v in dict(organism="", uploader="", um_in=0.0, temperature=None, light_from="none",
                     light_color="None", fps_in=0.0, cell_um=3.0, preview=0).items():
        ss.setdefault(k, v)
        ss[k] = ss[k]              # keep values while their widgets are hidden

    source = ss.get("source")       # ("file", bytes, name) | ("demo",) | None
    if source is None:
        with st.container(key="drop"):
            up = st.file_uploader("Microscope video", type=["mp4", "avi", "mov", "tif", "tiff"],
                                  key=f"up-{ss.up_key}", label_visibility="collapsed")
        if up is not None:
            ss.source = ("file", up.getvalue(), up.name)
            st.rerun()
        if st.button("No clip yet? Use a demo clip", type="tertiary"):
            ss.source = ("demo",)
            st.rerun()
        landing()
        return

    # ── a video is loaded ──
    if source[0] == "file":
        data, name = source[1], source[2]
        vid_hash = io.file_hash(data)
        try:
            frames, info = load_video(vid_hash, data, name)
        except Exception as e:
            st.error(f"Couldn't read this file: {e}")
            clear_video()
            return
        meta, video_bytes, size = None, data, f"{len(data) / 1e6:.1f} MB"
    else:
        frames, info, meta = load_demo()
        vid_hash, name, video_bytes, size = "demo-blue", "demo_cells_blue_light.mp4", None, "demo"

    if ss.get("vid_hash") != vid_hash:      # new video → fresh state, prefilled from the file
        ss.update(vid_hash=vid_hash, phase="setup", job=None, saved_code=None, save_pending=False,
                  setup_frame=0, result_frame=0,
                  fps_in=float(info["fps"] or 0.0))
        if meta:
            ss.update(organism="Demo cells", uploader="Seek demo", um_in=meta["um_per_px"], temperature=25.0,
                      light_from="right", light_color="Blue", cell_um=meta["cell_um"])

    chip(name, size)
    if ss.phase == "setup":
        setup(frames, info, vid_hash)
    else:
        results(frames, info, vid_hash, name, video_bytes)


def clear_video():
    ss.source, ss.vid_hash, ss.phase, ss.job = None, None, "setup", None
    ss.up_key += 1


def chip(name, size):
    with st.container(key="chip"):
        c1, c2 = st.columns([10, 1], vertical_alignment="center")
        c1.markdown(f"<span class='chip'>▣ {esc(name)} <small>{esc(size)}</small></span>", unsafe_allow_html=True)
        c2.button("✕", key="clear", help="Remove this video", on_click=clear_video)
    st.write("")


def landing():
    st.markdown("<div class='hero'>Seek</div>"
                "<div class='q'>Can you take back what you’ve set free?</div>"
                "<div class='lede'>Every living thing begins with instinct, a pull it never chose. Some learn to "
                "want something else and become outliers. Seek asks what happens next: once something has chosen "
                "a path against its nature, can the world still call it back, and does it answer?</div>"
                "<div class='lede' style='margin-top:1.2rem'>The robots were taught to go to the blue light. "
                "The red light is brighter and easier to notice.</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='steps'>
      <div><i>1</i><b>Drop a video</b><span>Seek finds and follows every cell in the microscope footage.</span></div>
      <div><i>2</i><b>Set the conditions</b><span>Scale, and the light the cells saw: its colour and
        where it came from.</span></div>
      <div><i>3</i><b>Analyze and save</b><span>Speed, spacing, alignment and the pull toward the light
        are stored in the Archives.</span></div>
      <div><i>4</i><b>Open in Simulator</b><span>Robots take those numbers as their rules and their
        instinct. Then you drop the blue and red lights.</span></div>
    </div>""", unsafe_allow_html=True)


def settings_for(info, minmass):
    um_eff = ss.um_in / info["scale"] if ss.um_in else None
    return Settings(um_per_px=um_eff, fps=ss.fps_in or None,
                    diameter=detect.odd_diameter(ss.cell_um, um_eff) if um_eff else 0, minmass=float(minmass),
                    R_um=max(10.0, 3 * ss.cell_um),
                    signal_dir=None if ss.light_from == "none" else ss.light_from)


def play_step(fps):
    return max(1, round((fps or 10) * 0.15))       # frames to advance per refresh, so playback runs near real time


def video_player(key, n, fps, draw, count_text, layers=None, default_layers=("Tracks", "Markers")):
    """Every video in Seek: live count line on top, the frame, then Play + frame slider.
    `draw(f, layers)` returns the image for frame f; `count_text(f)` the live line above it.
    `layers`: overlays available to switch on (an optional extra, tucked into a small Layers menu)."""
    ss.setdefault(f"{key}_play", True)
    if layers:
        if not ss.get(f"_{key}_layers_set"):          # first time layers exist → switch the defaults on
            ss[f"{key}_layers"] = [x for x in default_layers if x in layers]
            ss[f"_{key}_layers_set"] = True
            ss[f"_{key}_layers_ver"] = ss.get(f"_{key}_layers_ver", 0) + 1
        ss[f"{key}_layers"] = [x for x in ss.get(f"{key}_layers", []) if x in layers]
    ss.setdefault(f"{key}_frame", 0)
    ss[f"_{key}_mode"] = ss[f"{key}_play"]

    @st.fragment(run_every=0.15 if ss[f"{key}_play"] else None)
    def player():
        if ss[f"{key}_play"]:
            ss[f"{key}_frame"] = (ss[f"{key}_frame"] + play_step(fps)) % n
        f = min(ss[f"{key}_frame"], n - 1)
        st.markdown(f"<div class='live'>{count_text(f)}</div>", unsafe_allow_html=True)
        st.image(draw(f, ss.get(f"{key}_layers")), width="stretch", output_format="JPEG")
        c1, c2, c3 = st.columns([1.3, 3.4, 1.5] if layers else [1.3, 4.9, 0.01], vertical_alignment="center")
        playing = c1.toggle("Play", value=ss[f"{key}_play"], key=f"{key}_play_{ss[f'{key}_play']}")
        ss[f"{key}_play"] = playing
        c2.slider("Frame", 0, n - 1, key=f"{key}_frame", label_visibility="collapsed")
        if layers:
            with c3, st.container(key=f"layers-{key}"), st.popover("Layers", width="stretch"):
                chosen = st.pills("Extra views of what Seek measured", layers, selection_mode="multi",
                                  default=ss[f"{key}_layers"],
                                  key=f"{key}_pills_{ss.get(f'_{key}_layers_ver', 0)}")
                ss[f"{key}_layers"] = list(chosen or [])
        if ss[f"{key}_play"] != ss[f"_{key}_mode"]:   # play state changed → restart the timer
            st.rerun()
    player()


def setup(frames, info, vid_hash):
    n = len(frames)
    with st.container(key="card"):
        left, right = st.columns([1.1, 1], gap="large")
        with right:
            a, b = st.columns(2)
            a.text_input("Organism Name *", key="organism", placeholder="e.g. Euglena gracilis")
            b.text_input("Uploader Name *", key="uploader", placeholder="Your name")
            a.number_input("µm per pixel *", 0.0, 100.0, step=0.01, format="%.4f", key="um_in")
            b.number_input("Temperature (°C)", -10.0, 100.0, step=0.5, key="temperature")
            a.selectbox("Signal Applied *", list(LIGHT_FROM), key="light_from", format_func=LIGHT_FROM.get)
            b.selectbox("Light Color *", LIGHT_COLORS, key="light_color")
            st.markdown("<div class='note'>The light the cells saw. Seek measures how strongly they move toward "
                        "it, and the Simulator gives the robots that pull as their instinct for this colour.</div>",
                        unsafe_allow_html=True)
            with st.expander("Calibration", expanded=not info["fps"]):
                c, d = st.columns(2)
                c.number_input("Frame rate (fps) *", 0.0, 10000.0, step=1.0, key="fps_in")
                d.number_input("Cell size (µm)", 0.5, 100.0, step=0.5, key="cell_um",
                               help="One body length. The robots' distances are measured in it.")

            # detection preview needs only the scale; Analyze needs every required field
            preview = ss.um_in > 0
            if preview:
                s0 = settings_for(info, 0)
                bg = background(vid_hash, frames)
                pol = detect.guess_polarity(frames[0], bg)
                first = detect.locate(detect.preprocess(frames[0], bg, pol), s0.diameter)
                hi = float(first["mass"].max()) if len(first) else 1.0
                if ss.get("minmass_for") != (vid_hash, s0.diameter):
                    ss.minmass = min(detect.suggest_minmass(first["mass"]), hi) if len(first) else 0.0
                    ss.minmass_for = (vid_hash, s0.diameter)
                st.markdown("<div class='lbl' style='margin-top:1.2rem'>Brightness threshold</div>"
                            "<div class='note'>Lower finds fainter cells. Raise it if noise is being marked. "
                            "The count on the left updates as you move it.</div>", unsafe_allow_html=True)
                st.slider("Brightness threshold", 0.0, max(hi, 1.0), key="minmass", label_visibility="collapsed")
            else:
                st.markdown("<div class='lbl' style='margin-top:1.2rem'>Brightness threshold</div>"
                            "<div class='note'>Fill in µm per pixel to see which cells Seek finds.</div>",
                            unsafe_allow_html=True)

            missing = [f for f, ok in (("organism name", ss.organism.strip()), ("uploader name", ss.uploader.strip()),
                                       ("µm per pixel", ss.um_in), ("frame rate", ss.fps_in)) if not ok]

            def start():
                s = settings_for(info, ss.minmass)
                reg = job_registry()
                if (vid_hash, s) not in reg or reg[(vid_hash, s)].error:
                    reg[(vid_hash, s)] = Job(frames, s).start()
                ss.update(job=reg[(vid_hash, s)], phase="results", seen=0, result_frame=0, saved_code=None,
                          save_pending=False, _result_layers_set=False)

            st.button("Analyze", type="primary", width="stretch", on_click=start, disabled=bool(missing))
            if missing:
                st.markdown(f"<div class='note'>To analyze, fill in: {', '.join(missing)}.</div>",
                            unsafe_allow_html=True)
        with left:
            um_eff = ss.um_in / info["scale"] if ss.um_in else None
            found = {}

            def detect_frame(f):
                if f not in found:
                    feats = detect.locate(detect.preprocess(frames[f], bg, pol), s0.diameter)
                    found[f] = feats[feats["mass"] >= ss.minmass].assign(frame=f)
                return found[f]

            def draw(f, _layers=None):
                kept = detect_frame(f) if preview else None
                return overlay.draw(frames[f], f, None, kept, ["Markers"] if preview else [], um_eff)

            def count(f):
                if not preview:
                    return "<span>Fill in µm per pixel to detect cells</span>"
                return f"<b>{len(detect_frame(f))}</b> cells detected<span>frame {f + 1} of {n}</span>"

            video_player("setup", n, ss.fps_in or info["fps"], draw, count)
            st.markdown(f"<div class='note'>{info['n_frames_in_file']} frames · {info['width']}×{info['height']} px"
                        f"{' · ' + format(info['fps'], 'g') + ' fps' if info['fps'] else ''}</div>",
                        unsafe_allow_html=True)


def conditions():
    color = ss.light_color if ss.light_color != "None" and ss.light_from != "none" else None
    return {"organism": ss.organism.strip(), "strain": "", "medium": "", "temperature_c": ss.temperature,
            "signal": f"{color} light" if color else "", "signal_dir": ss.light_from if color else "none",
            "light_color": color, "source": "", "license": "", "uploader": ss.uploader.strip()}


def results(frames, info, vid_hash, name, video_bytes):
    job: Job = ss.job
    n = len(frames)
    um_eff = job.s.um_per_px
    st.button("← Change conditions", on_click=lambda: ss.update(phase="setup", job=None))
    if job.error:
        st.error(f"Analysis failed: {job.error}")
    with st.container(key="card"):
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            ready = [layer for k, layer, _ in STAGES if k in job.done] + (["IDs"] if "link" in job.done else [])

            def draw(f, layers):
                linked = "link" in job.done
                layers = layers or (["Markers"] if job.features is not None else [])
                return overlay.draw(frames[f], f, job.df if linked else None, job.features, layers, um_eff,
                                    job.s.R_um, job.s.signal_dir)

            def count(f):
                if "link" in job.done and job.df is not None:
                    k = int((job.df["frame"] == f).sum())
                    return f"<b>{k}</b> cells tracked<span>frame {f + 1} of {n}</span>"
                if job.features is not None:
                    k = int((job.features["frame"] == f).sum())
                    return f"<b>{k}</b> cells detected<span>frame {f + 1} of {n} · linking…</span>"
                return f"<span>Detecting cells… {job.progress:.0%}</span>"

            video_player("result", n, job.s.fps, draw, count, layers=ready)
        with right:
            # refreshes itself while the analysis runs; one full refresh when it finishes
            @st.fragment(run_every=None if job.finished else 0.5)
            def panel():
                measured(job)
                if job.finished and ss.get("seen") != "done":
                    ss.seen = "done"
                    st.rerun()
            panel()
    if not job.error:
        save_box(job, frames, info, vid_hash, name, video_bytes)


def measured(job):
    """What Seek measured, said as what the robots will take from it."""
    done = set(job.done)
    if len(done) < len(STAGES):
        step = STAGES[len(done)][2] if len(done) < len(STAGES) else ""
        pct = f" {job.progress:.0%}" if not done else ""
        st.markdown(f"<div class='lbl'>Analyzing · {esc(step)}{pct}…</div>"
                    f"<div class='note'>Finding every cell, following it frame to frame, then measuring how the "
                    f"group moves.</div>", unsafe_allow_html=True)
        st.progress(len(done) / len(STAGES))
        return
    S = summary(job.df)
    tr = S["1_trajectory"]
    color = conditions()["light_color"]
    drift, speed = S["5_signal_drift_mean_um_s"], S["2_velocity_mean_speed_um_s"]
    pull = drift / speed if (drift is not None and speed) else None
    rows = [("Speed", fmt(speed, "{:.1f} µm/s"), "how fast the robots move"),
            ("Spacing", fmt(S["3_nearest_neighbor_mean_um"], "{:.1f} µm"), "their personal space"),
            ("Alignment", fmt(S["4_alignment_mean"], "{:+.2f}"), "how much they copy their neighbours"),
            ("Light response", fmt(pull, "{:+.2f}") if color else "no light set",
             f"their instinct for the {color.lower()} light" if color else "no light instinct")]
    st.markdown(f"<div class='lbl'>What Seek measured · {tr['tracks']} cells followed</div>"
                "<table class='flow'><tr><th>Measured</th><th></th><th></th><th>Becomes, for the robots</th></tr>"
                + "".join(f"<tr><td>{a}</td><td class='v'>{b}</td><td class='arrow'>→</td><td>{c}</td></tr>"
                          for a, b, c in rows) + "</table>"
                "<div class='note' style='margin-top:.8rem'>Light response = drift toward the light ÷ speed: "
                "+1 every cell swims straight at it, 0 no preference.</div>", unsafe_allow_html=True)


def save_box(job, frames, info, vid_hash, name, video_bytes):
    st.write("")
    arc = archive()

    def save():
        me = arc.owner()
        video = video_bytes or io.write_mp4([np.repeat(f[..., None], 3, -1) for f in frames], job.s.fps)
        n = len(frames)
        tier0 = {"0a_scale_um_per_px": ss.um_in,
                 "0b_file": {"fps": job.s.fps, "frames": n, "duration_s": round(n / job.s.fps, 2),
                             "resolution_px": f"{info['width']}×{info['height']}", "analyzed_width_px": frames.shape[2]},
                 "0c_conditions": conditions()}
        settings = {**job.s.as_dict(), "polarity_used": job.polarity, "resize_scale": info["scale"]}
        with st.spinner("Saving to the Archives…"):
            ss.saved_code = arc.submit(me["id"], video=video, video_name=name, tier0=tier0, settings=settings,
                                       summary=summary(job.df), tracks=job.df, title=ss.get("save_title", ""),
                                       note="", elapsed_s=job.elapsed_s, file_hash=vid_hash)
        ss.save_pending = False

    if ss.get("save_pending") and job.finished and not ss.get("saved_code"):
        save()                                   # pressed while analysing → saved the moment it finished
    if ss.get("saved_code"):
        c1, c2 = st.columns([3, 1.3], vertical_alignment="center")
        c1.markdown(f"Saved to the Archives as **{ss.saved_code}**. Open it there to send it to the Simulator.")
        with c2, st.container(key="open-saved"):
            st.page_link(ARCHIVES, label=f"Open {ss.saved_code} →", query_params={"id": ss.saved_code})
        return

    c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
    c1.text_input("Save this recording to the Archives?", key="save_title",
                  placeholder=f"{ss.organism} · {conditions()['signal'] or 'no light'}")
    if ss.get("save_pending"):
        c2.button("Saving when ready…", type="primary", width="stretch", disabled=True)
        st.markdown("<div class='note'>It will be saved automatically the moment the analysis finishes.</div>",
                    unsafe_allow_html=True)
        return
    if c2.button("Save to Archives", type="primary", width="stretch"):
        if job.finished:
            save()
        else:
            ss.save_pending = True
        st.rerun()
    st.markdown("<div class='note'>Saving keeps the video with every cell's track, so it can be opened in the "
                "Simulator from the Archives. You can press it now, even while Seek is still analysing.</div>",
                unsafe_allow_html=True)


# ══ Archives ══════════════════════════════════════════════════════════════════════════
def archives():
    style()
    header("archives")
    arc = archive()
    me = arc.owner()
    code = st.query_params.get("id")
    if code:
        record_page(arc, me, code)
    else:
        samples(arc, me)


def open_record(code):
    st.query_params["id"] = code


def samples(arc, me):
    recs = arc.list_records(me)
    st.markdown(f"<span class='pill'>All Samples · {len(recs)}</span>", unsafe_allow_html=True)
    st.write("")
    if not recs:
        st.markdown("<div class='note'>No samples yet. Analyze a video in the Seek Terminal and save it.</div>",
                    unsafe_allow_html=True)
        return
    for i in range(0, len(recs), 4):
        cols = st.columns(4)
        for col, r in zip(cols, recs[i:i + 4]):
            with col, st.container(key=f"sample-{r['record_code']}"):
                st.markdown(f"<img src='data:image/png;base64,{thumb_b64(arc, r['thumb_key'])}'>",
                            unsafe_allow_html=True)
                st.button(r["record_code"], key=f"open-{r['record_code']}", on_click=open_record,
                          args=(r["record_code"],))
                st.markdown(f"<div class='note' style='font-size:.95rem;margin-top:-.4rem'>"
                            f"{esc(r['organism'] or 'Unnamed organism')}</div>", unsafe_allow_html=True)


def record_page(arc, me, code):
    from swarm import bridge
    rec = arc.get_record(code, me)
    if st.button("← All Samples"):
        st.query_params.clear()
        st.rerun()
    if rec is None:
        st.markdown(f"No sample **{esc(code)}**.")
        return
    st.markdown(f"<div style='font-size:2.6rem;font-weight:700;letter-spacing:-.02em'>{esc(code)}</div>"
                f"<div class='q' style='color:{MUTED}'>{esc(rec['organism'] or 'Unnamed organism')}</div>",
                unsafe_allow_html=True)

    prof = arc.swarm_profile(code)
    if prof is None or "instinct" not in prof:          # fit once (older profiles lack the light instinct)
        try:
            with st.spinner("Turning the cells' movement into robot rules…"):
                prof = bridge.profile_for_record(arc, rec)
            arc.attach_swarm_profile(code, me["id"], prof)
        except ValueError as e:
            prof = None
            st.warning(f"Can't make robot rules from this sample: {e}")

    with st.container(key="card"):
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            record_player(arc, rec)
        with right:
            c = json.loads(arc.file(rec["summary_key"])).get("0c_conditions", {}) if rec["summary_key"] else {}
            light = (f"{rec['signal']} · {LIGHT_FROM.get(rec['signal_dir'], '').lower()}" if rec["signal"]
                     else "no light")
            st.markdown("<div class='lbl'>Conditions</div><table class='kv'>" + "".join(
                f"<tr><td>{k}</td><td>{esc(v)}</td></tr>" for k, v in [
                    ("Organism", rec["organism"] or "—"), ("Uploader", c.get("uploader") or rec["researcher_name"]),
                    ("Light", light), ("Scale", fmt(rec["um_per_px"], "{:.4f} µm per pixel")),
                    ("Temperature", fmt(rec["temperature_c"], "{:g} °C")),
                    ("Video", f"{rec['n_frames']} frames · {fmt(rec['fps'], '{:g} fps')}"),
                    ("Cells followed", fmt(rec["n_tracks"], "{:d}"))]) + "</table>", unsafe_allow_html=True)

    if prof:
        st.write("")
        cells_to_robots(prof)
        simulator_launch(arc, code, prof)
        fidelity(arc, rec, me, prof)

    st.write("")
    st.markdown("<div class='lbl'>Files</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    ext = rec["video_key"].rsplit(".", 1)[-1]
    f1.download_button("Video", arc.file(rec["video_key"]), f"{code}.{ext}", width="stretch")
    f2.download_button("tracks.csv", _tracks(code, arc, rec["tracks_key"]).round(4).to_csv(index=False),
                       f"{code}_tracks.csv", "text/csv", width="stretch")
    f3.download_button("summary.json", arc.file(rec["summary_key"]), f"{code}_summary.json", "application/json",
                       width="stretch")
    if prof:
        f4.download_button("simulator profile", json.dumps(prof, indent=2), f"{code}_profile.json",
                           "application/json", width="stretch")
    notes(arc, rec, me)


def record_player(arc, rec):
    s = rec["settings"]
    try:
        frames, _ = record_frames(rec["record_code"], arc, rec["video_key"], rec["original_name"],
                                  s.get("max_width") or rec["analyzed_width_px"] or 800)
    except Exception as e:
        st.error(f"Couldn't open the stored video: {e}")
        return
    df = _tracks(rec["record_code"], arc, rec["tracks_key"])
    n = len(frames)
    per_frame = df.groupby("frame").size()

    have = {"Velocity": "speed_um_s", "Neighbors": "nn_dist_um", "Alignment": "alignment",
            "Signal": "signal_drift_um_s", "Density": "neighbors_in_R"}
    layers = ["Markers", "Tracks"] + [k for k, col in have.items() if col in df] + ["IDs"]

    def draw(f, chosen):
        return overlay.draw(frames[f], f, df, None, chosen or ["Tracks", "Markers"], s.get("um_per_px"),
                            s.get("R_um", 10.0), s.get("signal_dir"))

    def count(f):
        return f"<b>{int(per_frame.get(f, 0))}</b> cells tracked<span>frame {f + 1} of {n}</span>"

    if ss.get("rec_code") != rec["record_code"]:      # a different record → start from its first frame
        ss.rec_code, ss.record_frame, ss._record_layers_set = rec["record_code"], 0, False
    video_player("record", n, rec["fps"], draw, count, layers=layers)


def cells_to_robots(prof):
    """The data trail: each number the Simulator uses, and where it came from."""
    f, t = prof["fit"], prof["targets"]
    inst = prof.get("instinct") or {}
    rows = [
        ("Swimming speed", f"{f['cell_speed_bl_per_s']:g} body lengths/s", "Robot speed",
         f"{prof['speed_bl_per_s']:g} BL/s · slowed so 1 s of robot ≈ {f['time_scale']:.2f} s of cells"),
        ("Distance cells keep apart", f"{f['repulsion_measured_bl']:g} BL", "Personal space",
         f"{prof['zone_repulsion_bl']:g} BL"),
        ("How far headings match", f["orientation_note"], "Copy neighbours within", f"{prof['zone_orientation_bl']:g} BL"),
        ("Turning per body length", f"max {prof['max_turn_rate_deg_per_s']:g} °/s", "Turning and wander",
         f"wander {prof['heading_noise_deg']:g}° per tick"),
    ]
    for c in ("blue", "red"):
        if c in inst:
            rows.append((f"Pull toward the {c} light", f"{inst[c]['pull']:+.2f} (drift ÷ speed)",
                         f"Instinct for the {c} light", f"{inst[c]['pull']:+.2f}"))
        else:
            rows.append((f"Pull toward the {c} light", f"not measured — no cells filmed under {c} light",
                         f"Instinct for the {c} light", "0 until measured"))
    gaps = [("Staying together", "can't be seen in one video", "Stay within", f"{prof['zone_attraction_bl']:g} BL (assumed)")]
    body = "".join(f"<tr><td>{a}</td><td class='v'>{esc(b)}</td><td class='arrow'>→</td><td>{c}</td>"
                   f"<td class='v'>{esc(d)}</td></tr>" for a, b, c, d in rows)
    body += "".join(f"<tr class='gap'><td>{a}</td><td>{b}</td><td class='arrow'>→</td><td>{c}</td><td>{d}</td></tr>"
                    for a, b, c, d in gaps)
    st.markdown("<div class='lbl'>From the cells to the robots</div>"
                "<table class='flow'><tr><th>Seek measured in the cells</th><th></th><th></th>"
                "<th>The Simulator uses it as</th><th></th></tr>" + body + "</table>"
                f"<div class='note' style='margin-top:.6rem'>The cells’ own group numbers become the dashed lines in "
                f"the Simulator’s charts: alignment {fmt(t.get('polarization'), '{:.2f}')} · spacing "
                f"{fmt(t.get('nnd_bl'), '{:.2f}')} body lengths.</div>", unsafe_allow_html=True)


def simulator_launch(arc, code, prof):
    st.write("")
    c1, c2 = st.columns([1.3, 3], vertical_alignment="center")
    if c1.button("Open in Simulator", type="primary", width="stretch"):
        open_simulator(arc, code)
    c2.markdown("<div class='note'>Opens in its own window with these rules. <b>Blue Light</b> is the default "
                "training target; <b>Red Light</b> intervenes with what’s more visible. Each light pulls the robots "
                "by their instinct (measured above) plus training.</div>", unsafe_allow_html=True)


def open_simulator(arc, code):
    import importlib.util
    import os
    import subprocess
    import sys
    if importlib.util.find_spec("rps") is None:
        st.warning("The simulator isn't installed in this Python environment. In a terminal inside "
                   "project-seek, run:  pip install -r requirements-swarm.txt  — then press the button again.")
        return
    path = arc.swarm_profile_file(code)
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}   # Streamlit sets Agg (off-screen)
    subprocess.Popen([sys.executable, "-m", "swarm.live_tuner", "--profile", str(path)], cwd=str(ROOT), env=env)
    st.toast(f"Opening the Simulator for {code} in its own window…")


def fidelity(arc, rec, me, prof):
    from swarm import compare
    code = rec["record_code"]
    saved = arc.swarm_check(code)
    with st.expander("Check: do the robots move like the cells?"):
        st.markdown("<div class='note'>Runs the robots 4 times with these rules and measures them exactly like Seek "
                    "measures the cells. Close = the cells’ value falls inside the robots’ range. About a minute."
                    "</div>", unsafe_allow_html=True)
        if st.button("Run the check" if saved is None else "Run it again"):
            bar = st.progress(0.0)
            result, png = compare.check(_tracks(code, arc, rec["tracks_key"]), prof, code,
                                        progress=lambda k, n: bar.progress(k / n, text=f"Robot run {k + 1} of {n}…"))
            arc.attach_swarm_check(code, me["id"], result, png)
            saved = (result, png)
        if saved:
            result, png = saved
            st.markdown(f"**{esc(result['summary'])}**")
            st.image(png, width="stretch")


def notes(arc, rec, me):
    st.write("")
    st.markdown("<div class='lbl'>Notes</div>", unsafe_allow_html=True)
    for n in rec["notes"]:
        st.markdown(f"<div style='border-left:2px solid {INK};padding-left:.7rem;margin-bottom:.6rem'>"
                    f"{esc(n['body'])}<br><span class='note'>{esc(n['author_name'])}</span></div>",
                    unsafe_allow_html=True)

    def add():
        arc.add_note(rec["record_code"], me["id"], ss.get("new_note", ""))
        ss.new_note = ""
    st.text_area("Add a note", key="new_note", height=70, label_visibility="collapsed",
                 placeholder="What you saw, what to compare it with…")
    st.button("Add note", on_click=add)


if __name__ == "__main__":     # streamlit runs the page as __main__; detection workers import it and skip this
    st.set_page_config(page_title="Seek", page_icon="◎", layout="wide")
    TERMINAL = st.Page(terminal, title="Seek Terminal", default=True)
    ARCHIVES = st.Page(archives, title="Archives", url_path="archives")
    st.navigation([TERMINAL, ARCHIVES], position="hidden").run()
