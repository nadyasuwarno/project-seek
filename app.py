"""Seeker Terminal: a working model of the Seeker pipeline.

    streamlit run app.py

01 Contributor -> 02 Observation -> 03 Extraction -> 04 Signature -> 05 Compare
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from seek import db
from seek.compare import MATCH_THRESHOLD, Comparison, search
from seek.extraction import ExtractionParams, locate_first_frame, render_detections, render_tracks
from seek.identity import normalize_orcid
from seek.pipeline import ContributorInput, DuplicateObservation, ObservationInput, analyze
from seek.schema import FEATURES
from seek.storage import ALLOWED_EXTENSIONS, probe_video, sha256_bytes, store_video_bytes
from seek.ui import (CSS, comparison_card_head, credit_block, dimension_bars, esc, feature_detail_table,
                     neighbor_map, section, signature_table, stepper)

st.set_page_config(page_title="Seeker Terminal", page_icon="◎", layout="wide")
db.init_db()
st.markdown(CSS, unsafe_allow_html=True)

LICENSES = ["CC-BY-4.0", "CC-BY-NC-4.0", "CC0-1.0", "All rights reserved (analysis only)"]


# ---------------------------------------------------------------- cached helpers

@st.cache_data(show_spinner=False)
def tracks_image(video_path: str, tracks_path: str):
    return render_tracks(video_path, pd.read_csv(tracks_path))


@st.cache_data(show_spinner=False)
def population_speed(tracks_path: str, frame_interval_s: float) -> pd.DataFrame:
    df = pd.read_csv(tracks_path).sort_values(["particle", "frame"])
    df["dx"] = df.groupby("particle")["x_um"].diff()
    df["dy"] = df.groupby("particle")["y_um"].diff()
    df["df"] = df.groupby("particle")["frame"].diff()
    steps = df[df["df"] == 1].copy()
    steps["speed"] = (steps["dx"] ** 2 + steps["dy"] ** 2) ** 0.5 / frame_interval_s
    out = steps.groupby("t_s")["speed"].mean().rename("mean speed (um/s)").to_frame()
    return out


# ---------------------------------------------------------------- sidebar

def sidebar() -> str:
    with st.sidebar:
        st.markdown('<div class="eyebrow">[ <b>SEEKER</b> TERMINAL ]</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted" style="margin:.4rem 0 1rem">Working model v0.1 &middot; research prototype</div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigate", ["Analyze", "Archive", "Data model"], label_visibility="collapsed")
        counts = db.counts()
        st.markdown("---")
        st.markdown(
            f'<div class="mono muted">ARCHIVE<br>{counts["observations"]:>3} observations<br>'
            f'{counts["contributors"]:>3} contributors<br>{counts["signatures"]:>3} signatures<br>'
            f'{counts["decisions"]:>3} decisions</div>', unsafe_allow_html=True)
        st.markdown("---")
        has_demo = any(r["is_demo"] for r in db.latest_signatures(None))
        if not has_demo:
            st.markdown('<div class="muted">The archive needs something to compare against. '
                        'Load 7 synthetic systems (clearly labeled as demo data).</div>', unsafe_allow_html=True)
            if st.button("Load demo archive", use_container_width=True):
                from seek.seed import seed_demo
                bar = st.progress(0.0, "Starting")
                seed_demo(lambda f, m: bar.progress(min(1.0, f), m))
                st.rerun()
        st.markdown('<div class="muted" style="margin-top:1rem">Similarity is not causality. '
                    'Seeker surfaces relationships; you decide whether they matter.</div>', unsafe_allow_html=True)
    return page


# ---------------------------------------------------------------- results (04 + 05)

def render_signature(rec: dict) -> None:
    st.markdown(section("04", "Signature", "what was measured, with bootstrap uncertainty"), unsafe_allow_html=True)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        try:
            st.image(tracks_image(rec["video_path"], rec["tracks_path"]), use_container_width=True,
                     caption=f'{rec["n_tracks"]} trajectories extracted by {rec["tool"]} {rec["tool_version"]}')
        except Exception as e:  # video moved or unreadable: keep the numbers visible
            st.warning(f"Could not render trajectories: {e}")
        speed = population_speed(rec["tracks_path"], rec["frame_interval_s"] * rec["params"].get("frame_step", 1))
        if not speed.empty:
            st.markdown('<div class="eyebrow" style="margin-top:.6rem">Population speed over time (raw measurement)</div>',
                        unsafe_allow_html=True)
            st.line_chart(speed, color="#C8FF00", height=150)
        if rec.get("perturbation_time_s") is not None:
            st.markdown(f'<div class="muted">Perturbation at {rec["perturbation_time_s"]:.2f} s</div>',
                        unsafe_allow_html=True)
    with right:
        st.html(signature_table(rec["features"], rec["uncertainty"]))
        for w in rec["quality"].get("warnings", []):
            st.markdown(f'<div class="warnline">&#9888; {esc(w)}</div>', unsafe_allow_html=True)
        for note in rec["quality"].get("extraction_notes", []):
            st.markdown(f'<div class="muted">{esc(note)}</div>', unsafe_allow_html=True)


def decision_buttons(query: dict, c: Comparison, key: str) -> None:
    existing = [d for d in db.list_decisions(query["observation_id"])
                if d["target_observation_id"] == c.target["observation_id"]]
    cols = st.columns([1, 1, 1.4, 3])
    labels = {"save": "Save", "reject": "Reject", "hypothesis": "Make hypothesis"}
    for col, (decision, label) in zip(cols, labels.items()):
        if col.button(label, key=f"{key}-{decision}"):
            db.record_decision(query["observation_id"], c.target["observation_id"], decision, c.to_dict())
            st.toast(f"{label}: recorded for {c.target['system_name']}")
            st.rerun()
    if existing:
        cols[3].markdown(f'<div class="muted" style="padding-top:.5rem">Last decision: '
                         f'<b style="color:var(--lime)">{esc(existing[0]["decision"])}</b> '
                         f'&middot; {esc(existing[0]["created_at"][:16])}</div>', unsafe_allow_html=True)


def render_card(query: dict, c: Comparison, rank: int | None, key: str) -> None:
    with st.container(border=True):
        st.html(comparison_card_head(c, rank))
        st.html(dimension_bars(c.dimensions))
        cols = st.columns([1, 2.2], gap="large")
        with cols[0]:
            try:
                st.image(tracks_image(c.target["video_path"], c.target["tracks_path"]), use_container_width=True)
            except Exception:
                st.markdown('<div class="muted">Preview unavailable</div>', unsafe_allow_html=True)
        with cols[1]:
            st.html(credit_block(c.target))
        with st.expander("Feature-level evidence"):
            st.html(feature_detail_table(c))
        decision_buttons(query, c, key)


def render_results(signature_id: int, key: str) -> None:
    rec = db.get_signature(signature_id)
    if rec is None:
        st.error("Signature not found.")
        return
    render_signature(rec)

    st.markdown(section("05", "Comparable results", "ranked by dimension-level similarity"), unsafe_allow_html=True)
    scale_free = st.toggle("Scale-free comparison (ignore absolute speed and period)", value=True, key=f"{key}-sf",
                           help="On: compare how things move, not how fast in absolute units. "
                                "Off: absolute speed and period are scored too.")
    candidates = db.latest_signatures("archive")
    result = search(rec, candidates, scale_free=scale_free)

    svg = neighbor_map(result)
    st.markdown(f'<img alt="Neighbour map" style="width:100%" src="data:image/svg+xml;base64,'
                f'{base64.b64encode(svg.encode()).decode()}">', unsafe_allow_html=True)

    if result.discovery_prompt:
        st.markdown(f'<div class="prompt"><div class="eyebrow"><b>Discovery prompt</b> &middot; exploratory, '
                    f'not a finding &middot; causal link unknown</div><div class="q">{esc(result.discovery_prompt)}</div></div>',
                    unsafe_allow_html=True)

    if result.no_match:
        st.markdown(f'<div class="nomatch"><div class="t">No meaningful match found</div><div class="muted" '
                    f'style="margin-top:.35rem">Nothing in the archive reaches {MATCH_THRESHOLD:.2f} average similarity '
                    f'across at least 3 measured dimensions. The closest candidates are listed below as weak, '
                    f'so you can see why.</div></div>', unsafe_allow_html=True)

    for i, c in enumerate(result.matches, start=1):
        render_card(rec, c, i, f"{key}-m{c.target['observation_id']}")

    if result.counterexamples:
        st.markdown(section("", "Counterexamples", "alike on one dimension, clearly different on another"),
                    unsafe_allow_html=True)
        for c in result.counterexamples:
            render_card(rec, c, None, f"{key}-c{c.target['observation_id']}")

    rest = result.weak + result.insufficient
    if rest:
        with st.expander(f"Weak or insufficient candidates ({len(rest)})", expanded=result.no_match):
            for c in rest:
                render_card(rec, c, None, f"{key}-w{c.target['observation_id']}")

    decisions = db.list_decisions(rec["observation_id"])
    if decisions:
        st.markdown(section("", "Research actions", "what you did with these comparisons"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"when": d["created_at"][:16], "decision": d["decision"],
                                    "candidate": d["target_system"]} for d in decisions]),
                     hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- page: analyze

def page_analyze() -> None:
    st.markdown('<div class="eyebrow">[ 01 &mdash; <b>ANALYZE</b> ]</div>', unsafe_allow_html=True)
    st.markdown("# Search what you can observe.")
    st.markdown('<div class="lede">Upload a time-based observation. Seeker tracks what moves, summarizes the '
                'dynamics as a behavioral signature, and shows which archived systems behave alike, with the '
                'evidence and the people behind every result.</div>', unsafe_allow_html=True)

    done = st.session_state.get("query_sig")
    step_slot = st.empty()

    # 01 Contributor
    st.markdown(section("01", "Contributor", "every upload is credited to a researcher's ORCID iD"),
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.2])
    orcid_raw = c1.text_input("ORCID iD", placeholder="0000-0002-1825-0097")
    name = c2.text_input("Name (as credited)")
    affiliation = c3.text_input("Affiliation / lab")
    orcid, orcid_error = None, None
    if orcid_raw:
        try:
            orcid = normalize_orcid(orcid_raw)
            c1.markdown(f'<div class="muted">&#10003; valid &middot; '
                        f'<a style="color:var(--lime)" href="https://orcid.org/{orcid}" target="_blank">view on orcid.org</a></div>',
                        unsafe_allow_html=True)
        except ValueError as e:
            orcid_error = str(e)
            c1.markdown(f'<div class="warnline">{esc(orcid_error)}</div>', unsafe_allow_html=True)
    contributor_ok = bool(orcid and name.strip())

    # 02 Observation
    st.markdown(section("02", "Observation", "the original video plus what is needed to interpret it"),
                unsafe_allow_html=True)
    upload = st.file_uploader("Time-based observation (video)", type=[e.lstrip(".") for e in sorted(ALLOWED_EXTENSIONS)])
    video_bytes = upload.getvalue() if upload else None
    info = None
    tmp_path = None
    if video_bytes:
        tmp_dir = Path(tempfile.gettempdir()) / "seek_preview"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"{sha256_bytes(video_bytes)[:16]}{Path(upload.name).suffix.lower()}"
        if not tmp_path.exists():
            tmp_path.write_bytes(video_bytes)
        try:
            info = probe_video(tmp_path)
        except ValueError as e:
            st.error(str(e))

    o1, o2 = st.columns([1.3, 1], gap="large")
    with o1:
        a, b = st.columns(2)
        title = a.text_input("Title", placeholder="Swarming after nutrient pulse")
        system_name = b.text_input("System", placeholder="Bacillus subtilis swarm")
        a, b = st.columns(2)
        domain = a.text_input("Domain", placeholder="microbiology")
        conditions = b.text_input("Conditions", placeholder="37C, 0.5% agar")
        a, b = st.columns(2)
        um_per_px = a.number_input("Scale (micrometres per pixel)", min_value=0.0, value=0.0, step=0.01,
                                   format="%.4f", help="Required. From your microscope calibration.")
        default_interval = round(1.0 / info.container_fps, 4) if info and info.container_fps > 0 else 0.1
        frame_interval = b.number_input("Real time between frames (s)", min_value=0.0001, value=default_interval,
                                        step=0.01, format="%.4f",
                                        help="For time-lapse this is the acquisition interval, not the playback rate.")
        a, b = st.columns(2)
        has_pert = a.checkbox("Includes a perturbation")
        pert_time = b.number_input("Perturbation at (s)", min_value=0.0, value=0.0, step=0.1,
                                   disabled=not has_pert)
        a, b = st.columns(2)
        license_ = a.selectbox("License", LICENSES)
        doi = b.text_input("DOI (optional)", placeholder="10.1234/abcd.5678")
        visibility = st.radio("Visibility", ["private", "archive"], horizontal=True,
                              format_func=lambda v: {"private": "Private: analyse only, not searchable by others",
                                                     "archive": "Archive: searchable, always shown with your credit"}[v])
        attest = st.checkbox("I confirm this video contains no identifiable human subjects (PRD R10).")
    with o2:
        if video_bytes and upload.name.lower().endswith((".mp4", ".webm", ".mov", ".m4v")):
            st.video(video_bytes)
        if info:
            st.markdown(f'<div class="mono muted">{info.frame_count} frames &middot; {info.width}&times;{info.height}px '
                        f'&middot; container {info.container_fps:.2f} fps</div>', unsafe_allow_html=True)

    # 03 Extraction
    st.markdown(section("03", "Extraction", "trackpy particle tracking: tune until the preview circles the units"),
                unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    diameter = e1.number_input("Unit diameter (px)", min_value=3, max_value=101, value=9, step=2)
    search_range = e2.number_input("Max move / frame (px)", min_value=1.0, value=float(diameter), step=1.0)
    min_len = e3.number_input("Min track length", min_value=4, value=8, step=1)
    max_frames = e4.number_input("Max frames", min_value=20, value=400, step=20)
    invert = e5.checkbox("Dark units on light background")
    auto_mass = e5.checkbox("Auto brightness threshold", value=True)
    minmass = None if auto_mass else st.number_input("Min mass (brightness)", min_value=0.0, value=500.0, step=50.0)
    params = ExtractionParams(diameter=int(diameter), minmass=minmass, invert=invert, search_range=float(search_range),
                              min_track_length=int(min_len), max_frames=int(max_frames))

    if tmp_path and info and st.button("Preview detection on first frame"):
        with st.spinner("Locating units"):
            frame, found, mm = locate_first_frame(tmp_path, params)
        pc1, pc2 = st.columns([1, 1.3])
        pc1.image(render_detections(tmp_path, found, params.resolved().diameter), use_container_width=True)
        pc2.markdown(f'<div class="mono">{len(found)} units found<br><span class="muted">threshold {mm:.0f} &middot; '
                     f'diameter {params.resolved().diameter}px</span></div>', unsafe_allow_html=True)

    missing = []
    if not contributor_ok:
        missing.append("contributor (valid ORCID + name)")
    if not video_bytes or not info:
        missing.append("video")
    if not system_name.strip():
        missing.append("system")
    if um_per_px <= 0:
        missing.append("scale")
    if not attest:
        missing.append("no-human-subjects confirmation")

    current = 5 if done else 4 if not missing else (1 if not contributor_ok else 2 if not (video_bytes and info) else 3)
    step_slot.markdown(stepper(current), unsafe_allow_html=True)

    go = st.button("Analyze and search the archive", type="primary", disabled=bool(missing))
    if missing:
        st.markdown(f'<div class="muted">Still needed: {esc(", ".join(missing))}</div>', unsafe_allow_html=True)
    if go:
        with st.spinner("Tracking, computing signature, bootstrapping uncertainty"):
            try:
                path, sha = store_video_bytes(video_bytes, upload.name)
                out = analyze(
                    path, sha,
                    ContributorInput(name=name.strip(), orcid=orcid, affiliation=affiliation.strip() or None),
                    ObservationInput(
                        title=title.strip() or system_name.strip(), system_name=system_name.strip(),
                        domain=domain.strip() or None, conditions=conditions.strip() or None,
                        um_per_px=float(um_per_px), frame_interval_s=float(frame_interval),
                        perturbation_time_s=float(pert_time) if has_pert else None,
                        license=license_, doi=doi.strip() or None, visibility=visibility, no_human_subjects=True,
                    ),
                    params,
                )
                st.session_state["query_sig"] = out.signature_id
                st.rerun()
            except DuplicateObservation as e:
                existing = db.latest_signature_for_observation(e.observation_id)
                st.info(f"{e} Showing its existing results.")
                if existing:
                    st.session_state["query_sig"] = existing["signature_id"]
                    st.rerun()
            except ValueError as e:
                st.error(str(e))

    if done:
        render_results(done, key="q")


# ---------------------------------------------------------------- page: archive

def page_archive() -> None:
    st.markdown('<div class="eyebrow">[ 02 &mdash; <b>ARCHIVE</b> ]</div>', unsafe_allow_html=True)
    st.markdown("# Behavioral archive")
    recs = db.latest_signatures(None)
    if not recs:
        st.markdown('<div class="lede">Nothing here yet. Analyze a video, or load the demo archive from the sidebar.</div>',
                    unsafe_allow_html=True)
        return
    table = pd.DataFrame([{
        "#": r["observation_id"], "system": r["system_name"], "contributor": r["contributor_name"],
        "ORCID": r["orcid"] or "demo", "license": r["license"], "visibility": r["visibility"],
        "tracks": r["n_tracks"], "uploaded": r["uploaded_at"][:10],
    } for r in recs])
    st.dataframe(table, hide_index=True, use_container_width=True)
    choice = st.selectbox("Search the archive using", recs, format_func=lambda r: f'#{r["observation_id"]} {r["system_name"]}')
    with st.container(border=True):
        st.html(credit_block(choice))
    render_results(choice["signature_id"], key=f"a{choice['observation_id']}")


# ---------------------------------------------------------------- page: data model

def page_model() -> None:
    st.markdown('<div class="eyebrow">[ 03 &mdash; <b>DATA MODEL</b> ]</div>', unsafe_allow_html=True)
    st.markdown("# How a signature stays traceable")
    st.markdown('<div class="lede">Every signature points to the extraction run that measured it, the observation '
                'it came from, and the contributor who uploaded it. Nothing is anonymous and nothing is '
                'detached from its context (PRD R2, R9, R11).</div>', unsafe_allow_html=True)
    st.code(db.__doc__.split("\n\n", 1)[1].strip(), language=None)
    st.markdown(section("", "Signature schema", "v0.1"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"dimension": f.family, "feature": f.label, "unit": f.unit,
                                "scale-free": "yes" if f.scale_free else "no", "tolerance": f.tolerance,
                                "transform": f.transform, "meaning": f.description} for f in FEATURES]),
                 hide_index=True, use_container_width=True)
    st.markdown(section("", "Comparison rules"), unsafe_allow_html=True)
    from seek import compare as cmp
    st.code(cmp.__doc__.strip(), language=None)
    st.markdown(section("", "SQL"), unsafe_allow_html=True)
    st.code(db.SCHEMA.strip(), language="sql")


page = sidebar()
{"Analyze": page_analyze, "Archive": page_archive, "Data model": page_model}[page]()
