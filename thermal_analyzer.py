"""
Thermal Video Analyzer – Full version
======================================
Features:
  - Frame extraction at custom interval
  - OCR reads temperature scale from each frame automatically
  - Auto-detect objects + manual ROI adjustment
  - Temperature vs time graph
  - First and second derivatives (heating rate)

Run:  python -m streamlit run thermal_analyzer.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import re
from pathlib import Path
from scipy.signal import savgol_filter

import plotly.express as px

st.set_page_config(
    page_title="Thermal Analyzer",
    page_icon="🌡️",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────
# OCR loader – cached so model downloads only once
# ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OCR model (first time only, ~1 min)…")
def load_ocr_reader():
    try:
        import easyocr
        model_dir = Path(__file__).parent / "easyocr_models"
        model_dir.mkdir(exist_ok=True)
        return easyocr.Reader(["en"], gpu=False, verbose=False, model_storage_directory=str(model_dir))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────────
_defaults = dict(
    video_path=None,
    first_frame=None,
    fps=25.0,
    frames=[],
    timestamps=[],
    frames_extracted=False,
    roi_list=[],
    analysis_done=False,
    results={},
    scale_data=[],
    ocr_hit=[],
)
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────

def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)


def draw_rois(frame: np.ndarray, rois: list) -> np.ndarray:
    PALETTE = [
        (0, 220, 80), (0, 120, 255), (255, 60, 0),
        (255, 210, 0), (180, 0, 255), (0, 210, 210),
    ]
    out = frame.copy()
    for i, roi in enumerate(rois):
        c = PALETTE[i % len(PALETTE)]
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        lbl = roi["name"]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y - th - 6), (x + tw + 4, y), c, -1)
        cv2.putText(out, lbl, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def auto_detect_rois(frame: np.ndarray, threshold: int, min_area: int) -> list:
    gray = to_gray(frame).astype(np.uint8)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = sorted(
        [c for c in contours if cv2.contourArea(c) >= min_area],
        key=lambda c: cv2.boundingRect(c)[0],
    )
    rois = []
    for i, cnt in enumerate(valid):
        bx, by, bw, bh = cv2.boundingRect(cnt)
        rois.append({"name": f"Object {i + 1}", "x": bx, "y": by, "w": bw, "h": bh})
    return rois


def extract_numbers(texts: list) -> list:
    nums = []
    for txt in texts:
        for f in re.findall(r"-?\d+\.?\d*", txt):
            try:
                nums.append(float(f))
            except ValueError:
                pass
    return nums


def read_scale_ocr(frame: np.ndarray, reader) -> tuple:
    """
    Read min/max temperature from the camera's scale bar.
    Returns (t_min, t_max) or (None, None) on failure.
    """
    if reader is None:
        return None, None

    h, w = frame.shape[:2]
    right = frame[:, int(w * 0.70):]

    def ocr_band(band):
        scale = 3
        big = cv2.resize(band, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
        big = cv2.filter2D(big, -1,
                           np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        res = reader.readtext(big, detail=0, allowlist="0123456789.-°C")
        return [v for v in extract_numbers(res) if -50 <= v <= 200]

    top_nums = ocr_band(right[: int(h * 0.25), :])
    bot_nums = ocr_band(right[int(h * 0.75):,  :])

    t_max = max(top_nums) if top_nums else None
    t_min = min(bot_nums) if bot_nums else None

    if t_max is not None and t_min is not None and (t_max <= t_min or t_max - t_min < 5):
        return None, None
    return t_min, t_max


def px_to_temp(px: float, t_min: float, t_max: float) -> float:
    return t_min + (px / 255.0) * (t_max - t_min)


def smooth_and_diff(T: np.ndarray, t: np.ndarray, window: int):
    n = len(T)
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)
    if n < 3:
        return T, np.zeros_like(T), np.zeros_like(T)
    smoothed = savgol_filter(T, window_length=w, polyorder=2)
    d1 = np.gradient(smoothed, t)
    d2 = np.gradient(d1, t)
    return smoothed, d1, d2


# ──────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────
st.title("🌡️ Thermal Video Analyzer")

# ══════════════════════════════════════════════
# STEP 1 – Upload  (top-level so it is never disabled by inner reruns)
# ══════════════════════════════════════════════
st.header("1️⃣ Upload & Extract")

uploaded = st.file_uploader(
    "Upload thermal video", type=["mp4", "avi", "mov", "mkv", "wmv"],
    key="video_uploader",
)

# Process new file at the top level — only runs when a genuinely new file arrives
if uploaded and st.session_state.get("uploaded_file_name") != uploaded.name:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.close()
    st.session_state.video_path = tmp.name
    st.session_state.uploaded_file_name = uploaded.name
    st.session_state.frames_extracted = False
    st.session_state.analysis_done = False
    st.session_state.frames = []
    st.session_state.timestamps = []
    st.session_state.roi_list = []
    st.session_state.results = {}

    cap = cv2.VideoCapture(tmp.name)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, ff_new = cap.read()
    cap.release()

    if not ret:
        st.error("Could not read the video file. Try a different format or re-upload.")
    else:
        st.session_state.fps = fps
        st.session_state.first_frame = ff_new
        st.session_state._video_meta = dict(dur=dur, fw=fw, fh=fh, fps=fps)

elif not uploaded and not st.session_state.get("uploaded_file_name"):
    st.info("👆 Upload a thermal video to begin.")


# ══════════════════════════════════════════════
# Steps 1-display through 4 — inside a fragment so that sliders,
# ROI inputs, and other widgets only rerun this fragment, leaving
# the file uploader above always interactive.
# ══════════════════════════════════════════════
@st.fragment
def app_body():
    meta = st.session_state.get("_video_meta")
    ff   = st.session_state.first_frame
    if ff is None or meta is None:
        return

    # ── Step 1 display ──
    dur, fw, fh, fps_v = meta["dur"], meta["fw"], meta["fh"], meta["fps"]
    col_img, col_info = st.columns([3, 1])
    with col_img:
        st.image(cv2.cvtColor(ff, cv2.COLOR_BGR2RGB),
                 caption="First frame", use_container_width=True)
    with col_info:
        st.metric("Duration",   f"{dur:.1f} s")
        st.metric("FPS",        f"{fps_v:.1f}")
        st.metric("Resolution", f"{fw}×{fh}")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        interval = st.number_input(
            "Extract one frame every (seconds)",
            min_value=0.05, max_value=60.0, value=1.0, step=0.05,
        )
    with col_b:
        st.metric("Frames to analyse", max(1, int(dur / interval) + 1))

    if st.button("▶️ Extract frames", type="primary"):
        cap2 = cv2.VideoCapture(st.session_state.video_path)
        fps2   = cap2.get(cv2.CAP_PROP_FPS) or 25.0
        step   = max(1, int(fps2 * interval))
        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        frames, ts_list = [], []
        idx = 0
        bar = st.progress(0, text="Extracting…")
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
                ts_list.append(idx / fps2)
            idx += 1
            if total2 > 0 and idx % 30 == 0:
                bar.progress(min(idx / total2, 1.0))
        cap2.release()
        bar.empty()
        st.session_state.frames = frames
        st.session_state.timestamps = ts_list
        st.session_state.frames_extracted = True
        st.session_state.analysis_done = False
        st.success(f"✅ Extracted **{len(frames)}** frames")

    # ══════════════════════════════════════════════
    # STEP 2 – Define & adjust ROIs
    # ══════════════════════════════════════════════
    if st.session_state.frames_extracted:
        st.divider()
        st.header("2️⃣ Define Objects (ROI)")

        ff2 = st.session_state.first_frame
        fh2, fw2 = ff2.shape[:2]

        col_s1, col_s2, col_s3 = st.columns(3)
        threshold = col_s1.slider("Detection threshold", 0, 255, 160,
                                   help="Brighter pixels = warmer objects. "
                                        "Lower this if objects aren't detected.")
        min_area  = col_s2.number_input("Min object area (px²)", value=300, step=50)

        if col_s3.button("🤖 Auto-detect", type="primary"):
            detected = auto_detect_rois(ff2, threshold, min_area)
            if detected:
                st.session_state.roi_list = detected
                st.session_state.analysis_done = False
                st.success(f"Detected {len(detected)} object(s).")
            else:
                st.warning("Nothing found – try lowering the threshold.")

        st.markdown("---")

        # ── Editable ROI list ──
        if st.session_state.roi_list:
            st.markdown(
                "**Fine-tune positions** (X/Y = top-left corner, W/H = size). "
                "Changes are reflected in the preview immediately."
            )
            updated = []
            for i, roi in enumerate(st.session_state.roi_list):
                with st.expander(f"✏️  {roi['name']}", expanded=True):
                    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 0.6])
                    name_i = c1.text_input("Name", roi["name"], key=f"n{i}")
                    xi = c2.number_input("X", 0, fw2-1, min(roi["x"], fw2-1), key=f"x{i}")
                    yi = c3.number_input("Y", 0, fh2-1, min(roi["y"], fh2-1), key=f"y{i}")
                    wi = c4.number_input("W", 1, fw2,   min(roi["w"], fw2),   key=f"w{i}")
                    hi = c5.number_input("H", 1, fh2,   min(roi["h"], fh2),   key=f"h{i}")
                    remove = c6.checkbox("❌", key=f"d{i}", help="Delete")
                    if not remove:
                        updated.append({"name": name_i,
                                        "x": int(xi), "y": int(yi),
                                        "w": int(wi), "h": int(hi)})

            if updated != st.session_state.roi_list:
                st.session_state.roi_list = updated
                st.session_state.analysis_done = False

        # ── Draw ROI by dragging on the frame ──
        with st.expander("🖱️ Draw ROI on frame"):
            st.caption(
                "Drag a rectangle on the image (box-select tool, top-right of chart). "
                "The selected area appears below — name it and click **Add as ROI**. "
                "Repeat for each object."
            )
            roi_fig = px.imshow(cv2.cvtColor(ff2, cv2.COLOR_BGR2RGB))
            roi_fig.update_layout(
                dragmode="select",
                margin=dict(l=0, r=0, t=0, b=0),
                height=min(500, int(fw2 * 0.6)),
            )
            roi_fig.update_xaxes(showticklabels=False)
            roi_fig.update_yaxes(showticklabels=False)

            sel_event = st.plotly_chart(
                roi_fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode=("box",),
                key="roi_select_chart",
            )

            boxes = (sel_event.selection.get("box", [])
                     if sel_event and sel_event.selection else [])
            if boxes:
                b = boxes[0]
                rx  = max(0,    int(min(b["x"])))
                ry  = max(0,    int(min(b["y"])))
                rx2 = min(fw2,  int(max(b["x"])))
                ry2 = min(fh2,  int(max(b["y"])))
                rw, rh = max(1, rx2 - rx), max(1, ry2 - ry)

                ca, cb = st.columns([2, 1])
                roi_name = ca.text_input(
                    "Name for this ROI",
                    f"Object {len(st.session_state.roi_list) + 1}",
                    key="roi_draw_name",
                )
                cb.markdown(f"**x={rx}, y={ry}, w={rw}, h={rh}**")
                if st.button("✅ Add as ROI", type="primary"):
                    st.session_state.roi_list.append(
                        {"name": roi_name, "x": rx, "y": ry, "w": rw, "h": rh}
                    )
                    st.session_state.analysis_done = False
                    st.rerun(scope="fragment")
            else:
                st.info("No selection yet — drag on the image above.")

        # ── Manual add ──
        with st.expander("➕ Add ROI manually"):
            with st.form("add_roi", clear_on_submit=True):
                c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
                new_name = c1.text_input("Name",
                                         f"Object {len(st.session_state.roi_list)+1}")
                nx = c2.number_input("X", 0, fw2-1, fw2//4)
                ny = c3.number_input("Y", 0, fh2-1, fh2//4)
                nw = c4.number_input("W", 1, fw2,   fw2//6)
                nh = c5.number_input("H", 1, fh2,   fh2//6)
                if st.form_submit_button("Add"):
                    st.session_state.roi_list.append(
                        {"name": new_name,
                         "x": int(nx), "y": int(ny),
                         "w": int(nw), "h": int(nh)}
                    )
                    st.rerun(scope="fragment")

        # ── Preview ──
        if st.session_state.roi_list:
            gray_n = cv2.normalize(to_gray(ff2), None, 0, 255,
                                   cv2.NORM_MINMAX).astype(np.uint8)
            hm = cv2.applyColorMap(gray_n, cv2.COLORMAP_INFERNO)

            pa, pb = st.columns(2)
            pa.image(cv2.cvtColor(draw_rois(ff2, st.session_state.roi_list),
                                   cv2.COLOR_BGR2RGB),
                     caption="Original + ROIs", use_container_width=True)
            pb.image(cv2.cvtColor(draw_rois(hm,  st.session_state.roi_list),
                                   cv2.COLOR_BGR2RGB),
                     caption="Heatmap + ROIs",  use_container_width=True)
        else:
            st.info("No ROIs defined yet – use Auto-detect or add manually above.")

    # ══════════════════════════════════════════════
    # STEP 3 – Run analysis
    # ══════════════════════════════════════════════
    if st.session_state.frames_extracted and st.session_state.roi_list:
        st.divider()
        st.header("3️⃣ Analysis")

        use_ocr = st.toggle(
            "📷 Auto-read scale bar per frame (OCR)",
            value=True,
            help="Reads the camera's temperature scale from each frame separately, "
                 "so accuracy is maintained even if the camera auto-adjusts range.",
        )

        if not use_ocr:
            cm1, cm2 = st.columns(2)
            manual_tmin = cm1.number_input("Scale minimum (°C)", value=15.0, step=0.5)
            manual_tmax = cm2.number_input("Scale maximum (°C)", value=60.0, step=0.5)
        else:
            manual_tmin, manual_tmax = 15.0, 60.0  # fallback defaults

        smooth_window = st.slider(
            "Smoothing window for derivatives (frames, odd number)",
            3, 21, 7, step=2,
            help="Higher = smoother. Too high may hide real temperature changes.",
        )

        if st.button("🔬 Run analysis", type="primary"):
            frames   = st.session_state.frames
            ts_list  = st.session_state.timestamps
            roi_list = st.session_state.roi_list

            reader = load_ocr_reader() if use_ocr else None
            if use_ocr and reader is None:
                st.warning(
                    "⚠️ OCR not available (easyocr not installed). "
                    "Falling back to manual scale values."
                )

            n = len(frames)

            raw_tmin  = np.full(n, manual_tmin)
            raw_tmax  = np.full(n, manual_tmax)
            ocr_hit   = np.zeros(n, dtype=bool)
            last_bar_px = None
            last_good   = (manual_tmin, manual_tmax)
            ocr_calls   = 0

            if use_ocr and reader is not None:
                bar = st.progress(0, text="Reading scale bar (OCR)…")
                for fi, frame in enumerate(frames):
                    bar.progress(
                        (fi + 1) / n,
                        text=f"Reading scale bar {fi+1}/{n}  –  OCR calls so far: {ocr_calls}",
                    )
                    h_f, w_f = frame.shape[:2]
                    bar_px = frame[:, int(w_f * 0.70):]
                    if (last_bar_px is None
                            or bar_px.shape != last_bar_px.shape
                            or np.mean(np.abs(bar_px.astype(np.int16)
                                              - last_bar_px.astype(np.int16))) > 4.0):
                        t_min_f, t_max_f = read_scale_ocr(frame, reader)
                        last_bar_px = bar_px.copy()
                        ocr_calls += 1
                        if t_min_f is not None and t_max_f is not None:
                            raw_tmin[fi], raw_tmax[fi] = t_min_f, t_max_f
                            ocr_hit[fi] = True
                            last_good = (t_min_f, t_max_f)
                        else:
                            raw_tmin[fi], raw_tmax[fi] = last_good
                    else:
                        raw_tmin[fi], raw_tmax[fi] = last_good
                bar.empty()

            idx     = np.arange(n)
            hit_idx = idx[ocr_hit]

            if len(hit_idx) >= 2:
                sm_tmin = np.interp(idx, hit_idx, raw_tmin[ocr_hit])
                sm_tmax = np.interp(idx, hit_idx, raw_tmax[ocr_hit])
            else:
                sm_tmin, sm_tmax = raw_tmin.copy(), raw_tmax.copy()

            results    = {r["name"]: {"time": [], "temp": []} for r in roi_list}
            scale_data = list(zip(sm_tmin.tolist(), sm_tmax.tolist()))

            bar = st.progress(0, text="Computing temperatures…")
            for fi, (frame, ts) in enumerate(zip(frames, ts_list)):
                bar.progress((fi + 1) / n)
                t_min_f, t_max_f = sm_tmin[fi], sm_tmax[fi]
                gray = to_gray(frame)
                gh, gw = gray.shape
                for roi in roi_list:
                    x  = max(0, roi["x"])
                    y  = max(0, roi["y"])
                    x2 = min(x + roi["w"], gw)
                    y2 = min(y + roi["h"], gh)
                    patch = gray[y:y2, x:x2]
                    if patch.size == 0:
                        continue
                    temp = px_to_temp(float(np.mean(patch)), t_min_f, t_max_f)
                    results[roi["name"]]["time"].append(ts)
                    results[roi["name"]]["temp"].append(temp)
            bar.empty()

            st.session_state.results    = results
            st.session_state.scale_data = scale_data
            st.session_state.ocr_hit    = ocr_hit.tolist()
            st.session_state.smooth_w   = smooth_window
            st.session_state.analysis_done = True
            ocr_note = (f" OCR ran on **{ocr_calls}/{n}** frames." if use_ocr else "")
            st.success(f"✅ Done – graphs below.{ocr_note}")

    # ══════════════════════════════════════════════
    # STEP 4 – Graphs & Export
    # ══════════════════════════════════════════════
    if st.session_state.analysis_done and st.session_state.results:
        st.divider()
        st.header("4️⃣ Results")

        results = st.session_state.results
        sw      = st.session_state.get("smooth_w", 7)
        COLORS  = ["#e05c00","#0077cc","#22aa44","#cc22aa","#aa8800","#007799"]

        sw = st.slider("Smoothing window (adjust without re-running)",
                       3, 21, sw, step=2, key="sw_disp")

        if st.session_state.scale_data:
            with st.expander("📋 Scale values used per frame (for verification)"):
                sd  = st.session_state.scale_data
                tsl = st.session_state.timestamps
                oh  = st.session_state.get("ocr_hit", [])
                diag = {
                    "Time (s)": tsl[:len(sd)],
                    "T_min °C (smoothed)": [round(s[0], 2) for s in sd],
                    "T_max °C (smoothed)": [round(s[1], 2) for s in sd],
                    "Range °C":            [round(s[1] - s[0], 1) for s in sd],
                }
                if oh:
                    diag["Source"] = ["OCR" if h else "interpolated"
                                      for h in oh[:len(sd)]]
                st.dataframe(pd.DataFrame(diag),
                             use_container_width=True, hide_index=True)

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=[
                "Temperature  T(t)  [°C]",
                "Heating rate  dT/dt  [°C/s]",
                "Rate of change  d²T/dt²  [°C/s²]",
            ],
            row_heights=[0.5, 0.25, 0.25],
        )

        all_rows = []

        for i, (name, data) in enumerate(results.items()):
            col = COLORS[i % len(COLORS)]
            t   = np.array(data["time"])
            T   = np.array(data["temp"])

            smoothed, d1, d2 = smooth_and_diff(T, t, sw)

            fig.add_trace(go.Scatter(
                x=t, y=smoothed, name=name,
                line=dict(color=col, width=2),
                mode="lines+markers", marker=dict(size=4),
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=t, y=d1, name=name, showlegend=False,
                line=dict(color=col, width=1.5),
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=t, y=d2, name=name, showlegend=False,
                line=dict(color=col, width=1.5, dash="dot"),
            ), row=3, col=1)

            for j in range(len(t)):
                all_rows.append({
                    "Object":          name,
                    "Time_s":          round(float(t[j]),        4),
                    "Temp_C_raw":      round(float(T[j]),        4),
                    "Temp_C_smoothed": round(float(smoothed[j]), 4),
                    "dT_dt":           round(float(d1[j]),       6),
                    "d2T_dt2":         round(float(d2[j]),       6),
                })

        for r in [2, 3]:
            fig.add_hline(y=0, line_dash="dash", line_color="grey",
                          opacity=0.4, row=r, col=1)

        fig.update_layout(
            height=760,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.01, xanchor="right", x=1),
            margin=dict(l=55, r=25, t=60, b=40),
        )
        fig.update_yaxes(title_text="°C",    row=1, col=1)
        fig.update_yaxes(title_text="°C/s",  row=2, col=1)
        fig.update_yaxes(title_text="°C/s²", row=3, col=1)
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🖼️ Inspect individual frames"):
            fi = st.slider("Frame index", 0, len(st.session_state.frames) - 1, 0)
            sel   = st.session_state.frames[fi]
            ts_fi = st.session_state.timestamps[fi]

            pa, pb = st.columns(2)
            pa.image(
                cv2.cvtColor(draw_rois(sel, st.session_state.roi_list),
                             cv2.COLOR_BGR2RGB),
                caption=f"t = {ts_fi:.2f} s", use_container_width=True,
            )
            gn = cv2.normalize(to_gray(sel), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
            pb.image(
                cv2.cvtColor(draw_rois(cv2.applyColorMap(gn, cv2.COLORMAP_INFERNO),
                                       st.session_state.roi_list), cv2.COLOR_BGR2RGB),
                caption="Heatmap", use_container_width=True,
            )

            frame_row = {"Object": [], "Temp (°C)": []}
            for name, data in results.items():
                if fi < len(data["temp"]):
                    frame_row["Object"].append(name)
                    frame_row["Temp (°C)"].append(f"{data['temp'][fi]:.2f}")
            if st.session_state.scale_data and fi < len(st.session_state.scale_data):
                sd_fi = st.session_state.scale_data[fi]
                st.caption(f"Scale at this frame: {sd_fi[0]:.1f} – {sd_fi[1]:.1f} °C")
            st.dataframe(pd.DataFrame(frame_row),
                         use_container_width=True, hide_index=True)

        st.divider()
        df_out = pd.DataFrame(all_rows)
        st.download_button(
            "⬇️ Download CSV (temperatures + derivatives)",
            df_out.to_csv(index=False).encode("utf-8"),
            "thermal_analysis.csv",
            "text/csv",
            type="primary",
        )


app_body()
