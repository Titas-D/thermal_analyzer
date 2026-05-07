# Thermal Video Analyzer

A local Streamlit web app for analysing temperature data from thermal camera recordings.
Upload a video, define regions of interest, and get interactive temperature graphs,
derivative plots, automated object classification, and an academic-style interpretation
section — all running offline on your machine.

---

## Requirements

- Python 3.9 or newer
- ~200 MB of disk space (for Python packages + OCR models)

---

## Installation

### 1. Install Python packages

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

> This requires an internet connection and takes a few minutes the first time.

### 2. OCR models (already included)

The `easyocr_models/` folder inside the project already contains the two model files
needed for temperature scale reading. No download is needed — the app works offline
from the start.

> If the folder is missing for any reason, the app will download the models
> automatically on first use (~95 MB, requires internet once).

---

## Starting the app

Run this command from the project folder:

```bash
python -m streamlit run thermal_analyzer.py
```

The browser opens automatically at `http://localhost:8501`.
To stop the app press `Ctrl + C` in the terminal.

---

## How to use

### Step 1 — Upload & Extract

1. Click **Browse files** and upload a thermal video (MP4, AVI, MOV, MKV, WMV).
2. The first frame and video info (duration, FPS, resolution) are shown.
3. Set the **frame interval** — how many seconds between extracted frames.
   - Example: `1.0` = one frame per second.
   - Lower values = more detail, longer analysis time.
4. Click **Extract frames**.

---

### Step 2 — Define Objects (ROI)

A Region of Interest (ROI) is a rectangle drawn over the object you want to track.
You can define ROIs in three ways:

**Auto-detect**
- Adjust the **Detection threshold** slider (lower = picks up cooler areas too).
- Set a **Min object area** to ignore small noise.
- Click **Auto-detect** — the app finds bright (hot) regions automatically.

**Draw on frame**
- Expand the **Draw ROI on frame** section.
- Use the box-select tool (top-right of the image) to drag a rectangle.
- Name the region and click **Add as ROI**.

**Manual entry**
- Expand **Add ROI manually**.
- Type in the X, Y, Width, and Height coordinates directly.

After adding ROIs you will see a preview with both the original frame and a heatmap
with the rectangles drawn on top. You can rename, resize, or delete any ROI using the
edit controls above the preview.

---

### Step 3 — Analysis

**OCR scale bar (recommended)**
- Leave the **Auto-read scale bar per frame (OCR)** toggle ON.
- The app reads the temperature scale directly from each frame, so it stays accurate
  even if the camera auto-adjusts its range during recording.

**Manual scale**
- Turn the toggle OFF and enter the **Scale minimum** and **Scale maximum** values
  in °C manually (use values shown on your camera's scale bar).

Set the **Smoothing window** (affects how smooth the derivative graphs look), then
click **Run analysis**.

---

### Step 4 — Results

After analysis you get three interactive graphs:

| Graph | What it shows |
|---|---|
| **T(t)** | Temperature in °C over time for each object |
| **dT/dt** | Heating/cooling rate (°C per second) |
| **d²T/dt²** | How fast the heating rate is changing |

**Other features on the results page:**

- **Smoothing window slider** — adjust smoothing without re-running the analysis.
- **Scale values per frame** — expand to see what min/max temperatures OCR read for
  each frame and whether a value came from OCR or was interpolated.
- **Inspect individual frames** — step through frames with a slider and see
  per-frame temperatures and the heatmap with ROIs overlaid.
- **Download CSV** — exports all data (raw temp, smoothed temp, dT/dt, d²T/dt²)
  for every object at every timestamp.

---

### Step 5 — Conclusions

Automatically generated after the download button. All values are derived from the
recorded data at runtime — no hardcoded thresholds or object names.

**Summary table**

A table with one row per object:

| Column | Description |
|---|---|
| Object | ROI name |
| Mean temp (°C) | Time-averaged smoothed temperature |
| Std dev (°C) | Temporal variability of smoothed temperature |
| Trend (start→end) | Mean temperature of the first window vs. the last window (30 s windows; falls back to 25 % of recording duration for clips shorter than 60 s) |
| Δ Change | Last window minus first window |
| Thermal behaviour | Automated classification (see below) |

**Classification logic** (priority order, first match wins):

| Category | Condition |
|---|---|
| Cold – possible necrotic/dormant tissue | Object is in the coldest 25 % by mean temperature |
| Warming – elevated metabolic activity | Net Δ change > 0.05 °C |
| Cooling – high thermal variability | Highest standard deviation among all objects |
| Cooling – stable thermal profile | None of the above |

**Per-object interpretation**

One bordered card per object containing:
- **Badge tags** — any that apply: `[warmest]` `[coldest]` `[highest variability]` `[warming trend]`
- **2–3 sentence academic interpretation** linking the measured thermal behaviour to
  possible biological states (metabolic activity, fungal colonisation, necrotic tissue,
  or stable uninfected tissue).

**Key findings**

Four bullet points covering:
- Inter-object temperature spread and whether it exceeds the ≥1 °C detection threshold
  of uncooled microbolometer cameras
- Which objects (if any) show a warming trend and what that may indicate
- Which object has the highest temporal variability and the biological implication
- Whether the observed spread supports passive IR thermography as a non-destructive
  pre-screening tool

---

## File structure

```
thermal_analyzer/
├── thermal_analyzer.py   # main application
├── requirements.txt      # Python dependencies
├── easyocr_models/       # offline OCR model files (~95 MB)
│   ├── craft_mlt_25k.pth
│   └── english_g2.pth
└── README.md
```
