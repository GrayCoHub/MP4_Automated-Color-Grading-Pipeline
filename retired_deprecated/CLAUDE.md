# CLAUDE.md — Phase 1: Automated Color Grade Evaluation Pipeline
**Spec version:** v0.3  
**Status:** CC Handoff Ready  
**Footage:** Mavic 4 Pro D-Log M | SE Alaska Marine  
**Last updated:** 2026-03-26

---

## TWO-PHASE ARCHITECTURE — ALWAYS IN SCOPE

| | Phase 1 | Phase 2 |
|---|---|---|
| **Goal** | Find the winning grade approach | Apply it intelligently across full clips |
| **Method** | Contact sheets, N variants, zero user grading | Scene segmentation + keyframed grades |
| **Input** | Representative frames per clip | Full clips |
| **Output** | Labeled PNG contact sheets + metrics CSVs | Production-ready graded video |
| **Status** | **ACTIVE — building now** | Designed after Phase 1 evaluation complete |

Phase 1 does NOT produce production video output.  
The per-clip metrics CSVs are the primary input for Phase 2 scene segmentation design.

---

## HARD CONSTRAINTS

```
CRITICAL: OpenCV is a custom CMake CUDA build.
pip install opencv-python is STRICTLY FORBIDDEN.
It will overwrite the CUDA build and destroy the live production pipeline.

CC must verify CUDA build is active during system discovery:
    python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
    --> Must return > 0
    --> If 0 or cv2.cuda unavailable: STOP and report. Do not proceed.

No new dependencies may be introduced without explicit approval.
```

---

## SYSTEM DISCOVERY — Run before writing any code

```bash
# 1. Python version
python --version

# 2. OpenCV CUDA build verification — MUST show device count > 0
python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"

# 3. Pillow
python -c "import PIL; print(PIL.__version__)"

# 4. Config and LUT
# Confirm phase1_config.json exists and is readable
# Confirm LUT file exists at path specified in config

```

---

## OBJECTIVE

Given a root folder of Mavic 4 Pro D-Log M source clips organized in subfolders, automatically produce one labeled contact sheet per qualifying clip showing 5 representative frames x 4 grade variants. Zero user involvement in frame selection or grading.

---

## CONFIGURATION FILE — phase1_config.json

All paths and tunable parameters live in `phase1_config.json`.  
**No hardcoded values anywhere in any script.**

```json
{
  "root_folder": "D:\\",
  "lut_path": "D:\\luts\\DJI_DLogM_to_Rec709.cube",
  "output_root": "D:\\phase1_output",
  "min_clip_duration_seconds": 10,
  "trim_margin_percent": 10,
  "frame_sample_interval": 10,
  "contact_sheet_width_px": 3200,
  "annotation_font_size": 28
}
```

---

## INPUT — FILE DISCOVERY

- Recurse entire folder tree from `root_folder`
- **Include:** files matching `*_D.MP4` pattern only
- **Exclude:** `.LRF`, `.txt`, `screen-*`, any file not matching `*_D.MP4`
- **Exclude:** clips shorter than `min_clip_duration_seconds`
- All skipped files logged to `run_log.txt` with reason

### Known folder structure

```
D:\
├── BRG-64_split_filter_eval\    <- contains non-DJI file, will likely yield 0 qualifying clips
│       screen-20260221-132226_64_BRG_filter.mp4   <- EXCLUDE (no _D.MP4 pattern)
│       DJI_20260221130112_0001_D.MP4              <- INCLUDE
│       read_me.txt                                <- EXCLUDE
├── Mavic4Pro\
│   ├── ND_64_Filter\
│   ├── 032226\
│   └── 03226-seals\             <- largest session, 10 clips
├── BRG-32\                      <- 11 clips
├── BRG-16\
│       notes.txt                <- EXCLUDE
```

---

## METADATA EXTRACTION

Derived from folder path for every qualifying clip:

| Field | Source | Example |
|---|---|---|
| `clip_name` | filename stem | `DJI_20260324143543_0001_D` |
| `session` | immediate parent folder | `03226-seals` |
| `group` | grandparent folder if present | `Mavic4Pro` |
| `nd_tag` | extracted from `BRG-N` pattern | `ND-32` or `none` |
| `clip_path` | full absolute path | `D:\Mavic4Pro\03226-seals\...` |

---

## FRAME SELECTION — OpenCV, zero user input

Analysis window: trim first and last `trim_margin_percent` of clip duration before any selection.  
Sample every `frame_sample_interval` frames for performance.

5 frames selected per clip:

| Frame ID | Method | Logged Metric |
|---|---|---|
| `shadow` | Lowest mean luminance | `lum_mean` |
| `highlight` | Highest mean luminance | `lum_mean` |
| `midtone` | Closest to clip median luminance | `lum_mean` |
| `motion` | Highest inter-frame delta | `frame_delta` |
| `wide_range` | Highest histogram spread (max-min luminance bin) | `hist_spread` |

Each selected frame logs: frame index, timecode, selection metric value.

---

## GRADE VARIANTS

4 variants applied to every selected frame:

| ID | Name | Method | Parameters |
|---|---|---|---|
| V0 | Raw | None | D-Log M unmodified |
| V1 | Official LUT | LUT via OpenCV | DJI D-Log M to Rec.709 `.cube` from `lut_path` |
| V2 | No-LUT S-curve | OpenCV | Contrast +20, Sat +15, S-curve on L channel in LAB space |
| V3 | Warm Grade (OpenCV only) | OpenCV | Shadow warmth, highlight cooling, sat +10% HSV, radial vignette |

- **V0, V1, V2, V3** — implemented entirely in OpenCV/numpy. No Resolve dependency.

---

## CONTACT SHEET COMPOSITION — Pillow

Layout per sheet: **5 rows (frames) x 4 columns (variants)** plus header and histogram row.

**Header block:**
```
[clip_name]  |  [nd_tag]
[session] / [group]  |  [fps] [resolution]  |  duration  |  frames analyzed  |  trim: 10%-90%
```

**Column labels:** V0 Raw / V1 Official LUT / V2 No-LUT S-curve / V3 Warm Grade

**Each cell:**
- Frame image (16:9 aspect ratio)
- Footer bar: frame ID + selection metric value + variant parameters

**Histogram row** (below all image rows):
- One luminance histogram per selected frame (raw frame only)
- Shows tonal distribution context for each selected frame

**Output filename:** `[clip_stem]_eval.png`  
**Width:** `contact_sheet_width_px` (height auto-calculated)

---

## OUTPUT FOLDER STRUCTURE

Mirrors input tree exactly:

```
D:\phase1_output\
├── BRG-16\
├── BRG-32\
├── BRG-64\
├── BRG-64_split_filter_eval\
└── Mavic4Pro\
    ├── 032226\
    ├── 03226-seals\
    └── ND_64_Filter\
```

---

## PER-CLIP METADATA CSV

One CSV per clip, co-located with contact sheet.  
**Filename:** `[clip_stem]_metrics.csv`

**Columns:**
```
frame_id, frame_index, timecode, lum_mean, lum_median, hist_spread,
frame_delta, selection_reason, clip_name, session, group, nd_tag
```

> This CSV is the primary input for Phase 2 scene segmentation design.

---

## RUN SUMMARY SHEET

Single PNG at `D:\phase1_output\run_summary.png`:
- One cell per qualifying clip — midtone frame, V0 raw only
- Labeled: clip name + session + nd_tag
- Grid layout, auto-columns based on clip count
- Full-shoot overview at a glance

---

## RUN LOG

`D:\phase1_output\run_log.txt` — produced every run:
- Clips processed (count + list)
- Frames selected per clip (frame ID + index + metric)
- Files skipped (path + reason)
- Total runtime

---

## TOOLCHAIN

| Tool | Role |
|---|---|
| `opencv` **(CMake CUDA build — DO NOT pip install)** + `numpy` | Frame extraction, analysis, V1/V2 grade application |
| `Pillow` | Contact sheet composition, annotation, histogram rendering |
| `phase1_config.json` | All tunable parameters |

**Entry point:** `phase1_run.py`

---

## IMPLEMENTATION ORDER

```
1. phase1_config.py      — config loader and validator
2. file_discovery.py     — recursive file finder, filtering, metadata extraction
3. frame_analyzer.py     — OpenCV frame selection (all 5 frame types)
4. grade_engine.py       — V0/V1/V2/V3 all in OpenCV; V3 Warm Grade (no Resolve dependency)
5. sheet_composer.py     — Pillow contact sheet layout, annotation, histogram row
6. csv_writer.py         — per-clip metrics CSV output
7. summary_sheet.py      — run summary grid
8. phase1_run.py         — orchestrator, calls all modules, produces run_log.txt
```

---

## ACCEPTANCE CRITERIA

1. Every `*_D.MP4` file meeting duration threshold produces one contact sheet PNG and one metrics CSV
2. Non-qualifying files skipped silently with log entry
3. Frame selection requires zero user input
4. All 4 variants present for all 5 frames
5. Contact sheet header correctly reflects folder-derived metadata
6. ND tag present and correct on sheets from `BRG-*` folders
7. Output folder structure mirrors input folder structure exactly
8. Histogram row present on every contact sheet
9. Per-clip metrics CSV produced with all required columns
10. Run summary sheet produced at output root
11. `run_log.txt` produced listing all processed clips, skipped files, and V3 status
12. All parameters sourced from `phase1_config.json` — no hardcoded paths or values

Run `verify_outputs()` against all 12 criteria before reporting complete.

---

## REPORT BACK

At completion, report:
- Clips processed (count)
- Acceptance criteria that did not pass (if any)
- Ambiguities encountered and how resolved
- V3 Warm Grade confirmed active
