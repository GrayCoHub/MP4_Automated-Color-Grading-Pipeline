# CLAUDE.md — Automated Color Grade Pipeline
**Spec version:** v0.5
**Status:** Phase 1 and Phase 2 Complete. Phase 3 Active.
**Footage:** Mavic 4 Pro D-Log M | SE Alaska Marine
**Last updated:** 2026-03-28

---

## STRATEGIC DECISION

DaVinci Resolve was identified as the industry standard tool for color grading drone
footage. Upon initial evaluation it became immediately clear that achieving professional
results through Resolve's standard 3-node manual color grading workflow requires a
significant investment of time and skill that falls outside the project scope and
engagement model. The decision was made on the same day as the first Resolve session:
rather than investing in becoming proficient in manual color grading, all effort would
be directed toward automating the entire grading workflow through Python scripting.

This aligns with Level 4 AI engagement -- write specs, evaluate outcomes, delegate
implementation to CC. Manual color grading is incompatible with Level 4 because it
requires hands-on craft judgment that cannot be delegated. DaVinci Resolve is therefore
treated as a scriptable color science engine. Its Python API is the interface and its
node architecture is the automation target. The contact sheet methodology was designed
specifically to support this: automated frame selection, automated grade application,
and visual output that can be evaluated quickly without color grading expertise.

---

## THREE-PHASE PIPELINE — ALWAYS IN SCOPE

| | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| **Goal** | Find winning base LUT | Find best refinement | Bracket finalist with adjustments |
| **Input** | D-Log M source clips | Phase 1 graded video | Phase 1+2 processed video |
| **Output** | Contact sheets + graded video | Contact sheets + refined video | Contact sheets + final video |
| **Winner** | Zeb Gardner LUT | CLAHE clipLimit=2.0 | TBD |
| **Status** | **COMPLETE** | **COMPLETE** | **ACTIVE** |

**Master processing chain (being built):**
```
D-Log M source
  → Zeb Gardner LUT (Phase 1 winner)
    → CLAHE clipLimit=2.0 (Phase 2 winner)
      → [Phase 3 winner] (TBD)
        → Final production video
```

---

## HARD CONSTRAINTS

```
CRITICAL: OpenCV is a custom CMake CUDA build.
pip install opencv-python is STRICTLY FORBIDDEN.
Verify: python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
Must return > 0. If 0: STOP and report.

CRITICAL: NVIDIA driver 560.94 must NOT be updated.
It is the confirmed working driver for the custom OpenCV CUDA CMake build.
Updating risks breaking NVDEC, NVCUVID, and all GPU pipelines.
ffmpeg NVENC is blocked -- use libx264 CPU encoding only.
Do NOT update the driver to resolve ffmpeg NVENC issues.

No flush=True in any print() call -- Resolve's fu_stdout does not support it.
No multiprocessing Pool -- causes orphaned processes requiring PC reboot.
Use direct for loops only.
No hardcoded paths -- all paths from config or prompt.
No new dependencies without explicit approval.
```

---

## SYSTEM DISCOVERY — Run before writing any code

```bash
python --version
python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
python -c "import PIL; print(PIL.__version__)"
```

---

## PROJECT FOLDER STRUCTURE

```
C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\
  input_video\              source D-Log M clips (*_D.MP4 or *_D.MKV) ONLY
  phase1_contact_sheets\    Phase 1 contact sheet PNGs + metrics CSVs
  phase1_output_video\      Phase 1 LUT-graded video output
  phase2_contact_sheets\    Phase 2 contact sheet PNGs
  phase2_output_video\      Phase 2 refinement video output
  phase3_contact_sheets\    Phase 3 bracket contact sheet PNGs
  phase3_output_video\      Phase 3 final video output
  project_notes\            documentation, lessons learned, howtos
  retired_deprecated\       old scripts and docs no longer active
```

Canonical LUT library: C:\dev\LUTs_folder\

---

## LUT ACQUISITION HISTORY

The DJI official D-Log M to Rec.709 LUT for the Mavic 4 Pro was not available on the
DJI US download site. It was located and downloaded from the DJI UK website:
  DJI_Mavic_4_Pro_D-Log_M_to_Rec.709_V1.cube

Independent testing by Zeb Gardner (zebgardner.com) measured Delta E color accuracy
for multiple D-Log M conversion methods. The DJI official LUT performed near the
bottom -- barely better than no color management -- with a documented shadow crushing
flaw (bottom 5% of tone curve is a flat line). Gardner's reverse-engineered LUT
produced the lowest Delta E of all tested methods:
  Mean Delta E:    5.83 vs DJI's 7.61
  Shadow Delta L:  0.28 vs DJI's 4.57

  DLOGM_REC709_65_CUBE_ZRG_V1.cube  -- downloaded from zebgardner.com

Phase 1 contact sheet evaluation confirmed the scientific data visually.
Zeb Gardner LUT is the confirmed Phase 1 winner. DJI official LUT is retired.

---

## DAVINCI RESOLVE SCRIPTING INTEGRATION

Resolve Free includes a fully functional Python scripting API -- poorly documented
and not prominently advertised. Initial attempts to connect from external PowerShell
failed. The reliable path on Windows is Resolve's internal script runner via the
Fusion\Scripts\Utility\ folder.

A symbolic link was created using symbolic_link_setup.py run as Administrator:

  Resolve sees:
  C:\Users\Gray\AppData\Roaming\Blackmagic Design\DaVinci Resolve\Support\
  Fusion\Scripts\Utility\Devinci_Resolve_automation_M4Pro_video\

  Pointing to:
  C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\

All project scripts accessible from:
  Workspace -> Scripts -> Utility -> Devinci_Resolve_automation_M4Pro_video

Two critical fixes required for Resolve's internal runner:
1. Remove all flush=True from print() calls
2. Replace multiprocessing Pool with direct for loop

Config path resolved relative to script location using os.path.dirname(os.path.abspath
(__file__)) so it works regardless of working directory when launched from Resolve.

---

## PHASE 1 — COMPLETE

### Objective
Automatically produce one labeled contact sheet per qualifying clip showing 5
representative frames x 4 grade variants. Zero user involvement.

### Configuration — phase1_config.json
```json
{
  "root_folder": "C:\\dev\\All_dev_projects_testing_folder\\Devinci_Resolve_automation_M4Pro_video\\input_video\\",
  "lut_path": "C:\\dev\\All_dev_projects_testing_folder\\Devinci_Resolve_automation_M4Pro_video\\DJI_Mavic_4_Pro_D-Log_M_to_Rec.709_V1.cube",
  "landscape_lut_path": "C:\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT\\Grays_LUT\\01_Drone LUTs_Landscape.cube",
  "output_root": "C:\\dev\\All_dev_projects_testing_folder\\Devinci_Resolve_automation_M4Pro_video\\phase1_contact_sheets\\",
  "min_clip_duration_seconds": 10,
  "trim_margin_percent": 10,
  "frame_sample_interval": 10,
  "contact_sheet_width_px": 3200,
  "annotation_font_size": 28
}
```

### File Discovery
- Include: *_D.MP4 or *_D.MKV
- Exclude: .LRF, .txt, screen-*, clips shorter than min_clip_duration_seconds
- frame_count, width, height extracted via cv2.VideoCapture for all file types

### Frame Selection — cv2.VideoCapture CPU decode (NOT cv2.cudacodec)
5 frames per clip: shadow, highlight, midtone, motion, wide_range

### Grade Variants
| ID | Name | Winner |
|----|------|--------|
| V0 | Raw D-Log M | reference |
| V1 | DJI Official LUT | retired |
| V2 | Zeb Gardner LUT | **WINNER** |
| V3 | No-LUT S-curve | reference |

### Running Phase 1
From Resolve: Workspace -> Scripts -> Utility ->
              Devinci_Resolve_automation_M4Pro_video -> phase1_run
Open Console first: Workspace -> Console

From PowerShell:
  cd project folder
  & "C:\Program Files\Python311\python.exe" phase1_run.py

Output: phase1_contact_sheets\

---

## PHASE 2 — COMPLETE

### Objective
Apply automatic refinements on top of Phase 1 Zeb Gardner output. Find the best
single refinement that consistently improves all scene types.

### Variants Tested (poc_phase2.py)
| ID | Name | Result |
|----|------|--------|
| A1 | Baseline | reference |
| A2 | Auto Levels | REJECTED -- warm color cast |
| A3 | CLAHE clipLimit=2.0 | **WINNER** |
| A3b | CLAHE clipLimit=1.0 | too gentle -- rejected |
| A4 | Gentle S-curve | marginal improvement |
| A5 | Saturation +15% | good but CLAHE better |
| A6 | Cool Shadow Toning | subtle, inconclusive |
| A7 | Vibrance | good but CLAHE better |

### Winner
CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clipLimit = 2.0
  tileGrid  = (8, 8)
  Applied to L channel in LAB color space

### POC Structure
poc_phase2.py outputs to:
  phase2_contact_sheets\poc\frame_{FRAME_NUMBER}\
Each run creates a new subfolder -- previous runs preserved.

### Running Phase 2 POC
From PowerShell:
  cd project folder
  & "C:\Program Files\Python311\python.exe" poc_phase2.py

---

## PHASE 3 — ACTIVE

See CLAUDE_Phase2.md for full Phase 3 specification.

Phase 3 brackets the confirmed CLAHE finalist with targeted adjustments to determine
if any combination further improves the output. The master processing chain after
Phase 3 will be:

  Zeb Gardner LUT → CLAHE clipLimit=2.0 → [Phase 3 winner]

---

## APPLY LUT TO FULL VIDEO — apply_lut_to_video.py

```powershell
& "C:\Program Files\Python311\python.exe" apply_lut_to_video.py
```

1. Scans C:\dev\LUTs_folder\ -- displays numbered list
2. User selects LUTs by number
3. User pastes input video path
4. Outputs one graded MP4 per LUT to phase1_output_video\

ENCODE_METHOD = "cpu"  -- libx264, never NVENC
Output naming: {original_stem}_{lut_stem}.MP4
Progress every 100 frames with elapsed time and ETA.

---

## PRE-FLIGHT CHECKLIST — DJI Mavic 4 Pro

1. Color profile:  D-Log M
2. White balance:  Set Auto, let settle, note Kelvin, switch to Manual and lock.
                   SE Alaska defaults: 5600K daylight / 6500K overcast.
                   NEVER record on Auto WB.
3. Exposure:       Manual. ISO 100-400. Shutter = 2x fps. Correct ND filter.
                   NEVER Auto exposure -- causes AE flicker baked into footage.
4. Resolution:     4K, highest bitrate, 10-bit if available.
5. Test clip:      3 seconds before flying. Review -- confirm no flicker.

---

## COMPARISON TOOL — MPV

```powershell
Start-Process "mpv" "path\to\file1.MP4"
Start-Process "mpv" "path\to\file2.MP4"
```

Shortcuts: Space=pause, ,/.=frame step, Left/Right=seek 5s, f=fullscreen

---

## TOOLCHAIN

| Tool | Role |
|---|---|
| opencv CMake CUDA build | Frame extraction, analysis, grade application |
| numpy | Array operations |
| Pillow | Contact sheet composition |
| ffmpeg Gyan 8.1 | Full video LUT application |
| MPV player | Side-by-side comparison |

---

## SCRIPT INVENTORY

| Script | Role | Phase |
|--------|------|-------|
| phase1_run.py | Phase 1 orchestrator | 1 |
| phase1_config.py | Config loader | 1 |
| file_discovery.py | File finder + metadata | 1 |
| frame_analyzer.py | CPU frame selection | 1,2,3 |
| grade_engine.py | Grade variants in OpenCV | 1 |
| sheet_composer.py | Contact sheet layout | 1,2,3 |
| csv_writer.py | Metrics CSV output | 1 |
| summary_sheet.py | Run summary grid | 1 |
| apply_lut_to_video.py | Full video LUT via ffmpeg | 1 |
| poc_phase2.py | Phase 2+3 POC script | 2,3 |
| symbolic_link_setup.py | Resolve symlink setup | utility |

---

## WHAT CC MUST NOT DO

- pip install or modify opencv
- Use multiprocessing Pool
- Use flush=True in any print() call
- Hardcode any paths
- Update NVIDIA driver
- Enable NVENC encoding
- Use cv2.cudacodec for frame analysis
- Place graded videos in input_video\ folder
  (input_video\ is for source D-Log M only)
