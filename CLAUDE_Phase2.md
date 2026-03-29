# CLAUDE_Phase2.md — Phase 2 & Phase 3 Specification
**Spec version:** v0.2
**Status:** Phase 2 Complete. Phase 3 Active.
**Footage:** Mavic 4 Pro D-Log M | SE Alaska Marine
**Last updated:** 2026-03-28

---

## CONTEXT

This document covers Phase 2 (complete) and Phase 3 (active) of the automated
color grading pipeline. Phase 1 is documented in CLAUDE.md.

Master processing chain being built:
```
D-Log M source
  → Zeb Gardner LUT       (Phase 1 winner -- complete)
    → CLAHE clipLimit=2.0 (Phase 2 winner -- complete)
      → [Phase 3 winner]  (active)
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
ffmpeg NVENC is blocked -- use libx264 CPU encoding only.

No flush=True in any print() call.
No multiprocessing Pool -- use direct for loops only.
No hardcoded paths.
Do NOT place graded videos in input_video\ folder.
Do NOT modify Phase 1 scripts.
```

---

## PHASE 2 — COMPLETE

### Objective
Find the best single automatic refinement applied on top of the Zeb Gardner
Rec.709 output that consistently improves all scene types.

### Winner
```
CLAHE (Contrast Limited Adaptive Histogram Equalization)
  clipLimit = 2.0
  tileGrid  = (8, 8)
  Applied to L channel in LAB color space
```

### Why CLAHE Won
CLAHE enhances local contrast adaptively -- where there is scene detail (trees,
mountains, snow texture) it applies meaningful enhancement. Where content is flat
(open sky, calm water) it applies minimal change. This self-limiting behavior makes
it ideal for SE Alaska footage which contains both high-detail and flat-content zones
in the same frame.

At clipLimit=1.0 the effect was too gentle. At clipLimit=2.0 the enhancement is
visible and meaningful without introducing noise or artifacts in sky/water zones.

### POC Results
| ID | Name | Result |
|----|------|--------|
| A1 | Baseline (Zeb only) | reference |
| A2 | Auto Levels | REJECTED -- warm color cast |
| A3 | CLAHE clipLimit=2.0 | **WINNER** |
| A3b | CLAHE clipLimit=1.0 | rejected -- too gentle |
| A4 | Gentle S-curve | marginal |
| A5 | Saturation +15% | good but CLAHE better |
| A6 | Cool Shadow Toning | subtle, inconclusive |
| A7 | Vibrance | good but CLAHE better |

### POC Script
poc_phase2.py -- outputs to phase2_contact_sheets\poc\frame_{FRAME_NUMBER}\
Each run creates a new subfolder -- previous runs preserved.

---

## PHASE 3 — ACTIVE

### Objective
Bracket the CLAHE finalist with targeted adjustments. Determine if any combination
applied on top of CLAHE further improves the output. If a winner is found it becomes
the final step in the master processing chain.

### Input
Phase 3 input = Phase 1 + Phase 2 applied:
  Zeb Gardner LUT + CLAHE clipLimit=2.0

The script applies both in sequence before testing Phase 3 adjustments.
No file copying required -- script reads from phase1_output_video\ and
applies CLAHE internally before testing bracket variants.

Alternatively: user selects input video at prompt which already has
Zeb Gardner applied, and script applies CLAHE + bracket variant on top.

Input prompt behavior:
  Script scans phase1_output_video\ and displays available videos:
    1. DJI_20260223132708_0001_D_DLOGM_REC709_65_CUBE_ZRG_V1.MP4
    2. ...
  Select input video (enter number or full path):

### Bracket Variants

| ID | Name | Method | Parameters |
|----|------|--------|------------|
| B1 | CLAHE only | reference | clipLimit=2.0 tileGrid=(8,8) -- finalist |
| B2 | CLAHE + Sat+10 | OpenCV | HSV S channel x 1.10 |
| B3 | CLAHE + Sat+20 | OpenCV | HSV S channel x 1.20 |
| B4 | CLAHE + Cool shadows | OpenCV | LAB b-=10 a-=3 x shadow_weight |
| B5 | CLAHE + Vibrance | OpenCV | HSV S += 60 x (1 - S/255) |
| B6 | CLAHE + S-curve | OpenCV | Fixed tone curve lift/pull on L channel |
| B7 | CLAHE + Sat+10 + Cool shadows | OpenCV | B2 + B4 combined |

All variants implemented entirely in OpenCV/numpy.
No LUT files required for Phase 3 variants.
No Resolve dependency for grading.

### Rationale for Bracket Variants
CLAHE increases local contrast which can make colors appear slightly desaturated --
a saturation boost may compensate and produce richer natural colors (B2, B3).
Cool shadow toning pushes SE Alaska shadows toward blue/teal which matches the
actual cool color palette of the region (B4).
Vibrance is smarter than flat saturation -- boosts undersaturated areas only (B5).
The combination B7 is the most promising candidate -- CLAHE + gentle saturation +
cool shadows may produce the "professionally shot SE Alaska" target look.

### Contact Sheet Layout
5 rows (shadow, highlight, midtone, motion, wide_range) x 7 columns (B1-B7)
B1 CLAHE baseline always present as reference column.

If 7 columns too wide for readable output:
  Split into two sheets:
    Sheet A: B1 + B2 + B3 + B4
    Sheet B: B1 + B5 + B6 + B7
  B1 always present in both sheets.

Header block:
  [input_clip_name]  |  Phase 3 Bracket Evaluation
  [fps] [resolution]  |  duration  |  trim 10%-90%
  Processing chain: Zeb Gardner LUT + CLAHE clipLimit=2.0 + [variant]

Output filename: [input_stem]_phase3_eval.png
Output folder:   phase3_contact_sheets\

### Output Subfolders
Same pattern as Phase 2 POC:
  phase3_contact_sheets\poc\frame_{FRAME_NUMBER}\
Each run creates a new subfolder -- previous runs preserved.

### Claude Chat Evaluation
After poc_phase3.py produces the contact sheet:
1. Upload contact sheet to Claude Chat
2. Claude Chat scores each variant on:
     Color accuracy, highlight preservation,
     shadow detail, SE Alaska naturalness
3. Claude Chat recommends winner with reasoning
4. Gray confirms or overrides
5. Winner baked into master processing chain

### Running Phase 3 POC
```powershell
cd "C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video"
& "C:\Program Files\Python311\python.exe" poc_phase3.py
```

### Output Locations
  phase3_contact_sheets\    contact sheet PNGs
  phase3_output_video\      final processed video output

---

## APPLYING FINAL PROCESSING CHAIN — apply_final_to_video.py

After Phase 3 winner confirmed, a full video export script applies the complete
master processing chain:

```
Zeb Gardner LUT → CLAHE clipLimit=2.0 → [Phase 3 winner]
```

```powershell
& "C:\Program Files\Python311\python.exe" apply_final_to_video.py
```

Prompts:
1. Select input D-Log M video from input_video\
2. Confirm Phase 3 winner variant (B1-B7)

Output: final video in phase3_output_video\
Naming: {input_stem}_final.MP4
Encoding: libx264 -crf 18 -preset fast -pix_fmt yuv420p (CPU, no NVENC)

---

## FUTURE CONSIDERATION — Phantom LUTs

Phantom LUTs by Joel Famularo (joelfamularo.com/colour-mavic4) not yet evaluated.
Calibrated against ARRI Alexa Mini side-by-side across 80 scenes. Targets cinematic
soft look rather than technical accuracy. If Phase 3 does not reach target quality,
evaluate Phantom LUTs Neutral as alternative Phase 1 base LUT before proceeding.
Add to C:\dev\LUTs_folder\ and run Phase 1 contact sheet comparison vs Zeb Gardner.

---

## SCRIPT INVENTORY

| Script | Role |
|--------|------|
| poc_phase2.py | Phase 2 POC -- variants A1-A7 + A3b |
| poc_phase3.py | Phase 3 POC -- variants B1-B7 (to be built) |
| phase2_run.py | Full Phase 2 contact sheet pipeline (future) |
| phase3_run.py | Full Phase 3 contact sheet pipeline (future) |
| apply_final_to_video.py | Full master chain video export (future) |

Reuse frame_analyzer.py and sheet_composer.py from Phase 1.
Do NOT modify Phase 1 scripts.

---

## CC HANDOFF FOR PHASE 3 POC

```
Read CLAUDE_Phase2.md in the current folder.
Read poc_phase2.py as reference implementation.

Write poc_phase3.py implementing the Phase 3
bracket variants B1-B7 as specified in
CLAUDE_Phase2.md under "Bracket Variants".

poc_phase3.py behavior:
1. Scans phase1_output_video\ and prompts
   user to select input video by number
2. Extracts frame at FRAME_NUMBER = 775
3. Applies Zeb Gardner LUT + CLAHE internally
   to produce the Phase 1+2 base
4. Applies each bracket variant B1-B7 on top
5. Saves individual PNGs and combined sheet to:
   phase3_contact_sheets\poc\frame_{FRAME_NUMBER}\

Use poc_phase2.py as the structural template.
No flush=True. No multiprocessing Pool.
No hardcoded paths except FRAME_NUMBER constant.
Do not modify poc_phase2.py or any Phase 1 scripts.
```

---

## ACCEPTANCE CRITERIA — Phase 3 POC

1. Script prompts for input from phase1_output_video\
2. All 7 bracket variants B1-B7 applied to extracted frame
3. B1 CLAHE baseline present as reference
4. Individual PNG per variant saved
5. combined_all_variants.png saved
6. Output to phase3_contact_sheets\poc\frame_{FRAME_NUMBER}\
7. No flush=True, no multiprocessing Pool
8. Does not modify any Phase 1 or Phase 2 scripts
