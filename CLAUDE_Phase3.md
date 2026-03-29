# CLAUDE_Phase3.md — Phase 3: Bracket Evaluation Pipeline
**Spec version:** v0.2
**Status:** Phase 3 Complete. Master chain confirmed.
**Footage:** Mavic 4 Pro D-Log M | SE Alaska Marine
**Last updated:** 2026-03-28

---

## CONFIRMED MASTER PROCESSING CHAIN

```
D-Log M source
  → Zeb Gardner LUT          (Phase 1 winner)
    → CLAHE clipLimit=2.0    (Phase 2 winner)
      → Sat+10 + Cool shadows (Phase 3 winner -- B7)
        → Final production video
```

All three phases complete. The master chain is confirmed and ready for
implementation in apply_final_to_video.py.

---

## PHASE 3 RESULTS

### Winner: B7 CLAHE + Sat+10 + Cool shadows

B7 was selected over B3 (CLAHE+Sat20) in the final comparison.
At full resolution close-up, B3 Sat+20 made the scene look over-saturated
and too warm for winter SE Alaska conditions -- trees appeared too green,
beach too warm. B7 Sat+10+Cool accurately represents SE Alaska in winter:
cool grey-green snow-covered conifers, cool blue water, crisp white snow.

### Full POC Results

| ID | Name | Result |
|----|------|--------|
| B1 | CLAHE only | reference -- solid baseline |
| B2 | CLAHE + Sat+10 | good -- natural color richness |
| B3 | CLAHE + Sat+20 | rejected -- over-saturated, too warm for winter |
| B4 | CLAHE + Cool shadows | good -- accurate but slightly flat |
| B5 | CLAHE + Vibrance | REJECTED -- aggressive blue/teal cast, unnatural |
| B6 | CLAHE + S-curve | marginal improvement over B1 |
| B7 | CLAHE + Sat+10 + Cool shadows | **WINNER** |

### Why B7 Won
SE Alaska winter footage requires a cool color palette -- snow-covered conifers
are grey-green not vivid green, water is steel blue not tropical blue, shadows
are cool not warm. B7 combines:
  Saturation +10%: adds natural color richness without oversaturation
  Cool shadow toning: pushes shadows blue/teal matching actual SE Alaska palette
The combination produces accurate winter SE Alaska color science -- professionally
shot without looking processed.

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
No hardcoded paths except constants at top of script.
Do NOT modify Phase 1 or Phase 2 scripts.
Do NOT place output in input_video\ folder.
```

---

## SYSTEM DISCOVERY — Run before writing any code

```bash
python --version
python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
python -c "import PIL; print(PIL.__version__)"
```

---

## FOLDER STRUCTURE

```
C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\
  input_video\              source D-Log M clips -- read only
  phase1_output_video\      Phase 1 graded videos -- read only for Phase 3
  phase3_contact_sheets\    Phase 3 bracket contact sheet PNGs
    poc\
      frame_{FRAME_NUMBER}\ one subfolder per POC run -- preserved
  phase3_output_video\      Phase 3 final processed video output
```

Canonical LUT library: C:\dev\LUTs_folder\
Zeb Gardner LUT: C:\dev\LUTs_folder\DLOGM_REC709_65_CUBE_ZRG_V1.cube

---

## APPLY FINAL PROCESSING CHAIN — apply_final_to_video.py

Applies the complete confirmed master chain to any D-Log M source clip:

```
D-Log M → Zeb Gardner LUT → CLAHE clipLimit=2.0 → Sat+10 + Cool shadows
```

### Running

```powershell
cd "C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video"
& "C:\Program Files\Python311\python.exe" apply_final_to_video.py
```

### Script behavior

1. Scans input_video\ for source D-Log M clips (*_D.MP4 or *_D.MKV)
   Displays numbered list:
     Available source clips in input_video\:
       1. DJI_20260223132708_0001_D.MP4
       2. ...
   User selects by number or pastes full path.

2. Applies full master chain frame by frame:
   Step 1: Load Zeb Gardner LUT from LUT_PATH
           Apply via OpenCV LUT lookup (same as grade_engine.py)
   Step 2: Apply CLAHE clipLimit=2.0 tileGrid=(8,8) on LAB L channel
   Step 3: Apply Saturation +10% (HSV S channel x 1.10)
   Step 4: Apply Cool shadow toning (LAB b-=10 x shadow_weight,
           a-=3 x shadow_weight where shadow_weight = 1 - L/128)

3. Writes output to phase3_output_video\
   Naming: {input_stem}_final.MP4

4. Encoding: ffmpeg libx264 -crf 18 -preset fast -pix_fmt yuv420p
   ENCODE_METHOD = "cpu"  -- never NVENC

5. Progress every 100 frames with elapsed time and ETA.

### Constants at top of script

```python
LUT_PATH = r"C:\dev\LUTs_folder\DLOGM_REC709_65_CUBE_ZRG_V1.cube"
INPUT_SCAN_FOLDER = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\input_video"
OUTPUT_FOLDER = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase3_output_video"
ENCODE_METHOD = "cpu"   # options: cpu (libx264), nvenc (blocked -- driver 560.94)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
SAT_BOOST = 1.10        # +10% saturation
COOL_B_SHIFT = 10       # LAB b channel shadow shift
COOL_A_SHIFT = 3        # LAB a channel shadow shift
```

### Implementation approach

The master chain must be applied frame by frame via OpenCV -- NOT via ffmpeg
filter chain. The CLAHE and color adjustments are OpenCV operations that cannot
be expressed as ffmpeg filters. The workflow is:

  Read frame (cv2.VideoCapture CPU decode)
  Apply LUT (OpenCV numpy lookup)
  Apply CLAHE (cv2.createCLAHE on LAB L channel)
  Apply Sat+10 (HSV S channel multiply)
  Apply Cool shadows (LAB b/a channel shift weighted by L)
  Write frame (cv2.VideoWriter)

Use cv2.VideoWriter with fourcc('m','p','4','v') for intermediate output,
then pipe through ffmpeg for final H.264 encoding:
  OR use ffmpeg subprocess with stdin pipe reading OpenCV frames
  OR write to intermediate file then re-encode with ffmpeg

Recommended: write all processed frames to a temporary AVI with VideoWriter,
then run ffmpeg on the AVI to produce the final H.264 MP4.
This avoids complex ffmpeg pipe handling while ensuring H.264 output.

Temporary file: {output_folder}\{stem}_temp.avi
Final output:   {output_folder}\{stem}_final.MP4
Delete temp AVI after ffmpeg completes successfully.

---

## CC HANDOFF FOR apply_final_to_video.py

```
Before writing any code run system discovery:
1. python --version
2. python -c "import cv2; print(cv2.__version__);
   print(cv2.cuda.getCudaEnabledDeviceCount())"

Read CLAUDE_Phase3.md in the current folder.
Read apply_lut_to_video.py and poc_phase3.py
as reference implementations.

Write apply_final_to_video.py implementing
the confirmed master processing chain as
specified in CLAUDE_Phase3.md under
"APPLY FINAL PROCESSING CHAIN".

The script must:
1. Scan input_video\ and prompt user to select
   source D-Log M clip by number
2. Apply full master chain frame by frame:
   Zeb Gardner LUT → CLAHE → Sat+10 → Cool shadows
3. Write output to phase3_output_video\
   named {input_stem}_final.MP4
4. Use two-step encoding: OpenCV VideoWriter
   to temp AVI, then ffmpeg libx264 to final MP4
5. Delete temp AVI after successful encode
6. Report progress every 100 frames with
   elapsed time and ETA
7. Report total processing time at completion

All processing constants defined at top of script.
No flush=True. No multiprocessing Pool.
Do not modify any existing scripts.
```

---

## ACCEPTANCE CRITERIA

1. Script scans input_video\ and prompts by number
2. All four chain steps applied in correct order:
   Zeb LUT → CLAHE → Sat+10 → Cool shadows
3. Output written to phase3_output_video\
4. Output named {input_stem}_final.MP4
5. Temp AVI deleted after successful encode
6. Progress reported every 100 frames
7. No flush=True, no multiprocessing Pool
8. ENCODE_METHOD = "cpu" -- never NVENC
9. Does not modify Phase 1, 2, or 3 POC scripts

---

## WHAT CC MUST NOT DO

- pip install or modify opencv
- Use multiprocessing Pool
- Use flush=True in any print() call
- Hardcode paths other than the defined constants
- Update NVIDIA driver
- Enable NVENC encoding
- Use cv2.cudacodec for frame reading
- Modify any existing scripts
- Place output in input_video\ or phase1_output_video\
