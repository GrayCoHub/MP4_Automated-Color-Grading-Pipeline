# README — Automated Color Grading Pipeline
# DJI Mavic 4 Pro D-Log M | SE Alaska Marine Footage
# Last updated: 2026-03-28

---

## COMPATIBILITY NOTE

This pipeline was built on a machine running a custom OpenCV build compiled
from source with full CUDA GPU acceleration. This is not the standard OpenCV
that most Python users have installed via pip.

**If you installed OpenCV with:**
```
pip install opencv-python
```
...you have the standard CPU-only version. This is by far the most common
installation and works well for most use cases. However it does not include
the CUDA-accelerated modules (cv2.cuda, cv2.cudacodec) that parts of this
pipeline depend on -- specifically the live vessel detection pipeline and
hardware video decode via NVDEC.

**What this means for you:**
```
apply_final_to_video.py:   Will work with standard pip opencv
                           CLAHE and color operations are standard OpenCV
                           No GPU required for the color grading pipeline

phase1_run.py:             Will fail -- attempts NVDEC hardware decode
                           via cv2.cudacodec which does not exist in
                           pip opencv

The full pipeline as documented here requires:
  NVIDIA GPU
  CUDA 12.x
  OpenCV compiled from source with CUDA support
  Driver 560.94 (locked for this machine)
```

If you are running the color grading portion only (apply_final_to_video.py)
and have standard pip opencv installed, the script should work for you
as long as ffmpeg is available on your system.

---

## THE SHORT ANSWER

The single command:
  & "C:\Program Files\Python311\python.exe" apply_final_to_video.py

...is NOT a shortcut that skips the phases. It is the END RESULT of completing
all three phases. Each phase was a research and evaluation process that determined
one component of the processing chain. apply_final_to_video.py simply executes
the confirmed chain -- it could not have been written without completing Phases 1,
2, and 3 first.

Think of it this way:
  Phase 1, 2, 3 = the research        (done once)
  apply_final_to_video.py = the result (runs forever)

---

## UNDERSTANDING D-LOG M

Before explaining the pipeline it is important to understand what D-Log M actually
is -- because this directly shaped every decision made in Phase 1.

### What "log" means
A logarithmic color profile compresses the tonal range of a scene into the available
recording space. Bright areas (sky, sun glint on water) are pulled down and dark areas
(shadows, forest) are pushed up. The result looks flat and washed out straight out of
camera. The benefit is maximum dynamic range -- details in highlights and shadows are
both preserved where a standard recording would clip one or both ends.

### D-Log M is not real D-Log
DJI produces two log profiles:

```
D-Log:    True logarithmic format used on professional
          Inspire and Zenmuse cinema cameras.
          DJI published the color science mathematics.
          Blackmagic Design built a proper Color Space
          Transform (CST) for it in DaVinci Resolve.
          Industry standard tools handle it correctly.

D-Log M:  The "M" stands for Modified.
          Used on consumer/prosumer Mavic series cameras
          including the Mavic 4 Pro.
          NOT a true logarithmic format.
          DJI has never published the mathematical
          formula behind it.
          Essentially a flattened Rec.709 -- it compresses
          the image but does not follow true log math.
```

This distinction is widely discussed among drone videographers and color graders.
YouTube experts in drone color grading have specifically noted that D-Log M "is not
real D-Log" -- and they are correct. This is not a marketing complaint but a
technical fact with direct consequences for post-processing.

### Why this matters for grading
Because DJI never published the D-Log M color science, Blackmagic Design cannot
build a mathematically correct CST for it in DaVinci Resolve. The DJI D-Gamut/D-Log
setting in Resolve's CST was designed for true D-Log -- applying it to D-Log M
produces incorrect results. Many colorists make this mistake and produce footage
with wrong color response as a result.

The only correct options are:
  1. Use a LUT specifically built for D-Log M
  2. Treat it as slightly flat Rec.709 and correct by eye

DJI provides an official LUT for the conversion, but as Phase 1 evaluation
confirmed, it has a significant flaw: the bottom 5% of the tone curve is a flat
line, crushing shadow detail entirely.

### Dual Gain ISO Fusion -- a hidden D-Log M feature
Despite its "modified" status, D-Log M has a significant advantage over standard
D-Log on the Mavic 4 Pro that is not widely documented. When D-Log M is combined
with any auto mode that controls ISO (Auto ISO or full Auto Exposure), the camera
activates a hidden HDR feature called Dual Gain ISO Fusion.

```
How it works:
  The sensor simultaneously captures two readouts
  at different gain levels and blends them together.
  Result: extended dynamic range with natural
  highlight rolloff -- clouds and bright sky details
  are recovered that would otherwise clip.

Dynamic range comparison:
  Manual ISO + D-Log M:    ~16.0 stops
  Auto ISO + D-Log M:      ~17.7 stops
  Difference:              +1.7 stops of highlight recovery
  Additional noise:        zero -- clean HDR blend

Trigger condition:
  D-Log M color profile must be active
  AND any auto mode controlling ISO must be enabled:
    Auto ISO only:       activates fusion ✅
    Full Auto Exposure:  activates fusion ✅
    Manual ISO:          does NOT activate fusion ❌
```

The practical implication for shooting: lock shutter speed and aperture manually
(to prevent flicker) but leave ISO on Auto to activate Dual Gain Fusion. This
gives the best of both worlds -- stable exposure without flicker AND maximum
dynamic range from the sensor. See Pre-Flight Checklist for the correct settings.

### Why the Zeb Gardner LUT is the right solution
Zeb Gardner (zebgardner.com) approached D-Log M correctly -- rather than assuming
it follows standard log math, he measured its actual tone response using a color
checker reference and built a conversion LUT from the measured data. This
reverse-engineering approach produces the most accurate conversion precisely because
it works from D-Log M's actual non-standard behavior rather than theoretical log math.


```
DJI official LUT:      Delta E mean 7.61  Shadow Delta L 4.57
Zeb Gardner LUT:       Delta E mean 5.83  Shadow Delta L 0.28
```

The Zeb Gardner LUT was selected as the Phase 1 winner and is the foundation of
the entire master processing chain.

---

## WHAT THE PHASES ACTUALLY DO

The three phases are NOT processing steps that run on every video.
They are EVALUATION steps that were run once to determine the best
processing approach. Each phase asked a specific question and answered it
through automated contact sheet comparison.

```
Phase 1 asked:    Which base LUT converts D-Log M to Rec.709 most accurately?
Phase 1 answered: Zeb Gardner LUT (DLOGM_REC709_65_CUBE_ZRG_V1.cube)

Phase 2 asked:    What single refinement best improves the Zeb Gardner output?
Phase 2 answered: CLAHE clipLimit=2.0 tileGrid=(8,8) on LAB L channel

Phase 3 asked:    Can any adjustment on top of CLAHE further improve the output?
Phase 3 answered: Yes -- Saturation +10% + Cool shadow toning (B7)
```

Once those questions were answered the answers were baked permanently into
apply_final_to_video.py. The research is complete. The phases do not need
to be re-run for every new video.

---

## THE MASTER PROCESSING CHAIN

The confirmed chain applies three operations in sequence to every frame:

```
Step 1: Zeb Gardner LUT
        Converts D-Log M to Rec.709 color space.
        Built by measuring D-Log M's actual tone response
        against a color checker reference -- not by assuming
        standard log math which does not apply to D-Log M.
        Most color-accurate conversion of all tested methods.
        Delta E mean 5.83 vs DJI official LUT's 7.61.
        Shadow Delta L 0.28 vs DJI's 4.57 -- preserves
        shadow detail that the DJI LUT crushes entirely.

Step 2: CLAHE clipLimit=2.0
        Contrast Limited Adaptive Histogram Equalization.
        Enhances local contrast in detailed areas
        (trees, mountains, snow texture).
        Self-limiting -- minimal effect on flat areas
        (sky, open water) preventing noise or artifacts.
        Applied to L channel in LAB color space only.

Step 3: Saturation +10% + Cool shadow toning
        Adds natural color richness without oversaturation.
        Pushes shadows toward blue/teal matching SE Alaska
        winter palette -- cool grey-green conifers, steel
        blue water, crisp white snow.
        Smart weighting fades the cool toning to zero
        at midtones and above -- highlights unaffected.
```

---



### Prerequisites
1. Shoot in D-Log M with fixed Kelvin white balance (see Pre-Flight Checklist)
2. Copy source *_D.MP4 file to input_video\ folder
3. DaVinci Resolve does NOT need to be open
4. Custom OpenCV CUDA CMake build must be active on this machine
   (see OpenCV section below -- this is not a standard pip install)
5. Python 3.11.8 at C:\Program Files\Python311\python.exe

Verify OpenCV before running:
```powershell
& "C:\Program Files\Python311\python.exe" -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```
Must return > 0. If 0 or error: STOP -- do not proceed.

### Run
```powershell
cd "C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video"
& "C:\Program Files\Python311\python.exe" apply_final_to_video.py
```

### What happens
```
Script scans input_video\ and displays available clips:
  Available source clips in input_video\:
    1. DJI_20260223132708_0001_D.MP4
    2. DJI_20260224091532_0001_D.MP4

Select input video (enter number or full path): 1

Processing begins:
  Frame 100 / 1248  |  elapsed 8.2s  |  ETA 94.3s
  Frame 200 / 1248  |  elapsed 16.1s  |  ETA 82.4s
  ...

Output: phase3_output_video\DJI_20260223132708_0001_D_final_B7.MP4
```

---

## WHEN TO RE-RUN THE PHASES

The phases only need to be re-run if:

```
New camera body:
  Different sensor, different D-Log M characteristics.
  Phase 1 must be re-run to find the best LUT
  for the new camera's specific tone response.

New footage type or location:
  Summer SE Alaska footage may need different
  Phase 3 settings (less cool shadow toning).
  Re-run Phase 3 with summer clips.

Dissatisfied with output quality:
  Run Phase 3 again with new bracket variants.
  Or evaluate Phantom LUTs as Phase 1 alternative.

New LUT options available:
  Add to C:\dev\LUTs_folder\
  Re-run Phase 1 contact sheet comparison.
```

For routine filming at the same SE Alaska location with the same Mavic 4 Pro --
the phases do not need to be re-run. apply_final_to_video.py is the permanent
production tool.

---

## PHASE SCRIPTS (for reference and re-evaluation)

| Script | Purpose | When to use |
|--------|---------|-------------|
| phase1_run.py | Phase 1 contact sheet evaluation | Re-evaluating base LUT |
| poc_phase2.py | Phase 2 refinement POC | Re-evaluating refinements |
| poc_phase3.py | Phase 3 bracket POC | Re-evaluating bracket variants |
| apply_lut_to_video.py | Apply any single LUT to a video | Testing individual LUTs |
| apply_final_to_video.py | Apply confirmed master chain | **Production use** |

---

## PRE-FLIGHT CHECKLIST — DJI Mavic 4 Pro

Must be checked before every flight. These settings prevent artifacts that
cannot be fixed in post-processing.

```
1. Color profile:  D-Log M

2. White balance:  Set to Auto, let camera settle, note Kelvin value,
                   switch to Manual and lock it.
                   SE Alaska defaults:
                     5600K for sunny / partly cloudy
                     6500K for overcast / flat light
                   NEVER record on Auto WB --
                   causes color temperature hunting baked into footage.

3. Exposure:
   Shutter:        Manual -- 2x frame rate rule
                     1/50 for 25fps
                     1/60 for 30fps
                     1/100 for 50fps
                   LOCK shutter -- never let it hunt

   Aperture:       Manual -- install correct ND filter
                   LOCK aperture

   ISO:            Auto -- leave this on Auto intentionally
                   D-Log M + Auto ISO activates Dual Gain ISO Fusion
                   a hidden HDR mode that blends two sensor readouts
                   simultaneously for ~17.7 stops dynamic range
                   vs ~16 stops with manual ISO
                   Better highlight recovery in clouds and bright sky
                   Zero additional noise penalty
                   Any auto mode controlling ISO activates this feature

                   NEVER lock ISO manually when shooting D-Log M --
                   you lose the Dual Gain Fusion benefit

                   NEVER full Auto Exposure --
                   shutter hunting causes AE flicker baked into footage
                   ISO=Auto with manual shutter+aperture is the
                   correct combination

4. Resolution:     4K (3840x2160)
                   Highest available bitrate
                   10-bit if available

5. Test clip:      Shoot 3 seconds before flying.
                   Review on screen.
                   Confirm: not blown out, not too dark, no flicker.
                   Then fly.
```

---

## TECHNICAL CONSTRAINTS

### OpenCV — Custom CMake CUDA Build (Critical)

This pipeline depends on a custom-compiled version of OpenCV built from source
with CUDA support enabled. This is fundamentally different from the standard
OpenCV package available via pip and the two are NOT interchangeable.

```
Standard pip install (opencv-python):
  pip install opencv-python
  Installs a pre-compiled binary from PyPI
  No CUDA support
  No NVDEC hardware video decode
  No GPU-accelerated operations
  Works on any machine without a GPU
  Will NOT work for this pipeline's GPU operations

Custom CMake CUDA build (what this pipeline uses):
  Compiled from source against CUDA 12.6
  cuDNN 9.20.0 integrated
  NVDEC enabled via NVCUVID
  NVENC enabled
  GPU-accelerated Farneback optical flow
  Hardware video decode via cv2.cudacodec
  Requires NVIDIA GPU + matching driver
  Built specifically for this machine
```

**Can pip opencv replace the custom build?**
No -- and attempting to do so will destroy the pipeline:
```
pip install opencv-python     DESTROYS the custom build
pip install opencv-contrib    DESTROYS the custom build
```
Both commands overwrite the custom compiled cv2.pyd with a generic binary
that has no CUDA capability. Every GPU pipeline on this machine stops working.
Recovery requires a full rebuild from source which takes hours.

**Verification before every session:**
```powershell
& "C:\Program Files\Python311\python.exe" -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```
Must return > 0. If 0: the CUDA build has been overwritten. Stop immediately.

### NVIDIA Driver — Locked at 560.94

The NVIDIA driver must NOT be updated. Driver 560.94 is the confirmed working
version for the custom OpenCV CUDA build. Updating the driver risks breaking
NVDEC, NVCUVID, and all GPU pipelines. This also means ffmpeg NVENC hardware
encoding is unavailable (requires driver 570+) -- all video encoding in this
pipeline uses libx264 CPU encoding instead.

### Python
```
Version:  3.11.8
Location: C:\Program Files\Python311\python.exe
Manager:  pyenv-win
```

---

## PROJECT DOCUMENTS

```
CLAUDE.md             Main spec -- Phase 1 complete, pipeline overview
CLAUDE_Phase2.md      Phase 2 spec and results
CLAUDE_Phase3.md      Phase 3 spec, results, apply_final_to_video spec
Project_Brief.md      Full narrative of all three phases and decisions
project_notes\        Lessons learned, howtos, CC session notes
```
