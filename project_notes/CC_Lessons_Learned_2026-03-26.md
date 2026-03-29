# CC Hard-Learned Lessons — Session 2026-03-26
## Project: Phase 1 Automated Color Grade Evaluation Pipeline
## Context: Human-supervised, human-orchestrated, dual-agent architecture
## Reference this document when starting any new complex CC project on this PC

---

## LESSON 1 — Silent Pipeline Runs

**What happened:**
`phase1_run.py` ran silently for 26+ minutes with no terminal output. CC waited passively. No feedback loop. Could have run for hours.

**Root cause:**
Python buffers stdout by default. `print()` statements were held in memory buffer, never flushed to terminal. CC had no timeout rule for silent processes.

**Resolution:**
- Always run Python with `-u` flag: `python -u script.py`
- Add `flush=True` to all critical `print()` statements
- Every pipeline module must print timestamped start/complete lines
- CC must verify stdout is live within 15 seconds of launch — if no output, stop and diagnose immediately

**Permanent rule added to memory:** Yes (#9)

**CLAUDE.md boilerplate:**
```
All Python executions: python -u script.py (never plain python)
Every module prints: [HH:MM:SS] STARTED / COMPLETED with flush=True
If no output within 15 seconds of launch: STOP and diagnose
```

---

## LESSON 2 — Wrong Output Path (config mismatch)

**What happened:**
Output files were written to `C:\dev\Devinci_Resolve_automation_M4Pro_video\phase1_output\` instead of the correct `C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase1_output\`. Files appeared to be missing.

**Root cause:**
`output_root` in `phase1_config.json` was missing `All_dev_projects_testing_folder` in the path. Config was set from a template with a placeholder path.

**Resolution:**
Always verify config paths match the actual working directory before first run. CC should print the resolved `output_root` path at startup and confirm it exists or can be created.

**CLAUDE.md boilerplate:**
```
At startup, print and verify all config paths exist before processing begins.
```

---

## LESSON 3 — Unintended Footage Inclusion (M30 drone)

**What happened:**
File discovery picked up clips from `D:\M30_Drone_Video\` — a completely different camera (DJI M30, not Mavic 4 Pro). The `*_D.MP4` pattern matched both.

**Root cause:**
File filter was pattern-only, not folder-aware. No exclude list in config.

**Resolution:**
Added `exclude_folders` to `phase1_config.json`. Discovery now skips explicitly excluded folder names regardless of file pattern match.

**Permanent fix:**
Always include both include and exclude folder controls in any recursive file discovery config.

---

## LESSON 4 — V3 CST Sandwich Permanently Unavailable

**What happened:**
V3 required DaVinci Resolve external scripting API. All V3 cells rendered as grey `RESOLVE_UNAVAILABLE` placeholders even with Resolve open.

**Root cause:**
External scripting API is a **Resolve Studio (paid)** feature only. Free version does not support it. This was not known at spec time.

**Resolution:**
V3 replaced with OpenCV-based Warm Grade — no Resolve dependency. Pure OpenCV LAB color space implementation.

**Permanent rule:**
DaVinci Resolve free version cannot be scripted externally from Python. Any spec involving Resolve automation must verify Studio license first.

---

## LESSON 5 — Frame Analysis Bottleneck (183 seconds per clip)

**What happened:**
Frame analysis took 183 seconds per clip. `cap.set()` seeking on 4K H.264 forced keyframe decoding on every sample — 100 random seeks per clip at ~1.8s each.

**Root cause:**
`cv2.CAP_PROP_POS_FRAMES` seeking on H.264 is not truly random access — it decodes from the nearest keyframe forward on every seek. 100 seeks = 100 keyframe hunts.

**Resolution:**
Replaced random seeking with single sequential forward decode pass. Read every frame in order, compute metrics only at interval boundaries. Dropped from 183s to 30.5s.

**Rule:**
Never use `cap.set(cv2.CAP_PROP_POS_FRAMES, n)` in a loop for H.264 analysis. Always use sequential forward decode for metric passes.

---

## LESSON 6 — V1 LUT Bottleneck (23 seconds per clip)

**What happened:**
V1 LUT application (trilinear interpolation) took 23.7 seconds per clip on full 4K frames (3840×2160 = 8.3 megapixels).

**Root cause:**
Pure Python/NumPy trilinear interpolation is CPU-bound. Processing 8.3M pixels per frame is inherently slow without GPU acceleration.

**Resolution:**
Downsample frame to 1920×1080 before LUT application, upsample back afterward. 4× pixel reduction → dropped from 23.7s to 5.0s.

**Rule:**
For CPU-bound per-pixel operations on 4K footage, always downsample to working resolution first. Full 4K is only needed for final output, not intermediate processing.

---

## LESSON 7 — NVDEC API Call Error (wrong argument position)

**What happened:**
CC wrote `cv2.cudacodec.createVideoReader(clip_path, params=params)` based on training data patterns. Failed with: `Can't parse 'sourceParams'. Input argument doesn't provide sequence protocol`.

**Root cause:**
The correct signature is `createVideoReader(filename, sourceParams, params)` — `params` is the **third** positional argument, not the second. `sourceParams` (FFmpeg VideoCapture pairs) sits between filename and params. CC assumed keyword argument form without probing the actual build.

**Resolution:**
Probed `help(cv2.cudacodec.createVideoReader)` — confirmed correct call:
```python
cv2.cudacodec.createVideoReader(clip_path, [], params)
```
Empty list `[]` required as `sourceParams` placeholder. Frame analysis dropped from 30.5s to 8.2s with NVDEC.

**Permanent rule added to memory:** Yes (#8 — API Probe Rule)

**Rule:**
Before implementing any call to cudacodec or any OpenCV CUDA API in a pre-release build, always run `help()` probe first. Pre-release builds (4.14-pre) may have different signatures than training data. Never assume.

---

## PERFORMANCE PROGRESSION SUMMARY

| Version | Approach | Per-clip time |
|---|---|---|
| v1 | cap.set() seeking | 221s |
| v2 | Sequential forward decode | 46.8s |
| v3 | + 320px analysis resize | 44.0s |
| v4 | + NVDEC (after API fix) | ~25s |
| v5 | + Pool(processes=4) | ~17s effective |

**Total improvement: 13× faster**

---

## PERMANENT MEMORY RULES ADDED THIS SESSION

| # | Rule | Memory entry |
|---|---|---|
| 8 | API Probe Rule + POC-First Rule + No Silent Runs | Memory #8 |
| 9 | CC silent run prevention (python -u, flush=True, 15s timeout) | Memory #9 |
| 7 | pip install opencv-python FORBIDDEN — machine-wide, all projects | Memory #7 (updated) |

---

## CLAUDE.md BOILERPLATE — add to every new CC project

```markdown
## HARD CONSTRAINTS
- pip install opencv-python is STRICTLY FORBIDDEN on this machine
- CMake CUDA build must be verified active: cv2.cuda.getCudaEnabledDeviceCount() > 0
- If device count is 0: STOP and report

## CC ENGAGEMENT RULES
1. API Probe Rule: run help() before implementing any new API call
2. POC-First Rule: single-item POC with timing before any batch run
3. No Silent Runs: python -u always, flush=True on all prints,
   verify stdout live within 15 seconds

## PIPELINE STANDARDS
- All modules print timestamped start/complete with flush=True
- Config paths verified at startup before processing begins
- Per-stage timing logged for all long-running operations
```
