# project_notes

Documentation, lessons learned, and howtos for the automated color grade pipeline.

**Project:** Mavic 4 Pro D-Log M | SE Alaska Marine
**Pipeline:** Three-phase automated color grading via Python + OpenCV + ffmpeg

## Phase status

| Phase | Goal | Winner | Status |
|-------|------|--------|--------|
| Phase 1 | Find winning base LUT | Zeb Gardner LUT | COMPLETE |
| Phase 2 | Find best refinement | CLAHE clipLimit=2.0 | COMPLETE |
| Phase 3 | Bracket finalist with adjustments | TBD | ACTIVE |

## Master processing chain

```
D-Log M source
  → Zeb Gardner LUT       (DLOGM_REC709_65_CUBE_ZRG_V1.cube)
    → CLAHE clipLimit=2.0  (LAB L channel, tileGrid 8x8)
      → [Phase 3 winner]   (TBD)
        → Final production video
```

## Key scripts

| Script | Purpose |
|--------|---------|
| `apply_lut_to_video.py` | Phase 1: apply LUT to full video via ffmpeg |
| `poc_phase2.py` | Phase 2 POC: single-frame bracket variants |
| `poc_phase3.py` | Phase 3 POC: single-frame bracket variants B1-B7 |
| `apply_final_to_video.py` | Apply confirmed master chain to full video |

## Hard constraints

- OpenCV is a custom CMake CUDA build — `pip install opencv-python` is forbidden
- NVIDIA driver 560.94 must NOT be updated
- No NVENC — libx264 CPU encoding only
- No `flush=True` in any `print()` call (DaVinci Resolve runner incompatibility)
- No `multiprocessing.Pool` (causes orphaned processes requiring reboot)
