"""
frame_analyzer.py — OpenCV frame selection (all 5 frame types).

Selects 5 representative frames per clip:
  shadow      — lowest mean luminance
  highlight   — highest mean luminance
  midtone     — closest to clip median luminance
  motion      — highest inter-frame delta
  wide_range  — highest histogram spread (max-min luminance bin)

Two-phase approach:
  Phase 1 (metric pass): NVDEC hardware decode at 320x180 for fast metric
    computation — no full-res frames stored. CPU forward-decode fallback
    if NVDEC is unavailable.
  Phase 2 (extraction): 5 targeted seeks at full resolution to get BGR data.

Analysis window: trim first/last trim_margin_percent of clip duration.
Samples every frame_sample_interval frames for performance.
"""

import cv2
import numpy as np


def _frame_index_to_timecode(frame_idx: int, fps: float) -> str:
    """Convert frame index to HH:MM:SS:FF timecode string."""
    if fps <= 0:
        fps = 25.0
    total_seconds = frame_idx / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int(round((total_seconds - int(total_seconds)) * fps))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def _metrics_from_gray(gray: np.ndarray, prev_gray) -> tuple:
    """Compute lum_mean, frame_delta, hist_spread from a uint8 grayscale frame."""
    lum_mean = float(np.mean(gray))
    if prev_gray is not None and prev_gray.shape == gray.shape:
        delta = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
    else:
        delta = 0.0
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    nz = np.nonzero(hist)[0]
    hist_spread = float(nz[-1] - nz[0]) if len(nz) >= 2 else 0.0
    return lum_mean, delta, hist_spread


# ---------------------------------------------------------------------------
# Phase 1: metric pass (NVDEC or CPU)
# ---------------------------------------------------------------------------

def _metric_pass_nvdec(clip_path: str, start_frame: int, end_frame: int, interval: int) -> list:
    """
    NVDEC hardware decode at 320x180 targetSz.
    Reads from frame 0; discards frames before start_frame without downloading.
    Downloads GpuMat only at interval sample points within the analysis window.
    Frame format: (180, 320, 4) uint16 BGRA — top 8 bits used for uint8 BGR.
    """
    params = cv2.cudacodec.VideoReaderInitParams()
    params.targetSz = (320, 180)
    reader = cv2.cudacodec.createVideoReader(clip_path, [], params)

    samples = []
    prev_gray = None
    current = 0

    while current < end_frame:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break
        if current >= start_frame and (current - start_frame) % interval == 0:
            cpu = gpu_frame.download()                          # (180, 320, 4) uint16
            bgr8 = (cpu[:, :, :3] >> 8).astype(np.uint8)      # top 8 bits -> uint8 BGR
            gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
            lum_mean, delta, hist_spread = _metrics_from_gray(gray, prev_gray)
            samples.append({
                "frame_idx": current,
                "lum_mean": lum_mean,
                "frame_delta": delta,
                "hist_spread": hist_spread,
            })
            prev_gray = gray
        current += 1

    del reader
    return samples


def _metric_pass_cpu(clip_path: str, start_frame: int, end_frame: int, interval: int) -> list:
    """
    CPU forward decode for metric computation.
    Single seek to start_frame, then sequential reads.
    Resizes each sampled frame to 320x180 for fast metric computation.
    """
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples = []
    prev_gray = None
    current = start_frame

    while current < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (current - start_frame) % interval == 0:
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            lum_mean, delta, hist_spread = _metrics_from_gray(gray, prev_gray)
            samples.append({
                "frame_idx": current,
                "lum_mean": lum_mean,
                "frame_delta": delta,
                "hist_spread": hist_spread,
            })
            prev_gray = gray
        current += 1

    cap.release()
    return samples


# ---------------------------------------------------------------------------
# Phase 2: full-resolution frame extraction
# ---------------------------------------------------------------------------

def _extract_full_res_frames(clip_path: str, indices: list) -> dict:
    """Seek to each index and extract the full-resolution BGR frame."""
    cap = cv2.VideoCapture(clip_path)
    frames = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not extract frame {idx} from {clip_path}")
        frames[idx] = frame
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def select_frames(clip_info: dict, cfg: dict) -> list:
    """
    Two-phase frame selection.

    Phase 1: metric pass at 320x180 via NVDEC (or CPU fallback) to select
             5 representative frame indices — no full-res data stored.
    Phase 2: 5 targeted full-resolution seeks to extract BGR frames.

    Returns list of 5 dicts compatible with sheet_composer and csv_writer.
    """
    clip_path = clip_info["clip_path"]
    fps = clip_info["fps"]
    frame_count = clip_info["frame_count"]
    trim_margin = cfg["trim_margin_percent"] / 100.0
    interval = cfg["frame_sample_interval"]

    start_frame = int(frame_count * trim_margin)
    end_frame = int(frame_count * (1.0 - trim_margin))
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = frame_count - 1

    # --- Phase 1: metric pass ---
    use_nvdec = False  # NVDEC returns black frames for standard MP4; cv2.VideoCapture is reliable
    path_label = "CPU"
    samples = []

    if use_nvdec:
        try:
            samples = _metric_pass_nvdec(clip_path, start_frame, end_frame, interval)
            path_label = "NVDEC"
        except Exception as e:
            print(f"  [frame_analyzer] NVDEC failed ({e}), falling back to CPU")
            samples = _metric_pass_cpu(clip_path, start_frame, end_frame, interval)
            path_label = "CPU-fallback"
    else:
        samples = _metric_pass_cpu(clip_path, start_frame, end_frame, interval)

    if not samples:
        return []

    # Select 5 frames by metric
    all_lum = [s["lum_mean"] for s in samples]
    clip_median = float(np.median(all_lum))

    shadow_s     = min(samples, key=lambda s: s["lum_mean"])
    highlight_s  = max(samples, key=lambda s: s["lum_mean"])
    midtone_s    = min(samples, key=lambda s: abs(s["lum_mean"] - clip_median))
    motion_s     = max(samples, key=lambda s: s["frame_delta"])
    wide_range_s = max(samples, key=lambda s: s["hist_spread"])

    selections = [
        ("shadow",     shadow_s,     "lum_mean",    shadow_s["lum_mean"]),
        ("highlight",  highlight_s,  "lum_mean",    highlight_s["lum_mean"]),
        ("midtone",    midtone_s,    "lum_mean",    midtone_s["lum_mean"]),
        ("motion",     motion_s,     "frame_delta", motion_s["frame_delta"]),
        ("wide_range", wide_range_s, "hist_spread", wide_range_s["hist_spread"]),
    ]

    print(
        f"  frame_analysis [{path_label}] ({len(samples)} samples / "
        f"{end_frame - start_frame} frame window):",
    )
    for frame_id, s, mk, mv in selections:
        print(f"    {frame_id:<12} idx={s['frame_idx']:>6}  {mk}={mv:.3f}")

    # --- Phase 2: full-res extraction ---
    target_indices = list({s["frame_idx"] for _, s, _, _ in selections})
    bgr_frames = _extract_full_res_frames(clip_path, target_indices)

    results = []
    for frame_id, s, metric_key, metric_val in selections:
        idx = s["frame_idx"]
        results.append({
            "frame_id": frame_id,
            "frame_idx": idx,
            "timecode": _frame_index_to_timecode(idx, fps),
            "bgr": bgr_frames[idx],
            "lum_mean": s["lum_mean"],
            "lum_median": clip_median,
            "hist_spread": s["hist_spread"],
            "frame_delta": s["frame_delta"],
            "selection_metric_key": metric_key,
            "selection_metric_value": metric_val,
        })

    return results
