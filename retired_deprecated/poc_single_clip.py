"""
poc_single_clip.py — Single-clip proof-of-concept for Phase 1 pipeline.

Hardcoded clip: first *_D.MP4 in D:\\Mavic4_Pro\\BRG-16\\
Loads phase1_config.json for parameters (LUT path, contact sheet width, etc.)
Times each stage independently.
No DaVinciResolveScript dependency.

Optimizations vs v1:
  - frame_analysis: single forward decode pass (one seek to window start only)
  - v1_lut: applied at 1920x1080 then upsampled, reducing pixel work 4x
"""

import os
import sys
import time
import traceback
import csv
import multiprocessing
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Hardcoded clip path (first *_D.MP4 in D:\Mavic4_Pro\BRG-16\)
# ---------------------------------------------------------------------------
CLIP_PATH = r"D:\Mavic4_Pro\BRG-16\DJI_20260223132708_0001_D.MP4"

CONFIG_PATH = "phase1_config.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts():
    return time.perf_counter()


def _elapsed(t0):
    return time.perf_counter() - t0


def _get_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Stage: Load config
# ---------------------------------------------------------------------------

def stage_load_config():
    import json
    t0 = _ts()
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"load_config: {_elapsed(t0):.3f}s", flush=True)
    return cfg


# ---------------------------------------------------------------------------
# Stage: Load LUT
# ---------------------------------------------------------------------------

def stage_load_lut(lut_path):
    t0 = _ts()
    lut_size = None
    data = []
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                lut_size = int(line.split()[-1])
                continue
            if any(line.upper().startswith(k) for k in ("DOMAIN_", "TITLE", "LUT_1D")):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    continue
    if lut_size is None:
        raise ValueError("Could not find LUT_3D_SIZE in .cube file")
    arr = np.array(data, dtype=np.float32)
    arr = arr.reshape((lut_size, lut_size, lut_size, 3))
    arr = arr[:, :, :, ::-1]  # RGB -> BGR output
    print(f"load_lut (size={lut_size}): {_elapsed(t0):.3f}s", flush=True)
    return arr


# ---------------------------------------------------------------------------
# Stage: Video open
# ---------------------------------------------------------------------------

def stage_video_open(clip_path):
    t0 = _ts()
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {clip_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"video_open ({width}x{height} {fps:.2f}fps {frame_count}fr): {_elapsed(t0):.3f}s", flush=True)
    return fps, frame_count, width, height


# ---------------------------------------------------------------------------
# Stage: Duration check
# ---------------------------------------------------------------------------

def stage_duration_check(frame_count, fps, min_duration, max_duration=None):
    t0 = _ts()
    duration = frame_count / fps if fps > 0 else 0.0
    ok = duration >= min_duration and (max_duration is None or duration <= max_duration)
    bounds = f">= {min_duration}s"
    if max_duration is not None:
        bounds += f" and <= {max_duration}s"
    print(
        f"duration_check ({duration:.1f}s {bounds}: {'PASS' if ok else 'FAIL'}): "
        f"{_elapsed(t0):.3f}s",
        flush=True,
    )
    if duration < min_duration:
        raise RuntimeError(f"Clip too short: {duration:.1f}s < {min_duration}s")
    if max_duration is not None and duration > max_duration:
        raise RuntimeError(f"Clip too long: {duration:.1f}s > {max_duration}s")
    return duration


# ---------------------------------------------------------------------------
# Stage: Frame analysis (metric pass — no full frame storage)
# ---------------------------------------------------------------------------

def _compute_metrics_from_gray(gray, prev_gray):
    """Shared metric computation from a uint8 grayscale frame."""
    lum_mean = float(np.mean(gray))
    if prev_gray is not None and prev_gray.shape == gray.shape:
        delta = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
    else:
        delta = 0.0
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    nz = np.nonzero(hist)[0]
    hist_spread = float(nz[-1] - nz[0]) if len(nz) >= 2 else 0.0
    return lum_mean, delta, hist_spread


def _build_selections(samples):
    """Select 5 representative frames from metric samples list."""
    all_lum = [s["lum_mean"] for s in samples]
    clip_median = float(np.median(all_lum))

    shadow_s    = min(samples, key=lambda s: s["lum_mean"])
    highlight_s = max(samples, key=lambda s: s["lum_mean"])
    midtone_s   = min(samples, key=lambda s: abs(s["lum_mean"] - clip_median))
    motion_s    = max(samples, key=lambda s: s["frame_delta"])
    widerange_s = max(samples, key=lambda s: s["hist_spread"])

    return clip_median, {
        "shadow":     {"idx": shadow_s["idx"],    "metric_key": "lum_mean",    "metric_val": shadow_s["lum_mean"],    "lum_mean": shadow_s["lum_mean"],    "lum_median": clip_median, "hist_spread": shadow_s["hist_spread"],    "frame_delta": shadow_s["frame_delta"]},
        "highlight":  {"idx": highlight_s["idx"], "metric_key": "lum_mean",    "metric_val": highlight_s["lum_mean"], "lum_mean": highlight_s["lum_mean"], "lum_median": clip_median, "hist_spread": highlight_s["hist_spread"], "frame_delta": highlight_s["frame_delta"]},
        "midtone":    {"idx": midtone_s["idx"],   "metric_key": "lum_mean",    "metric_val": midtone_s["lum_mean"],   "lum_mean": midtone_s["lum_mean"],   "lum_median": clip_median, "hist_spread": midtone_s["hist_spread"],   "frame_delta": midtone_s["frame_delta"]},
        "motion":     {"idx": motion_s["idx"],    "metric_key": "frame_delta", "metric_val": motion_s["frame_delta"], "lum_mean": motion_s["lum_mean"],    "lum_median": clip_median, "hist_spread": motion_s["hist_spread"],    "frame_delta": motion_s["frame_delta"]},
        "wide_range": {"idx": widerange_s["idx"], "metric_key": "hist_spread", "metric_val": widerange_s["hist_spread"], "lum_mean": widerange_s["lum_mean"], "lum_median": clip_median, "hist_spread": widerange_s["hist_spread"], "frame_delta": widerange_s["frame_delta"]},
    }


def _frame_analysis_nvdec(clip_path, frame_count, trim_margin_percent, frame_sample_interval):
    """
    NVDEC path: uses cv2.cudacodec.createVideoReader with targetSz=(320,180).
    NVDEC hardware-scales frames to 320x180 before download — no CPU resize.
    Reads from frame 0; discards frames before analysis window start.
    Downloads GpuMat only at interval sample points; all other frames
    are decoded on GPU and dropped without touching CPU memory.

    Frame format from this build: (H, W, 4) uint16 BGRA.
    Luminance: top 8 bits of each channel (>> 8) → uint8 BGR → cvtColor GRAY.
    """
    trim = trim_margin_percent / 100.0
    start = int(frame_count * trim)
    end   = int(frame_count * (1.0 - trim))

    params = cv2.cudacodec.VideoReaderInitParams()
    params.targetSz = (320, 180)
    reader = cv2.cudacodec.createVideoReader(clip_path, [], params)

    samples = []
    prev_gray = None
    current = 0

    while current < end:
        ret, gpu_frame = reader.nextFrame()
        if not ret:
            break

        # Only download + process at interval sample points within the window
        if current >= start and (current - start) % frame_sample_interval == 0:
            cpu = gpu_frame.download()                          # (180, 320, 4) uint16
            bgr8 = (cpu[:, :, :3] >> 8).astype(np.uint8)      # top 8 bits → uint8 BGR
            gray = cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY)
            lum_mean, delta, hist_spread = _compute_metrics_from_gray(gray, prev_gray)
            samples.append({
                "idx": current,
                "lum_mean": lum_mean,
                "frame_delta": delta,
                "hist_spread": hist_spread,
            })
            prev_gray = gray

        current += 1

    del reader
    return samples, end - start


def _frame_analysis_cpu(clip_path, frame_count, trim_margin_percent, frame_sample_interval):
    """
    CPU fallback path: single seek to window start, pure forward decode.
    Resizes to 320x180 on CPU for metric computation.
    """
    trim = trim_margin_percent / 100.0
    start = int(frame_count * trim)
    end   = int(frame_count * (1.0 - trim))

    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    samples = []
    prev_gray = None
    current = start

    while current < end:
        ret, frame = cap.read()
        if not ret:
            break

        if (current - start) % frame_sample_interval == 0:
            small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            lum_mean, delta, hist_spread = _compute_metrics_from_gray(gray, prev_gray)
            samples.append({
                "idx": current,
                "lum_mean": lum_mean,
                "frame_delta": delta,
                "hist_spread": hist_spread,
            })
            prev_gray = gray

        current += 1

    cap.release()
    return samples, end - start


def stage_frame_analysis(clip_path, frame_count, fps, trim_margin_percent, frame_sample_interval):
    """
    Dispatch to NVDEC path if cudacodec is available, else CPU fallback.
    NVDEC path: hardware-decoded 320x180 frames, no CPU resize.
    CPU path: single seek to window start, software resize to 320x180.
    """
    t0 = _ts()

    use_nvdec = hasattr(cv2, "cudacodec") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    path_label = "NVDEC" if use_nvdec else "CPU"

    try:
        if use_nvdec:
            samples, window_size = _frame_analysis_nvdec(
                clip_path, frame_count, trim_margin_percent, frame_sample_interval
            )
        else:
            samples, window_size = _frame_analysis_cpu(
                clip_path, frame_count, trim_margin_percent, frame_sample_interval
            )
    except Exception as e:
        if use_nvdec:
            # NVDEC failed at runtime — fall back to CPU
            print(f"  NVDEC failed ({e}), falling back to CPU", flush=True)
            path_label = "CPU-fallback"
            samples, window_size = _frame_analysis_cpu(
                clip_path, frame_count, trim_margin_percent, frame_sample_interval
            )
        else:
            raise

    if not samples:
        raise RuntimeError("No samples collected during frame analysis")

    clip_median, selections = _build_selections(samples)

    elapsed = _elapsed(t0)
    print(
        f"frame_analysis [{path_label}] ({len(samples)} samples from {window_size} frame window): "
        f"{elapsed:.3f}s",
        flush=True,
    )
    print("  Selected frame indices:", flush=True)
    for frame_id, s in selections.items():
        print(f"    {frame_id:<12} idx={s['idx']:>6}  {s['metric_key']}={s['metric_val']:.3f}", flush=True)

    return selections, clip_median


# ---------------------------------------------------------------------------
# Stage: Frame extraction (pull 5 frames at full res)
# ---------------------------------------------------------------------------

def stage_frame_extraction(clip_path, selections, fps):
    """Seek to each selected index and extract full-res BGR frame."""
    t0 = _ts()
    cap = cv2.VideoCapture(clip_path)
    frames = {}
    for frame_id, s in selections.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["idx"])
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not extract frame {s['idx']} for {frame_id}")
        frames[frame_id] = frame
    cap.release()

    elapsed = _elapsed(t0)
    h, w = next(iter(frames.values())).shape[:2]
    print(f"frame_extraction (5 frames @ {w}x{h}): {elapsed:.3f}s", flush=True)
    return frames


# ---------------------------------------------------------------------------
# Grade helpers
# ---------------------------------------------------------------------------

def _apply_lut3d(bgr_frame, lut):
    lut_size = lut.shape[0]
    scale = (lut_size - 1) / 255.0
    b = np.clip(bgr_frame[:, :, 0].astype(np.float32) * scale, 0, lut_size - 1)
    g = np.clip(bgr_frame[:, :, 1].astype(np.float32) * scale, 0, lut_size - 1)
    r = np.clip(bgr_frame[:, :, 2].astype(np.float32) * scale, 0, lut_size - 1)
    b0 = np.floor(b).astype(np.int32); b1 = np.minimum(b0 + 1, lut_size - 1)
    g0 = np.floor(g).astype(np.int32); g1 = np.minimum(g0 + 1, lut_size - 1)
    r0 = np.floor(r).astype(np.int32); r1 = np.minimum(r0 + 1, lut_size - 1)
    tb = (b - b0)[..., np.newaxis]; tg = (g - g0)[..., np.newaxis]; tr = (r - r0)[..., np.newaxis]
    result = (
        lut[b0, g0, r0] * (1-tb)*(1-tg)*(1-tr) + lut[b0, g0, r1] * (1-tb)*(1-tg)*tr
        + lut[b0, g1, r0] * (1-tb)*tg*(1-tr)   + lut[b0, g1, r1] * (1-tb)*tg*tr
        + lut[b1, g0, r0] * tb*(1-tg)*(1-tr)   + lut[b1, g0, r1] * tb*(1-tg)*tr
        + lut[b1, g1, r0] * tb*tg*(1-tr)       + lut[b1, g1, r1] * tb*tg*tr
    )
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def _build_scurve_lut():
    x = np.arange(256, dtype=np.float32) / 255.0
    y = 0.5 + (x - 0.5) * 1.15
    y = y + 0.08 * np.sin(2 * np.pi * y) * (1 - y) * y
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

_SCURVE_LUT = _build_scurve_lut()


# ---------------------------------------------------------------------------
# Grade stages
# ---------------------------------------------------------------------------

def stage_v0_grade(frames):
    t0 = _ts()
    result = {fid: f.copy() for fid, f in frames.items()}
    print(f"v0_raw (copy): {_elapsed(t0):.3f}s", flush=True)
    return result


def stage_v1_grade(frames, lut):
    """
    Apply LUT at 1920x1080 then upsample back to original resolution.
    Reduces pixel count 4x (3840x2160 -> 1920x1080) for the trilinear interp,
    then INTER_LINEAR upsample. Acceptable quality loss for contact sheet use.
    """
    t0 = _ts()
    result = {}
    for fid, f in frames.items():
        orig_h, orig_w = f.shape[:2]
        small = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA)
        graded_small = _apply_lut3d(small, lut)
        result[fid] = cv2.resize(graded_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    print(f"v1_lut (1920x1080 + upsample): {_elapsed(t0):.3f}s", flush=True)
    return result


def stage_v2_grade(frames):
    t0 = _ts()
    result = {}
    for fid, f in frames.items():
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l = cv2.LUT(l, _SCURVE_LUT)
        a = np.clip(128 + (a.astype(np.float32) - 128) * 1.15, 0, 255).astype(np.uint8)
        b = np.clip(128 + (b.astype(np.float32) - 128) * 1.15, 0, 255).astype(np.uint8)
        result[fid] = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)
    print(f"v2_scurve: {_elapsed(t0):.3f}s", flush=True)
    return result


def stage_v3_warm_grade(frames):
    """
    V3 Warm grade — OpenCV implementation (no Resolve dependency).
    Lifts shadows warm (add orange tint to shadows), cools highlights slightly,
    boosts overall saturation +10, adds mild vignette.
    """
    t0 = _ts()
    result = {}
    h_ref, w_ref = next(iter(frames.values())).shape[:2]

    # Vignette mask (computed once, reused for all frames of same size)
    Y, X = np.ogrid[:h_ref, :w_ref]
    cx, cy = w_ref / 2.0, h_ref / 2.0
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.35 * dist ** 2, 0.0, 1.0).astype(np.float32)

    for fid, f in frames.items():
        img = f.astype(np.float32) / 255.0

        # Apply vignette
        img = img * vignette[:, :, np.newaxis]

        # Warm shadow lift: add orange (R+, G+slight, B-) proportional to (1 - luminance)
        lum = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
        shadow_mask = np.clip(1.0 - lum * 2.0, 0.0, 1.0)[:, :, np.newaxis]
        img[:, :, 2] = np.clip(img[:, :, 2] + 0.06 * shadow_mask[:, :, 0], 0, 1)  # R up
        img[:, :, 1] = np.clip(img[:, :, 1] + 0.02 * shadow_mask[:, :, 0], 0, 1)  # G slight
        img[:, :, 0] = np.clip(img[:, :, 0] - 0.04 * shadow_mask[:, :, 0], 0, 1)  # B down

        # Cool highlight tint: subtle blue push in highlights
        hi_mask = np.clip((lum - 0.7) * 3.0, 0.0, 1.0)[:, :, np.newaxis]
        img[:, :, 0] = np.clip(img[:, :, 0] + 0.03 * hi_mask[:, :, 0], 0, 1)  # B up

        # Saturation boost +10% in HSV space
        bgr_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
        result[fid] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    print(f"v3_warm: {_elapsed(t0):.3f}s", flush=True)
    return result


# ---------------------------------------------------------------------------
# Stage: Contact sheet composition
# ---------------------------------------------------------------------------

def stage_contact_sheet(
    frames_v0, frames_v1, frames_v2, frames_v3,
    selections, fps, frame_count, width, height, duration,
    cfg, clip_path, output_path
):
    t0 = _ts()

    sheet_width = cfg["contact_sheet_width_px"]
    font_size = cfg["annotation_font_size"]
    num_cols = 4
    frame_ids = list(selections.keys())
    num_rows = len(frame_ids)

    cell_w = sheet_width // num_cols
    cell_h = int(cell_w * 9 / 16)
    line_h = font_size + 6

    header_h = line_h * 2 + 16
    col_label_h = line_h + 12
    cell_footer_h = line_h + 8
    hist_h = int(cell_h * 0.35)
    total_h = header_h + col_label_h + num_rows * (cell_h + cell_footer_h) + hist_h

    sheet = Image.new("RGB", (sheet_width, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font_sm = _get_font(font_size)
    font_lg = _get_font(int(font_size * 1.4), bold=True)

    HEADER_BG   = (20, 20, 20)
    COL_BG      = (35, 35, 35)
    FOOTER_BG   = (15, 15, 15)
    HIST_BG     = (10, 10, 10)
    ACCENT      = (180, 140, 60)
    TEXT        = (210, 210, 210)
    DIM         = (110, 110, 110)

    stem = Path(clip_path).stem
    nd_tag = "ND-16"  # derived from BRG-16 folder
    session = "BRG-16"
    group = "Mavic4_Pro"

    # Header
    draw.rectangle([0, 0, sheet_width, header_h], fill=HEADER_BG)
    draw.text((12, 8), f"{stem}    |    {nd_tag}", font=font_lg, fill=ACCENT)
    draw.text(
        (12, 8 + line_h + 4),
        f"{session} / {group}    |    {fps:.2f}fps  {width}x{height}    |    {duration:.1f}s    |    {frame_count} frames    |    trim: 10%-90%",
        font=font_sm, fill=TEXT,
    )

    # Column labels
    y_col = header_h
    draw.rectangle([0, y_col, sheet_width, y_col + col_label_h], fill=COL_BG)
    for col_i, label in enumerate(["V0 Raw", "V1 Official LUT", "V2 No-LUT S-curve", "V3 Warm"]):
        draw.text((col_i * cell_w + 8, y_col + 6), label, font=font_sm, fill=ACCENT)

    # Frame rows
    all_variant_frames = [frames_v0, frames_v1, frames_v2, frames_v3]
    y_rows = header_h + col_label_h

    for row_i, frame_id in enumerate(frame_ids):
        y_top = y_rows + row_i * (cell_h + cell_footer_h)
        s = selections[frame_id]

        for col_i, vframes in enumerate(all_variant_frames):
            x = col_i * cell_w
            bgr = vframes[frame_id]
            resized = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(rgb), (x, y_top))

            y_footer = y_top + cell_h
            draw.rectangle([x, y_footer, x + cell_w, y_footer + cell_footer_h], fill=FOOTER_BG)
            tc_secs = s["idx"] / fps if fps > 0 else 0
            tc = f"{int(tc_secs//3600):02d}:{int(tc_secs%3600//60):02d}:{int(tc_secs%60):02d}"
            draw.text(
                (x + 6, y_footer + 4),
                f"{frame_id}  |  {tc}  |  {s['metric_key']}={s['metric_val']:.2f}",
                font=font_sm, fill=TEXT,
            )

    # Histogram row
    y_hist = y_rows + num_rows * (cell_h + cell_footer_h)
    draw.rectangle([0, y_hist, sheet_width, y_hist + hist_h], fill=HIST_BG)
    hist_cell_w = sheet_width // num_rows
    for i, frame_id in enumerate(frame_ids):
        bgr = frames_v0[frame_id]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hmax = hist.max() if hist.max() > 0 else 1
        himg = Image.new("RGB", (hist_cell_w - 4, hist_h - 20), HIST_BG)
        hdraw = ImageDraw.Draw(himg)
        bw = (hist_cell_w - 4) / 256.0
        for bi, bv in enumerate(hist):
            bh = int((bv / hmax) * (hist_h - 24))
            if bh > 0:
                x0 = int(bi * bw); x1 = max(x0 + 1, int((bi + 1) * bw))
                hdraw.rectangle([x0, hist_h - 24 - bh, x1, hist_h - 24], fill=(100, 180, 100))
        sheet.paste(himg, (i * hist_cell_w + 2, y_hist + 10))
        draw.text((i * hist_cell_w + 6, y_hist + 2), frame_id, font=font_sm, fill=(100, 200, 100))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, "PNG")
    print(f"contact_sheet ({sheet_width}x{total_h}): {_elapsed(t0):.3f}s", flush=True)
    print(f"  saved: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Stage: CSV write
# ---------------------------------------------------------------------------

def stage_csv_write(selections, fps, clip_path, cfg, output_path):
    t0 = _ts()
    stem = Path(clip_path).stem
    nd_tag = "ND-16"
    session = "BRG-16"
    group = "Mavic4_Pro"

    columns = [
        "frame_id", "frame_index", "timecode", "lum_mean", "lum_median",
        "hist_spread", "frame_delta", "selection_reason",
        "clip_name", "session", "group", "nd_tag",
    ]
    rows = []
    for frame_id, s in selections.items():
        tc_secs = s["idx"] / fps if fps > 0 else 0
        tc = f"{int(tc_secs//3600):02d}:{int(tc_secs%3600//60):02d}:{int(tc_secs%60):02d}:{int((tc_secs % 1) * fps):02d}"
        rows.append({
            "frame_id": frame_id,
            "frame_index": s["idx"],
            "timecode": tc,
            "lum_mean": round(s["lum_mean"], 4),
            "lum_median": round(s["lum_median"], 4),
            "hist_spread": round(s["hist_spread"], 4),
            "frame_delta": round(s["frame_delta"], 4),
            "selection_reason": s["metric_key"],
            "clip_name": stem,
            "session": session,
            "group": group,
            "nd_tag": nd_tag,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"csv_write: {_elapsed(t0):.3f}s", flush=True)
    print(f"  saved: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Multiprocessing worker — must be top-level for Windows spawn pickling
# ---------------------------------------------------------------------------

def _worker_process_clip(args):
    """
    Process a single clip end-to-end. Returns dict of per-stage timings.
    Receives (clip_path, cfg, lut, output_subdir) as a tuple.
    All stage functions are module-level — safe to pickle on Windows.
    stdout from workers is suppressed; timings returned to parent.
    """
    clip_path, cfg, lut, output_subdir = args
    timings = {}
    stem = Path(clip_path).stem

    try:
        t = _ts()
        fps, frame_count, width, height = stage_video_open(clip_path)
        timings["video_open"] = _elapsed(t)

        duration = frame_count / fps if fps > 0 else 0.0
        min_dur = cfg["min_clip_duration_seconds"]
        max_dur = cfg.get("max_clip_duration_seconds", None)
        if duration < min_dur:
            raise RuntimeError(f"Clip too short: {duration:.1f}s < {min_dur}s")
        if max_dur is not None and duration > max_dur:
            raise RuntimeError(f"Clip too long: {duration:.1f}s > {max_dur}s")
        timings["duration_check"] = 0.0

        t = _ts()
        selections, _ = stage_frame_analysis(
            clip_path, frame_count, fps,
            cfg["trim_margin_percent"],
            cfg["frame_sample_interval"],
        )
        timings["frame_analysis"] = _elapsed(t)

        t = _ts()
        frames_raw = stage_frame_extraction(clip_path, selections, fps)
        timings["frame_extraction"] = _elapsed(t)

        t = _ts(); frames_v0 = stage_v0_grade(frames_raw);       timings["v0_raw"]    = _elapsed(t)
        t = _ts(); frames_v1 = stage_v1_grade(frames_raw, lut);  timings["v1_lut"]    = _elapsed(t)
        t = _ts(); frames_v2 = stage_v2_grade(frames_raw);       timings["v2_scurve"] = _elapsed(t)
        t = _ts(); frames_v3 = stage_v3_warm_grade(frames_raw);  timings["v3_warm"]   = _elapsed(t)

        out_root = cfg["output_root"]
        sheet_path = os.path.join(out_root, "poc_test", output_subdir, f"{stem}_eval.png")
        csv_path   = os.path.join(out_root, "poc_test", output_subdir, f"{stem}_metrics.csv")

        t = _ts()
        _compose_contact_sheet_worker(
            frames_v0, frames_v1, frames_v2, frames_v3,
            selections, fps, frame_count, width, height, duration,
            cfg, clip_path, sheet_path,
        )
        timings["contact_sheet"] = _elapsed(t)

        t = _ts()
        _write_csv_worker(selections, fps, clip_path, cfg, csv_path)
        timings["csv_write"] = _elapsed(t)

        timings["error"] = None

    except Exception:
        timings["error"] = traceback.format_exc()

    timings["clip"] = stem
    return timings


def _compose_contact_sheet_worker(
    frames_v0, frames_v1, frames_v2, frames_v3,
    selections, fps, frame_count, width, height, duration,
    cfg, clip_path, output_path,
):
    """Contact sheet composition without print statements — safe for worker processes."""
    sheet_width = cfg["contact_sheet_width_px"]
    font_size = cfg["annotation_font_size"]
    frame_ids = list(selections.keys())
    num_rows = len(frame_ids)
    cell_w = sheet_width // 4
    cell_h = int(cell_w * 9 / 16)
    line_h = font_size + 6
    header_h = line_h * 2 + 16
    col_label_h = line_h + 12
    cell_footer_h = line_h + 8
    hist_h = int(cell_h * 0.35)
    total_h = header_h + col_label_h + num_rows * (cell_h + cell_footer_h) + hist_h

    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (sheet_width, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font_sm = _get_font(font_size)
    font_lg = _get_font(int(font_size * 1.4), bold=True)

    HEADER_BG = (20, 20, 20); COL_BG = (35, 35, 35); FOOTER_BG = (15, 15, 15)
    HIST_BG = (10, 10, 10);   ACCENT = (180, 140, 60); TEXT = (210, 210, 210)

    stem = Path(clip_path).stem
    nd_tag = "ND-16"; session = "BRG-16"; group = "Mavic4_Pro"

    draw.rectangle([0, 0, sheet_width, header_h], fill=HEADER_BG)
    draw.text((12, 8), f"{stem}    |    {nd_tag}", font=font_lg, fill=ACCENT)
    draw.text(
        (12, 8 + line_h + 4),
        f"{session} / {group}    |    {fps:.2f}fps  {width}x{height}    |    {duration:.1f}s    |    trim: 10%-90%",
        font=font_sm, fill=TEXT,
    )
    y_col = header_h
    draw.rectangle([0, y_col, sheet_width, y_col + col_label_h], fill=COL_BG)
    for col_i, label in enumerate(["V0 Raw", "V1 Official LUT", "V2 No-LUT S-curve", "V3 Warm"]):
        draw.text((col_i * cell_w + 8, y_col + 6), label, font=font_sm, fill=ACCENT)

    all_variants = [frames_v0, frames_v1, frames_v2, frames_v3]
    y_rows = header_h + col_label_h
    for row_i, frame_id in enumerate(frame_ids):
        y_top = y_rows + row_i * (cell_h + cell_footer_h)
        s = selections[frame_id]
        for col_i, vframes in enumerate(all_variants):
            x = col_i * cell_w
            bgr = vframes[frame_id]
            resized = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            sheet.paste(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), (x, y_top))
            y_footer = y_top + cell_h
            draw.rectangle([x, y_footer, x + cell_w, y_footer + cell_footer_h], fill=FOOTER_BG)
            tc_secs = s["idx"] / fps if fps > 0 else 0
            tc = f"{int(tc_secs//3600):02d}:{int(tc_secs%3600//60):02d}:{int(tc_secs%60):02d}"
            draw.text((x + 6, y_footer + 4),
                      f"{frame_id}  |  {tc}  |  {s['metric_key']}={s['metric_val']:.2f}",
                      font=font_sm, fill=TEXT)

    y_hist = y_rows + num_rows * (cell_h + cell_footer_h)
    draw.rectangle([0, y_hist, sheet_width, y_hist + hist_h], fill=HIST_BG)
    hist_cell_w = sheet_width // num_rows
    for i, frame_id in enumerate(frame_ids):
        gray = cv2.cvtColor(frames_v0[frame_id], cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hmax = hist.max() if hist.max() > 0 else 1
        himg = Image.new("RGB", (hist_cell_w - 4, hist_h - 20), HIST_BG)
        hdraw = ImageDraw.Draw(himg)
        bw = (hist_cell_w - 4) / 256.0
        for bi, bv in enumerate(hist):
            bh = int((bv / hmax) * (hist_h - 24))
            if bh > 0:
                x0 = int(bi * bw); x1 = max(x0 + 1, int((bi + 1) * bw))
                hdraw.rectangle([x0, hist_h - 24 - bh, x1, hist_h - 24], fill=(100, 180, 100))
        sheet.paste(himg, (i * hist_cell_w + 2, y_hist + 10))
        draw.text((i * hist_cell_w + 6, y_hist + 2), frame_id, font=font_sm, fill=(100, 200, 100))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, "PNG")


def _write_csv_worker(selections, fps, clip_path, cfg, output_path):
    """CSV write without print statements — safe for worker processes."""
    stem = Path(clip_path).stem
    nd_tag = "ND-16"; session = "BRG-16"; group = "Mavic4_Pro"
    columns = [
        "frame_id", "frame_index", "timecode", "lum_mean", "lum_median",
        "hist_spread", "frame_delta", "selection_reason",
        "clip_name", "session", "group", "nd_tag",
    ]
    rows = []
    for frame_id, s in selections.items():
        tc_secs = s["idx"] / fps if fps > 0 else 0
        tc = f"{int(tc_secs//3600):02d}:{int(tc_secs%3600//60):02d}:{int(tc_secs%60):02d}:{int((tc_secs % 1) * fps):02d}"
        rows.append({
            "frame_id": frame_id, "frame_index": s["idx"], "timecode": tc,
            "lum_mean": round(s["lum_mean"], 4), "lum_median": round(s["lum_median"], 4),
            "hist_spread": round(s["hist_spread"], 4), "frame_delta": round(s["frame_delta"], 4),
            "selection_reason": s["metric_key"], "clip_name": stem,
            "session": session, "group": group, "nd_tag": nd_tag,
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CLIPS = [
    r"D:\Mavic4_Pro\BRG-16\DJI_20260223132708_0001_D.MP4",
    r"D:\Mavic4_Pro\BRG-16\DJI_20260223132958_0004_D.MP4",
]


def _print_worker_timings(label, results):
    """Print per-stage timings returned from worker(s)."""
    stages = ["video_open", "frame_analysis", "frame_extraction",
              "v0_raw", "v1_lut", "v2_scurve", "v3_warm", "contact_sheet", "csv_write"]
    print(f"\n  {label} per-clip stage breakdown:", flush=True)
    for r in results:
        if r.get("error"):
            print(f"    {r['clip']}: ERROR\n{r['error']}", flush=True)
            continue
        print(f"    {r['clip']}:", flush=True)
        for s in stages:
            if s in r:
                print(f"      {s:<20} {r[s]:.3f}s", flush=True)


def main():
    run_start = _ts()
    print("=" * 60, flush=True)
    print("poc_single_clip.py — multiprocessing comparison", flush=True)
    print("=" * 60, flush=True)

    try:
        cfg = stage_load_config()
        lut = stage_load_lut(cfg["lut_path"])
        print(f"Clips: {[Path(c).stem for c in CLIPS]}", flush=True)

        # ------------------------------------------------------------------
        # Sequential: process clip 1 then clip 2
        # ------------------------------------------------------------------
        print("\n--- SEQUENTIAL (2 clips) ---", flush=True)
        t_seq = _ts()
        seq_results = []
        for clip in CLIPS:
            r = _worker_process_clip((clip, cfg, lut, "seq"))
            seq_results.append(r)
            status = "ERROR" if r.get("error") else "OK"
            print(f"  {r['clip']}: {status}", flush=True)
        wall_seq = _elapsed(t_seq)
        print(f"Sequential wall time: {wall_seq:.2f}s", flush=True)
        _print_worker_timings("Sequential", seq_results)

        # ------------------------------------------------------------------
        # Parallel: Pool(processes=2), both clips dispatched simultaneously
        # ------------------------------------------------------------------
        print("\n--- PARALLEL (2 clips, Pool processes=2) ---", flush=True)
        t_par = _ts()
        args = [(clip, cfg, lut, "par") for clip in CLIPS]
        with multiprocessing.Pool(processes=2) as pool:
            par_results = pool.map(_worker_process_clip, args)
        wall_par = _elapsed(t_par)
        for r in par_results:
            status = "ERROR" if r.get("error") else "OK"
            print(f"  {r['clip']}: {status}", flush=True)
        print(f"Parallel wall time:   {wall_par:.2f}s", flush=True)
        _print_worker_timings("Parallel", par_results)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 60, flush=True)
        print("MULTIPROCESSING SUMMARY", flush=True)
        print("=" * 60, flush=True)
        print(f"  Sequential (2 clips):  {wall_seq:.2f}s", flush=True)
        print(f"  Parallel   (2 clips):  {wall_par:.2f}s", flush=True)
        speedup = wall_seq / wall_par if wall_par > 0 else 0
        print(f"  Speedup:               {speedup:.2f}x", flush=True)
        print(f"  Total script elapsed:  {_elapsed(run_start):.2f}s", flush=True)

    except Exception:
        print("\n--- TRACEBACK ---", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # safe no-op on non-frozen; required pattern on Windows
    main()
