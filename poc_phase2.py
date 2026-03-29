"""
poc_phase2.py ??? Phase 2 single-frame variant POC.

Extracts one frame from the Phase 1 winner video, applies all 7 Phase 2
refinement variants (A1-A7), and saves:
  - 7 individual PNGs  (one per variant)
  - 1 combined side-by-side PNG (all 7 variants)

Output folder: phase2_contact_sheets\poc\

Set INPUT_VIDEO and FRAME_NUMBER before running.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration ??? set before running
# ---------------------------------------------------------------------------

INPUT_VIDEO = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase1_output_video\DJI_20260223132708_0001_D_DLOGM_REC709_65_CUBE_ZRG_V1.MP4"
FRAME_NUMBER = 775

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "phase2_contact_sheets" / "poc" / f"frame_{FRAME_NUMBER}"

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame(video_path: str, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number >= total:
        raise RuntimeError(f"Frame {frame_number} out of range (total {total})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_number}")
    return frame

# ---------------------------------------------------------------------------
# Variant implementations (A1-A7)
# All operate on BGR uint8 numpy arrays, return BGR uint8.
# ---------------------------------------------------------------------------

def apply_a1_baseline(frame: np.ndarray) -> np.ndarray:
    """A1: Baseline ??? Phase 1 winner unmodified."""
    return frame.copy()


def apply_a2_auto_levels(frame: np.ndarray) -> np.ndarray:
    """A2: Auto levels ??? stretch each channel histogram to 0-255."""
    result = np.empty_like(frame)
    for ch in range(3):
        tmp = cv2.normalize(frame[:, :, ch], None, 0, 255, cv2.NORM_MINMAX)
        result[:, :, ch] = tmp
    return result


def apply_a3_clahe(frame: np.ndarray) -> np.ndarray:
    """A3: CLAHE ??? local contrast enhancement on L channel in LAB."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_Lab2BGR)


def _build_scurve_lut() -> np.ndarray:
    """
    Gentle S-curve: lifts shadows slightly, pulls highlights slightly.
    Applied to L channel only to avoid color shift.
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    y = x.copy()
    # Shadow lift: input 0.0-0.33 gets a small upward nudge
    shadow = x < 0.33
    y[shadow] = x[shadow] + 0.06 * (1.0 - x[shadow] / 0.33)
    # Highlight pull: input 0.66-1.0 gets a small downward nudge
    highlight = x > 0.66
    y[highlight] = x[highlight] - 0.06 * ((x[highlight] - 0.66) / 0.34)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

_SCURVE_LUT = _build_scurve_lut()

def apply_a3b_clahe_light(frame: np.ndarray) -> np.ndarray:
    """A3b: CLAHE-Light ??? local contrast enhancement, gentler clipLimit=1.0."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_Lab2BGR)


def apply_a4_scurve(frame: np.ndarray) -> np.ndarray:
    """A4: Gentle S-curve ??? lift shadows, pull highlights on L channel."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_curved = cv2.LUT(l, _SCURVE_LUT)
    return cv2.cvtColor(cv2.merge([l_curved, a, b]), cv2.COLOR_Lab2BGR)


def apply_a5_saturation(frame: np.ndarray) -> np.ndarray:
    """A5: Saturation +15% ??? uniform HSV saturation boost."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_a6_cool_shadows(frame: np.ndarray) -> np.ndarray:
    """
    A6: Cool shadow toning ??? push shadows toward blue/teal, preserve warm highlights.
    Works in LAB: reduce b channel (yellow-blue axis) in shadow regions only.
    OpenCV LAB: b=128 is neutral, <128 = blue, >128 = yellow.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    l, a, b = cv2.split(lab)
    # Shadow weight: 1.0 at black, 0.0 at midtone and above
    shadow_weight = np.clip(1.0 - (l / 128.0), 0.0, 1.0)
    b = b - 10.0 * shadow_weight   # push blue in shadows
    a = a - 3.0 * shadow_weight    # slight green tint (teal rather than pure blue)
    lab_result = cv2.merge([
        l.astype(np.uint8),
        np.clip(a, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
    ])
    return cv2.cvtColor(lab_result, cv2.COLOR_Lab2BGR)


def apply_a7_vibrance(frame: np.ndarray) -> np.ndarray:
    """
    A7: Vibrance ??? smart saturation boost that targets undersaturated pixels.
    Fully saturated pixels receive little or no boost.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    # Boost weight inversely proportional to existing saturation
    boost_weight = 1.0 - (sat / 255.0)
    hsv[:, :, 1] = np.clip(sat + 60.0 * boost_weight, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = [
    ("A1", "Baseline",       apply_a1_baseline),
    ("A2", "Auto Levels",    apply_a2_auto_levels),
    ("A3", "CLAHE",          apply_a3_clahe),
    ("A3b", "CLAHE-Light",   apply_a3b_clahe_light),
    ("A4", "Gentle S-curve", apply_a4_scurve),
    ("A5", "Saturation +15", apply_a5_saturation),
    ("A6", "Cool Shadows",   apply_a6_cool_shadows),
    ("A7", "Vibrance",       apply_a7_vibrance),
]

# ---------------------------------------------------------------------------
# Combined image
# ---------------------------------------------------------------------------

def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _get_font(size: int):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def save_combined(results: list, output_path: Path, frame_number: int) -> None:
    """
    Build a side-by-side PNG: one column per variant, label below each.
    results: list of (variant_id, variant_name, bgr_array)
    """
    cell_w = 640
    cell_h = 360
    label_h = 36
    padding = 4
    font = _get_font(20)

    num = len(results)
    total_w = num * (cell_w + padding) - padding
    total_h = cell_h + label_h

    canvas = Image.new("RGB", (total_w, total_h), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)

    for i, (vid, vname, bgr) in enumerate(results):
        resized = cv2.resize(bgr, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cell_pil = _bgr_to_pil(resized)
        x = i * (cell_w + padding)
        canvas.paste(cell_pil, (x, 0))
        label = f"{vid} {vname}"
        draw.rectangle([x, cell_h, x + cell_w, cell_h + label_h], fill=(25, 25, 25))
        draw.text((x + 6, cell_h + 6), label, font=font, fill=(210, 210, 210))

    canvas.save(str(output_path), "PNG")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Input video:  {INPUT_VIDEO}")
    print(f"Frame number: {FRAME_NUMBER}")

    # Extract frame
    print("Extracting frame...")
    frame = extract_frame(INPUT_VIDEO, FRAME_NUMBER)
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Apply variants and save individual PNGs
    results = []
    print()
    for vid, vname, fn in VARIANTS:
        graded = fn(frame)
        out_path = OUTPUT_DIR / f"{vid}_{vname.replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), graded)
        results.append((vid, vname, graded))
        print(f"  {vid} {vname:<20} -> {out_path}")

    # Save combined side-by-side
    combined_path = OUTPUT_DIR / "combined_all_variants.png"
    save_combined(results, combined_path, FRAME_NUMBER)
    print(f"\n  Combined      -> {combined_path}")

    print("\nDone. 8 variants + 1 combined = 9 images.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
