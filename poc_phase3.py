"""
poc_phase3.py -- Phase 3 single-frame bracket POC.

Processing chain per frame:
  1. Read frame from Phase 1 graded video (cv2.VideoCapture CPU)
  2. Apply CLAHE clipLimit=2.0 tileGrid=(8,8) on LAB L channel (Phase 2 winner)
  3. Apply each bracket variant B1-B7 on top
  4. Save individual PNG per variant
  5. Save combined_all_variants.png

Output folder: phase3_contact_sheets\poc\frame_{FRAME_NUMBER}\
Each run creates a new subfolder -- previous runs preserved.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration -- set before running
# ---------------------------------------------------------------------------

FRAME_NUMBER = 775
LUT_PATH = r"C:\dev\LUTs_folder\DLOGM_REC709_65_CUBE_ZRG_V1.cube"
INPUT_SCAN_FOLDER = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase1_output_video"
OUTPUT_ROOT = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase3_contact_sheets\poc"

OUTPUT_DIR = Path(OUTPUT_ROOT) / f"frame_{FRAME_NUMBER}"

# ---------------------------------------------------------------------------
# Input video selection
# ---------------------------------------------------------------------------

def select_input_video() -> str:
    scan = Path(INPUT_SCAN_FOLDER)
    videos = sorted(
        p for p in scan.iterdir()
        if p.suffix.upper() in (".MP4", ".MOV", ".MXF", ".AVI")
    )
    if not videos:
        print(f"No video files found in {INPUT_SCAN_FOLDER}")
        sys.exit(1)

    print(f"\nAvailable videos in {INPUT_SCAN_FOLDER}:")
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")
    print()

    raw = input("Select input video (enter number or full path): ").strip()

    if raw.isdigit():
        idx = int(raw) - 1
        if idx < 0 or idx >= len(videos):
            print(f"Invalid selection: {raw}")
            sys.exit(1)
        return str(videos[idx])
    else:
        if not Path(raw).is_file():
            print(f"File not found: {raw}")
            sys.exit(1)
        return raw

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
# Phase 2 winner: CLAHE applied to every frame before bracket variants
# ---------------------------------------------------------------------------

def apply_clahe_base(frame: np.ndarray) -> np.ndarray:
    """CLAHE clipLimit=2.0 tileGrid=(8,8) on LAB L channel -- Phase 2 winner."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_Lab2BGR)

# ---------------------------------------------------------------------------
# Bracket variant implementations (B1-B7)
# All receive the CLAHE-processed frame and return BGR uint8.
# ---------------------------------------------------------------------------

def apply_b1_clahe_only(frame: np.ndarray) -> np.ndarray:
    """B1: CLAHE only -- reference, no additional adjustment."""
    return frame.copy()


def apply_b2_sat10(frame: np.ndarray) -> np.ndarray:
    """B2: CLAHE + Sat+10 -- HSV S channel x 1.10."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_b3_sat20(frame: np.ndarray) -> np.ndarray:
    """B3: CLAHE + Sat+20 -- HSV S channel x 1.20."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.20, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_b4_cool_shadows(frame: np.ndarray) -> np.ndarray:
    """B4: CLAHE + Cool shadows -- b-=10 x shadow_weight, a-=3 x shadow_weight."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    l, a, b = cv2.split(lab)
    shadow_weight = np.clip(1.0 - (l / 128.0), 0.0, 1.0)
    b = b - 10.0 * shadow_weight
    a = a - 3.0 * shadow_weight
    lab_result = cv2.merge([
        l.astype(np.uint8),
        np.clip(a, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
    ])
    return cv2.cvtColor(lab_result, cv2.COLOR_Lab2BGR)


def apply_b5_vibrance(frame: np.ndarray) -> np.ndarray:
    """B5: CLAHE + Vibrance -- S += 60 x (1 - S/255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    boost_weight = 1.0 - (sat / 255.0)
    hsv[:, :, 1] = np.clip(sat + 60.0 * boost_weight, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _build_scurve_lut() -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    y = x.copy()
    shadow = x < 0.33
    y[shadow] = x[shadow] + 0.06 * (1.0 - x[shadow] / 0.33)
    highlight = x > 0.66
    y[highlight] = x[highlight] - 0.06 * ((x[highlight] - 0.66) / 0.34)
    return (np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)

_SCURVE_LUT = _build_scurve_lut()

def apply_b6_scurve(frame: np.ndarray) -> np.ndarray:
    """B6: CLAHE + S-curve -- fixed tone curve on LAB L channel only."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_curved = cv2.LUT(l, _SCURVE_LUT)
    return cv2.cvtColor(cv2.merge([l_curved, a, b]), cv2.COLOR_Lab2BGR)


def apply_b7_sat10_cool(frame: np.ndarray) -> np.ndarray:
    """B7: CLAHE + Sat+10 + Cool shadows -- B2 then B4 combined."""
    return apply_b4_cool_shadows(apply_b2_sat10(frame))


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS = [
    ("B1", "CLAHE only",       apply_b1_clahe_only),
    ("B2", "CLAHE+Sat10",      apply_b2_sat10),
    ("B3", "CLAHE+Sat20",      apply_b3_sat20),
    ("B4", "CLAHE+Cool",       apply_b4_cool_shadows),
    ("B5", "CLAHE+Vibrance",   apply_b5_vibrance),
    ("B6", "CLAHE+S-curve",    apply_b6_scurve),
    ("B7", "CLAHE+Sat10+Cool", apply_b7_sat10_cool),
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
    video_path = select_input_video()
    print(f"\nInput video:  {video_path}")
    print(f"Frame number: {FRAME_NUMBER}")

    # Extract frame
    print("Extracting frame...")
    frame = extract_frame(video_path, FRAME_NUMBER)
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # Apply Phase 2 winner (CLAHE) as base for all variants
    print("Applying CLAHE base (Phase 2 winner)...")
    clahe_frame = apply_clahe_base(frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Apply variants and save individual PNGs
    results = []
    print()
    for vid, vname, fn in VARIANTS:
        graded = fn(clahe_frame)
        out_path = OUTPUT_DIR / f"{vid}_{vname.replace(' ', '_').replace('+', '_')}.png"
        cv2.imwrite(str(out_path), graded)
        results.append((vid, vname, graded))
        print(f"  {vid} {vname:<22} -> {out_path.name}")

    # Save combined side-by-side
    combined_path = OUTPUT_DIR / "combined_all_variants.png"
    save_combined(results, combined_path, FRAME_NUMBER)
    print(f"\n  Combined       -> {combined_path.name}")

    print("\nDone. 7 variants + 1 combined = 8 images.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
