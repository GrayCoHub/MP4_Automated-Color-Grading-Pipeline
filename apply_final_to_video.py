"""
apply_final_to_video.py -- Apply the confirmed master processing chain to a D-Log M source clip.

Full processing chain:
  D-Log M source
    -> Zeb Gardner LUT       (ffmpeg lut3d -- applied during frame read)
      -> CLAHE clipLimit=2.0 (Phase 2 winner -- OpenCV, applied per frame)
        -> Phase 3 winner    (user-selected B1-B7 -- OpenCV, applied per frame)

Implementation:
  ffmpeg read process  : decode input + apply LUT, emit raw BGR24 frames to stdout pipe
  Python frame loop    : read raw frames, apply CLAHE then variant, write to stdin pipe
  ffmpeg write process : encode BGR24 pipe frames to libx264; audio copied from original

Prompts:
  1. Select input D-Log M video from input_video\
  2. Select Phase 3 winner variant (B1-B7)

Output: phase3_output_video\{input_stem}_final_{variant_id}.MP4
Encoding: libx264 -crf 18 -preset fast -pix_fmt yuv420p (CPU, no NVENC)
Progress: every 100 frames with elapsed time and ETA
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LUT_PATH        = r"C:\dev\LUTs_folder\DLOGM_REC709_65_CUBE_ZRG_V1.cube"
INPUT_SCAN_FOLDER = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\input_video"
OUTPUT_FOLDER   = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase3_output_video"

# ---------------------------------------------------------------------------
# ffmpeg filtergraph path escaping (Windows drive-letter colon)
# Same logic as apply_lut_to_video.py.
# ---------------------------------------------------------------------------

def _filtergraph_path(path: str) -> str:
    """
    Escape a path for use in an ffmpeg filtergraph value on Windows.
    Converts backslashes to forward slashes, then escapes the drive-letter
    colon as \\: so ffmpeg's filtergraph parser does not treat it as a
    separator, then wraps in single quotes.
    e.g.  C:\\dev\\LUTs_folder\\my.cube  ->  'C\\:/dev/LUTs_folder/my.cube'
    """
    p = str(path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return f"'{p}'"

# ---------------------------------------------------------------------------
# Video probe
# ---------------------------------------------------------------------------

def _probe_video(path: str) -> tuple:
    """Return (width, height, fps, total_frames) via cv2.VideoCapture."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, total

# ---------------------------------------------------------------------------
# Input and variant selection prompts
# ---------------------------------------------------------------------------

def select_input_video() -> str:
    scan = Path(INPUT_SCAN_FOLDER)
    if not scan.is_dir():
        print(f"ERROR: Input folder not found: {INPUT_SCAN_FOLDER}")
        sys.exit(1)
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
    path = Path(raw.strip('"'))
    if not path.is_file():
        print(f"File not found: {raw}")
        sys.exit(1)
    return str(path)


def select_variant() -> tuple:
    """Display B1-B7 menu. Return (variant_id, variant_name, variant_fn)."""
    print("\nPhase 3 bracket variants:")
    for vid, vname, _ in VARIANTS:
        print(f"  {vid:<4} {vname}")
    print()
    raw = input("Select Phase 3 winner variant (B1-B7): ").strip().upper()
    for vid, vname, fn in VARIANTS:
        if vid == raw:
            return vid, vname, fn
    print(f"Invalid selection: {raw}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 2 winner: CLAHE clipLimit=2.0 tileGrid=(8,8) on LAB L channel
# Applied to every frame before the Phase 3 variant.
# ---------------------------------------------------------------------------

def apply_clahe_base(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_Lab2BGR)

# ---------------------------------------------------------------------------
# Phase 3 bracket variant implementations (B1-B7)
# Each receives the CLAHE-processed BGR uint8 frame and returns BGR uint8.
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


VARIANTS = [
    ("B1", "CLAHE only",            apply_b1_clahe_only),
    ("B2", "CLAHE + Sat+10",        apply_b2_sat10),
    ("B3", "CLAHE + Sat+20",        apply_b3_sat20),
    ("B4", "CLAHE + Cool shadows",  apply_b4_cool_shadows),
    ("B5", "CLAHE + Vibrance",      apply_b5_vibrance),
    ("B6", "CLAHE + S-curve",       apply_b6_scurve),
    ("B7", "CLAHE + Sat+10 + Cool", apply_b7_sat10_cool),
]

# ---------------------------------------------------------------------------
# Pipe frame reader -- reads exactly n bytes, handles short reads
# ---------------------------------------------------------------------------

def _read_exact(pipe, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf

# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(input_path: str, variant_id: str, variant_name: str, variant_fn) -> tuple:
    """
    Run the full pipeline via two ffmpeg subprocesses connected through Python.

    ffmpeg-read  :  input.mp4  ->  lut3d filter  ->  raw BGR24 stdout
    Python loop  :  raw BGR24  ->  CLAHE  ->  variant  ->  raw BGR24
    ffmpeg-write :  raw BGR24 stdin  ->  libx264  ->  output.MP4
                    (audio stream copied directly from original input)

    Returns (output_path, elapsed_seconds, frames_processed).
    """
    if not Path(LUT_PATH).is_file():
        raise RuntimeError(f"LUT not found: {LUT_PATH}")

    w, h, fps, total_frames = _probe_video(input_path)
    frame_size = w * h * 3   # BGR24: 3 bytes per pixel

    input_p = Path(input_path)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = str(Path(OUTPUT_FOLDER) / f"{input_p.stem}_final_{variant_id}.MP4")

    lut_filter = f"lut3d={_filtergraph_path(LUT_PATH)}"

    # ffmpeg read: decode input, apply LUT, pipe raw BGR24 frames to stdout
    cmd_read = [
        "ffmpeg",
        "-i", input_path,
        "-vf", lut_filter,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
        "-loglevel", "error",
    ]

    # ffmpeg write: encode BGR24 frames from stdin pipe
    #   Second -i carries the original audio track (1:a:0? -- optional, no error if absent)
    cmd_write = [
        "ffmpeg",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        "-loglevel", "error",
        "-y",
        output_path,
    ]

    print(f"  Output:  {output_path}")
    print(f"  Chain:   Zeb Gardner LUT  ->  CLAHE clipLimit=2.0  ->  {variant_id} {variant_name}")
    if total_frames:
        print(f"  Frames:  {total_frames}")
    print()

    t_start = time.time()
    frame_count = 0
    last_printed = 0

    proc_read  = subprocess.Popen(cmd_read,  stdout=subprocess.PIPE, stderr=None)
    proc_write = subprocess.Popen(cmd_write, stdin=subprocess.PIPE,  stderr=None)

    try:
        while True:
            raw = _read_exact(proc_read.stdout, frame_size)
            if len(raw) < frame_size:
                break   # end of stream

            # frombuffer returns read-only view; .copy() makes it writable for OpenCV
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
            frame = apply_clahe_base(frame)
            frame = variant_fn(frame)
            proc_write.stdin.write(frame.tobytes())

            frame_count += 1
            if frame_count - last_printed >= 100:
                elapsed = time.time() - t_start
                fps_enc = frame_count / elapsed if elapsed > 0 else 0.0
                if total_frames and fps_enc > 0:
                    eta = (total_frames - frame_count) / fps_enc
                    print(
                        f"  Frame {frame_count} / {total_frames}"
                        f"  |  elapsed {elapsed:.1f}s"
                        f"  |  ETA {eta:.1f}s"
                    )
                else:
                    print(f"  Frame {frame_count}  |  elapsed {elapsed:.1f}s")
                last_printed = frame_count
    finally:
        proc_write.stdin.close()

    proc_read.wait()
    proc_write.wait()

    if proc_read.returncode != 0:
        raise RuntimeError(f"ffmpeg read process exited with code {proc_read.returncode}")
    if proc_write.returncode != 0:
        raise RuntimeError(f"ffmpeg write process exited with code {proc_write.returncode}")

    elapsed = time.time() - t_start
    return output_path, elapsed, frame_count

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    input_path = select_input_video()
    print(f"\nInput: {input_path}")

    variant_id, variant_name, variant_fn = select_variant()
    print(f"\nVariant: {variant_id} -- {variant_name}")
    print()

    try:
        out, elapsed, frames = process_video(input_path, variant_id, variant_name, variant_fn)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"Complete.  {frames} frames  |  {elapsed:.1f}s")
    print(f"Output: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
