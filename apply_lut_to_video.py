"""
apply_lut_to_video.py — Apply one or more .cube LUTs to a video file via ffmpeg.

Usage:
    python apply_lut_to_video.py

Scans C:\\dev\\LUTs_folder\\ for .cube files, prompts for selection,
prompts for input video path, writes one output file per selected LUT
to phase1_output_video\\.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Encode method
# ---------------------------------------------------------------------------

ENCODE_METHOD = "cpu"  # options: cpu, nvenc

# ENCODE_METHOD options:
#   "cpu"   -- libx264 software encode (current)
#              universally compatible
#   "nvenc" -- h264_nvenc hardware encode
#              requires driver 570+ (not compatible
#              with current driver 560.94)
#              do not enable until driver updated

LUT_FOLDER = r"C:\dev\LUTs_folder"
OUTPUT_FOLDER = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\phase1_output_video"


# ---------------------------------------------------------------------------
# ffprobe: get total frame count
# ---------------------------------------------------------------------------

def _get_total_frames(input_path: str) -> int:
    """
    Query total frame count via ffprobe.
    Tries nb_frames from stream metadata first; falls back to duration * fps.
    Returns 0 if both fail.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,duration,r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            input_path,
        ],
        capture_output=True,
        text=True,
    )

    nb_frames = None
    duration = None
    r_frame_rate = None

    for line in result.stdout.splitlines():
        if line.startswith("nb_frames=") and line.split("=", 1)[1] not in ("N/A", ""):
            try:
                nb_frames = int(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("duration=") and line.split("=", 1)[1] not in ("N/A", ""):
            try:
                duration = float(line.split("=", 1)[1])
            except ValueError:
                pass
        elif line.startswith("r_frame_rate="):
            r_frame_rate = line.split("=", 1)[1]

    if nb_frames is not None and nb_frames > 0:
        return nb_frames

    # Fallback: duration * fps
    if duration and r_frame_rate:
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den)
            return int(duration * fps)
        except (ValueError, ZeroDivisionError):
            pass

    return 0


# ---------------------------------------------------------------------------
# ffmpeg path escaping for lut3d filtergraph (Windows drive-letter colon)
# ---------------------------------------------------------------------------

def _filtergraph_path(path: str) -> str:
    """
    Escape a path for use in an ffmpeg filtergraph value on Windows.
    Steps:
      1. Convert all backslashes to forward slashes (preserves drive letter).
      2. Escape the drive-letter colon as \\: so ffmpeg's filtergraph parser
         does not interpret it as an option separator.
      3. Wrap in single quotes so spaces in the path are handled correctly.
    e.g.  C:\\dev\\LUTs_folder\\my.cube  ->  'C\\:/dev/LUTs_folder/my.cube'
    """
    p = str(path).replace("\\", "/")
    # Escape the colon after the drive letter — must come after the \\ -> /
    # conversion so the drive letter is intact.
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return f"'{p}'"


# ---------------------------------------------------------------------------
# ffmpeg: apply LUT and stream progress
# ---------------------------------------------------------------------------

def process_video(input_path: str, lut_path: Path, total_frames: int) -> tuple:
    """
    Run ffmpeg to apply lut_path to input_path.
    Streams -progress pipe:1 output and prints every 100 frames.
    Returns (output_path, elapsed_seconds).
    """
    input_p = Path(input_path)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = str(Path(OUTPUT_FOLDER) / f"{input_p.stem}_{lut_path.stem}.MP4")

    lut_label = lut_path.stem
    lut_filter = f"lut3d={_filtergraph_path(str(lut_path))}"

    if ENCODE_METHOD == "nvenc":
        video_codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]
    else:
        video_codec_args = ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", lut_filter,
        *video_codec_args,
        "-c:a", "copy",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-y",
        output_path,
    ]

    print(f"  Output: {output_path}")
    if total_frames:
        print(f"  Total frames: {total_frames}")

    t_start = time.time()
    last_printed = 0
    current_frame = 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,   # stderr passes through to terminal (shows ffmpeg errors)
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                current_frame = int(line.split("=", 1)[1])
            except ValueError:
                continue

            if current_frame - last_printed >= 100:
                elapsed = time.time() - t_start
                if total_frames and total_frames > 0:
                    fps_enc = current_frame / elapsed if elapsed > 0 else 0.0
                    eta = (total_frames - current_frame) / fps_enc if fps_enc > 0 else 0.0
                    frame_display = f"{current_frame} / {total_frames}"
                    eta_str = f"  |  ETA {eta:.1f}s"
                else:
                    frame_display = str(current_frame)
                    eta_str = ""
                print(
                    f"  Frame {frame_display}"
                    f"  |  LUT: {lut_label}"
                    f"  |  elapsed {elapsed:.1f}s"
                    f"{eta_str}"
                )
                last_printed = current_frame

    proc.wait()
    elapsed = time.time() - t_start

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    return output_path, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Scan for .cube files
    if not os.path.isdir(LUT_FOLDER):
        print(f"ERROR: LUT folder not found: {LUT_FOLDER}")
        sys.exit(1)

    cube_files = sorted(
        p for p in Path(LUT_FOLDER).iterdir()
        if p.suffix.lower() == ".cube" and p.is_file()
    )

    if not cube_files:
        print(f"No .cube files found in {LUT_FOLDER}")
        sys.exit(1)

    print(f"\nLUTs found in {LUT_FOLDER}:")
    for i, p in enumerate(cube_files, 1):
        print(f"  {i}. {p.name}")

    # 2. Prompt for selection
    print()
    raw = input("Select LUT numbers to apply (e.g. 1,2,3): ").strip()
    try:
        selections = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        print("ERROR: Invalid input — enter comma-separated numbers.")
        sys.exit(1)

    selected_luts = []
    for num in selections:
        if num < 1 or num > len(cube_files):
            print(f"ERROR: {num} is out of range (1–{len(cube_files)}).")
            sys.exit(1)
        selected_luts.append(cube_files[num - 1])

    if not selected_luts:
        print("No LUTs selected.")
        sys.exit(0)

    # 3. Prompt for input video
    print()
    input_path = input("Input video path: ").strip().strip('"')
    if not os.path.isfile(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Get total frame count once — shared across all LUT runs
    print("\nProbing video...")
    total_frames = _get_total_frames(input_path)
    if total_frames:
        print(f"  Total frames: {total_frames}")
    else:
        print("  Total frames: unknown (ffprobe could not determine)")

    # 4. Process each selected LUT
    output_paths = []
    for lut_path in selected_luts:
        print(f"\nApplying LUT: {lut_path.name}")
        try:
            out, elapsed = process_video(input_path, lut_path, total_frames)
            output_paths.append((out, elapsed))
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    # 5. Report
    print("\n" + "=" * 60)
    print("Complete.")
    if output_paths:
        print("Output files:")
        for out, elapsed in output_paths:
            print(f"  {out}  ({elapsed:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
