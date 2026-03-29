"""
file_discovery.py — Recursive file finder, filtering, and metadata extraction.

Includes only files matching *_D.MP4 pattern.
Excludes .LRF, .txt, screen-*, and files not matching *_D.MP4.
Excludes clips shorter than min_clip_duration_seconds.
Excludes clips longer than max_clip_duration_seconds.
All skipped files logged with reason.
"""

import os
import re
from pathlib import Path
from typing import Generator

import cv2


def _extract_nd_tag(folder_name: str) -> str:
    """Extract ND tag from BRG-N folder pattern, or return 'none'."""
    match = re.search(r"BRG-(\d+)", folder_name, re.IGNORECASE)
    if match:
        return f"ND-{match.group(1)}"
    return "none"


def _get_clip_duration(clip_path: str) -> float:
    """Return clip duration in seconds using OpenCV. Returns -1 on failure."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return -1.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return -1.0
    return frame_count / fps


def discover_clips(cfg: dict, log_entries: list) -> list:
    """
    Recurse root_folder, find qualifying *_D.MP4 clips.
    Returns list of metadata dicts for each qualifying clip.
    Appends skip reasons to log_entries.
    """
    root = cfg["root_folder"]
    min_duration = cfg["min_clip_duration_seconds"]
    max_duration = cfg.get("max_clip_duration_seconds", None)
    qualifying = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Sort for deterministic ordering
        dirnames.sort()
        for filename in sorted(filenames):
            full_path = os.path.join(dirpath, filename)
            stem = Path(filename).stem
            suffix = Path(filename).suffix.upper()

            # Exclude non-MP4 entirely
            if suffix != ".MP4":
                log_entries.append(f"SKIP [{full_path}] — not .MP4")
                continue

            # Exclude screen-* files
            if filename.lower().startswith("screen-"):
                log_entries.append(f"SKIP [{full_path}] — matches screen-* pattern")
                continue

            # Exclude files not matching *_D.MP4
            if not stem.endswith("_D"):
                log_entries.append(f"SKIP [{full_path}] — does not match *_D.MP4 pattern")
                continue

            # Check duration
            duration = _get_clip_duration(full_path)
            if duration < 0:
                log_entries.append(f"SKIP [{full_path}] — could not read duration")
                continue
            if duration < min_duration:
                log_entries.append(
                    f"SKIP [{full_path}] — duration {duration:.1f}s < {min_duration}s minimum"
                )
                continue
            if max_duration is not None and duration > max_duration:
                log_entries.append(
                    f"SKIP [{full_path}] — duration {duration:.1f}s > {max_duration}s maximum"
                )
                continue

            # Extract metadata from folder structure
            parts = Path(dirpath).parts
            session = parts[-1] if len(parts) >= 1 else "unknown"
            group = parts[-2] if len(parts) >= 2 else "none"

            # ND tag: check session folder first, then group folder
            nd_tag = "none"
            if re.search(r"BRG-\d+", session, re.IGNORECASE):
                nd_tag = _extract_nd_tag(session)
            elif re.search(r"BRG-\d+", group, re.IGNORECASE):
                nd_tag = _extract_nd_tag(group)

            # Get video properties for metadata
            cap = cv2.VideoCapture(full_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            qualifying.append({
                "clip_path": full_path,
                "clip_name": stem,
                "session": session,
                "group": group,
                "nd_tag": nd_tag,
                "duration_seconds": duration,
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
            })

    return qualifying
