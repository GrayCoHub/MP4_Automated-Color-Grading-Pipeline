"""
csv_writer.py — Per-clip metrics CSV output.

Columns:
  frame_id, frame_index, timecode, lum_mean, lum_median, hist_spread,
  frame_delta, selection_reason, clip_name, session, group, nd_tag
"""

import csv
import os


COLUMNS = [
    "frame_id",
    "frame_index",
    "timecode",
    "lum_mean",
    "lum_median",
    "hist_spread",
    "frame_delta",
    "selection_reason",
    "clip_name",
    "session",
    "group",
    "nd_tag",
]


def write_clip_csv(clip_info: dict, selected_frames: list, output_path: str) -> None:
    """
    Write per-clip metrics CSV.

    clip_info: metadata dict from file_discovery
    selected_frames: list of frame dicts from frame_analyzer
    output_path: full path to output .csv file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for frame in selected_frames:
            writer.writerow({
                "frame_id": frame["frame_id"],
                "frame_index": frame["frame_idx"],
                "timecode": frame["timecode"],
                "lum_mean": round(frame["lum_mean"], 4),
                "lum_median": round(frame["lum_median"], 4),
                "hist_spread": round(frame["hist_spread"], 4),
                "frame_delta": round(frame["frame_delta"], 4),
                "selection_reason": frame["selection_metric_key"],
                "clip_name": clip_info["clip_name"],
                "session": clip_info["session"],
                "group": clip_info["group"],
                "nd_tag": clip_info["nd_tag"],
            })
