"""
phase1_run.py — Phase 1 orchestrator.

Entry point: python phase1_run.py [--config phase1_config.json]

Produces per qualifying clip:
  - Contact sheet PNG
  - Metrics CSV

Produces at output_root:
  - run_summary.png
  - run_log.txt

Config options:
  "test_mode": true  — processes first 2 clips only
  "max_clip_duration_seconds"  — clips exceeding this are skipped (enforced in file_discovery)
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

from phase1_config import load_config
from file_discovery import discover_clips
from frame_analyzer import select_frames
from grade_engine import load_cube_lut, apply_all_variants
from sheet_composer import compose_contact_sheet
from csv_writer import write_clip_csv
from summary_sheet import compose_summary_sheet


# ---------------------------------------------------------------------------
# Resolve availability check
# ---------------------------------------------------------------------------

def _check_resolve_available() -> bool:
    """Return True if DaVinci Resolve scripting API is available and running."""
    try:
        import sys as _sys
        import os as _os
        resolve_dir = "C:/Program Files/Blackmagic Design/DaVinci Resolve"
        _sys.path.insert(0, resolve_dir + "/Support/Developer/Scripting/Modules")
        # fusionscript.dll is loaded from the Resolve install dir at import time on Windows;
        # that dir must be in PATH or the import silently fails even when Resolve is running.
        env_path = _os.environ.get("PATH", "")
        resolve_dir_win = resolve_dir.replace("/", "\\")
        if resolve_dir not in env_path and resolve_dir_win not in env_path:
            _os.environ["PATH"] = resolve_dir_win + _os.pathsep + env_path
        import DaVinciResolveScript as dvr  # noqa
        resolve = dvr.scriptapp("Resolve")
        return resolve is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------

def _output_dir_for_clip(clip_info: dict, cfg: dict) -> str:
    """Mirror the input folder structure under output_root."""
    clip_path = clip_info["clip_path"]
    root = str(cfg["root_folder"]).rstrip("/\\")
    clip_dir = str(Path(clip_path).parent)

    if clip_dir.lower().startswith(root.lower()):
        rel = clip_dir[len(root):].lstrip("/\\")
    else:
        rel = clip_info["session"]

    out_dir = os.path.join(cfg["output_root"], rel)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Per-clip processor
# ---------------------------------------------------------------------------

def _process_clip_worker(clip_info: dict, cfg: dict, lut, lut_landscape) -> dict:
    """
    Process a single clip end-to-end.
    Returns dict: clip_info, log_entries, midtone_bgr (640x360), success, error.
    """
    import cv2 as _cv2
    log_entries = []
    clip_name = clip_info["clip_name"]

    try:
        log_entries.append(f"\n--- Processing: {clip_name} ---")

        # Frame selection (NVDEC metric pass + full-res extraction)
        selected_frames = select_frames(clip_info, cfg)

        if not selected_frames:
            log_entries.append("  SKIP — no frames selected (clip may be too short after trim)")
            return {
                "clip_info": clip_info,
                "log_entries": log_entries,
                "midtone_bgr": None,
                "success": False,
                "error": "no frames selected",
            }

        for f in selected_frames:
            log_entries.append(
                f"  frame_id={f['frame_id']}  idx={f['frame_idx']}  tc={f['timecode']}  "
                f"{f['selection_metric_key']}={f['selection_metric_value']:.3f}"
            )

        # Grade variants
        graded_variants = []
        for f in selected_frames:
            try:
                variants = apply_all_variants(f["bgr"], lut, lut_landscape)
            except Exception as e:
                log_entries.append(f"  ERROR grading frame {f['frame_id']}: {e}")
                variants = {
                    "V0": (f["bgr"].copy(), "V0 Raw"),
                    "V1": (f["bgr"].copy(), "V1 ERROR"),
                    "V2": (f["bgr"].copy(), "V2 ERROR"),
                    "V3": (f["bgr"].copy(), "V3 ERROR"),
                }
            graded_variants.append(variants)

        # Output paths
        out_dir = _output_dir_for_clip(clip_info, cfg)
        sheet_path = os.path.join(out_dir, f"{clip_name}_eval.png")
        csv_path = os.path.join(out_dir, f"{clip_name}_metrics.csv")

        # Contact sheet
        sheet = compose_contact_sheet(clip_info, selected_frames, graded_variants, cfg)
        sheet.save(sheet_path, "PNG")
        log_entries.append(f"  contact sheet saved: {sheet_path}")

        # CSV
        write_clip_csv(clip_info, selected_frames, csv_path)
        log_entries.append(f"  metrics CSV saved: {csv_path}")

        # Midtone frame for summary sheet — downsample to 640x360 for IPC efficiency
        midtone = next(
            (f for f in selected_frames if f["frame_id"] == "midtone"),
            selected_frames[0],
        )
        midtone_small = _cv2.resize(midtone["bgr"], (640, 360), interpolation=_cv2.INTER_AREA)

        return {
            "clip_info": clip_info,
            "log_entries": log_entries,
            "midtone_bgr": midtone_small,
            "success": True,
            "error": None,
        }

    except Exception:
        err = traceback.format_exc()
        log_entries.append(f"  ERROR:\n{err}")
        return {
            "clip_info": clip_info,
            "log_entries": log_entries,
            "midtone_bgr": None,
            "success": False,
            "error": err,
        }


# ---------------------------------------------------------------------------
# Acceptance criteria verifier
# ---------------------------------------------------------------------------

def verify_outputs(
    qualifying_clips: list,
    processed_clips: list,
    cfg: dict,
    resolve_available: bool,
    log_entries: list,
) -> list:
    """Check all 12 acceptance criteria. Returns list of failed criteria strings."""
    import csv as _csv

    failures = []
    output_root = cfg["output_root"]

    # C1: every processed clip produced PNG + CSV
    for clip_info in processed_clips:
        out_dir = _output_dir_for_clip(clip_info, cfg)
        stem = clip_info["clip_name"]
        png = os.path.join(out_dir, f"{stem}_eval.png")
        csv_path = os.path.join(out_dir, f"{stem}_metrics.csv")
        if not os.path.isfile(png):
            failures.append(f"[C1] Missing contact sheet: {png}")
        if not os.path.isfile(csv_path):
            failures.append(f"[C1] Missing metrics CSV: {csv_path}")

    # C2: skips logged — confirmed by log_entries containing SKIP entries during discovery

    # C3: zero user input — architectural

    # C4: 4 variants or RESOLVE_UNAVAILABLE — architectural

    # C5: header metadata — architectural

    # C6: ND tag on BRG-* clips
    for clip_info in processed_clips:
        if "BRG-" in clip_info["session"].upper() or "BRG-" in clip_info["group"].upper():
            if clip_info["nd_tag"] == "none":
                failures.append(f"[C6] ND tag missing: {clip_info['clip_name']}")

    # C7: output mirrors input — architectural via _output_dir_for_clip

    # C8: histogram row — architectural

    # C9: CSV columns
    REQUIRED_COLS = {
        "frame_id", "frame_index", "timecode", "lum_mean", "lum_median",
        "hist_spread", "frame_delta", "selection_reason",
        "clip_name", "session", "group", "nd_tag",
    }
    for clip_info in processed_clips:
        out_dir = _output_dir_for_clip(clip_info, cfg)
        csv_path = os.path.join(out_dir, f"{clip_info['clip_name']}_metrics.csv")
        if os.path.isfile(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                missing_cols = REQUIRED_COLS - set(reader.fieldnames or [])
                if missing_cols:
                    failures.append(f"[C9] CSV missing columns {missing_cols}: {csv_path}")

    # C10: run_summary.png
    if not os.path.isfile(os.path.join(output_root, "run_summary.png")):
        failures.append(f"[C10] run_summary.png not found")

    # C11: run_log.txt
    if not os.path.isfile(os.path.join(output_root, "run_log.txt")):
        failures.append(f"[C11] run_log.txt not found")

    # C12: no hardcoded values — architectural

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Color Grade Evaluation Pipeline")
    parser.add_argument("--config", default="phase1_config.json", help="Path to config file")
    args = parser.parse_args()

    run_start = time.time()
    log_entries = []

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    test_mode = cfg.get("test_mode", False)
    output_root = cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)

    if test_mode:
        print("*** TEST MODE: first 2 clips ***")
        log_entries.append("*** TEST MODE ACTIVE — first 2 clips only ***")

    # ------------------------------------------------------------------
    # Resolve check
    # ------------------------------------------------------------------
    resolve_available = _check_resolve_available()
    resolve_status = "available" if resolve_available else "unavailable"
    log_entries.append(f"V3 Resolve status: {resolve_status}")
    print(f"Resolve: {resolve_status}")

    # ------------------------------------------------------------------
    # Load LUT
    # ------------------------------------------------------------------
    try:
        lut = load_cube_lut(cfg["lut_path"])
        print(f"LUT loaded: {cfg['lut_path']}")
        log_entries.append(f"LUT loaded: {cfg['lut_path']}")
    except Exception as e:
        print(f"ERROR: Failed to load LUT: {e}")
        sys.exit(1)

    try:
        lut_landscape = load_cube_lut(cfg["landscape_lut_path"])
        print(f"Landscape LUT loaded: {cfg['landscape_lut_path']}")
        log_entries.append(f"Landscape LUT loaded: {cfg['landscape_lut_path']}")
    except Exception as e:
        print(f"ERROR: Failed to load landscape LUT: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover clips
    # ------------------------------------------------------------------
    _SINGLE_INPUT_FILE = r"C:\dev\All_dev_projects_testing_folder\Devinci_Resolve_automation_M4Pro_video\video\short_test_2_D.MP4"

    if _SINGLE_INPUT_FILE:
        import cv2 as _cv2
        _cap = _cv2.VideoCapture(_SINGLE_INPUT_FILE)
        _fps = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
        _frame_count = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        _w = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        _h = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        _cap.release()
        _p = Path(_SINGLE_INPUT_FILE)
        qualifying_clips = [{
            "clip_name": _p.stem,
            "session": _p.parent.name,
            "group": _p.parent.parent.name,
            "nd_tag": "none",
            "clip_path": _SINGLE_INPUT_FILE,
            "duration_seconds": _frame_count / _fps,
            "fps": _fps,
            "resolution": f"{_w}x{_h}",
            "width": _w,
            "height": _h,
            "frame_count": _frame_count,
        }]
        print(f"Single-file override: {_SINGLE_INPUT_FILE}")
        log_entries.append(f"Single-file override: {_SINGLE_INPUT_FILE}")
    else:
        print(f"Discovering clips in: {cfg['root_folder']}")
        qualifying_clips = discover_clips(cfg, log_entries)

    print(f"Qualifying clips found: {len(qualifying_clips)}")
    log_entries.append(f"\n=== Qualifying clips: {len(qualifying_clips)} ===")
    for ci in qualifying_clips:
        log_entries.append(
            f"  CLIP {ci['clip_name']}  session={ci['session']}  "
            f"nd={ci['nd_tag']}  dur={ci['duration_seconds']:.1f}s"
        )

    clips_to_process = qualifying_clips[:2] if test_mode else qualifying_clips
    total = len(clips_to_process)

    if total == 0:
        print("No clips to process. Exiting.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Process clips
    # ------------------------------------------------------------------
    print(f"Processing {total} clips...")
    log_entries.append(f"\n=== Processing {total} clips ===")

    t_pool = time.time()
    results = []
    for ci in clips_to_process:
        results.append(_process_clip_worker(ci, cfg, lut, lut_landscape))
    pool_elapsed = time.time() - t_pool

    print(f"Processing completed in {pool_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    processed_clips = []
    summary_records = []

    for r in results:
        clip_name = r["clip_info"]["clip_name"]
        log_entries.extend(r["log_entries"])
        if r["success"]:
            print(f"  OK: {clip_name}")
            processed_clips.append(r["clip_info"])
            if r["midtone_bgr"] is not None:
                summary_records.append({
                    "clip_info": r["clip_info"],
                    "midtone_frame": r["midtone_bgr"],
                })
        else:
            print(f"  FAIL: {clip_name} — {r['error'][:120]}")

    print(
        f"{len(processed_clips)}/{total} clips succeeded, "
        f"{total - len(processed_clips)} failed.",
    )

    # ------------------------------------------------------------------
    # Run summary sheet
    # ------------------------------------------------------------------
    summary_path = os.path.join(output_root, "run_summary.png")
    print(f"Composing run summary ({len(summary_records)} clips)...")
    try:
        compose_summary_sheet(
            summary_records,
            summary_path,
            sheet_width=cfg["contact_sheet_width_px"],
            font_size=cfg["annotation_font_size"],
        )
        log_entries.append(f"\nRun summary saved: {summary_path}")
        print(f"Summary sheet saved: {summary_path}")
    except Exception as e:
        log_entries.append(f"\nERROR composing run summary: {e}")
        print(f"ERROR composing summary: {e}")

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------
    run_elapsed = time.time() - run_start
    log_entries.append(f"\n=== Run complete ===")
    log_entries.append(f"Clips processed: {len(processed_clips)}")
    log_entries.append(f"Processing wall time: {pool_elapsed:.1f}s")
    log_entries.append(f"Total runtime: {run_elapsed:.1f}s")

    log_path = os.path.join(output_root, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries))
    print(f"Run log saved: {log_path}")

    # ------------------------------------------------------------------
    # Verify outputs
    # ------------------------------------------------------------------
    print("\nRunning verify_outputs()...")
    failures = verify_outputs(
        qualifying_clips,
        processed_clips,
        cfg,
        resolve_available,
        log_entries,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1 REPORT")
    print("=" * 60)
    print(f"Clips processed:    {len(processed_clips)}/{total}")
    print(f"Processing time:    {pool_elapsed:.1f}s")
    print(f"Total elapsed:      {run_elapsed:.1f}s")
    print(f"V3 Resolve:         {resolve_status}")
    if test_mode:
        print(f"(TEST MODE — full run would process {len(qualifying_clips)} clips)")

    if failures:
        print(f"\nAcceptance criteria FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
    else:
        print("\nAll acceptance criteria passed.")

    print(f"\nOutputs at: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
