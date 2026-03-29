"""
phase1_config.py — Config loader and validator for Phase 1 pipeline.
All parameters sourced from phase1_config.json. No hardcoded values.
"""

import json
import os
from pathlib import Path


REQUIRED_KEYS = [
    "root_folder",
    "lut_path",
    "output_root",
    "min_clip_duration_seconds",
    "trim_margin_percent",
    "frame_sample_interval",
    "contact_sheet_width_px",
    "annotation_font_size",
]


def load_config(config_path: str = "phase1_config.json") -> dict:
    """Load and validate phase1_config.json. Raises on missing keys or invalid paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = Path(os.path.join(script_dir, config_path))
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    # Validate root_folder exists
    if not os.path.isdir(cfg["root_folder"]):
        raise FileNotFoundError(f"root_folder does not exist: {cfg['root_folder']}")

    # Validate LUT path
    if not os.path.isfile(cfg["lut_path"]):
        raise FileNotFoundError(f"LUT file not found: {cfg['lut_path']}")

    # Ensure output_root is writable (create if needed)
    os.makedirs(cfg["output_root"], exist_ok=True)

    return cfg
