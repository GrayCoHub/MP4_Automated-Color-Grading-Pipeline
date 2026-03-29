"""
summary_sheet.py — Run summary grid.

Single PNG at output_root/run_summary.png.
One cell per qualifying clip — midtone frame, V0 raw only.
Labeled: clip_name + session + nd_tag.
Grid layout, auto-columns based on clip count.
"""

import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_BG = (20, 20, 20)
LABEL_BG = (15, 15, 15)
TEXT_COLOR = (210, 210, 210)
ACCENT_COLOR = (180, 140, 60)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _get_font(size: int):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def compose_summary_sheet(
    clip_records: list,
    output_path: str,
    sheet_width: int = 3200,
    font_size: int = 22,
) -> None:
    """
    Compose and save the run summary sheet.

    clip_records: list of dicts, each with:
        - clip_info: dict from file_discovery
        - midtone_frame: BGR numpy array (raw V0)
    output_path: full path for run_summary.png
    """
    if not clip_records:
        return

    num_clips = len(clip_records)
    # Auto-columns: roughly square grid
    num_cols = math.ceil(math.sqrt(num_clips))
    num_rows = math.ceil(num_clips / num_cols)

    cell_width = sheet_width // num_cols
    cell_height = int(cell_width * 9 / 16)
    label_h = font_size + 10

    title_h = font_size * 2 + 16
    total_h = title_h + num_rows * (cell_height + label_h)

    sheet = Image.new("RGB", (sheet_width, total_h), (5, 5, 5))
    draw = ImageDraw.Draw(sheet)

    font_title = _get_font(int(font_size * 1.6))
    font_label = _get_font(font_size)

    # Title bar
    draw.rectangle([0, 0, sheet_width, title_h], fill=(25, 25, 25))
    draw.text(
        (16, 8),
        f"Phase 1 Run Summary  —  {num_clips} qualifying clips  |  V0 Raw  |  Midtone frame",
        font=font_title,
        fill=ACCENT_COLOR,
    )

    for idx, record in enumerate(clip_records):
        row = idx // num_cols
        col = idx % num_cols

        x = col * cell_width
        y = title_h + row * (cell_height + label_h)

        bgr = record["midtone_frame"]
        resized = cv2.resize(bgr, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        cell_pil = _bgr_to_pil(resized)
        sheet.paste(cell_pil, (x, y))

        # Label bar
        clip_info = record["clip_info"]
        label = f"{clip_info['clip_name']}  |  {clip_info['session']}  |  {clip_info['nd_tag']}"
        draw.rectangle([x, y + cell_height, x + cell_width, y + cell_height + label_h], fill=LABEL_BG)
        draw.text((x + 6, y + cell_height + 4), label, font=font_label, fill=TEXT_COLOR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, "PNG")
