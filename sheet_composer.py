"""
sheet_composer.py — Pillow contact sheet layout, annotation, and histogram row.

Layout: 5 rows (frames) x 4 columns (variants) + header + histogram row.
Output width: contact_sheet_width_px from config (height auto-calculated).
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VARIANT_ORDER = ["V0", "V1", "V2", "V3"]
VARIANT_LABELS = {
    "V0": "V0 Raw",
    "V1": "V1 DJI Official LUT",
    "V2": "V2 Zeb Gardner LUT",
    "V3": "V3 No-LUT S-curve",
}

HEADER_BG = (20, 20, 20)
COL_LABEL_BG = (35, 35, 35)
CELL_FOOTER_BG = (15, 15, 15)
HIST_BG = (10, 10, 10)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (180, 140, 60)
RESOLVE_UNAVAIL_COLOR = (180, 60, 60)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _get_font(size: int, bold: bool = False):
    """Try to load a system font; fall back to PIL default."""
    font_candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _draw_histogram(gray_frame: np.ndarray, width: int, height: int) -> Image.Image:
    """Render a luminance histogram as a PIL image."""
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist_max = hist.max() if hist.max() > 0 else 1

    img = Image.new("RGB", (width, height), HIST_BG)
    draw = ImageDraw.Draw(img)

    bar_w = width / 256.0
    for i, val in enumerate(hist):
        bar_h = int((val / hist_max) * (height - 4))
        x0 = int(i * bar_w)
        x1 = max(x0 + 1, int((i + 1) * bar_w))
        if bar_h > 0:
            draw.rectangle([x0, height - bar_h - 2, x1, height - 2], fill=(100, 180, 100))

    return img


def compose_contact_sheet(
    clip_info: dict,
    selected_frames: list,
    graded_variants: list,
    cfg: dict,
) -> Image.Image:
    """
    Compose a contact sheet image.

    selected_frames: list of 5 frame dicts from frame_analyzer
    graded_variants: list of 5 dicts, each mapping variant_id -> (bgr, label)
    cfg: config dict

    Returns a PIL Image.
    """
    sheet_width = cfg["contact_sheet_width_px"]
    font_size = cfg["annotation_font_size"]

    num_cols = 4
    num_rows = len(selected_frames)

    cell_width = sheet_width // num_cols

    # Cell height: maintain 16:9 aspect ratio
    cell_height = int(cell_width * 9 / 16)

    font_sm = _get_font(font_size)
    font_md = _get_font(int(font_size * 1.2))
    font_lg = _get_font(int(font_size * 1.5), bold=True)

    line_h = font_size + 6

    # Heights of each section
    header_h = line_h * 3 + 16
    col_label_h = line_h + 12
    cell_footer_h = line_h + 8
    hist_h = int(cell_height * 0.35)

    total_h = (
        header_h
        + col_label_h
        + num_rows * (cell_height + cell_footer_h)
        + hist_h
    )

    sheet = Image.new("RGB", (sheet_width, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)

    # -----------------------------------------------------------------------
    # Header block
    # -----------------------------------------------------------------------
    draw.rectangle([0, 0, sheet_width, header_h], fill=HEADER_BG)

    clip_name = clip_info["clip_name"]
    nd_tag = clip_info["nd_tag"]
    session = clip_info["session"]
    group = clip_info["group"]
    fps = clip_info["fps"]
    width_px = clip_info["width"]
    height_px = clip_info["height"]
    duration = clip_info["duration_seconds"]
    frame_count = clip_info["frame_count"]
    trim = cfg["trim_margin_percent"]

    line1 = f"{clip_name}    |    {nd_tag}"
    line2 = f"{session} / {group}    |    {fps:.2f}fps  {width_px}x{height_px}    |    {duration:.1f}s    |    {frame_count} frames    |    trim: {trim}%-{100-trim}%"
    line3 = f"Phase 1 Automated Color Grade Evaluation  |  Mavic 4 Pro D-Log M"

    draw.text((12, 8), line1, font=font_lg, fill=ACCENT_COLOR)
    draw.text((12, 8 + line_h + 4), line2, font=font_sm, fill=TEXT_COLOR)
    draw.text((12, 8 + line_h * 2 + 6), line3, font=font_sm, fill=(130, 130, 130))

    # -----------------------------------------------------------------------
    # Column labels
    # -----------------------------------------------------------------------
    y_col_labels = header_h
    draw.rectangle([0, y_col_labels, sheet_width, y_col_labels + col_label_h], fill=COL_LABEL_BG)
    for col_idx, variant_id in enumerate(VARIANT_ORDER):
        x = col_idx * cell_width
        label = VARIANT_LABELS[variant_id]
        draw.text((x + 8, y_col_labels + 6), label, font=font_md, fill=ACCENT_COLOR)

    # -----------------------------------------------------------------------
    # Frame rows
    # -----------------------------------------------------------------------
    y_rows_start = header_h + col_label_h

    for row_idx, (frame_info, variants) in enumerate(zip(selected_frames, graded_variants)):
        y_cell_top = y_rows_start + row_idx * (cell_height + cell_footer_h)

        for col_idx, variant_id in enumerate(VARIANT_ORDER):
            x = col_idx * cell_width

            bgr_img, variant_label = variants[variant_id]

            # Resize frame to cell dimensions
            resized = cv2.resize(bgr_img, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
            cell_pil = _bgr_to_pil(resized)
            sheet.paste(cell_pil, (x, y_cell_top))

            # Cell footer bar
            y_footer = y_cell_top + cell_height
            draw.rectangle(
                [x, y_footer, x + cell_width, y_footer + cell_footer_h],
                fill=CELL_FOOTER_BG,
            )

            frame_id = frame_info["frame_id"]
            metric_key = frame_info["selection_metric_key"]
            metric_val = frame_info["selection_metric_value"]
            tc = frame_info["timecode"]

            footer_text = f"{frame_id}  |  {tc}  |  {metric_key}: {metric_val:.2f}"

            # Highlight RESOLVE_UNAVAILABLE cells
            text_col = TEXT_COLOR
            if "RESOLVE_UNAVAILABLE" in variant_label:
                text_col = RESOLVE_UNAVAIL_COLOR

            draw.text((x + 6, y_footer + 4), footer_text, font=font_sm, fill=text_col)

    # -----------------------------------------------------------------------
    # Histogram row (one per selected frame, raw frame only, below all rows)
    # -----------------------------------------------------------------------
    y_hist = y_rows_start + num_rows * (cell_height + cell_footer_h)
    draw.rectangle([0, y_hist, sheet_width, y_hist + hist_h], fill=HIST_BG)

    hist_cell_w = sheet_width // num_rows
    for i, frame_info in enumerate(selected_frames):
        bgr = frame_info["bgr"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hist_img = _draw_histogram(gray, hist_cell_w - 4, hist_h - 20)

        x = i * hist_cell_w
        sheet.paste(hist_img, (x + 2, y_hist + 10))

        # Label histogram with frame_id
        draw.text((x + 6, y_hist + 2), frame_info["frame_id"], font=font_sm, fill=(100, 200, 100))

    return sheet
