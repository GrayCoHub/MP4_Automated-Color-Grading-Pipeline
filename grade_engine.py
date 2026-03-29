"""
grade_engine.py — Grade variant application.

V0: Raw — D-Log M unmodified
V1: DJI Official LUT — DJI_Mavic_4_Pro_D-Log_M_to_Rec.709_V1.cube
V2: Zeb Gardner LUT — DLOGM_REC709_65_CUBE_ZRG_V1.cube
V3: No-LUT S-curve — Contrast +20, Sat +15, S-curve on L channel in LAB space
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# LUT loading
# ---------------------------------------------------------------------------

def load_cube_lut(lut_path: str) -> np.ndarray:
    """
    Parse a .cube LUT file and return a (N, N, N, 3) float32 array in BGR order.
    Supports 3D LUTs only.
    """
    lut_size = None
    data = []

    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                lut_size = int(line.split()[-1])
                continue
            # Skip DOMAIN_MIN / DOMAIN_MAX / TITLE lines
            if any(line.upper().startswith(k) for k in ("DOMAIN_", "TITLE", "LUT_1D")):
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append([float(p) for p in parts])
                except ValueError:
                    continue

    if lut_size is None:
        raise ValueError("Could not determine LUT_3D_SIZE from .cube file")

    arr = np.array(data, dtype=np.float32)
    expected = lut_size ** 3
    if len(arr) != expected:
        raise ValueError(f"LUT data length {len(arr)} does not match LUT_3D_SIZE^3 = {expected}")

    # .cube order: R changes fastest, then G, then B
    # Reshape to (B, G, R, 3) for OpenCV LUT3D convention
    arr = arr.reshape((lut_size, lut_size, lut_size, 3))
    # .cube is (R_outer... actually R innermost): arr[b][g][r] = (R_out, G_out, B_out)
    # We need BGR output, so swap axis 3: (R,G,B) -> (B,G,R)
    arr = arr[:, :, :, ::-1]  # now (B,G,R) output at [b][g][r]
    return arr


def _apply_lut3d(bgr_frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to a BGR uint8 frame. Returns uint8 BGR result."""
    lut_size = lut.shape[0]
    scale = (lut_size - 1) / 255.0

    # Normalize to [0,1] then scale to LUT index space
    b = bgr_frame[:, :, 0].astype(np.float32) * scale
    g = bgr_frame[:, :, 1].astype(np.float32) * scale
    r = bgr_frame[:, :, 2].astype(np.float32) * scale

    # Clamp to valid range
    b = np.clip(b, 0, lut_size - 1)
    g = np.clip(g, 0, lut_size - 1)
    r = np.clip(r, 0, lut_size - 1)

    # Trilinear interpolation indices
    b0 = np.floor(b).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    r0 = np.floor(r).astype(np.int32)
    b1 = np.minimum(b0 + 1, lut_size - 1)
    g1 = np.minimum(g0 + 1, lut_size - 1)
    r1 = np.minimum(r0 + 1, lut_size - 1)

    tb = (b - b0)[..., np.newaxis]
    tg = (g - g0)[..., np.newaxis]
    tr = (r - r0)[..., np.newaxis]

    # Trilinear interpolation
    c000 = lut[b0, g0, r0]
    c001 = lut[b0, g0, r1]
    c010 = lut[b0, g1, r0]
    c011 = lut[b0, g1, r1]
    c100 = lut[b1, g0, r0]
    c101 = lut[b1, g0, r1]
    c110 = lut[b1, g1, r0]
    c111 = lut[b1, g1, r1]

    result = (
        c000 * (1 - tb) * (1 - tg) * (1 - tr)
        + c001 * (1 - tb) * (1 - tg) * tr
        + c010 * (1 - tb) * tg * (1 - tr)
        + c011 * (1 - tb) * tg * tr
        + c100 * tb * (1 - tg) * (1 - tr)
        + c101 * tb * (1 - tg) * tr
        + c110 * tb * tg * (1 - tr)
        + c111 * tb * tg * tr
    )

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# S-curve helper
# ---------------------------------------------------------------------------

def _build_scurve_lut() -> np.ndarray:
    """Build a 256-entry S-curve lookup table for the L channel."""
    x = np.arange(256, dtype=np.float32) / 255.0
    # Sigmoid-style S-curve: pulls shadows down slightly, highlights up slightly
    # Parameterised to approximate "Contrast +20" effect
    y = 0.5 + (x - 0.5) * 1.15  # slight stretch
    # Apply soft S shape
    y = y + 0.08 * np.sin(2 * np.pi * y) * (1 - y) * y
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


_SCURVE_LUT = _build_scurve_lut()


# ---------------------------------------------------------------------------
# Grade variants
# ---------------------------------------------------------------------------

def apply_v0_raw(bgr_frame: np.ndarray) -> np.ndarray:
    """V0: Return frame unmodified."""
    return bgr_frame.copy()


def apply_v1_lut(bgr_frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """V1: Apply official DJI D-Log M to Rec.709 LUT at 1920x1080, then upsample.
    Reduces trilinear interpolation pixel count 4x vs full 4K. Acceptable
    quality for contact sheet use."""
    orig_h, orig_w = bgr_frame.shape[:2]
    small = cv2.resize(bgr_frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    graded_small = _apply_lut3d(small, lut)
    return cv2.resize(graded_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def apply_v2_zrg_lut(bgr_frame: np.ndarray, lut_zrg: np.ndarray) -> np.ndarray:
    """V2: Apply Zeb Gardner D-Log M to Rec.709 LUT at 1920x1080, then upsample."""
    orig_h, orig_w = bgr_frame.shape[:2]
    small = cv2.resize(bgr_frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    graded_small = _apply_lut3d(small, lut_zrg)
    return cv2.resize(graded_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def apply_v3_scurve(bgr_frame: np.ndarray) -> np.ndarray:
    """
    V3: No-LUT S-curve grade.
    Contrast +20, Sat +15, S-curve on L channel in LAB space.
    """
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2Lab)
    l_ch, a_ch, b_ch = cv2.split(lab)

    l_ch = cv2.LUT(l_ch, _SCURVE_LUT)

    sat_scale = 1.15  # +15% saturation
    a_ch = np.clip(128 + (a_ch.astype(np.float32) - 128) * sat_scale, 0, 255).astype(np.uint8)
    b_ch = np.clip(128 + (b_ch.astype(np.float32) - 128) * sat_scale, 0, 255).astype(np.uint8)

    lab_graded = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab_graded, cv2.COLOR_Lab2BGR)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def apply_all_variants(bgr_frame: np.ndarray, lut: np.ndarray, lut_landscape: np.ndarray) -> dict:
    """
    Apply all 4 grade variants to a single BGR frame.
    lut          — DJI Official LUT (V1)
    lut_landscape — Zeb Gardner LUT (V2)
    Returns dict: variant_id -> (bgr_result, label_string)
    """
    return {
        "V0": (apply_v0_raw(bgr_frame), "V0 Raw"),
        "V1": (apply_v1_lut(bgr_frame, lut), "V1 DJI Official LUT"),
        "V2": (apply_v2_zrg_lut(bgr_frame, lut_landscape), "V2 Zeb Gardner LUT"),
        "V3": (apply_v3_scurve(bgr_frame), "V3 No-LUT S-curve"),
    }
