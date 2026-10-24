"""
inspection/preprocess.py — Grayscale, blur, polarity detection, Otsu threshold,
illumination flattening, and morphological cleanup.

AGENT_SPEC §4.1.  All steps numbered to match the spec.
"""

import cv2
import numpy as np

from inspection import config as cfg


# ---------------------------------------------------------------------------
# Public result dataclass (plain dict for simplicity; no extra dependencies)
# ---------------------------------------------------------------------------

def make_preprocess_result(
    gray_f: np.ndarray,
    mask: np.ndarray,
    bg_is_bright: bool,
    otsu_thresh: float,
    illumination_flattened: bool,
    warnings: list,
) -> dict:
    """Bundle preprocessing outputs into a single dict for downstream use."""
    return {
        "gray_f": gray_f,             # float32 grayscale, UN-blurred (for sub-pixel work)
        "mask": mask,                  # uint8 binary mask: 255 = part, 0 = background
        "bg_is_bright": bg_is_bright,  # True if background is lighter than part
        "otsu_thresh": otsu_thresh,    # Otsu threshold value
        "illumination_flattened": illumination_flattened,
        "warnings": warnings,          # list of warning strings
    }


# ---------------------------------------------------------------------------
# Step 1: grayscale conversion
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1- or 3-channel uint8 image to grayscale.

    Returns:
        gray    -- uint8 grayscale
        gray_f  -- float32 copy kept for sub-pixel edge work (never blurred)

    AGENT_SPEC §4.1 step 1.
    """
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    gray_f = gray.astype(np.float32)
    return gray, gray_f


# ---------------------------------------------------------------------------
# Step 3 helper: detect polarity from image border
# ---------------------------------------------------------------------------

def detect_polarity(gray: np.ndarray) -> bool:
    """
    Return True if the background is brighter than the part (dark-on-light).

    Strategy: sample the outer BORDER_FRAC of the image (which should be
    background only) and take the median.  Compare with the Otsu threshold
    to decide which side is background.

    AGENT_SPEC §4.1 step 3, §2 A13.
    """
    h, w = gray.shape
    bw = max(1, int(cfg.BORDER_FRAC * min(h, w)))  # border band width in px

    border_pixels = np.concatenate([
        gray[:bw, :].ravel(),          # top
        gray[-bw:, :].ravel(),         # bottom
        gray[:, :bw].ravel(),          # left
        gray[:, -bw:].ravel(),         # right
    ])
    bg_level = float(np.median(border_pixels))
    return bg_level > 127  # > 127 → bright border = bright background


def get_corner_medians(gray: np.ndarray) -> list[float]:
    """
    Return the median grey level of each of the 4 image corners.
    Used for detecting illumination gradients (§4.1 step 5).
    """
    h, w = gray.shape
    bw = max(1, int(cfg.BORDER_FRAC * min(h, w)))
    return [
        float(np.median(gray[:bw, :bw])),      # top-left
        float(np.median(gray[:bw, -bw:])),      # top-right
        float(np.median(gray[-bw:, :bw])),      # bottom-left
        float(np.median(gray[-bw:, -bw:])),     # bottom-right
    ]


# ---------------------------------------------------------------------------
# Step 5 helper: illumination flattening (flat-field correction)
# ---------------------------------------------------------------------------

def flatten_illumination(gray_f: np.ndarray) -> np.ndarray:
    """
    Flat-field correction: divide by a heavily blurred copy then rescale.

        flat = gray / GaussianBlur(gray, large_sigma) * mean(gray)

    Removes slow brightness gradients while preserving local contrast.
    AGENT_SPEC §4.1 step 5.
    """
    mean_level = float(np.mean(gray_f))
    # Use a kernel size large enough to span the part (~600 px) but must be odd
    ksize = int(cfg.LARGE_SIGMA * 6) | 1   # ensure odd
    bg_blur = cv2.GaussianBlur(
        gray_f, (ksize, ksize), sigmaX=cfg.LARGE_SIGMA, sigmaY=cfg.LARGE_SIGMA
    )
    # Avoid division by zero (should not happen with a real image)
    bg_blur = np.clip(bg_blur, 1.0, None)
    flat = (gray_f / bg_blur) * mean_level
    return flat.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5 helper: Otsu inter-class variance (separability check)
# ---------------------------------------------------------------------------

def otsu_separability(gray_uint8: np.ndarray) -> tuple[float, float]:
    """
    Run Otsu on the image and return (threshold, separability_ratio).

    Separability = inter-class variance / total variance.
    A bimodal histogram gives separability ≈ 1.0; a flat histogram ≈ 0.0.
    If ratio < OTSU_SEPARABILITY_MIN, the image is hard to threshold.

    AGENT_SPEC §4.1 step 5.
    """
    t, _ = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pixels = gray_uint8.ravel().astype(np.float64)
    total_mean = pixels.mean()
    total_var = pixels.var()
    if total_var < 1e-6:
        return float(t), 0.0

    below = pixels[pixels <= t]
    above = pixels[pixels > t]
    w0 = len(below) / len(pixels)
    w1 = len(above) / len(pixels)
    m0 = below.mean() if len(below) > 0 else 0.0
    m1 = above.mean() if len(above) > 0 else 0.0
    inter_var = w0 * w1 * (m0 - m1) ** 2
    sep = inter_var / total_var
    return float(t), float(min(sep, 1.0))


# ---------------------------------------------------------------------------
# Main preprocess function
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict from make_preprocess_result() with keys:
        gray_f, mask, bg_is_bright, otsu_thresh, illumination_flattened, warnings

    On failure (PART_NOT_FOUND conditions detected here), the dict will have
    warnings populated and mask may be all zeros.

    AGENT_SPEC §4.1, steps 1–6.
    """
    warnings = []

    # ------------------------------------------------------------------
    # Step 1: grayscale + float32 copy
    # ------------------------------------------------------------------
    gray_uint8, gray_f = to_grayscale(img)

    # ------------------------------------------------------------------
    # Step 3 (first pass): polarity detection from border
    # ------------------------------------------------------------------
    bg_is_bright = detect_polarity(gray_uint8)

    # ------------------------------------------------------------------
    # Step 5a: check for brightness gradient (corner-median spread)
    # ------------------------------------------------------------------
    corner_meds = get_corner_medians(gray_uint8)
    gradient_detected = (max(corner_meds) - min(corner_meds)) > cfg.GRADIENT_CORNER_THRESH

    # ------------------------------------------------------------------
    # Step 5b: initial Otsu separability check
    # ------------------------------------------------------------------
    blur_uint8 = cv2.GaussianBlur(
        gray_uint8, cfg.BLUR_KERNEL, sigmaX=cfg.BLUR_SIGMA
    )
    t, sep = otsu_separability(blur_uint8)
    separability_poor = sep < cfg.OTSU_SEPARABILITY_MIN

    illumination_flattened = False
    if separability_poor or gradient_detected:
        # Step 5c: flatten and re-run
        flat_f = flatten_illumination(gray_f)
        flat_uint8 = np.clip(flat_f, 0, 255).astype(np.uint8)
        flat_blur = cv2.GaussianBlur(flat_uint8, cfg.BLUR_KERNEL, sigmaX=cfg.BLUR_SIGMA)
        t, sep2 = otsu_separability(flat_blur)
        # Use flattened image for segmentation only; keep original gray_f for edges
        blur_uint8 = flat_blur
        bg_is_bright = detect_polarity(flat_uint8)
        illumination_flattened = True
        warnings.append("ILLUMINATION_FLATTENED")

    # ------------------------------------------------------------------
    # Step 2: segmentation blur (already done above)
    # ------------------------------------------------------------------
    # blur_uint8 is the (possibly flattened) blurred image for thresholding.

    # ------------------------------------------------------------------
    # Step 3: Otsu threshold + polarity
    # ------------------------------------------------------------------
    # bg_level compared to threshold to decide which side is the part.
    h, w = gray_uint8.shape
    bw = max(1, int(cfg.BORDER_FRAC * min(h, w)))
    border_pixels = np.concatenate([
        blur_uint8[:bw, :].ravel(), blur_uint8[-bw:, :].ravel(),
        blur_uint8[:, :bw].ravel(), blur_uint8[:, -bw:].ravel(),
    ])
    bg_level_blurred = float(np.median(border_pixels))

    # Determine part polarity: part is on the opposite side of t from bg
    if bg_level_blurred > t:
        # background is bright → part is dark → part = pixels < t
        _, mask = cv2.threshold(blur_uint8, int(t), 255, cv2.THRESH_BINARY_INV)
    else:
        # background is dark → part is bright → part = pixels > t
        _, mask = cv2.threshold(blur_uint8, int(t), 255, cv2.THRESH_BINARY)

    # ------------------------------------------------------------------
    # Step 4: contrast gate
    # ------------------------------------------------------------------
    fg_pixels = gray_uint8[mask > 0]
    bg_pixels = gray_uint8[mask == 0]
    if len(fg_pixels) < 10 or len(bg_pixels) < 10:
        warnings.append("LOW_CONTRAST")
        return make_preprocess_result(
            gray_f, mask, bg_is_bright, t, illumination_flattened, warnings
        )

    contrast = abs(float(np.mean(fg_pixels)) - float(np.mean(bg_pixels)))
    if contrast < cfg.CONTRAST_MIN_LEVELS:
        warnings.append("LOW_CONTRAST")
        # Return mask as-is; part_detector will gate on LOW_CONTRAST

    # ------------------------------------------------------------------
    # Step 6: morphological cleanup (open then close, 3×3 kernel)
    # Kept small so the 60 px hole is never closed (§4.1 step 6).
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.MORPH_KERNEL_SIZE, cfg.MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)   # remove speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # close pinholes

    return make_preprocess_result(
        gray_f, mask, bg_is_bright, t, illumination_flattened, warnings
    )
