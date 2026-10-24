"""
inspection/part_detector.py — Detect the stamped component, apply validity gates,
and compute the rotation angle (pose).

AGENT_SPEC §4.2. All gate names match the spec exactly.

Key design decisions:
- We use RETR_CCOMP (not RETR_EXTERNAL) so the hole's inner contour is also
  available in the hierarchy; we filter to outer contours only (hierarchy[:,3]==-1).
- Angle is derived from the longest edge vector of the minAreaRect box points,
  NOT from rect[2], because the angle convention of rect[2] changed in OpenCV
  4.5.1 (was [-90,0), is now (0,90]). Using edge vectors is version-agnostic.
- Validity gates are deliberately loose (§2 A4): a part that is the wrong size
  must still be measured and rejected for LENGTH/WIDTH, not labeled PART_NOT_FOUND.
"""

import math

import cv2
import numpy as np

from inspection import config as cfg


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def make_part_result(
    found: bool,
    contour,                  # the selected contour (or None)
    cx: float, cy: float,     # coarse centre in image pixels
    angle_deg: float,         # rotation angle, normalised to [-90, 90)
    long_side_px: float,      # coarse length from minAreaRect
    short_side_px: float,     # coarse width from minAreaRect
    bbox_coarse: tuple,       # (x0, y0, x1, y1) axis-aligned enclosing bbox
    warnings: list,
    fail_reason: str = None,  # PART_NOT_FOUND sub-reason for diagnostics
) -> dict:
    return {
        "found":          found,
        "contour":        contour,
        "cx":             cx,
        "cy":             cy,
        "angle_deg":      angle_deg,
        "long_side_px":   long_side_px,
        "short_side_px":  short_side_px,
        "bbox_coarse":    bbox_coarse,   # (x0, y0, x1, y1)
        "warnings":       warnings,
        "fail_reason":    fail_reason,
    }


# ---------------------------------------------------------------------------
# Angle extraction from box points (version-agnostic)
# ---------------------------------------------------------------------------

def angle_from_box_points(cnt: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the long-axis angle of a contour without using rect[2].

    Strategy (AGENT_SPEC §4.2 step 3):
      1. Get the 4 corner points of the rotated bounding box.
      2. Compute the 4 edge vectors.
      3. Pick the longest edge → that is the long axis direction.
      4. theta = degrees(atan2(dy, dx)), normalised to [-90, 90).

    Returns:
        (theta_deg, long_side_px, short_side_px)

    The sign convention: positive θ means the part's long axis points
    slightly upward to the right (clockwise from horizontal when y-down).
    """
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)   # 4 corner points, shape (4, 2)

    # Compute 4 edge vectors and their lengths
    edges = []
    for i in range(4):
        dx = float(box[(i + 1) % 4, 0] - box[i, 0])
        dy = float(box[(i + 1) % 4, 1] - box[i, 1])
        length = math.hypot(dx, dy)
        edges.append((length, dx, dy))

    # Sort by length descending; the two longest are the "long" edges
    edges.sort(reverse=True)
    _, dx_long, dy_long = edges[0]

    theta = math.degrees(math.atan2(dy_long, dx_long))

    # Normalise to [-90, 90): an angle of 90° is the same as -90°,
    # and we want a canonical representation for "nearly horizontal".
    while theta >= 90.0:
        theta -= 180.0
    while theta < -90.0:
        theta += 180.0

    # Long and short side from minAreaRect
    rw, rh = rect[1]
    long_side  = max(rw, rh)
    short_side = min(rw, rh)

    return theta, long_side, short_side


# ---------------------------------------------------------------------------
# Validity gates (§4.2 step 2, §2 A4)
# ---------------------------------------------------------------------------

def check_clipped_by_border(cnt: np.ndarray, img_h: int, img_w: int) -> bool:
    """
    Return True if the contour touches within BORDER_MARGIN_PX of the image
    boundary, indicating the part is clipped (CLIPPED_BY_BORDER gate).
    """
    m = cfg.BORDER_MARGIN_PX
    x, y, bw, bh = cv2.boundingRect(cnt)
    return (x <= m or y <= m or
            x + bw >= img_w - m or
            y + bh >= img_h - m)


def compute_rectangularity(cnt: np.ndarray) -> float:
    """
    Rectangularity = contour area / minAreaRect area.

    A perfect rectangle ≈ 1.0; a circle ≈ π/4 ≈ 0.785.
    Threshold: ≥ 0.85 (AGENT_SPEC §2 A4).
    """
    contour_area = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    rect_area = rw * rh
    if rect_area < 1e-3:
        return 0.0
    return contour_area / rect_area


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_part(
    mask: np.ndarray,
    gray_f: np.ndarray,
    preprocess_warnings: list = None,
) -> dict:
    """
    Detect the stamped component in the binary mask.

    Args:
        mask: uint8 binary mask from preprocess() (255 = part, 0 = bg)
        gray_f: float32 grayscale (for contrast gate)
        preprocess_warnings: warnings already emitted by preprocess()

    Returns a dict from make_part_result().

    AGENT_SPEC §4.2.
    """
    h, w = mask.shape
    warnings = list(preprocess_warnings or [])

    # -----------------------------------------------------------------------
    # Gate: LOW_CONTRAST (already detected in preprocess)
    # -----------------------------------------------------------------------
    if "LOW_CONTRAST" in warnings:
        return make_part_result(
            False, None, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="LOW_CONTRAST"
        )

    # -----------------------------------------------------------------------
    # Step 1: find all contours with hierarchy (RETR_CCOMP)
    # Keep only outer contours (hierarchy[:,3] == -1 means no parent).
    # AGENT_SPEC §4.2 step 1.
    # -----------------------------------------------------------------------
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None or len(contours) == 0:
        warnings.append("NO_BLOB")
        return make_part_result(
            False, None, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="NO_BLOB"
        )

    # Filter to outer contours only (parent == -1)
    hierarchy = hierarchy[0]   # shape (N, 4): next, prev, child, parent
    outer_contours = [
        contours[i] for i in range(len(contours))
        if hierarchy[i][3] == -1
    ]

    if not outer_contours:
        warnings.append("NO_BLOB")
        return make_part_result(
            False, None, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="NO_BLOB"
        )

    # Choose the largest outer contour by area
    cnt = max(outer_contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # -----------------------------------------------------------------------
    # Gate: AREA_TOO_SMALL (§2 A4)
    # 20% of nominal part area = 0.2 × 600 × 160 = 19 200 px²
    # Deliberately loose: a part that is wrong size still passes this.
    # -----------------------------------------------------------------------
    if area < cfg.PART_AREA_MIN_FRAC * cfg.NOMINAL_PART_AREA_PX:
        warnings.append("AREA_TOO_SMALL")
        return make_part_result(
            False, cnt, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="AREA_TOO_SMALL"
        )

    # -----------------------------------------------------------------------
    # Gate: CLIPPED_BY_BORDER (§2 A4)
    # -----------------------------------------------------------------------
    if check_clipped_by_border(cnt, h, w):
        warnings.append("CLIPPED_BY_BORDER")
        return make_part_result(
            False, cnt, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="CLIPPED_BY_BORDER"
        )

    # -----------------------------------------------------------------------
    # Gate: NOT_RECTANGULAR (§2 A4)
    # -----------------------------------------------------------------------
    rect_score = compute_rectangularity(cnt)
    if rect_score < cfg.RECTANGULARITY_MIN:
        warnings.append("NOT_RECTANGULAR")
        return make_part_result(
            False, cnt, 0, 0, 0, 0, 0, (0, 0, w, h), warnings,
            fail_reason="NOT_RECTANGULAR"
        )

    # -----------------------------------------------------------------------
    # Compute coarse geometry (angle, long/short side, centre)
    # AGENT_SPEC §4.2 steps 3 & 4.
    # -----------------------------------------------------------------------
    angle_deg, long_side_px, short_side_px = angle_from_box_points(cnt)

    rect = cv2.minAreaRect(cnt)
    (cx, cy) = rect[0]    # coarse centre

    # Coarse axis-aligned bounding box
    bx, by, bw_bb, bh_bb = cv2.boundingRect(cnt)
    bbox_coarse = (bx, by, bx + bw_bb, by + bh_bb)

    # -----------------------------------------------------------------------
    # Gate: BAD_ASPECT (§2 A4)
    # Nominal aspect ratio = 30/8 = 3.75; allow [2.0, 6.0].
    # -----------------------------------------------------------------------
    if short_side_px < 1:
        warnings.append("BAD_ASPECT")
        return make_part_result(
            False, cnt, cx, cy, angle_deg, long_side_px, short_side_px,
            bbox_coarse, warnings, fail_reason="BAD_ASPECT"
        )
    aspect = long_side_px / short_side_px
    if not (cfg.ASPECT_RATIO_MIN <= aspect <= cfg.ASPECT_RATIO_MAX):
        warnings.append("BAD_ASPECT")
        return make_part_result(
            False, cnt, cx, cy, angle_deg, long_side_px, short_side_px,
            bbox_coarse, warnings, fail_reason="BAD_ASPECT"
        )

    # -----------------------------------------------------------------------
    # Gate: BAD_POSE (§2 A3)
    # |θ| ≤ 5°  → OK
    # 5° < |θ| ≤ 10° → warning but not rejected
    # |θ| > 10° → PART_NOT_FOUND (BAD_POSE)
    # -----------------------------------------------------------------------
    abs_angle = abs(angle_deg)
    if abs_angle > cfg.POSE_WARN_DEG:
        warnings.append("BAD_POSE")
        return make_part_result(
            False, cnt, cx, cy, angle_deg, long_side_px, short_side_px,
            bbox_coarse, warnings, fail_reason="BAD_POSE"
        )
    if abs_angle > cfg.POSE_OK_DEG:
        warnings.append(f"ROTATION_WARNING:{angle_deg:.2f}deg")

    # -----------------------------------------------------------------------
    # All gates passed → part found
    # -----------------------------------------------------------------------
    return make_part_result(
        True, cnt, cx, cy, angle_deg, long_side_px, short_side_px,
        bbox_coarse, warnings, fail_reason=None
    )
