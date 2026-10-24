"""
inspection/predict.py -- Sub-pixel refinement of part pose and hole localisation.

Implements §4.4 (edge refinement) and §4.5/§4.6 (hole detection & refinement).
All constants are taken from inspection/config.py.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from inspection import config as cfg

# ---------------------------------------------------------------------------
# Helper: normalize angle to the same convention used in part detection
# ---------------------------------------------------------------------------

def _norm_angle(theta: float) -> float:
    """Normalise angle to [-90, 90) degrees.

    The same normalisation used in ``part_detector.angle_from_box_points``.
    """
    while theta >= 90.0:
        theta -= 180.0
    while theta < -90.0:
        theta += 180.0
    return theta

# ---------------------------------------------------------------------------
# Sub-pixel edge / pose refinement
# ---------------------------------------------------------------------------

def refine_pose(contour: np.ndarray) -> float:
    """Return a refined rotation angle (degrees) for ``contour``.

    Strategy (lightweight, version-agnostic):
    1. Fit a line through all contour points using ``cv2.fitLine`` -- this
       operates on the raw integer coordinates but yields a direction vector
       with sub-pixel precision.
    2. The longer edge of the part aligns with the line direction.  Compute
       ``atan2(vy, vx)`` and normalise to ``[-90, 90)``.
    3. For extra robustness we also compute ``minAreaRect`` and take the
       longer side length; the angle from ``fitLine`` is kept because OpenCV's
       ``rect[2]`` changed semantics in 4.5.1.

    Returns
    -------
    float
        Angle in degrees, positive -> clockwise rotation when the Y axis
        points down (the same sign convention as the synthetic generator).
    """
    # Fit a line (vx, vy) direction; point (x0, y0) is also returned but not needed.
    vx, vy, _, _ = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    theta = math.degrees(math.atan2(vy, vx))
    return _norm_angle(theta)

# ---------------------------------------------------------------------------
# Hole localisation helpers
# ---------------------------------------------------------------------------

def _find_hole_contour(
    mask: np.ndarray,
    part_contour: np.ndarray,
) -> Optional[np.ndarray]:
    """Return the child contour that lies inside ``part_contour``.

    ``mask`` is the binary part mask (255 = part, 0 = background).
    Uses ``RETR_CCOMP`` hierarchy to keep parent/child relationships.
    If no child contour is found, returns ``None``.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]

    # Identify the index of the outer (part) contour -- it must match ``part_contour``
    outer_idx = None
    for i, cnt in enumerate(contours):
        if cnt.shape == part_contour.shape and np.array_equal(cnt, part_contour):
            outer_idx = i
            break
    if outer_idx is None:
        # Fallback: pick the largest outer contour and look for its child
        outer_areas = []
        for i in range(len(contours)):
            if hierarchy[i][3] == -1:  # no parent -> outer
                outer_areas.append((cv2.contourArea(contours[i]), i))
        if not outer_areas:
            return None
        outer_areas.sort(reverse=True)
        outer_idx = outer_areas[0][1]

    # Find the best child contour (most circular, within area range)
    best_child = None
    best_circularity = -1.0
    for i, (next_i, prev_i, child_i, parent_i) in enumerate(hierarchy):
        if parent_i == outer_idx:
            area = cv2.contourArea(contours[i])
            peri = cv2.arcLength(contours[i], True)
            if peri > 0 and area > 0:
                circ = 4 * math.pi * area / (peri ** 2)
                if circ > best_circularity:
                    best_circularity = circ
                    best_child = contours[i]

    return best_child


def _hole_measurements(
    hole_cnt: np.ndarray,
    part_contour: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute hole centre (in part-frame mm) and diameter (mm).

    Uses the part's **minAreaRect** (rotated bounding box) so that
    measurements are correct even when the part is rotated up to ±5°.
    The part frame origin is the bottom-left corner of the deskewed
    rectangle, consistent with the synthetic generator convention:
    ``hole_x`` = distance from the left edge, ``hole_y`` = distance
    from the bottom edge.

    Returns (hole_x_mm, hole_y_mm, hole_d_mm).
    """
    M = cv2.moments(hole_cnt)
    if M["m00"] == 0:
        raise ValueError("Zero area hole contour -- cannot compute centre")
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]

    # Get rotated bounding box of the part
    rect = cv2.minAreaRect(part_contour)
    (rcx, rcy), (rw, rh), angle_rect = rect

    # Determine which side is long / short
    # After boxPoints, the long axis direction is the edge between the first
    # two box points (or second two).  We use the part contour's fitLine.
    box = cv2.boxPoints(rect)

    # Compute edge vectors
    edges = []
    for i in range(4):
        dx = float(box[(i + 1) % 4, 0] - box[i, 0])
        dy = float(box[(i + 1) % 4, 1] - box[i, 1])
        length = math.hypot(dx, dy)
        edges.append((length, dx, dy, i))
    edges.sort(reverse=True)
    # Longest edge → long axis direction
    _, dx_long, dy_long, _ = edges[0]
    long_len = edges[0][0]
    short_len = edges[2][0]  # the shorter pair

    # Unit vectors for long and short axes
    long_axis = np.array([dx_long, dy_long]) / long_len
    # Short axis is perpendicular (rotated 90° CW in image coords: y down)
    short_axis = np.array([dy_long, -dx_long]) / long_len

    # Project hole centre into part frame relative to rect centre
    hole_vec = np.array([cx_px - rcx, cy_px - rcy])
    proj_long = float(np.dot(hole_vec, long_axis))   # signed, along long axis
    proj_short = float(np.dot(hole_vec, short_axis))  # signed, along short axis

    # Convert from rect-centre-relative to edge-relative:
    # hole_x_from_left = (long_side/2 + proj_long) / scale
    # hole_y_from_bottom = (short_side/2 + proj_short) / scale
    # But we need to handle the axis direction carefully.  The long_axis
    # may point left or right.  We want "from left edge" to be positive.
    # Strategy: try both orientations and pick the one where hole_x is
    # in [0, long_side_mm] and hole_y is in [0, short_side_mm].
    long_side_px = long_len
    short_side_px = short_len

    hx_a = (long_side_px / 2 + proj_long) / cfg.SCALE_PX_PER_MM
    hy_a = (short_side_px / 2 + proj_short) / cfg.SCALE_PX_PER_MM
    hx_b = (long_side_px / 2 - proj_long) / cfg.SCALE_PX_PER_MM
    hy_b = (short_side_px / 2 - proj_short) / cfg.SCALE_PX_PER_MM

    long_side_mm = long_side_px / cfg.SCALE_PX_PER_MM
    short_side_mm = short_side_px / cfg.SCALE_PX_PER_MM

    # Pick the orientation where both are in [0, side_mm]
    def score(hx, hy):
        s = 0
        if 0 <= hx <= long_side_mm:
            s += 1
        if 0 <= hy <= short_side_mm:
            s += 1
        return s

    candidates = [
        (hx_a, hy_a),
        (hx_a, hy_b),
        (hx_b, hy_a),
        (hx_b, hy_b),
    ]
    best = max(candidates, key=lambda c: score(c[0], c[1]))
    hole_x_mm, hole_y_mm = best

    # Hole diameter from equivalent circle area
    area_px = cv2.contourArea(hole_cnt)
    diameter_px = 2.0 * math.sqrt(area_px / math.pi) if area_px > 0 else 0.0
    hole_d_mm = diameter_px / cfg.SCALE_PX_PER_MM

    return (
        round(hole_x_mm, cfg.ROUND_DECIMALS),
        round(hole_y_mm, cfg.ROUND_DECIMALS),
        round(hole_d_mm, cfg.ROUND_DECIMALS),
    )


# ---------------------------------------------------------------------------
# Public API -- predict function
# ---------------------------------------------------------------------------

def predict(mask: np.ndarray, gray_f: np.ndarray, part_result: dict) -> dict:
    """Run sub-pixel pose refinement and hole localisation.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask from ``preprocess`` (255 = part).
    gray_f : np.ndarray
        Float32 grayscale image -- currently unused but kept for future
        photometric checks (e.g., hole contrast).
    part_result : dict
        Result dictionary returned by ``part_detector.detect_part``.  It must
        contain at least ``"found"``, ``"contour"`` and ``"bbox_coarse"``.

    Returns
    -------
    dict
        ``{"angle_deg": float, "hole_x_mm": float, "hole_y_mm": float,
         "hole_d_mm": float, "hole_found": bool, "warnings": List[str]}``
    """
    warnings: List[str] = []
    if not part_result.get("found", False) or part_result.get("contour") is None:
        return {
            "angle_deg": None,
            "hole_x_mm": None,
            "hole_y_mm": None,
            "hole_d_mm": None,
            "hole_found": False,
            "warnings": warnings,
        }

    contour = part_result["contour"]
    # ---------- Pose refinement ------------------------------------------------
    angle_deg = refine_pose(contour)

    # ---------- Hole localisation --------------------------------------------
    bbox = part_result.get("bbox_coarse")
    hole_cnt = _find_hole_contour(mask, contour)

    if hole_cnt is None:
        return {
            "angle_deg": angle_deg,
            "hole_x_mm": None,
            "hole_y_mm": None,
            "hole_d_mm": None,
            "hole_found": False,
            "warnings": warnings,
        }

    # Basic size/shape sanity checks (area, circularity)
    area_px = cv2.contourArea(hole_cnt)
    if not (cfg.HOLE_AREA_MIN_PX <= area_px <= cfg.HOLE_AREA_MAX_PX):
        warnings.append("HOLE_AREA_OUT_OF_RANGE")
    perimeter = cv2.arcLength(hole_cnt, True)
    circularity = 4 * math.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0.0
    if circularity < cfg.CIRCULARITY_MIN:
        warnings.append("HOLE_NOT_CIRCULAR")

    try:
        hole_x_mm, hole_y_mm, hole_d_mm = _hole_measurements(hole_cnt, contour)
    except Exception:
        warnings.append("HOLE_CENTRE_COMPUTE_FAIL")
        return {
            "angle_deg": angle_deg,
            "hole_x_mm": None,
            "hole_y_mm": None,
            "hole_d_mm": None,
            "hole_found": False,
            "warnings": warnings,
        }

    return {
        "angle_deg": angle_deg,
        "hole_x_mm": hole_x_mm,
        "hole_y_mm": hole_y_mm,
        "hole_d_mm": hole_d_mm,
        "hole_found": True,
        "warnings": warnings,
    }
