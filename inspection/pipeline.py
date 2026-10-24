"""
inspection/pipeline.py -- Full inspection orchestration.

AGENT_SPEC §5, §7. Combines preprocessing, part detection, pose refinement
and hole localisation, then applies tolerance limits and outputs the exact
JSON schema specified in §1.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from inspection import config as cfg
from inspection.preprocess import preprocess
from inspection.part_detector import detect_part
from inspection.predict import predict


def _round_mm(value: Optional[float]) -> Optional[float]:
    """Round a mm measurement to ``ROUND_DECIMALS`` (2 d.p.), or return None."""
    if value is None:
        return None
    return round(value, cfg.ROUND_DECIMALS)


def _in_limits(value: Optional[float], key: str) -> Optional[bool]:
    """Return True/False if *value* is within ``config.LIMITS[key]``, or None
    if the value is unavailable.
    Uses ``ROUND_EPSILON`` for inclusive boundary comparison (A7).
    """
    if value is None:
        return None
    lo, hi = cfg.LIMITS[key]
    v = round(value, cfg.ROUND_DECIMALS)
    return (v + cfg.ROUND_EPSILON) >= lo and (v - cfg.ROUND_EPSILON) <= hi


def _guard_warning(value: Optional[float], key: str) -> bool:
    """Return True when *value* is inside the guard band of a limit (A10)."""
    if value is None:
        return False
    lo, hi = cfg.LIMITS[key]
    return ((lo - cfg.GUARD_MM) < value < (lo + cfg.GUARD_MM) or
            (hi - cfg.GUARD_MM) < value < (hi + cfg.GUARD_MM))


def run_full_pipeline(img: np.ndarray) -> Dict[str, Any]:
    """Full inspection pipeline.

    Returns a dict matching the exact JSON schema from AGENT_SPEC §1:
    {
      "part_present": bool,
      "inspection_status": "ACCEPT" | "REJECT",
      "measurements_mm": { ... },
      "result": { ... per-check PASS/FAIL/NOT_EVALUATED ... },
      "reject_reason": str | null
    }
    """
    # -----------------------------------------------------------------
    # Step 1 -- preprocessing
    # -----------------------------------------------------------------
    pre = preprocess(img)
    diagnostics_warnings: List[str] = list(pre.get("warnings", []))

    # -----------------------------------------------------------------
    # Step 2 -- part detection (coarse geometry)
    # -----------------------------------------------------------------
    part_res = detect_part(pre["mask"], pre["gray_f"], diagnostics_warnings)
    diagnostics_warnings.extend(part_res.get("warnings", []))

    part_found = part_res.get("found", False)

    # -----------------------------------------------------------------
    # Step 3 -- sub-pixel refinement & hole localisation
    # -----------------------------------------------------------------
    pred = predict(pre["mask"], pre["gray_f"], part_res)
    diagnostics_warnings.extend(pred.get("warnings", []))

    # -----------------------------------------------------------------
    # Assemble measurements
    # -----------------------------------------------------------------
    if part_found:
        length_mm = _round_mm(part_res.get("long_side_px", 0.0) * cfg.SCALE_MM_PER_PX)
        width_mm  = _round_mm(part_res.get("short_side_px", 0.0) * cfg.SCALE_MM_PER_PX)
    else:
        length_mm = None
        width_mm  = None

    hole_x_mm = _round_mm(pred.get("hole_x_mm"))
    hole_y_mm = _round_mm(pred.get("hole_y_mm"))
    hole_d_mm = _round_mm(pred.get("hole_d_mm"))

    # Determine whether the hole was found
    hole_found = (hole_x_mm is not None and hole_y_mm is not None)

    # -----------------------------------------------------------------
    # Decision logic (§5, A5 priority)
    # -----------------------------------------------------------------
    reject_reason: Optional[str] = None

    # Per-check PASS / FAIL / NOT_EVALUATED (§5, A8)
    if not part_found:
        reject_reason = cfg.REASON_PART_NOT_FOUND
        r_length   = "NOT_EVALUATED"
        r_width    = "NOT_EVALUATED"
        r_hole_dia = "NOT_EVALUATED"
        r_hole_x   = "NOT_EVALUATED"
        r_hole_y   = "NOT_EVALUATED"
    else:
        # Length & width are always measurable when the part is found
        len_ok = _in_limits(length_mm, "length")
        wid_ok = _in_limits(width_mm, "width")
        r_length = "PASS" if len_ok else "FAIL"
        r_width  = "PASS" if wid_ok else "FAIL"

        if not hole_found:
            reject_reason = cfg.REASON_HOLE_NOT_FOUND
            r_hole_dia = "NOT_EVALUATED"
            r_hole_x   = "NOT_EVALUATED"
            r_hole_y   = "NOT_EVALUATED"
        else:
            dia_ok = _in_limits(hole_d_mm, "hole_diameter")
            hx_ok  = _in_limits(hole_x_mm, "hole_x")
            hy_ok  = _in_limits(hole_y_mm, "hole_y")
            r_hole_dia = "PASS" if dia_ok else ("FAIL" if dia_ok is not None else "NOT_EVALUATED")
            r_hole_x   = "PASS" if hx_ok  else "FAIL"
            r_hole_y   = "PASS" if hy_ok  else "FAIL"

        # Apply priority order for reject_reason (A5):
        # PART_NOT_FOUND already handled above
        if reject_reason is None and not hole_found:
            reject_reason = cfg.REASON_HOLE_NOT_FOUND
        if reject_reason is None and r_length == "FAIL":
            reject_reason = cfg.REASON_LENGTH_OOT
        if reject_reason is None and r_width == "FAIL":
            reject_reason = cfg.REASON_WIDTH_OOT
        if reject_reason is None and r_hole_dia == "FAIL":
            reject_reason = cfg.REASON_HOLE_DIA_OOT
        if reject_reason is None and (r_hole_x == "FAIL" or r_hole_y == "FAIL"):
            reject_reason = cfg.REASON_HOLE_POS_OOT

    inspection_status = "ACCEPT" if reject_reason is None else "REJECT"

    # Marginal warnings (A10) -- only when the measurement exists
    if _guard_warning(length_mm, "length"):
        diagnostics_warnings.append("MARGINAL:length")
    if _guard_warning(width_mm, "width"):
        diagnostics_warnings.append("MARGINAL:width")
    if _guard_warning(hole_d_mm, "hole_diameter"):
        diagnostics_warnings.append("MARGINAL:hole_diameter")
    if _guard_warning(hole_x_mm, "hole_x"):
        diagnostics_warnings.append("MARGINAL:hole_x")
    if _guard_warning(hole_y_mm, "hole_y"):
        diagnostics_warnings.append("MARGINAL:hole_y")

    # Flip warning (A11)
    if hole_x_mm is not None and hole_x_mm > cfg.FLIP_WARN_HOLE_X_MM:
        diagnostics_warnings.append("POSSIBLE_180_FLIP")

    # -----------------------------------------------------------------
    # Build the output dict matching §1 JSON schema exactly
    # -----------------------------------------------------------------
    result = {
        "part_present": part_found,
        "inspection_status": inspection_status,
        "measurements_mm": {
            "length":            length_mm,
            "width":             width_mm,
            "hole_diameter":     hole_d_mm,
            "hole_x_from_left":  hole_x_mm,
            "hole_y_from_bottom": hole_y_mm,
        },
        "result": {
            "length":          r_length,
            "width":           r_width,
            "hole_diameter":   r_hole_dia,
            "hole_x_position": r_hole_x,
            "hole_y_position": r_hole_y,
        },
        "reject_reason": reject_reason,
    }
    return result


def export_json(result: Dict[str, Any], out_path: Path) -> None:
    """Write *result* to *out_path* as pretty-printed JSON.
    Parent directories are created automatically.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
