"""
tools/make_synthetic.py — Synthetic stamped-part image generator with ground truth.

AGENT_SPEC §8: Render at 8× supersampling (draw at 8× size, then INTER_AREA
downsample) so edges are naturally anti-aliased at sub-pixel positions.

Usage:
    python tools/make_synthetic.py --out data/synthetic --seed 0
    python tools/make_synthetic.py --out data/synthetic --seed 0 --single \
        --length 30.0 --width 8.0 --hole-d 3.0 --hole-x 6.0 --hole-y 4.0 \
        --rotation 2.5 --noise 4 --name nominal_test
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# Make sure `inspection` is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inspection import config as cfg

# ---------------------------------------------------------------------------
# Constants for the generator (§8)
# ---------------------------------------------------------------------------
SS = 4                           # supersampling factor (§8 says 8×; 4× is sufficient for sub-pixel AA
                                 # and 4× faster/smaller: 4000×2400 vs 8000×4800 canvas)
DEFAULT_IMG_W = 1000             # image width in pixels at final resolution (§8)
DEFAULT_IMG_H = 600              # image height in pixels at final resolution (§8)
MARGIN_PX     = 40               # minimum margin between part bbox and image border (§8)
DEFAULT_CORNER_RADIUS_MM  = 0.3  # stamped corner radius (§8)
DEFAULT_FG_LEVEL = 170           # part grey level (§8)
DEFAULT_BG_LEVEL = 40            # background grey level (§8)
DEFAULT_BLUR_SIGMA = 0.7         # optical blur σ in pixels, [0.5, 1.0] (§8)
DEFAULT_NOISE_SIGMA = 4          # Gaussian noise σ, [0, 8] (§8)
DEFAULT_GRADIENT = 0.0           # linear illumination gradient, [0, ±15 %] (§8)


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------

def render_part(
    img_w: int,
    img_h: int,
    length_mm: float,
    width_mm: float,
    hole_d_mm: float,
    hole_x_mm: float,   # hole centre from left edge (part frame)
    hole_y_mm: float,   # hole centre from bottom edge (part frame)
    rotation_deg: float,
    cx_px: float,       # part centre in the final image (pixels)
    cy_px: float,
    corner_radius_mm: float = DEFAULT_CORNER_RADIUS_MM,
    fg_level: int = DEFAULT_FG_LEVEL,
    bg_level: int = DEFAULT_BG_LEVEL,
    optical_blur_sigma: float = DEFAULT_BLUR_SIGMA,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    gradient_frac: float = DEFAULT_GRADIENT,
    include_hole: bool = True,
    polarity_swap: bool = False,
    metal_texture: bool = True,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Render a synthetic stamped-part image (uint8, single-channel).

    Rendering is done at SS× (8×) resolution and downsampled with INTER_AREA
    for natural sub-pixel anti-aliasing (AGENT_SPEC §8).

    Returns the final image as a uint8 ndarray of shape (img_h, img_w).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    scale = cfg.SCALE_PX_PER_MM   # 20 px/mm

    # ---- Build super-sampled canvas ----------------------------------------
    sh = img_h * SS
    sw = img_w * SS
    canvas = np.full((sh, sw), bg_level, dtype=np.float32)

    # ---- Part rectangle in super-sampled coordinates -----------------------
    L_px  = length_mm * scale * SS
    W_px  = width_mm  * scale * SS
    cx_ss = cx_px * SS
    cy_ss = cy_px * SS
    r_corner = corner_radius_mm * scale * SS   # corner radius in SS pixels

    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # The four corners of the rectangle in the part's own frame (centred at 0):
    # long axis = x, short axis = y
    hL = L_px / 2
    hW = W_px / 2

    def rot(x, y):
        """Rotate a point in part frame to canvas frame."""
        rx = cos_t * x - sin_t * y + cx_ss
        ry = sin_t * x + cos_t * y + cy_ss
        return rx, ry

    # Draw filled rounded rectangle using polygon approximation
    # Decompose into: central rectangle + 4 edge rectangles + 4 corner circles
    _draw_rounded_rect(canvas, hL, hW, r_corner, rot, fg_level)

    # ---- Hole ---------------------------------------------------------------
    if include_hole and hole_d_mm > 0:
        # hole centre in part frame: left edge = −hL/SS, bottom = +hW/SS (image y down)
        # hole_x_mm from left → part frame x = −hL/SS + hole_x_mm*scale*SS
        hole_cx_part = -hL + hole_x_mm * scale * SS
        # hole_y_mm from bottom edge (larger y in image): bottom edge at +hW in part y
        # bottom = cy + hW (in image coords, y down), hole is above bottom by hole_y_mm
        # In part frame (y down), bottom = +hW, hole_cy_part = hW - hole_y_mm*scale*SS
        hole_cy_part = hW - hole_y_mm * scale * SS
        hole_cx_ss, hole_cy_ss = rot(hole_cx_part, hole_cy_part)
        hole_r_ss = (hole_d_mm / 2) * scale * SS
        # Draw hole as background-level filled circle
        cv2.circle(
            canvas,
            (int(round(hole_cx_ss)), int(round(hole_cy_ss))),
            int(round(hole_r_ss)),
            float(bg_level),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # ---- Metal texture (low-amplitude, low-frequency noise) (§8) -----------
    if metal_texture:
        texture = rng.standard_normal((sh, sw)).astype(np.float32) * 2.5
        # Low-frequency: blur the texture
        texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=8 * SS)
        canvas = canvas + texture

    # ---- Illumination gradient (§8): ±gradient_frac across width ----------
    if gradient_frac != 0.0:
        # Linear gradient from left to right
        grad = np.linspace(
            1.0 - abs(gradient_frac),
            1.0 + abs(gradient_frac),
            sw,
            dtype=np.float32,
        )
        if gradient_frac < 0:
            grad = grad[::-1]
        canvas = canvas * grad[np.newaxis, :]

    # ---- Downsample to final resolution (INTER_AREA for anti-aliasing) ----
    canvas = np.clip(canvas, 0, 255)
    img_ss = canvas.astype(np.uint8)
    img = cv2.resize(img_ss, (img_w, img_h), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    # ---- Optical blur (σ in final-resolution pixels) (§8) -----------------
    if optical_blur_sigma > 0:
        img = cv2.GaussianBlur(
            img, (0, 0), sigmaX=optical_blur_sigma, sigmaY=optical_blur_sigma
        )

    # ---- Gaussian noise (§8) -----------------------------------------------
    if noise_sigma > 0:
        noise = rng.standard_normal(img.shape).astype(np.float32) * noise_sigma
        img = img + noise

    img = np.clip(img, 0, 255).astype(np.uint8)

    # ---- Polarity swap (§8): invert so part is dark, bg is light ----------
    if polarity_swap:
        img = 255 - img

    return img


def _draw_rounded_rect(canvas, hL, hW, r, rot_fn, level):
    """
    Draw a filled rounded rectangle on canvas using OpenCV polygon drawing.

    Approximates rounded corners with small arcs (polylines), then fills
    the resulting closed polygon. This gives natural anti-aliased edges
    because the supersampled polygon has sub-pixel accuracy at the
    original resolution.

    Args:
        canvas: float32 ndarray (super-sampled)
        hL: half-length of the rectangle in SS pixels
        hW: half-width of the rectangle in SS pixels
        r: corner radius in SS pixels
        rot_fn: function(x_part, y_part) -> (x_canvas, y_canvas)
        level: fill grey level
    """
    # Build polygon points for the rounded rectangle in part frame.
    # The rectangle long axis is x, short axis is y.
    # Corner arcs: 20 points each gives smooth appearance.
    ARC_STEPS = 20
    pts = []

    # Corner centres in part frame (x, y):
    corners = [
        ( hL - r,  hW - r, 0),    # bottom-right: arc from 0° to 90°
        (-hL + r,  hW - r, 90),   # bottom-left: 90° to 180°
        (-hL + r, -hW + r, 180),  # top-left: 180° to 270°
        ( hL - r, -hW + r, 270),  # top-right: 270° to 360°
    ]
    for (cx, cy, start_deg) in corners:
        for i in range(ARC_STEPS + 1):
            a = math.radians(start_deg + i * 90 / ARC_STEPS)
            px = cx + r * math.cos(a)
            py = cy + r * math.sin(a)
            rx, ry = rot_fn(px, py)
            pts.append([int(round(rx)), int(round(ry))])

    pts_arr = np.array([pts], dtype=np.int32)
    cv2.fillPoly(canvas, pts_arr, float(level))


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def compute_ground_truth(
    img_w: int,
    img_h: int,
    length_mm: float,
    width_mm: float,
    hole_d_mm: float,
    hole_x_mm: float,
    hole_y_mm: float,
    rotation_deg: float,
    cx_px: float,
    cy_px: float,
    include_hole: bool = True,
) -> dict:
    """
    Return the ground truth pixel positions and expected inspection result.

    Pixel positions are computed from the exact render parameters (not from
    the image pixels) so they serve as reference for error measurement.

    AGENT_SPEC §8: emit name.json truth with true L, W, d, x, y, rotation,
    expected status and reason.
    """
    scale = cfg.SCALE_PX_PER_MM
    theta = math.radians(rotation_deg)

    # ---- Part edges in image pixels ----------------------------------------
    hL_px = (length_mm * scale) / 2
    hW_px = (width_mm  * scale) / 2
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # In axis-aligned (deskewed) frame, left/right/top/bottom of the part:
    x_left_px   = cx_px - hL_px
    x_right_px  = cx_px + hL_px
    y_top_px    = cy_px - hW_px
    y_bottom_px = cy_px + hW_px

    # ---- Hole centre in image pixels ----------------------------------------
    hole_cx_part = -hL_px + hole_x_mm * scale
    hole_cy_part =  hW_px - hole_y_mm * scale   # bottom = +hW; hole is above by hole_y
    hole_cx_px = cos_t * hole_cx_part - sin_t * hole_cy_part + cx_px
    hole_cy_px = sin_t * hole_cx_part + cos_t * hole_cy_part + cy_px

    # ---- Expected decision --------------------------------------------------
    def in_tol(val, key):
        lo, hi = cfg.LIMITS[key]
        v = round(val, cfg.ROUND_DECIMALS)
        return lo - cfg.ROUND_EPSILON <= v <= hi + cfg.ROUND_EPSILON

    if not include_hole or hole_d_mm <= 0:
        status = "REJECT"
        reason = cfg.REASON_HOLE_NOT_FOUND
    elif not in_tol(length_mm, "length"):
        status = "REJECT"
        reason = cfg.REASON_LENGTH_OOT
    elif not in_tol(width_mm, "width"):
        status = "REJECT"
        reason = cfg.REASON_WIDTH_OOT
    elif not in_tol(hole_d_mm, "hole_diameter"):
        status = "REJECT"
        reason = cfg.REASON_HOLE_DIA_OOT
    elif not in_tol(hole_x_mm, "hole_x") or not in_tol(hole_y_mm, "hole_y"):
        status = "REJECT"
        reason = cfg.REASON_HOLE_POS_OOT
    else:
        status = "ACCEPT"
        reason = None

    return {
        "true_length_mm":    round(length_mm, 4),
        "true_width_mm":     round(width_mm, 4),
        "true_hole_d_mm":    round(hole_d_mm, 4) if include_hole else None,
        "true_hole_x_mm":    round(hole_x_mm, 4) if include_hole else None,
        "true_hole_y_mm":    round(hole_y_mm, 4) if include_hole else None,
        "true_rotation_deg": round(rotation_deg, 4),
        "part_cx_px":        round(cx_px, 3),
        "part_cy_px":        round(cy_px, 3),
        "img_w":             img_w,
        "img_h":             img_h,
        # Pixel positions for error measurement
        "true_x_left_px":    round(x_left_px, 3),
        "true_x_right_px":   round(x_right_px, 3),
        "true_y_top_px":     round(y_top_px, 3),
        "true_y_bottom_px":  round(y_bottom_px, 3),
        "true_hole_cx_px":   round(hole_cx_px, 3) if include_hole else None,
        "true_hole_cy_px":   round(hole_cy_px, 3) if include_hole else None,
        "true_hole_r_px":    round((hole_d_mm / 2) * scale, 3) if include_hole else None,
        # Expected inspection outcome
        "expected_status":   status,
        "expected_reason":   reason,
        "include_hole":      include_hole,
    }


# ---------------------------------------------------------------------------
# Save one image + truth JSON
# ---------------------------------------------------------------------------

def save_image_and_truth(
    out_dir: str,
    name: str,
    img: np.ndarray,
    truth: dict,
) -> None:
    """Save PNG and JSON side-by-side in out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), img)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(truth, f, indent=2)


# ---------------------------------------------------------------------------
# Random placement helper
# ---------------------------------------------------------------------------

def random_center(rng, img_w, img_h, length_mm, width_mm, margin_px=MARGIN_PX):
    """
    Return a random (cx, cy) that keeps the axis-aligned bounding box of the
    part at least margin_px from the image border.  Conservative: uses the
    part diagonal as the safe half-size so it works for any rotation.

    AGENT_SPEC §8: part position random with ≥40 px margin.
    """
    scale = cfg.SCALE_PX_PER_MM
    # Worst-case half-size: diagonal / 2 + small buffer
    half = math.hypot(length_mm * scale, width_mm * scale) / 2 + 5
    safe_margin = int(margin_px + half)
    lo_x = safe_margin
    hi_x = img_w - safe_margin
    lo_y = safe_margin
    hi_y = img_h - safe_margin
    if lo_x >= hi_x or lo_y >= hi_y:
        # Fall back to dead centre if image too small
        return img_w / 2, img_h / 2
    cx = rng.uniform(lo_x, hi_x)
    cy = rng.uniform(lo_y, hi_y)
    return cx, cy


# ---------------------------------------------------------------------------
# Labelled suite generator (AGENT_SPEC §8: ≥40 images)
# ---------------------------------------------------------------------------

def generate_suite(out_dir: str, seed: int = 0, img_w: int = DEFAULT_IMG_W, img_h: int = DEFAULT_IMG_H):
    """
    Generate the full labelled test suite (≥40 images).

    Categories (per AGENT_SPEC §8):
      - nominal × 8 (varied rotation −5..+5, position, noise)
      - tolerance edges: inside by 0.05 mm and outside by 0.05 mm for L, W, d, x, y (≈20)
      - gross defects: hole missing, hole far off-position, elliptical hole, part missing,
        part clipped, part rotated 15°, tiny debris blob only
      - robustness: dark-on-light, light-on-dark, strong noise, illumination gradient,
        low contrast (≈35 levels)
    """
    rng = np.random.default_rng(seed)
    count = 0

    def make_and_save(name, **kw):
        nonlocal count
        # Defaults for each image
        params = dict(
            img_w=img_w, img_h=img_h,
            length_mm=30.0, width_mm=8.0,
            hole_d_mm=3.0, hole_x_mm=6.0, hole_y_mm=4.0,
            rotation_deg=0.0,
            corner_radius_mm=DEFAULT_CORNER_RADIUS_MM,
            fg_level=DEFAULT_FG_LEVEL, bg_level=DEFAULT_BG_LEVEL,
            optical_blur_sigma=DEFAULT_BLUR_SIGMA,
            noise_sigma=DEFAULT_NOISE_SIGMA,
            gradient_frac=DEFAULT_GRADIENT,
            include_hole=True,
            polarity_swap=False,
            metal_texture=True,
        )
        params.update(kw)
        # Remove keys not in render_part
        cx = params.pop("cx_px", None)
        cy = params.pop("cy_px", None)
        if cx is None or cy is None:
            cx, cy = random_center(rng, img_w, img_h,
                                   params["length_mm"], params["width_mm"])
        iw = params.pop("img_w")
        ih = params.pop("img_h")
        img = render_part(iw, ih, cx_px=cx, cy_px=cy, rng=rng, **params)
        truth = compute_ground_truth(
            iw, ih,
            length_mm=params["length_mm"],
            width_mm=params["width_mm"],
            hole_d_mm=params.get("hole_d_mm", 3.0),
            hole_x_mm=params.get("hole_x_mm", 6.0),
            hole_y_mm=params.get("hole_y_mm", 4.0),
            rotation_deg=params.get("rotation_deg", 0.0),
            cx_px=cx, cy_px=cy,
            include_hole=params.get("include_hole", True),
        )
        save_image_and_truth(out_dir, name, img, truth)
        count += 1
        print(f"  [{count:02d}] {name:40s}  expected={truth['expected_status']}"
              f"  reason={truth.get('expected_reason') or 'null'}")

    # =========================================================================
    # 1. Nominal × 8: varied rotation, position, noise
    # =========================================================================
    print("\n--- Nominal variants ---")
    rotations = [-5.0, -3.0, -1.5, 0.0, 1.0, 2.5, 4.0, 5.0]
    for i, rot in enumerate(rotations):
        make_and_save(
            f"nominal_{i+1:02d}_rot{rot:+.1f}",
            rotation_deg=rot,
            noise_sigma=rng.uniform(1, 6),
        )

    # =========================================================================
    # 2. Tolerance edges: inside (+0.05 mm) and outside (−0.05 mm) each limit
    #    AGENT_SPEC §8: 1 px margin → 0.05 mm = 1 px at 20 px/mm
    # =========================================================================
    MARGIN = 0.05   # mm; 1 px margin inside/outside the tolerance

    print("\n--- Tolerance edge cases ---")

    # --- Length ---
    lo_L, hi_L = cfg.LIMITS["length"]
    make_and_save("tol_length_inside_min",  length_mm=lo_L + MARGIN)   # ACCEPT
    make_and_save("tol_length_outside_min", length_mm=lo_L - MARGIN)   # REJECT LENGTH
    make_and_save("tol_length_inside_max",  length_mm=hi_L - MARGIN)   # ACCEPT
    make_and_save("tol_length_outside_max", length_mm=hi_L + MARGIN)   # REJECT LENGTH

    # --- Width ---
    lo_W, hi_W = cfg.LIMITS["width"]
    make_and_save("tol_width_inside_min",  width_mm=lo_W + MARGIN)
    make_and_save("tol_width_outside_min", width_mm=lo_W - MARGIN)
    make_and_save("tol_width_inside_max",  width_mm=hi_W - MARGIN)
    make_and_save("tol_width_outside_max", width_mm=hi_W + MARGIN)

    # --- Hole diameter ---
    lo_D, hi_D = cfg.LIMITS["hole_diameter"]
    make_and_save("tol_dia_inside_min",  hole_d_mm=lo_D + MARGIN)
    make_and_save("tol_dia_outside_min", hole_d_mm=lo_D - MARGIN)
    make_and_save("tol_dia_inside_max",  hole_d_mm=hi_D - MARGIN)
    make_and_save("tol_dia_outside_max", hole_d_mm=hi_D + MARGIN)

    # --- Hole X ---
    lo_X, hi_X = cfg.LIMITS["hole_x"]
    make_and_save("tol_hole_x_inside_min",  hole_x_mm=lo_X + MARGIN)
    make_and_save("tol_hole_x_outside_min", hole_x_mm=lo_X - MARGIN)
    make_and_save("tol_hole_x_inside_max",  hole_x_mm=hi_X - MARGIN)
    make_and_save("tol_hole_x_outside_max", hole_x_mm=hi_X + MARGIN)

    # --- Hole Y ---
    lo_Y, hi_Y = cfg.LIMITS["hole_y"]
    make_and_save("tol_hole_y_inside_min",  hole_y_mm=lo_Y + MARGIN)
    make_and_save("tol_hole_y_outside_min", hole_y_mm=lo_Y - MARGIN)
    make_and_save("tol_hole_y_inside_max",  hole_y_mm=hi_Y - MARGIN)
    make_and_save("tol_hole_y_outside_max", hole_y_mm=hi_Y + MARGIN)

    # =========================================================================
    # 3. Gross defects
    # =========================================================================
    print("\n--- Gross defects ---")

    # Hole missing
    make_and_save("defect_hole_missing", include_hole=False)

    # Hole far off-position (x=25 mm from left = beyond 15 mm → also POSSIBLE_180_FLIP)
    make_and_save("defect_hole_far_x", hole_x_mm=25.0)

    # Hole far off Y
    make_and_save("defect_hole_far_y", hole_y_mm=0.5)

    # Oversized hole (elliptical-looking due to large diameter)
    make_and_save("defect_hole_oversize", hole_d_mm=5.0)

    # Tiny hole
    make_and_save("defect_hole_tiny", hole_d_mm=0.5)

    # Part missing (empty background image)
    empty = np.full((img_h, img_w), DEFAULT_BG_LEVEL, dtype=np.uint8)
    # Add noise
    noise = rng.standard_normal((img_h, img_w)).astype(np.float32) * DEFAULT_NOISE_SIGMA
    empty = np.clip(empty.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    truth_empty = {
        "true_length_mm": None, "true_width_mm": None,
        "true_hole_d_mm": None, "true_hole_x_mm": None, "true_hole_y_mm": None,
        "true_rotation_deg": None, "part_cx_px": None, "part_cy_px": None,
        "img_w": img_w, "img_h": img_h,
        "true_x_left_px": None, "true_x_right_px": None,
        "true_y_top_px": None, "true_y_bottom_px": None,
        "true_hole_cx_px": None, "true_hole_cy_px": None, "true_hole_r_px": None,
        "expected_status": "REJECT", "expected_reason": "PART_NOT_FOUND",
        "include_hole": False,
    }
    save_image_and_truth(out_dir, "defect_part_missing", empty, truth_empty)
    count += 1
    print(f"  [{count:02d}] {'defect_part_missing':40s}  expected=REJECT  reason=PART_NOT_FOUND")

    # Part clipped by border (place part so it extends off the left edge)
    clipped_cx = 50  # very close to left border
    clipped_cy = img_h // 2
    img_clipped = render_part(
        img_w, img_h,
        length_mm=30.0, width_mm=8.0,
        hole_d_mm=3.0, hole_x_mm=6.0, hole_y_mm=4.0,
        rotation_deg=0.0,
        cx_px=clipped_cx, cy_px=clipped_cy, rng=rng,
    )
    truth_clipped = compute_ground_truth(
        img_w, img_h, 30.0, 8.0, 3.0, 6.0, 4.0, 0.0, clipped_cx, clipped_cy
    )
    truth_clipped["expected_status"] = "REJECT"
    truth_clipped["expected_reason"] = "PART_NOT_FOUND"
    truth_clipped["note"] = "part clipped by left border"
    save_image_and_truth(out_dir, "defect_part_clipped", img_clipped, truth_clipped)
    count += 1
    print(f"  [{count:02d}] {'defect_part_clipped':40s}  expected=REJECT  reason=PART_NOT_FOUND")

    # Part rotated 15° (BAD_POSE → PART_NOT_FOUND)
    cx15, cy15 = random_center(rng, img_w, img_h, 30.0, 8.0)
    img_rot15 = render_part(
        img_w, img_h,
        length_mm=30.0, width_mm=8.0,
        hole_d_mm=3.0, hole_x_mm=6.0, hole_y_mm=4.0,
        rotation_deg=15.0,
        cx_px=cx15, cy_px=cy15, rng=rng,
    )
    truth_rot15 = compute_ground_truth(
        img_w, img_h, 30.0, 8.0, 3.0, 6.0, 4.0, 15.0, cx15, cy15
    )
    truth_rot15["expected_status"] = "REJECT"
    truth_rot15["expected_reason"] = "PART_NOT_FOUND"
    truth_rot15["note"] = "rotation 15° exceeds 10° BAD_POSE gate"
    save_image_and_truth(out_dir, "defect_part_rot15", img_rot15, truth_rot15)
    count += 1
    print(f"  [{count:02d}] {'defect_part_rot15':40s}  expected=REJECT  reason=PART_NOT_FOUND")

    # =========================================================================
    # 4. Robustness: polarity, strong noise, gradient, low contrast
    # =========================================================================
    print("\n--- Robustness variants ---")

    # Dark part on light background (polarity swap)
    make_and_save("robust_dark_on_light",
                  fg_level=DEFAULT_BG_LEVEL, bg_level=DEFAULT_FG_LEVEL,
                  polarity_swap=False)   # already swapped by using reversed levels

    # Also test with polarity_swap flag
    make_and_save("robust_polarity_swap", polarity_swap=True)

    # Strong noise (σ=8)
    make_and_save("robust_strong_noise", noise_sigma=8)

    # Strong illumination gradient (15%)
    make_and_save("robust_gradient_pos", gradient_frac=0.15)
    make_and_save("robust_gradient_neg", gradient_frac=-0.15)

    # Low contrast (≈35 levels): part ≈80, bg ≈45
    make_and_save("robust_low_contrast",
                  fg_level=80, bg_level=45, noise_sigma=2)

    print(f"\nTotal images generated: {count}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic stamped-part images with ground truth."
    )
    p.add_argument("--out",    default="data/synthetic", help="Output directory")
    p.add_argument("--seed",   type=int, default=0,      help="Random seed (§0 rule 6)")
    p.add_argument("--img-w",  type=int, default=DEFAULT_IMG_W)
    p.add_argument("--img-h",  type=int, default=DEFAULT_IMG_H)
    # Single-image mode
    p.add_argument("--single", action="store_true",
                   help="Render one image with the given parameters")
    p.add_argument("--name",       default="single_test")
    p.add_argument("--length",     type=float, default=30.0)
    p.add_argument("--width",      type=float, default=8.0)
    p.add_argument("--hole-d",     type=float, default=3.0)
    p.add_argument("--hole-x",     type=float, default=6.0)
    p.add_argument("--hole-y",     type=float, default=4.0)
    p.add_argument("--rotation",   type=float, default=0.0)
    p.add_argument("--noise",      type=float, default=DEFAULT_NOISE_SIGMA)
    p.add_argument("--gradient",   type=float, default=0.0)
    p.add_argument("--no-hole",    action="store_true")
    p.add_argument("--swap",       action="store_true", help="Polarity swap")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.single:
        cx, cy = random_center(rng, args.img_w, args.img_h, args.length, args.width)
        img = render_part(
            args.img_w, args.img_h,
            length_mm=args.length, width_mm=args.width,
            hole_d_mm=args.hole_d, hole_x_mm=args.hole_x, hole_y_mm=args.hole_y,
            rotation_deg=args.rotation,
            cx_px=cx, cy_px=cy,
            noise_sigma=args.noise,
            gradient_frac=args.gradient,
            include_hole=not args.no_hole,
            polarity_swap=args.swap,
            rng=rng,
        )
        truth = compute_ground_truth(
            args.img_w, args.img_h,
            args.length, args.width, args.hole_d, args.hole_x, args.hole_y,
            args.rotation, cx, cy,
            include_hole=not args.no_hole,
        )
        save_image_and_truth(args.out, args.name, img, truth)
        print(f"Saved {args.out}/{args.name}.png  (expected: {truth['expected_status']})")
    else:
        print(f"Generating labelled suite -> {args.out}  (seed={args.seed})")
        n = generate_suite(args.out, seed=args.seed, img_w=args.img_w, img_h=args.img_h)
        print(f"\nDone. {n} images in {args.out}/")


if __name__ == "__main__":
    main()
