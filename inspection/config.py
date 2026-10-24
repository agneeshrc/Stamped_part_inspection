"""
inspection/config.py — All numeric constants used by the inspection pipeline.

AGENT_SPEC §1 (tolerances), §2 (resolved ambiguities), §4 (algorithm knobs).
Every number lives here with a comment explaining *why* that value was chosen.
No magic numbers anywhere else in the codebase.
"""

# ---------------------------------------------------------------------------
# §1 — Scale (given by assignment; do NOT change)
# ---------------------------------------------------------------------------
SCALE_PX_PER_MM = 20          # 20 px = 1 mm (assignment-provided calibration)
SCALE_MM_PER_PX = 1.0 / SCALE_PX_PER_MM   # 0.05 mm/px

# ---------------------------------------------------------------------------
# §1 — Nominal dimensions (mm) — used only for documentation / gating
# ---------------------------------------------------------------------------
NOMINAL_LENGTH_MM   = 30.00
NOMINAL_WIDTH_MM    =  8.00
NOMINAL_HOLE_DIA_MM =  3.00
NOMINAL_HOLE_X_MM   =  6.00   # hole centre from left edge
NOMINAL_HOLE_Y_MM   =  4.00   # hole centre from bottom edge

# Nominal in pixels (for ROI sizing, loose gates)
NOMINAL_LENGTH_PX   = NOMINAL_LENGTH_MM   * SCALE_PX_PER_MM   # 600 px
NOMINAL_WIDTH_PX    = NOMINAL_WIDTH_MM    * SCALE_PX_PER_MM   # 160 px
NOMINAL_HOLE_DIA_PX = NOMINAL_HOLE_DIA_MM * SCALE_PX_PER_MM   # 60 px

# ---------------------------------------------------------------------------
# §1 — Tolerance limits (mm) — inclusive both ends (see A7)
# ---------------------------------------------------------------------------
LIMITS = {
    # key            : (min_mm, max_mm)
    "length"         : (29.80, 30.20),    # ±0.20 mm, i.e. ±4 px at 20 px/mm
    "width"          : ( 7.85,  8.15),    # ±0.15 mm, i.e. ±3 px
    "hole_diameter"  : ( 2.90,  3.10),    # ±0.10 mm, i.e. ±2 px (tightest!)
    "hole_x"         : ( 5.80,  6.20),    # ±0.20 mm, i.e. ±4 px
    "hole_y"         : ( 3.80,  4.20),    # ±0.20 mm, i.e. ±4 px
}

# ---------------------------------------------------------------------------
# §2 A10 — Marginal-result guard band (mm)
# If |measurement - nearest limit| < GUARD_MM, add a MARGINAL warning.
# 0.03 mm ≈ 0.6 px; honest about uncertainty near the boundary.
# ---------------------------------------------------------------------------
GUARD_MM = 0.03

# ---------------------------------------------------------------------------
# §4.1 — Preprocessing
# ---------------------------------------------------------------------------
BLUR_KERNEL = (5, 5)          # GaussianBlur kernel for segmentation only (§4.1 step 2)
BLUR_SIGMA  = 1.0             # sigma for segmentation blur
LARGE_SIGMA = 50              # sigma for illumination-flattening GaussianBlur (§4.1 step 5)
BORDER_FRAC = 0.03            # fraction of image used as "border frame" for bg polarity (§4.1 step 3)
MORPH_KERNEL_SIZE = 3         # open/close kernel (§4.1 step 6); kept small so 60 px hole survives

# Segmentation quality gates
CONTRAST_MIN_LEVELS    = 30   # |mean(fg) - mean(bg)| must exceed this (§4.1 step 4)
OTSU_SEPARABILITY_MIN  = 0.6  # inter-class variance ratio; below this, apply flattening (§4.1 step 5)
GRADIENT_CORNER_THRESH = 25   # corner-median difference level triggering flattening (§4.1 step 5)

# ---------------------------------------------------------------------------
# §2 A4 / §4.2 — Part validity gates
# ---------------------------------------------------------------------------
PART_AREA_MIN_FRAC  = 0.20    # blob area >= 20% of nominal part area (A4); loose so short parts fail on LENGTH
BORDER_MARGIN_PX    = 3       # blob must not touch within this many px of the image edge (clipping check)
RECTANGULARITY_MIN  = 0.85    # area / minAreaRect area >= this (A4)
ASPECT_RATIO_MIN    = 2.0     # long/short side; nominal ~3.75 (30/8) (A4)
ASPECT_RATIO_MAX    = 6.0     # upper bound; very long slivers rejected (A4)

# Nominal part area for the 20% gate
NOMINAL_PART_AREA_PX = NOMINAL_LENGTH_PX * NOMINAL_WIDTH_PX   # 600 * 160 = 96000 px²

# ---------------------------------------------------------------------------
# §2 A3 / §4.2 — Pose (rotation angle) gates
# ---------------------------------------------------------------------------
POSE_OK_DEG     = 5.0         # |θ| <= 5° → normal (A3)
POSE_WARN_DEG   = 10.0        # 5° < |θ| <= 10° → accepted with warning (A3)
# |θ| > 10° → PART_NOT_FOUND "BAD_POSE"

# ---------------------------------------------------------------------------
# §4.4 — Sub-pixel edge measurement
# ---------------------------------------------------------------------------
EDGE_PROFILE_HALF   = 12      # profile extends ±12 px around coarse edge (§4.4)
EDGE_SAMPLE_STEP    = 2       # sample every 2 px along the edge (§4.4)
EDGE_TRIM_FRAC      = 0.15    # skip the inner 15% and outer 15% end zones (§4.4 corner radius avoidance)
EDGE_CENTRAL_FRAC_LO = 0.15   # lower bound of central sampling region (§4.4)
EDGE_CENTRAL_FRAC_HI = 0.85   # upper bound of central sampling region
EDGE_OUTLIER_THRESH = 1.5     # discard profiles > 1.5 px from median (§4.4)
EDGE_MIN_VALID_FRAC = 0.60    # if fewer than 60% profiles survive, declare EDGE_UNRELIABLE (§4.4)
EDGE_METHOD_DISAGREE_THRESH = 1.5   # |MethodA - MethodB| > this → LW_METHOD_DISAGREE warning (§4.4)

# Method B: minAreaRect contour +1 px correction
CONTOUR_UNDERCOUNT_CORRECTION_PX = 1.0  # contour pixels are centres; boundary is ~1 px bigger (§4.4 Method B)

# ---------------------------------------------------------------------------
# §4.5 — Coarse hole detection
# ---------------------------------------------------------------------------
HOLE_INTERIOR_SHRINK_PX = 6   # shrink part bbox by 6 px each side before searching for hole (§4.5)
HOLE_AREA_MIN_FACTOR    = 0.1 # area >= 0.1 * π * r_nominal² (§4.5)
HOLE_AREA_MAX_FACTOR    = 5.0 # area <= 5.0 * π * r_nominal² (§4.5)
CIRCULARITY_MIN         = 0.80 # 4πA/P² >= 0.80; digital perimeter over-estimates so not 0.9+ (§4.5)
HOLE_CONTRAST_MIN       = 30  # hole interior vs surrounding part (§4.5); same scale as CONTRAST_MIN

# Nominal hole area bounds in pixels²
import math as _math
_r_nom = NOMINAL_HOLE_DIA_PX / 2   # 30 px
HOLE_AREA_MIN_PX = HOLE_AREA_MIN_FACTOR * _math.pi * _r_nom ** 2   # ≈ 283 px²
HOLE_AREA_MAX_PX = HOLE_AREA_MAX_FACTOR * _math.pi * _r_nom ** 2   # ≈ 14137 px²

# ---------------------------------------------------------------------------
# §4.6 — Sub-pixel hole refinement
# ---------------------------------------------------------------------------
HOLE_RAY_COUNT       = 120    # number of radial rays; 360°/120 = 3° spacing (§4.6)
HOLE_RAY_R_MIN_FRAC  = 0.50   # innermost ray sample: 50% of coarse radius (§4.6)
HOLE_RAY_R_MAX_FRAC  = 1.60   # outermost ray sample: 160% of coarse radius (§4.6)
HOLE_RAY_STEP_PX     = 0.25   # radial step in pixels along each ray (§4.6)
HOLE_OUTLIER_MAD     = 3.0    # drop ray-edge points > 3 * MAD (§4.6)
HOLE_RESIDUAL_MAX_PX = 1.0    # RMS residual > this → HOLE_NOT_FOUND "UNCLEAR" (§4.6)
HOLE_VALID_RAY_MIN_FRAC = 0.70 # < 70% rays valid → HOLE_NOT_FOUND (§4.6)

# Hough cross-check parameters (diagnostic only, never overrides; see §4.6 rationale)
HOUGH_DP         = 1          # inverse accumulator resolution
HOUGH_PARAM1     = 100        # Canny upper threshold for Hough (§4.6)
HOUGH_PARAM2     = 20         # accumulator threshold (§4.6)
HOUGH_AGREE_PX   = 1.5        # |Hough vs fit| < this → hough_agrees=true (§4.6)

# Photometric area cross-check
HOLE_PHOTOMETRIC_DISAGREE_WARN_PX = 0.5   # > 0.5 px → HOLE_METHOD_DISAGREE warning (§4.6)
HOLE_PHOTOMETRIC_DISAGREE_FATAL_PX = 3.0  # > 3 px → treat as unclear, HOLE_NOT_FOUND (§4.6)

# ---------------------------------------------------------------------------
# §2 A11 — Flip detection
# ---------------------------------------------------------------------------
FLIP_WARN_HOLE_X_MM = 15.0    # if hole_x_from_left > 15 mm, warn POSSIBLE_180_FLIP (A11)

# ---------------------------------------------------------------------------
# §5 / §2 A5 — Reject reason priority (ascending index = higher priority)
# ---------------------------------------------------------------------------
REJECT_PRIORITY = [
    "PART_NOT_FOUND",
    "HOLE_NOT_FOUND",
    "LENGTH_OUT_OF_TOLERANCE",
    "WIDTH_OUT_OF_TOLERANCE",
    "HOLE_DIAMETER_OUT_OF_TOLERANCE",
    "HOLE_POSITION_OUT_OF_TOLERANCE",
]

# ---------------------------------------------------------------------------
# §1 — Exact reject reason code strings (must not be changed)
# ---------------------------------------------------------------------------
REASON_PART_NOT_FOUND     = "PART_NOT_FOUND"
REASON_HOLE_NOT_FOUND     = "HOLE_NOT_FOUND"
REASON_LENGTH_OOT         = "LENGTH_OUT_OF_TOLERANCE"
REASON_WIDTH_OOT          = "WIDTH_OUT_OF_TOLERANCE"
REASON_HOLE_DIA_OOT       = "HOLE_DIAMETER_OUT_OF_TOLERANCE"
REASON_HOLE_POS_OOT       = "HOLE_POSITION_OUT_OF_TOLERANCE"

# ---------------------------------------------------------------------------
# §2 A7 — Measurement rounding
# ---------------------------------------------------------------------------
ROUND_DECIMALS = 2   # round to 0.01 mm before comparison and JSON output (A7)
ROUND_EPSILON  = 1e-9  # for floating-point inclusive limit comparison (A7)
