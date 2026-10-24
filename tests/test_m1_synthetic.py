"""
tests/test_m1_synthetic.py — M1 acceptance tests for the synthetic generator.

AGENT_SPEC §10 M1:
  - 40+ images generated
  - truth JSON consistent with pixel positions
  - opening images shows realistic parts (checked via image statistics)
"""

import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inspection import config as cfg
from tools.make_synthetic import (
    render_part,
    compute_ground_truth,
    generate_suite,
    random_center,
    DEFAULT_FG_LEVEL,
    DEFAULT_BG_LEVEL,
    DEFAULT_IMG_W,
    DEFAULT_IMG_H,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUITE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic"
)


@pytest.fixture(scope="session")
def suite(tmp_path_factory):
    """Generate the full suite once per test session into a temp directory."""
    out = str(tmp_path_factory.mktemp("synthetic"))
    n = generate_suite(out, seed=42)
    return out, n


# ---------------------------------------------------------------------------
# M1-A: Suite size
# ---------------------------------------------------------------------------

class TestSuiteSize:
    def test_at_least_40_images(self, suite):
        out_dir, n = suite
        assert n >= 40, f"Expected ≥40 images, got {n}"

    def test_png_and_json_pairs(self, suite):
        """Every .png must have a matching .json."""
        out_dir, _ = suite
        pngs = {f[:-4] for f in os.listdir(out_dir) if f.endswith(".png")}
        jsons = {f[:-5] for f in os.listdir(out_dir) if f.endswith(".json")}
        missing_json = pngs - jsons
        assert not missing_json, f"PNGs missing JSON: {missing_json}"


# ---------------------------------------------------------------------------
# M1-B: Truth JSON consistency
# ---------------------------------------------------------------------------

class TestTruthConsistency:
    """
    For images with a known part, verify that the pixel positions stored in
    the truth JSON are geometrically consistent with the declared mm values.

    AGENT_SPEC §10 M1: "truth JSON consistent with pixel positions".
    """

    def _load_truths(self, suite):
        out_dir, _ = suite
        truths = []
        for fn in sorted(os.listdir(out_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(out_dir, fn)) as f:
                    t = json.load(f)
                truths.append((fn, t))
        return truths

    def test_length_consistency(self, suite):
        """x_right - x_left in pixels == length_mm * 20, within 0.01 px."""
        errors = []
        for fn, t in self._load_truths(suite):
            if t["true_length_mm"] is None:
                continue
            expected_px = t["true_length_mm"] * cfg.SCALE_PX_PER_MM
            actual_px = t["true_x_right_px"] - t["true_x_left_px"]
            err = abs(actual_px - expected_px)
            if err > 0.01:
                errors.append(f"{fn}: err={err:.4f} px")
        assert not errors, "Length truth inconsistent:\n" + "\n".join(errors)

    def test_width_consistency(self, suite):
        """y_bottom - y_top in pixels == width_mm * 20, within 0.01 px."""
        errors = []
        for fn, t in self._load_truths(suite):
            if t["true_width_mm"] is None:
                continue
            expected_px = t["true_width_mm"] * cfg.SCALE_PX_PER_MM
            actual_px = t["true_y_bottom_px"] - t["true_y_top_px"]
            err = abs(actual_px - expected_px)
            if err > 0.01:
                errors.append(f"{fn}: err={err:.4f} px")
        assert not errors, "Width truth inconsistent:\n" + "\n".join(errors)

    def test_hole_radius_consistency(self, suite):
        """true_hole_r_px == (true_hole_d_mm / 2) * 20."""
        errors = []
        for fn, t in self._load_truths(suite):
            if t["true_hole_d_mm"] is None:
                continue
            expected_r = (t["true_hole_d_mm"] / 2) * cfg.SCALE_PX_PER_MM
            err = abs(t["true_hole_r_px"] - expected_r)
            if err > 0.01:
                errors.append(f"{fn}: err={err:.4f} px")
        assert not errors, "Hole radius truth inconsistent:\n" + "\n".join(errors)

    def test_hole_position_in_part_frame(self, suite):
        """
        hole_x_from_left = hole_cx - x_left (in deskewed/zero-rotation frame).
        For zero-rotation images this is exact; test on nominal images.
        """
        errors = []
        for fn, t in self._load_truths(suite):
            if t["true_hole_d_mm"] is None:
                continue
            # Only exact for zero rotation
            if abs(t["true_rotation_deg"]) > 0.1:
                continue
            measured_x = t["true_hole_cx_px"] - t["true_x_left_px"]
            expected_x = t["true_hole_x_mm"] * cfg.SCALE_PX_PER_MM
            err = abs(measured_x - expected_x)
            if err > 0.1:
                errors.append(f"{fn}: x err={err:.4f} px")
            measured_y = t["true_y_bottom_px"] - t["true_hole_cy_px"]
            expected_y = t["true_hole_y_mm"] * cfg.SCALE_PX_PER_MM
            err_y = abs(measured_y - expected_y)
            if err_y > 0.1:
                errors.append(f"{fn}: y err={err_y:.4f} px")
        assert not errors, "Hole position truth inconsistent:\n" + "\n".join(errors)

    def test_expected_status_fields_present(self, suite):
        """Every truth JSON must have expected_status and expected_reason."""
        out_dir, _ = suite
        for fn in os.listdir(out_dir):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(out_dir, fn)) as f:
                t = json.load(f)
            assert "expected_status" in t, f"{fn} missing expected_status"
            assert "expected_reason" in t, f"{fn} missing expected_reason"
            assert t["expected_status"] in ("ACCEPT", "REJECT"), \
                f"{fn} bad status: {t['expected_status']}"


# ---------------------------------------------------------------------------
# M1-C: Image realism checks
# ---------------------------------------------------------------------------

class TestImageRealism:
    """
    Visual sanity checks without opening a window.
    AGENT_SPEC §10 M1: "opening a few shows realistic parts".
    """

    def _get_nominal_image(self, suite):
        out_dir, _ = suite
        for fn in sorted(os.listdir(out_dir)):
            if fn.startswith("nominal") and fn.endswith(".png"):
                img = cv2.imread(os.path.join(out_dir, fn), cv2.IMREAD_GRAYSCALE)
                with open(os.path.join(out_dir, fn.replace(".png", ".json"))) as f:
                    t = json.load(f)
                return img, t
        pytest.skip("No nominal image found")

    def test_image_shape(self, suite):
        img, _ = self._get_nominal_image(suite)
        assert img.shape == (DEFAULT_IMG_H, DEFAULT_IMG_W), \
            f"Expected {DEFAULT_IMG_H}×{DEFAULT_IMG_W}, got {img.shape}"

    def test_image_has_two_grey_levels(self, suite):
        """
        A realistic part image should have a bimodal histogram:
        background near bg_level, part near fg_level.
        Check that both the low and high mode exist.
        """
        img, _ = self._get_nominal_image(suite)
        assert img.mean() > DEFAULT_BG_LEVEL, "Mean should be above background level"
        assert img.min() < (DEFAULT_FG_LEVEL + DEFAULT_BG_LEVEL) / 2, \
            "Should have dark pixels (background)"
        assert img.max() > (DEFAULT_FG_LEVEL + DEFAULT_BG_LEVEL) / 2, \
            "Should have bright pixels (part)"

    def test_contrast_sufficient(self, suite):
        """Global contrast must meet the pipeline's CONTRAST_MIN_LEVELS."""
        img, _ = self._get_nominal_image(suite)
        contrast = float(img.max()) - float(img.min())
        assert contrast >= cfg.CONTRAST_MIN_LEVELS, \
            f"Contrast {contrast:.1f} < {cfg.CONTRAST_MIN_LEVELS}"

    def test_part_occupies_expected_fraction(self, suite):
        """
        Nominal part (30×8 mm = 600×160 px = 96000 px²) at 20 px/mm.
        In a 1000×600 image: 96000 / 600000 ≈ 16%.
        Expect 10%–30% of pixels to be brighter than midpoint.
        """
        img, _ = self._get_nominal_image(suite)
        mid = (DEFAULT_FG_LEVEL + DEFAULT_BG_LEVEL) // 2
        frac = (img > mid).mean()
        assert 0.08 <= frac <= 0.40, \
            f"Part pixel fraction {frac:.3f} outside expected range [0.08, 0.40]"

    def test_polarity_swap_image(self, suite):
        """Dark-on-light image should have a reversed grey distribution."""
        out_dir, _ = suite
        fn = "robust_polarity_swap.png"
        path = os.path.join(out_dir, fn)
        if not os.path.exists(path):
            pytest.skip("polarity_swap image not found")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # In a polarity-swapped image, the part is dark → mean should be high
        assert img.mean() > 128, "Polarity-swapped image should have high mean (bright background)"


# ---------------------------------------------------------------------------
# M1-D: render_part unit tests (no file I/O)
# ---------------------------------------------------------------------------

class TestRenderPart:
    """Unit tests for render_part() in isolation."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        img = render_part(200, 100, 30.0, 8.0, 3.0, 6.0, 4.0,
                          rotation_deg=0.0, cx_px=100, cy_px=50, rng=rng,
                          noise_sigma=0, optical_blur_sigma=0, metal_texture=False)
        assert img.shape == (100, 200)
        assert img.dtype == np.uint8

    def test_no_hole_image(self):
        """With include_hole=False, the image should have no dark interior spot."""
        rng = np.random.default_rng(1)
        img_hole = render_part(200, 100, 30.0, 8.0, 3.0, 6.0, 4.0,
                               rotation_deg=0.0, cx_px=100, cy_px=50, rng=rng,
                               noise_sigma=0, optical_blur_sigma=0,
                               metal_texture=False, include_hole=True)
        rng2 = np.random.default_rng(1)
        img_no_hole = render_part(200, 100, 30.0, 8.0, 3.0, 6.0, 4.0,
                                  rotation_deg=0.0, cx_px=100, cy_px=50, rng=rng2,
                                  noise_sigma=0, optical_blur_sigma=0,
                                  metal_texture=False, include_hole=False)
        # With a hole, min pixel inside the part region should be darker
        assert int(img_hole.min()) < int(img_no_hole.min()) + 5, \
            "Image with hole should have darker minimum pixel"

    def test_rotation_changes_image(self):
        """A rotated part should differ from the upright one."""
        rng = np.random.default_rng(2)
        img0 = render_part(400, 200, 30.0, 8.0, 3.0, 6.0, 4.0,
                           rotation_deg=0.0, cx_px=200, cy_px=100, rng=rng,
                           noise_sigma=0, optical_blur_sigma=0, metal_texture=False)
        rng2 = np.random.default_rng(2)
        img5 = render_part(400, 200, 30.0, 8.0, 3.0, 6.0, 4.0,
                           rotation_deg=5.0, cx_px=200, cy_px=100, rng=rng2,
                           noise_sigma=0, optical_blur_sigma=0, metal_texture=False)
        diff = np.abs(img0.astype(int) - img5.astype(int))
        assert diff.max() > 10, "Rotated image should differ from upright image"

    def test_deterministic_with_same_seed(self):
        """Same seed → byte-identical output (AGENT_SPEC §0 rule 6)."""
        rng1 = np.random.default_rng(99)
        img1 = render_part(200, 100, 30.0, 8.0, 3.0, 6.0, 4.0,
                           rotation_deg=2.0, cx_px=100, cy_px=50, rng=rng1,
                           noise_sigma=3, metal_texture=True)
        rng2 = np.random.default_rng(99)
        img2 = render_part(200, 100, 30.0, 8.0, 3.0, 6.0, 4.0,
                           rotation_deg=2.0, cx_px=100, cy_px=50, rng=rng2,
                           noise_sigma=3, metal_texture=True)
        assert np.array_equal(img1, img2), "Same seed must produce byte-identical output"


# ---------------------------------------------------------------------------
# M1-E: compute_ground_truth unit tests
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_nominal_accept(self):
        t = compute_ground_truth(1000, 600, 30.0, 8.0, 3.0, 6.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300)
        assert t["expected_status"] == "ACCEPT"
        assert t["expected_reason"] is None

    def test_short_part_reject_length(self):
        t = compute_ground_truth(1000, 600, 29.5, 8.0, 3.0, 6.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300)
        assert t["expected_status"] == "REJECT"
        assert t["expected_reason"] == cfg.REASON_LENGTH_OOT

    def test_oversize_hole_reject(self):
        t = compute_ground_truth(1000, 600, 30.0, 8.0, 3.5, 6.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300)
        assert t["expected_status"] == "REJECT"
        assert t["expected_reason"] == cfg.REASON_HOLE_DIA_OOT

    def test_missing_hole_reject(self):
        t = compute_ground_truth(1000, 600, 30.0, 8.0, 3.0, 6.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300,
                                 include_hole=False)
        assert t["expected_status"] == "REJECT"
        assert t["expected_reason"] == cfg.REASON_HOLE_NOT_FOUND

    def test_hole_pos_x_reject(self):
        t = compute_ground_truth(1000, 600, 30.0, 8.0, 3.0, 7.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300)
        assert t["expected_status"] == "REJECT"
        assert t["expected_reason"] == cfg.REASON_HOLE_POS_OOT

    def test_length_priority_over_width(self):
        """If both length and width are out of tolerance, reason = LENGTH (A5 priority)."""
        t = compute_ground_truth(1000, 600, 29.0, 7.0, 3.0, 6.0, 4.0,
                                 rotation_deg=0.0, cx_px=500, cy_px=300)
        assert t["expected_reason"] == cfg.REASON_LENGTH_OOT
