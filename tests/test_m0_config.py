"""
tests/test_m0_config.py — M0 acceptance test.

AGENT_SPEC §10 M0: "pytest green on trivial test".
This test verifies:
  1. config.py imports without errors.
  2. The five LIMITS entries have the exact (min, max) values from §1.
  3. Scale constants are mathematically consistent.
  4. Reject reason strings are exactly as specified in §1.
"""
import math
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inspection import config


class TestScale:
    """Scale constants must be self-consistent (AGENT_SPEC §1)."""

    def test_scale_reciprocal(self):
        """1 / SCALE_PX_PER_MM == SCALE_MM_PER_PX."""
        assert abs(1.0 / config.SCALE_PX_PER_MM - config.SCALE_MM_PER_PX) < 1e-12

    def test_scale_value(self):
        """20 px = 1 mm as stated in the assignment."""
        assert config.SCALE_PX_PER_MM == 20


class TestLimits:
    """Tolerance limits must match §1 exactly."""

    def test_length_limits(self):
        assert config.LIMITS["length"] == (29.80, 30.20)

    def test_width_limits(self):
        assert config.LIMITS["width"] == (7.85, 8.15)

    def test_hole_diameter_limits(self):
        assert config.LIMITS["hole_diameter"] == (2.90, 3.10)

    def test_hole_x_limits(self):
        assert config.LIMITS["hole_x"] == (5.80, 6.20)

    def test_hole_y_limits(self):
        assert config.LIMITS["hole_y"] == (3.80, 4.20)

    def test_all_limits_have_min_lt_max(self):
        """Every limit pair must satisfy min < max."""
        for key, (lo, hi) in config.LIMITS.items():
            assert lo < hi, f"LIMITS[{key!r}]: min={lo} >= max={hi}"


class TestReasonCodes:
    """Reject reason strings must be exactly as specified in §1."""

    EXPECTED = {
        "PART_NOT_FOUND",
        "HOLE_NOT_FOUND",
        "LENGTH_OUT_OF_TOLERANCE",
        "WIDTH_OUT_OF_TOLERANCE",
        "HOLE_DIAMETER_OUT_OF_TOLERANCE",
        "HOLE_POSITION_OUT_OF_TOLERANCE",
    }

    def test_reason_constants_exist(self):
        actual = {
            config.REASON_PART_NOT_FOUND,
            config.REASON_HOLE_NOT_FOUND,
            config.REASON_LENGTH_OOT,
            config.REASON_WIDTH_OOT,
            config.REASON_HOLE_DIA_OOT,
            config.REASON_HOLE_POS_OOT,
        }
        assert actual == self.EXPECTED

    def test_priority_list_covers_all_reasons(self):
        """REJECT_PRIORITY must contain all six reason codes."""
        assert set(config.REJECT_PRIORITY) == self.EXPECTED

    def test_priority_order(self):
        """PART_NOT_FOUND has highest priority (index 0); HOLE_POSITION lowest (index 5)."""
        p = config.REJECT_PRIORITY
        assert p[0] == "PART_NOT_FOUND"
        assert p[-1] == "HOLE_POSITION_OUT_OF_TOLERANCE"


class TestNominals:
    """Nominal dimensions in pixels must be consistent with scale."""

    def test_nominal_length_px(self):
        assert config.NOMINAL_LENGTH_PX == pytest.approx(600.0)

    def test_nominal_width_px(self):
        assert config.NOMINAL_WIDTH_PX == pytest.approx(160.0)

    def test_nominal_hole_dia_px(self):
        assert config.NOMINAL_HOLE_DIA_PX == pytest.approx(60.0)

    def test_hole_area_bounds_positive(self):
        assert config.HOLE_AREA_MIN_PX > 0
        assert config.HOLE_AREA_MAX_PX > config.HOLE_AREA_MIN_PX
