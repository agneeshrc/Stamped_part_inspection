import json
from pathlib import Path

import cv2
import pytest

from inspection.pipeline import run_full_pipeline

SUITE_DIR = Path("data/synthetic")

@pytest.mark.parametrize("name", [
    "nominal_01_rot-5.0",
    "nominal_02_rot-3.0",
    "nominal_03_rot-1.5",
    "nominal_04_rot+0.0",
    "nominal_05_rot+1.0",
    "nominal_06_rot+2.5",
    "nominal_07_rot+4.0",
    "nominal_08_rot+5.0",
])
def test_full_pipeline_accuracy(name: str):
    img_path = SUITE_DIR / f"{name}.png"
    json_path = SUITE_DIR / f"{name}.json"
    if not img_path.exists():
        pytest.skip(f"Synthetic suite not generated: {img_path}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    with open(json_path, "r", encoding="utf-8") as f:
        truth = json.load(f)
    out = run_full_pipeline(img)

    meas = out["measurements_mm"]

    # Length and width checks
    assert abs(meas["length"] - truth["true_length_mm"]) < 0.15, \
        f"Length error too large for {name}"
    assert abs(meas["width"] - truth["true_width_mm"]) < 0.15, \
        f"Width error too large for {name}"

    # Hole centre checks
    hx = meas["hole_x_from_left"]
    hy = meas["hole_y_from_bottom"]
    assert hx is not None and hy is not None, f"Hole not found in {name}"
    assert abs(hx - truth["true_hole_x_mm"]) < 0.15, f"Hole X error too large ({name})"
    assert abs(hy - truth["true_hole_y_mm"]) < 0.15, f"Hole Y error too large ({name})"

    # Nominal cases should ACCEPT
    assert out["inspection_status"] == "ACCEPT", \
        f"Unexpected {out['inspection_status']} / {out['reject_reason']} for {name}"
