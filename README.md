# Stamped Part Inspection

## Overview

A vision-based inspection system for stamped metal parts. Reads a grayscale PNG image, detects the rectangular part and its circular hole, measures geometric dimensions (length, width, hole diameter, hole position), and outputs an ACCEPT/REJECT decision with a JSON report matching the assignment specification.

## Quick Start

```bash
# Install dependencies (Python 3.10+)
pip install -r requirements.txt

# Generate the synthetic test suite (42 images)
python tools/make_synthetic.py --out data/synthetic --seed 0

# Run inspection on the full suite with annotated output
python inspect_part.py data/synthetic --out-dir results --annotate

# Run on a single image
python inspect_part.py data/synthetic/nominal_04_rot+0.0.png --out-dir results --annotate

# Run tests
python -m pytest tests/ -v

# Validate against ground truth
python validate.py
```

## Output JSON Schema

```json
{
  "part_present": true,
  "inspection_status": "ACCEPT",
  "measurements_mm": {
    "length": 29.95,
    "width": 8.04,
    "hole_diameter": 3.02,
    "hole_x_from_left": 6.03,
    "hole_y_from_bottom": 3.96
  },
  "result": {
    "length": "PASS",
    "width": "PASS",
    "hole_diameter": "PASS",
    "hole_x_position": "PASS",
    "hole_y_position": "PASS"
  },
  "reject_reason": null
}
```

## Annotated Output

Each annotated image shows:
- **Green contour** around the detected part
- **Yellow circle + red crosshair** on the detected hole
- **Length / width / diameter labels** with dimension lines
- **ACCEPT/REJECT banner** with reject reason
- **Per-check pass/fail bar** at bottom

## Directory Layout

```
stamped-part-inspection/
├─ inspection/                 # Core library
│   ├─ config.py               # All constants and tolerances
│   ├─ preprocess.py           # Grayscale, Otsu, morphology
│   ├─ part_detector.py        # Contour analysis, validity gates
│   ├─ predict.py              # Hole localisation, pose refinement
│   └─ pipeline.py             # Full orchestration + JSON schema
├─ tools/
│   └─ make_synthetic.py       # Synthetic test image generator
├─ tests/                      # Pytest suite (M0-M4)
├─ inspect_part.py             # CLI entry point
├─ validate.py                 # Ground truth comparison
├─ requirements.txt
└─ README.md
```

## Measurement Specification

| Measurement | Nominal | Tolerance |
|---|---|---|
| Length | 30.00 mm | ±0.20 mm |
| Width | 8.00 mm | ±0.15 mm |
| Hole diameter | 3.00 mm | ±0.10 mm |
| Hole X from left | 6.00 mm | ±0.20 mm |
| Hole Y from bottom | 4.00 mm | ±0.20 mm |

Scale: **20 pixels = 1 mm**
