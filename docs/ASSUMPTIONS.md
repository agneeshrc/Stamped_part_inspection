# ASSUMPTIONS.md — Resolved Ambiguities

This file is a copy of AGENT_SPEC §2 (the 14 resolved ambiguities) plus any
additional ones added during implementation. The AGENT_SPEC says: "Where the
assignment is silent, follow the resolved assumptions in §2. If you must add
new ones, append them here."

---

## Assumptions from §2 (verbatim from AGENT_SPEC)

| # | Ambiguity | Decision | Rationale |
|---|---|---|---|
| A1 | Pixel resolution vs tolerance: 1 px = 0.05 mm, hole tolerance is only ±2 px | **Sub-pixel measurement** is mandatory (§4.4, §4.6) | Integer-pixel contours would consume 25-50% of the tolerance band in quantisation alone |
| A2 | "Left" and "bottom" edges when part is rotated up to ±5° | Measure in the **part's own deskewed frame**: left = short edge at smaller x, bottom = long edge at larger image-y (image y grows downward) | The bounding box of the part is the stated reference frame; deskewing removes rotation error |
| A3 | What "orientation verification" means | Compute long-axis angle θ. θ within ±5° = normal. 5° < \|θ\| ≤ 10° = accepted with warning. \|θ\| > 10° → `PART_NOT_FOUND` ("invalid pose") | Spec allows ±5°; small margin for real life; beyond that the assumption is broken |
| A4 | What is "invalid" part for `PART_NOT_FOUND` | Any of: no foreground blob; blob area < 20% of nominal part area; blob touches image border (clipped, so dimensions unreliable); rectangularity (area / minAreaRect area) < 0.85; aspect ratio outside [2.0, 6.0]; global contrast < 30 grey levels | Wrong size parts must still be *measured* and rejected for LENGTH/WIDTH, so presence gates must be loose |
| A5 | Multiple simultaneous failures | Single `reject_reason` by priority: `PART_NOT_FOUND` > `HOLE_NOT_FOUND` > `LENGTH_*` > `WIDTH_*` > `HOLE_DIAMETER_*` > `HOLE_POSITION_*`. Per-check PASS/FAIL still reported for all five. | Spec says "primary failure reason"; priority follows pipeline order, gross defects before fine ones |
| A6 | Hole position uses two measurements but one reason code | If either x or y fails → `HOLE_POSITION_OUT_OF_TOLERANCE` | Only one position code exists |
| A7 | Rounding and boundary inclusivity | Round measurements to 0.01 mm **first**, then compare with inclusive limits (`min <= v <= max`, epsilon 1e-9). Decision is made on the same numbers printed in the JSON | The JSON is self-consistent and auditable; 0.01 mm is below measurement uncertainty anyway |
| A8 | Checks not evaluated (e.g. part not found) | `measurements_mm` values `null`, `result` values `"NOT_EVALUATED"`, `part_present` false/true accordingly | Schema example only shows the happy path; this is the least surprising extension |
| A9 | Extra diagnostics | Default JSON = exact schema. `--diagnostics` adds a top-level `"diagnostics"` object (rotation, px values, method agreement, warnings, marginal flags, timing) | Keeps compatibility with any strict schema validator |
| A10 | Marginal results | If any measurement is within `GUARD_MM = 0.03` of a limit, add warning `MARGINAL:<check>` in diagnostics; decision unchanged | Honest about measurement uncertainty without changing the contract |
| A11 | Part flipped 180° (hole near right edge) | Out of scope (±5° only). Detect: if hole x-from-left > 15 mm, add warning `POSSIBLE_180_FLIP`; no auto-correction | Documented limitation + future work |
| A12 | No sample images supplied | Build synthetic generator with ground truth (§8) and accept arbitrary real images too | Allows quantitative validation; real images may appear at the interview |
| A13 | Background polarity unknown (dark-on-light or light-on-dark) | Auto-detect from the image border median | "Uniform, contrasting background" does not say which |
| A14 | Image resolution | No assumption beyond 20 px/mm; part may be anywhere in frame | Fixed camera but part placement varies |

---

## Additional Assumptions Added During Implementation

*(Append new rows here if any silent assumption is made during coding.)*

| # | Ambiguity | Decision | Rationale |
|---|---|---|---|
| A15 | Synthetic image size | Default 1000×600 px as specified in §8 | Gives 50×30 mm field of view at 20 px/mm, comfortably larger than the nominal 30×8 mm part |
