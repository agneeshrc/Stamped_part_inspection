"""
inspect_part.py — CLI entry point for the stamped-part inspection pipeline.

Usage:
    python inspect_part.py <image_or_dir> [--out-dir results] [--annotate]

Processes one or more PNG images, producing a JSON result file for each and
optionally an annotated PNG overlay.
"""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from inspection.pipeline import run_full_pipeline, export_json
from inspection import config as cfg


# ---------------------------------------------------------------------------
# Annotation constants
# ---------------------------------------------------------------------------
_GREEN  = (0, 255, 0)
_RED    = (0, 0, 255)
_BLUE   = (255, 180, 0)
_YELLOW = (0, 255, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX
_FONT_S = 0.45
_THICK  = 1


def annotate_and_save(image_path: Path, result: dict, out_dir: Path) -> None:
    """Draw detection overlays on the original image and write to *out_dir*.

    Annotation includes (per assignment §7):
    • Part contour / bounding rectangle (green)
    • Detected hole with centre cross-hair and circle (cyan)
    • Length and width dimension labels
    • Hole diameter label
    • ACCEPT / REJECT banner with reason
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image for annotation: {image_path}", file=sys.stderr)
        return

    h, w = img.shape[:2]

    # Re-run preprocessing to obtain the binary mask for visualisation.
    from inspection.preprocess import preprocess
    from inspection.part_detector import detect_part

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    pre = preprocess(gray)
    mask = pre["mask"]

    # --- Part contour (green) -----------------------------------------------
    contours_ext, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_ext:
        cnt = max(contours_ext, key=cv2.contourArea)
        cv2.drawContours(img, [cnt], -1, _GREEN, 2)

        # Draw minAreaRect as dashed-style (solid thin line)
        rect = cv2.minAreaRect(cnt)
        box_pts = cv2.boxPoints(rect).astype(np.intp)
        cv2.polylines(img, [box_pts], True, _GREEN, 1, cv2.LINE_AA)

        # Compute axis-aligned bbox for hole coordinate conversion
        bx, by, bw_bb, bh_bb = cv2.boundingRect(cnt)
        bbox = (bx, by, bx + bw_bb, by + bh_bb)

        # --- Hole detection for annotation ---
        all_contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is not None:
            hier0 = hier[0]
            # Find the outer contour index
            outer_idx = None
            outer_areas = []
            for i in range(len(all_contours)):
                if hier0[i][3] == -1:
                    outer_areas.append((cv2.contourArea(all_contours[i]), i))
            if outer_areas:
                outer_areas.sort(reverse=True)
                outer_idx = outer_areas[0][1]

            # Find child contour (hole)
            if outer_idx is not None:
                for i, (ni, pi, ci, pari) in enumerate(hier0):
                    if pari == outer_idx:
                        hole_cnt = all_contours[i]
                        hole_area = cv2.contourArea(hole_cnt)
                        if hole_area > 10:  # skip noise
                            M = cv2.moments(hole_cnt)
                            if M["m00"] > 0:
                                hcx = int(M["m10"] / M["m00"])
                                hcy = int(M["m01"] / M["m00"])
                                hole_r_px = int(math.sqrt(hole_area / math.pi))

                                # Draw hole circle and cross-hair
                                cv2.circle(img, (hcx, hcy), hole_r_px, _YELLOW, 1, cv2.LINE_AA)
                                cv2.drawMarker(img, (hcx, hcy), _RED,
                                               markerType=cv2.MARKER_CROSS,
                                               markerSize=15, thickness=1)

                                # Hole diameter text
                                meas = result.get("measurements_mm", {})
                                hd = meas.get("hole_diameter")
                                if hd is not None:
                                    cv2.putText(img, f"D={hd:.2f}",
                                                (hcx + hole_r_px + 4, hcy - 4),
                                                _FONT, _FONT_S, _YELLOW, _THICK, cv2.LINE_AA)
                        break  # only one hole expected

        # --- Dimension labels (length and width) ---
        meas = result.get("measurements_mm", {})
        length_mm = meas.get("length")
        width_mm = meas.get("width")

        (rcx, rcy) = rect[0]
        rw, rh = rect[1]
        long_side = max(rw, rh)
        short_side = min(rw, rh)

        if length_mm is not None:
            # Place length label above the part
            label_y = max(20, by - 10)
            mid_x = bx + bw_bb // 2
            cv2.putText(img, f"L={length_mm:.2f} mm",
                        (mid_x - 50, label_y),
                        _FONT, _FONT_S, _WHITE, _THICK, cv2.LINE_AA)
            # Draw top dimension line
            cv2.line(img, (bx, label_y + 5), (bx + bw_bb, label_y + 5), _WHITE, 1)

        if width_mm is not None:
            # Place width label to the right of the part
            label_x = min(w - 100, bx + bw_bb + 8)
            mid_y = by + bh_bb // 2
            cv2.putText(img, f"W={width_mm:.2f} mm",
                        (label_x, mid_y),
                        _FONT, _FONT_S, _WHITE, _THICK, cv2.LINE_AA)
            # Draw right dimension line
            cv2.line(img, (label_x - 3, by), (label_x - 3, by + bh_bb), _WHITE, 1)

    # --- ACCEPT / REJECT banner ---------------------------------------------
    status = result.get("inspection_status", "REJECT")
    reason = result.get("reject_reason")
    part_present = result.get("part_present", False)

    if status == "ACCEPT":
        banner_color = (0, 160, 0)   # dark green
        banner_text = "ACCEPT"
    else:
        banner_color = (0, 0, 180)   # dark red
        banner_text = "REJECT"

    # Banner rectangle at top of image
    cv2.rectangle(img, (0, 0), (w, 28), banner_color, -1)
    cv2.putText(img, banner_text, (8, 20),
                _FONT, 0.7, _WHITE, 2, cv2.LINE_AA)

    # Reason / details
    if reason:
        cv2.putText(img, reason, (110, 20),
                    _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)

    # --- Per-check results line (bottom of image) ---------------------------
    res = result.get("result", {})
    checks = []
    for k in ["length", "width", "hole_diameter", "hole_x_position", "hole_y_position"]:
        v = res.get(k, "?")
        short_k = k.replace("_position", "").replace("hole_", "h_")
        color_code = "P" if v == "PASS" else ("F" if v == "FAIL" else "?")
        checks.append(f"{short_k}:{color_code}")
    check_str = "  ".join(checks)
    cv2.rectangle(img, (0, h - 22), (w, h), (40, 40, 40), -1)
    cv2.putText(img, check_str, (8, h - 6),
                _FONT, 0.4, _WHITE, 1, cv2.LINE_AA)

    out_path = out_dir / f"{image_path.stem}_annotated.png"
    cv2.imwrite(str(out_path), img)
    print(f"Annotated image saved to {out_path}")


def process_image(image_path: Path, out_dir: Path, annotate: bool) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Failed to read image: {image_path}", file=sys.stderr)
        return
    result = run_full_pipeline(img)
    json_path = out_dir / f"{image_path.stem}.json"
    export_json(result, json_path)
    print(f"Processed {image_path.name} -> {json_path.name}")
    if annotate:
        annotate_and_save(image_path, result, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stamped-part inspection pipeline.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a PNG image or a directory containing PNG images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory where JSON (and optional PNG) results are written.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Generate an annotated PNG alongside each JSON output.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input.is_file():
        process_image(args.input, out_dir, args.annotate)
    elif args.input.is_dir():
        pngs = sorted(args.input.glob("*.png"))
        if not pngs:
            print(f"[INFO] No PNG files found in {args.input}", file=sys.stderr)
            sys.exit(0)
        for p in pngs:
            process_image(p, out_dir, args.annotate)
    else:
        parser.error(f"Input must be a file or directory: {args.input}")


if __name__ == "__main__":
    main()
