"""
Generate a printable ChArUco calibration board.

Default is Letter paper with a black page background, white quiet border, and
an 8 x 5 square ChArUco board using 10 mm squares.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


MM_TO_INCH = 1.0 / 25.4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares-x", type=int, default=8)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=10.0)
    parser.add_argument("--marker-mm", type=float, default=7.0)
    parser.add_argument("--border-mm", type=float, default=10.0)
    parser.add_argument("--paper", choices=("letter", "a4"), default="letter")
    parser.add_argument("--offset-y-mm", type=float, default=25.0)
    parser.add_argument("--bottom-margin-mm", type=float, default=None,
                        help="Place target this far above the bottom of the page; overrides offset-y-mm")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", default="imgs/charuco_8x5_10mm_letter.png")
    args = parser.parse_args()

    if args.paper == "letter":
        page_w_mm, page_h_mm = 215.9, 279.4
    else:
        page_w_mm, page_h_mm = 210.0, 297.0

    page_w_px = round(page_w_mm * MM_TO_INCH * args.dpi)
    page_h_px = round(page_h_mm * MM_TO_INCH * args.dpi)
    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    target_w_mm = board_w_mm + 2 * args.border_mm
    target_h_mm = board_h_mm + 2 * args.border_mm
    target_w_px = round(target_w_mm * MM_TO_INCH * args.dpi)
    target_h_px = round(target_h_mm * MM_TO_INCH * args.dpi)
    board_w_px = round(board_w_mm * MM_TO_INCH * args.dpi)
    board_h_px = round(board_h_mm * MM_TO_INCH * args.dpi)
    border_px = round(args.border_mm * MM_TO_INCH * args.dpi)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_mm,
        args.marker_mm,
        dictionary,
    )
    board_img = board.generateImage((board_w_px, board_h_px), marginSize=0)

    page = np.zeros((page_h_px, page_w_px), dtype=np.uint8)
    target = np.full((target_h_px, target_w_px), 255, dtype=np.uint8)
    target[border_px:border_px + board_h_px, border_px:border_px + board_w_px] = board_img

    x = (page_w_px - target_w_px) // 2
    if args.bottom_margin_mm is None:
        y = (page_h_px - target_h_px) // 2 + round(args.offset_y_mm * MM_TO_INCH * args.dpi)
    else:
        y = page_h_px - target_h_px - round(args.bottom_margin_mm * MM_TO_INCH * args.dpi)
    if x < 0 or y < 0 or x + target_w_px > page_w_px or y + target_h_px > page_h_px:
        raise ValueError("Board does not fit on the page with the requested border/offset")
    page[y:y + target_h_px, x:x + target_w_px] = target

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), page)

    print(f"Wrote {out}")
    print(f"Paper: {args.paper} at {args.dpi} dpi")
    print(f"ChArUco squares: {args.squares_x} x {args.squares_y}")
    print(f"Square size: {args.square_mm} mm")
    print(f"Marker size: {args.marker_mm} mm")
    print("Calibration command:")
    print(
        "  venv/bin/python calibrate_charuco.py "
        f"--squares-x {args.squares_x} --squares-y {args.squares_y} "
        f"--square-size-mm {args.square_mm:g} --marker-size-mm {args.marker_mm:g}"
    )


if __name__ == "__main__":
    main()
