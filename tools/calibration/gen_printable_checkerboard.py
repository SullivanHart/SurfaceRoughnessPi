"""
Generate a printable checkerboard calibration target.

Default target:
  - 11 squares wide x 7 squares high
  - 10 mm square size
  - white quiet border around checkerboard
  - black page background

OpenCV inner-corner count for the default is 10 x 6.
"""

import argparse
from pathlib import Path


MM_TO_PT = 72.0 / 25.4


def mm(value):
    return value * MM_TO_PT


def rect(x, y, w, h, fill):
    return f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}" />'


def generate_svg(args):
    page_w = mm(args.page_width_mm)
    page_h = mm(args.page_height_mm)
    square = mm(args.square_mm)
    border = mm(args.border_mm)
    board_w = args.squares_wide * square
    board_h = args.squares_high * square
    target_w = board_w + 2 * border
    target_h = board_h + 2 * border
    origin_x = (page_w - target_w) / 2
    origin_y = (page_h - target_h) / 2 + mm(args.offset_y_mm)
    board_x = origin_x + border
    board_y = origin_y + border

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{args.page_width_mm}mm" '
        f'height="{args.page_height_mm}mm" viewBox="0 0 {page_w:.3f} {page_h:.3f}">',
        rect(0, 0, page_w, page_h, "black"),
        rect(origin_x, origin_y, target_w, target_h, "white"),
    ]

    for row in range(args.squares_high):
        for col in range(args.squares_wide):
            if (row + col) % 2 == 0:
                fill = "black" if args.top_left == "black" else "white"
            else:
                fill = "white" if args.top_left == "black" else "black"
            if fill == "black":
                parts.append(rect(board_x + col * square, board_y + row * square, square, square, fill))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squares-wide", type=int, default=11)
    parser.add_argument("--squares-high", type=int, default=7)
    parser.add_argument("--square-mm", type=float, default=10.0)
    parser.add_argument("--border-mm", type=float, default=10.0)
    parser.add_argument("--page-width-mm", type=float, default=210.0, help="A4 width")
    parser.add_argument("--page-height-mm", type=float, default=297.0, help="A4 height")
    parser.add_argument("--paper", choices=("a4", "letter"), default=None)
    parser.add_argument("--offset-y-mm", type=float, default=0.0,
                        help="Positive moves target down on the page")
    parser.add_argument("--top-left", choices=("black", "white"), default="black")
    parser.add_argument("--out", default="imgs/printable_checkerboard_11x7_10mm.svg")
    args = parser.parse_args()

    if args.paper == "a4":
        args.page_width_mm = 210.0
        args.page_height_mm = 297.0
    elif args.paper == "letter":
        args.page_width_mm = 215.9
        args.page_height_mm = 279.4

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_svg(args))

    inner_w = args.squares_wide - 1
    inner_h = args.squares_high - 1
    print(f"Wrote {out}")
    print(f"Squares: {args.squares_wide} x {args.squares_high}")
    print(f"OpenCV inner corners: {inner_w} x {inner_h}")
    print(f"Square size: {args.square_mm} mm")
    print(f"Use calibrate.py --checkerboard {inner_w}x{inner_h} --square-size-mm {args.square_mm:g}")


if __name__ == "__main__":
    main()
