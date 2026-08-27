"""
Draw detected checkerboard corner order for a stereo image pair.

Use this to verify that corner index 0, 1, 2, ... refer to the same physical
checkerboard points in both cameras and across calibration frames.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_checkerboard(value):
    cols, rows = value.lower().split("x", 1)
    return int(cols), int(rows)


def detect(gray, checkerboard):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    return cv2.findChessboardCornersSB(gray, checkerboard, flags)


def draw_order(image, corners, checkerboard):
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cols, rows = checkerboard
    pts = corners.reshape(-1, 2)

    for idx, (x, y) in enumerate(pts):
        color = (0, 255, 0)
        if idx == 0:
            color = (0, 0, 255)
        elif idx == cols - 1:
            color = (255, 0, 0)
        elif idx == len(pts) - 1:
            color = (0, 255, 255)
        cv2.circle(out, (round(x), round(y)), 7, color, -1)

        if idx in (0, 1, cols - 1, cols, len(pts) - cols, len(pts) - 1) or idx % cols == 0:
            cv2.putText(
                out,
                str(idx),
                (round(x) + 8, round(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

    # Draw row and column direction arrows.
    if len(pts) >= cols + 1:
        cv2.arrowedLine(out, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[cols - 1]).astype(int)), (255, 0, 0), 3)
        cv2.arrowedLine(out, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[-cols]).astype(int)), (0, 255, 255), 3)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="calib/debug/gray_left.png")
    parser.add_argument("--right", default="calib/debug/gray_right.png")
    parser.add_argument("--calib-dir", default=None, help="Process all pairs in calib-dir/left and calib-dir/right")
    parser.add_argument("--checkerboard", required=True, help="Inner corners, e.g. 10x5")
    parser.add_argument("--out", default="calib/debug/corner_order.png")
    args = parser.parse_args()

    checkerboard = parse_checkerboard(args.checkerboard)

    if args.calib_dir:
        root = Path(args.calib_dir)
        left_files = sorted((root / "left").glob("*.png"))
        right_files = sorted((root / "right").glob("*.png"))
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        ok_count = 0
        for idx, (left_path, right_path) in enumerate(zip(left_files, right_files)):
            left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
            ok_l, corners_l = detect(left, checkerboard)
            ok_r, corners_r = detect(right, checkerboard)
            print(f"{idx:02d} {left_path.name} {right_path.name}: left={ok_l} right={ok_r}")
            if not (ok_l and ok_r):
                continue
            ok_count += 1
            vis_l = draw_order(left, corners_l, checkerboard)
            vis_r = draw_order(right, corners_r, checkerboard)
            pair = np.concatenate([vis_l, vis_r], axis=1)
            cv2.imwrite(str(out_dir / f"corner_order_{idx:02d}.png"), pair)
        print(f"Wrote {ok_count} overlays to {out_dir}")
    else:
        left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        if left is None:
            raise FileNotFoundError(args.left)
        if right is None:
            raise FileNotFoundError(args.right)

        ok_l, corners_l = detect(left, checkerboard)
        ok_r, corners_r = detect(right, checkerboard)
        print(f"detect left={ok_l} right={ok_r}")
        if not (ok_l and ok_r):
            return

        vis_l = draw_order(left, corners_l, checkerboard)
        vis_r = draw_order(right, corners_r, checkerboard)
        pair = np.concatenate([vis_l, vis_r], axis=1)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), pair)
        print(f"Wrote {out}")
    print("Red = corner 0, blue = first row end, yellow = last corner.")


if __name__ == "__main__":
    main()
