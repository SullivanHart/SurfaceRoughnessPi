"""
Verify a ChArUco stereo calibration from saved calib/ images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def draw_rectified_pair(left, right, out):
    left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR) if left.ndim == 2 else left.copy()
    right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR) if right.ndim == 2 else right.copy()
    pair = np.concatenate([left_bgr, right_bgr], axis=1)
    h, w = left_bgr.shape[:2]
    for row in range(0, h, 50):
        cv2.line(pair, (0, row), (2 * w, row), (0, 255, 0), 1)
    cv2.imwrite(str(out), pair)


def detect(detector, gray):
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if corners is None or ids is None:
        return None, None
    return corners.astype(np.float32), ids.reshape(-1).astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", default="config/calibration.npz")
    parser.add_argument("--images", default="data/calibration")
    parser.add_argument("--squares-x", type=int, default=None)
    parser.add_argument("--squares-y", type=int, default=None)
    parser.add_argument("--square-size-mm", type=float, default=None)
    parser.add_argument("--marker-size-mm", type=float, default=None)
    parser.add_argument("--out-dir", default="data/calibration/verify_charuco")
    args = parser.parse_args()

    cal = np.load(project_path(args.calib))
    squares = cal["charuco_squares"] if "charuco_squares" in cal.files else None
    squares_x = args.squares_x or int(squares[0])
    squares_y = args.squares_y or int(squares[1])
    square_size = args.square_size_mm or float(cal["square_size_mm"])
    marker_size = args.marker_size_mm or float(cal["marker_size_mm"])

    mtx_l, dist_l = cal["mtxL"], cal["distL"]
    mtx_r, dist_r = cal["mtxR"], cal["distR"]
    R, T = cal["R"], cal["T"]
    rms = float(cal["rms"]) if "rms" in cal.files else float("nan")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    root = project_path(args.images)
    left_files = sorted((root / "left").glob("*.png"))
    right_files = sorted((root / "right").glob("*.png"))
    pairs = list(zip(left_files, right_files))
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stored stereo RMS: {rms:.4f} px")
    print(f"ChArUco board: {squares_x} x {squares_y}, square={square_size} mm, marker={marker_size} mm")
    print(f"T: {T.ravel()}  baseline={float(np.linalg.norm(T)):.3f} mm")
    print(f"Image pairs: {len(pairs)}")
    if not pairs:
        return

    sample = cv2.imread(str(pairs[0][0]), cv2.IMREAD_GRAYSCALE)
    image_size = sample.shape[::-1]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, flags=0, alpha=-1
    )
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

    y_errors = []
    common_counts = []
    for idx, (lf, rf) in enumerate(pairs):
        left = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        corners_l, ids_l = detect(detector, left)
        corners_r, ids_r = detect(detector, right)
        if ids_l is None or ids_r is None:
            print(f"{idx:02d}: detect failed")
            continue
        common, idx_l, idx_r = np.intersect1d(ids_l, ids_r, return_indices=True)
        common_counts.append(len(common))
        if len(common) < 4:
            print(f"{idx:02d}: common={len(common)}")
            continue

        pts_l = corners_l[idx_l]
        pts_r = corners_r[idx_r]
        und_l = cv2.undistortPoints(pts_l, mtx_l, dist_l, R=R1, P=P1).reshape(-1, 2)
        und_r = cv2.undistortPoints(pts_r, mtx_r, dist_r, R=R2, P=P2).reshape(-1, 2)
        dy = np.abs(und_l[:, 1] - und_r[:, 1])
        err = float(np.mean(dy))
        y_errors.append(err)
        print(f"{idx:02d}: common={len(common):2d} mean_rectified_y_error={err:.3f}px")

        if idx in (0, len(pairs) // 2, len(pairs) - 1):
            left_r = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
            right_r = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
            draw_rectified_pair(left_r, right_r, out_dir / f"rectified_pair_{idx:02d}.png")

    if common_counts:
        print(
            f"\nCommon corners: min={min(common_counts)} "
            f"mean={np.mean(common_counts):.1f} max={max(common_counts)}"
        )
    if y_errors:
        print(
            "Rectified Y error px: "
            f"min={min(y_errors):.3f} mean={np.mean(y_errors):.3f} max={max(y_errors):.3f}"
        )
    print(f"Debug rectified pairs: {out_dir}")


if __name__ == "__main__":
    main()
