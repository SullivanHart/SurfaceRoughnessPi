"""
Verify stereo calibration quality from saved calibration images and calibration.npz.

Reports:
  - calibration metadata and baseline
  - per-frame checkerboard detection status
  - epipolar error after undistort/rectify
  - rectified debug pair with horizontal guide lines
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


def parse_checkerboard(value):
    cols, rows = value.lower().split("x", 1)
    return int(cols), int(rows)


def detect(gray, checkerboard):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, checkerboard, flags)
    return ok, corners


def draw_rectified_pair(left, right, out):
    left_bgr = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR) if left.ndim == 2 else left.copy()
    right_bgr = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR) if right.ndim == 2 else right.copy()
    pair = np.concatenate([left_bgr, right_bgr], axis=1)
    h, w = left_bgr.shape[:2]
    for row in range(0, h, 50):
        cv2.line(pair, (0, row), (2 * w, row), (0, 255, 0), 1)
    cv2.imwrite(str(out), pair)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", default="config/calibration.npz")
    parser.add_argument("--images", default="data/calibration")
    parser.add_argument("--checkerboard", default=None, help="Inner corners, e.g. 10x6")
    parser.add_argument("--out-dir", default="data/calibration/verify")
    args = parser.parse_args()

    cal = np.load(project_path(args.calib))
    mtx_l, dist_l = cal["mtxL"], cal["distL"]
    mtx_r, dist_r = cal["mtxR"], cal["distR"]
    R, T = cal["R"], cal["T"]
    rms = float(cal["rms"]) if "rms" in cal.files else float("nan")
    if args.checkerboard:
        checkerboard = parse_checkerboard(args.checkerboard)
    elif "checkerboard" in cal.files:
        checkerboard = tuple(int(v) for v in cal["checkerboard"])
    else:
        raise ValueError("Pass --checkerboard, e.g. --checkerboard 10x6")

    image_root = project_path(args.images)
    left_files = sorted((image_root / "left").glob("*.png"))
    right_files = sorted((image_root / "right").glob("*.png"))
    pairs = list(zip(left_files, right_files))
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Calibration RMS stored: {rms:.4f} px")
    print(f"Checkerboard inner corners: {checkerboard}")
    if "square_size_mm" in cal.files:
        print(f"Square size: {float(cal['square_size_mm']):.4f} mm")
    print(f"T: {T.ravel()}")
    print(f"Baseline norm: {float(np.linalg.norm(T)):.3f} calibration units")
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

    epipolar_errors = []
    detected = 0
    for idx, (lf, rf) in enumerate(pairs):
        left = cv2.imread(str(lf), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rf), cv2.IMREAD_GRAYSCALE)
        ok_l, corners_l = detect(left, checkerboard)
        ok_r, corners_r = detect(right, checkerboard)
        print(f"{idx:02d} {lf.name} {rf.name}: detect L={ok_l} R={ok_r}", end="")
        if ok_l and ok_r:
            detected += 1
            und_l = cv2.undistortPoints(corners_l, mtx_l, dist_l, R=R1, P=P1).reshape(-1, 2)
            und_r = cv2.undistortPoints(corners_r, mtx_r, dist_r, R=R2, P=P2).reshape(-1, 2)
            dy = np.abs(und_l[:, 1] - und_r[:, 1])
            err = float(np.mean(dy))
            epipolar_errors.append(err)
            print(f"  mean_rectified_y_error={err:.3f}px")
        else:
            print()

        if idx in (0, len(pairs) // 2, len(pairs) - 1):
            left_r = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
            right_r = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
            draw_rectified_pair(left_r, right_r, out_dir / f"rectified_pair_{idx:02d}.png")

    print(f"\nDetected pairs: {detected}/{len(pairs)}")
    if epipolar_errors:
        print(
            "Rectified Y error px: "
            f"min={min(epipolar_errors):.3f} "
            f"mean={np.mean(epipolar_errors):.3f} "
            f"max={max(epipolar_errors):.3f}"
        )
    print(f"Debug rectified pairs: {out_dir}")


if __name__ == "__main__":
    main()
