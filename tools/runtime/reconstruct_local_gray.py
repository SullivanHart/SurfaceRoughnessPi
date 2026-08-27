"""
Reconstruct local Gray-code captures using OpenCV's reference path.

Expected capture order:
  left_00/right_00 .. inverse Gray-code pairs
  final two frames: white, black
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="data/captures/latest", help="Directory containing left/ and right/")
    parser.add_argument("--calib", default="config/calibration.npz")
    parser.add_argument("--out", default="output/pointclouds/latest.ply")
    parser.add_argument("--proj-width", type=int, default=456)
    parser.add_argument("--proj-height", type=int, default=570)
    parser.add_argument("--white-thresh", type=int, default=None)
    parser.add_argument("--black-thresh", type=int, default=None)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument(
        "--disparity-sign",
        choices=("positive", "negative", "both", "auto"),
        default="positive",
        help="Which disparity sign to keep. Use both only as a diagnostic unless it looks clean.",
    )
    parser.add_argument("--max-depth", type=float, default=1e9)
    parser.add_argument("--min-depth", type=float, default=-1e9)
    parser.add_argument("--plane-filter-mm", type=float, default=0.0,
                        help="For flat-surface tests, keep points within this distance of a robust fitted plane")
    parser.add_argument("--median-filter", type=int, default=0,
                        help="Odd kernel size for disparity outlier filtering; 0 disables")
    parser.add_argument("--max-median-diff", type=float, default=2.0,
                        help="Reject pixels whose disparity differs from local median by more than this")
    parser.add_argument("--min-component-area", type=int, default=200,
                        help="Reject connected mask components smaller than this")
    parser.add_argument(
        "--rectify-order",
        choices=("normal", "opencv-sample"),
        default="normal",
        help="normal maps left with left calibration and right with right calibration",
    )
    parser.add_argument(
        "--zero-disparity-rectify",
        action="store_true",
        help="Use cv2.CALIB_ZERO_DISPARITY during stereoRectify so the working surface is less likely to cross disparity 0",
    )
    parser.add_argument("--no-debug", action="store_true", help="Skip writing debug images")
    parser.add_argument("--no-bit-debug", action="store_true", help="Skip Gray-code bit contrast diagnostics")
    return parser.parse_args()


def load_sequence(directory, prefix, count):
    images = []
    for idx in range(count):
        path = Path(directory) / f"{prefix}_{idx:02d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return images


def remap_sequence(images, map_x, map_y):
    return [
        cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        for image in images
    ]


def save_mask_overlay(base, mask, out_path):
    if base.ndim == 2:
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        overlay = base.copy()
    overlay[mask] = (0.5 * overlay[mask] + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(out_path), overlay)


def bit_contrast_debug(patterns, lit_mask, out_dir, prefix):
    pair_count = len(patterns) // 2
    if pair_count == 0:
        return

    contrasts = []
    print(f"{prefix} Gray-code bit contrast p10/median by pattern pair:")
    for idx in range(pair_count):
        a = patterns[2 * idx].astype(np.int16)
        b = patterns[2 * idx + 1].astype(np.int16)
        diff = np.abs(a - b).astype(np.uint8)
        contrasts.append(diff)
        samples = diff[lit_mask]
        if samples.size:
            p10, med = np.percentile(samples, [10, 50])
            print(f"  {idx:02d}: p10={p10:5.1f} median={med:5.1f}")

    stack = np.stack(contrasts, axis=0)
    min_contrast = stack.min(axis=0)
    mean_contrast = stack.mean(axis=0)
    cv2.imwrite(str(out_dir / f"gray_bit_contrast_min_{prefix.lower()}.png"), min_contrast)
    cv2.imwrite(str(out_dir / f"gray_bit_contrast_mean_{prefix.lower()}.png"), np.clip(mean_contrast, 0, 255).astype(np.uint8))


def save_debug(disparity, raw_mask, filtered_mask, positive, negative, contrast_l, contrast_r, white_l, white_r, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = disparity[filtered_mask]
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        scaled = (disparity - lo) / max(hi - lo, 1e-6) * 255
        vis = np.where(filtered_mask, scaled, 0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "gray_disparity.png"), vis)
    cv2.imwrite(str(out_dir / "gray_mask.png"), filtered_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "gray_mask_raw.png"), raw_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "gray_mask_removed_by_filter.png"), (raw_mask & ~filtered_mask).astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "gray_mask_positive.png"), positive.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "gray_mask_negative.png"), negative.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "white_black_contrast_left.png"), np.clip(contrast_l, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "white_black_contrast_right.png"), np.clip(contrast_r, 0, 255).astype(np.uint8))
    save_mask_overlay(white_l, raw_mask, out_dir / "gray_mask_raw_overlay_left.png")
    save_mask_overlay(white_r, raw_mask, out_dir / "gray_mask_raw_overlay_right.png")


def save_rectified_pair(left, right, out_dir):
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    pair = np.concatenate([left, right], axis=1)
    h, w = left.shape[:2]
    for row in range(0, h, 50):
        cv2.line(pair, (0, row), (2 * w, row), (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / "rectified_white_pair.png"), pair)


def filter_mask(mask, disparity, median_filter, max_median_diff, min_component_area):
    filtered = mask.copy()

    if median_filter and median_filter >= 3:
        if median_filter % 2 == 0:
            median_filter += 1
        safe = np.where(filtered, disparity, 0).astype(np.float32)
        local_median = cv2.medianBlur(safe, median_filter)
        filtered &= np.abs(disparity - local_median) <= max_median_diff

    if min_component_area > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered.astype(np.uint8), 8)
        keep_labels = np.zeros(count, dtype=bool)
        keep_labels[0] = False
        for label in range(1, count):
            keep_labels[label] = stats[label, cv2.CC_STAT_AREA] >= min_component_area
        filtered = keep_labels[labels]

    return filtered


def write_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            gray = int(color)
            f.write(f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} {gray} {gray} {gray}\n")


def fit_plane_svd(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    distances = centered @ normal
    return centroid, normal, distances


def robust_plane_filter(points, colors, threshold_mm, iterations=3):
    if threshold_mm <= 0 or len(points) < 100:
        return points, colors

    keep = np.ones(len(points), dtype=bool)
    for _ in range(iterations):
        if keep.sum() < 100:
            break
        _, _, distances = fit_plane_svd(points[keep])
        candidate_indices = np.flatnonzero(keep)
        residual_keep = np.abs(distances) <= threshold_mm
        new_keep = np.zeros_like(keep)
        new_keep[candidate_indices[residual_keep]] = True
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    removed = len(points) - int(keep.sum())
    print(f"Plane filter: kept {int(keep.sum())}/{len(points)} points, removed {removed}")
    return points[keep], colors[keep]


def main():
    args = parse_args()
    if not hasattr(cv2, "structured_light"):
        raise RuntimeError("cv2.structured_light is missing. Install opencv-contrib-python-headless.")

    graycode = cv2.structured_light.GrayCodePattern.create(args.proj_width, args.proj_height)
    if args.white_thresh is not None:
        graycode.setWhiteThreshold(args.white_thresh)
    if args.black_thresh is not None:
        graycode.setBlackThreshold(args.black_thresh)

    pattern_count = int(graycode.getNumberOfPatternImages())
    total_count = pattern_count + 2
    caps = project_path(args.caps)
    out_path = project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {total_count} local captures per camera...")
    left = load_sequence(caps / "left", "left", total_count)
    right = load_sequence(caps / "right", "right", total_count)

    cal = np.load(project_path(args.calib))
    mtx_l, dist_l = cal["mtxL"], cal["distL"]
    mtx_r, dist_r = cal["mtxR"], cal["distR"]
    R, T = cal["R"], cal["T"]

    image_size = (left[0].shape[1], left[0].shape[0])
    print(f"Image size: {image_size[0]}x{image_size[1]}")

    rectify_flags = cv2.CALIB_ZERO_DISPARITY if args.zero_disparity_rectify else 0
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, flags=rectify_flags, alpha=-1
    )
    print(
        "stereoRectify flags:",
        "CALIB_ZERO_DISPARITY" if args.zero_disparity_rectify else "0",
    )
    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

    if args.rectify_order == "normal":
        left_map = (map1x, map1y)
        right_map = (map2x, map2y)
        print("Rectifying captures using normal left/right calibration order...")
    else:
        left_map = (map2x, map2y)
        right_map = (map1x, map1y)
        print("Rectifying captures using OpenCV sample swapped map order...")

    captured_pattern = [
        remap_sequence(left[:pattern_count], left_map[0], left_map[1]),
        remap_sequence(right[:pattern_count], right_map[0], right_map[1]),
    ]
    white_images = [
        cv2.remap(left[pattern_count], left_map[0], left_map[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(right[pattern_count], right_map[0], right_map[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
    ]
    black_images = [
        cv2.remap(left[pattern_count + 1], left_map[0], left_map[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(right[pattern_count + 1], right_map[0], right_map[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
    ]

    debug_dir = Path("debug_gray_local")
    if not args.no_debug:
        debug_dir.mkdir(exist_ok=True)
        save_rectified_pair(white_images[0], white_images[1], debug_dir)

    contrast_l = white_images[0].astype(np.int16) - black_images[0].astype(np.int16)
    contrast_r = white_images[1].astype(np.int16) - black_images[1].astype(np.int16)
    print(
        "White-black contrast percentiles L:",
        np.percentile(contrast_l, [1, 5, 50, 95, 99]),
    )
    print(
        "White-black contrast percentiles R:",
        np.percentile(contrast_r, [1, 5, 50, 95, 99]),
    )
    if not args.no_bit_debug:
        debug_dir.mkdir(exist_ok=True)
        bit_contrast_debug(
            captured_pattern[0],
            contrast_l > max(10, args.white_thresh or 0),
            debug_dir,
            "Left",
        )
        bit_contrast_debug(
            captured_pattern[1],
            contrast_r > max(10, args.white_thresh or 0),
            debug_dir,
            "Right",
        )

    print("Decoding Gray-code disparity...")
    decoded, disparity = graycode.decode(
        captured_pattern,
        None,
        black_images,
        white_images,
        cv2.structured_light.DECODE_3D_UNDERWORLD,
    )
    if not decoded:
        raise RuntimeError("GrayCodePattern.decode failed")

    disparity = disparity.astype(np.float32)
    print(f"Disparity min/max: {float(np.nanmin(disparity)):.3f} / {float(np.nanmax(disparity)):.3f}")

    finite = np.isfinite(disparity)
    nonzero = finite & (disparity != 0)
    positive = finite & (disparity >= args.min_disparity)
    negative = finite & (disparity <= -args.min_disparity)
    print(f"Nonzero disparity: {int(nonzero.sum())} / {nonzero.size} ({nonzero.mean() * 100:.1f}%)")
    print(f"Positive disparity: {int(positive.sum())} / {positive.size} ({positive.mean() * 100:.1f}%)")
    print(f"Negative disparity: {int(negative.sum())} / {negative.size} ({negative.mean() * 100:.1f}%)")

    if args.disparity_sign == "positive":
        mask = positive
        print("Using positive disparity mask.")
    elif args.disparity_sign == "negative":
        mask = negative
        print("Using negative disparity mask.")
    elif args.disparity_sign == "both":
        mask = positive | negative
        print("Using both positive and negative disparity masks.")
    else:
        mask = positive
        print("Using auto disparity sign selection.")
    if args.disparity_sign == "auto" and negative.sum() > positive.sum() * 2:
        print("Using negative disparity mask because it dominates positive disparity.")
        mask = negative
    print(f"Valid disparity: {int(mask.sum())} / {mask.size} ({mask.mean() * 100:.1f}%)")
    raw_mask = mask.copy()

    filtered_mask = filter_mask(
        mask,
        disparity,
        args.median_filter,
        args.max_median_diff,
        args.min_component_area,
    )
    removed = int(mask.sum() - filtered_mask.sum())
    mask = filtered_mask
    print(
        f"Filtered disparity: {int(mask.sum())} / {mask.size} "
        f"({mask.mean() * 100:.1f}%), removed {removed}"
    )

    print("Reprojecting to 3D...")
    pointcloud = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    points = pointcloud[mask]
    colors = white_images[0][mask]

    z = points[:, 2]
    finite_before_depth = np.isfinite(points).all(axis=1)
    if finite_before_depth.any():
        print(
            "Raw depth percentiles before filter:",
            np.percentile(points[finite_before_depth, 2], [1, 5, 50, 95, 99]),
        )
    keep = np.isfinite(points).all(axis=1) & (z >= args.min_depth) & (z <= args.max_depth)
    points = points[keep]
    colors = colors[keep]
    print(f"Depth filter: kept {len(points)} points in [{args.min_depth:g}, {args.max_depth:g}] mm")

    points, colors = robust_plane_filter(points, colors, args.plane_filter_mm)

    if len(points):
        print("Depth percentiles:", np.percentile(points[:, 2], [1, 5, 50, 95, 99]))
    print(f"Writing {len(points)} points to {out_path}")

    write_ply(str(out_path), points, colors)
    if not args.no_debug:
        save_debug(
            disparity,
            raw_mask,
            mask,
            positive,
            negative,
            contrast_l,
            contrast_r,
            white_images[0],
            white_images[1],
            debug_dir,
        )
        print("Debug images: debug_gray_local/")


if __name__ == "__main__":
    main()
