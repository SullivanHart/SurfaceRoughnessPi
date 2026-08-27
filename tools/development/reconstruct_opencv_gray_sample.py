"""
Reconstruct OpenCV's structured-light Gray-code sample dataset.

By default this decodes projector IDs in both raw camera views and triangulates
matched IDs with the supplied stereo calibration. This is a good 3D smoke test
because it avoids depending on one particular rectification/disparity convention.

Usage:
  python download_opencv_structured_light_sample.py
  venv/bin/python reconstruct_opencv_gray_sample.py \
      --data sample_data/opencv_structured_light \
      --out sample_data/opencv_structured_light/cloud.ply
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sample_data/opencv_structured_light")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--out", default="sample_data/opencv_structured_light/cloud.ply")
    parser.add_argument("--proj-width", type=int, default=1280)
    parser.add_argument("--proj-height", type=int, default=800)
    parser.add_argument("--white-thresh", type=float, default=5.0)
    parser.add_argument("--black-thresh", type=float, default=5.0)
    parser.add_argument("--max-depth", type=float, default=1e9)
    parser.add_argument(
        "--decoder",
        choices=("triangulate", "opencv-disparity", "manual-disparity"),
        default="triangulate",
        help="3D from raw projector-ID triangulation, or disparity debug modes",
    )
    return parser.parse_args()


def read_mat(fs, key):
    mat = fs.getNode(key).mat()
    if mat is None:
        raise KeyError(f"Missing calibration key: {key}")
    return mat


def load_calibration(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)
    cal = {
        "mtx_l": read_mat(fs, "cam1_intrinsics"),
        "dist_l": read_mat(fs, "cam1_distorsion"),
        "mtx_r": read_mat(fs, "cam2_intrinsics"),
        "dist_r": read_mat(fs, "cam2_distorsion"),
        "R": read_mat(fs, "R"),
        "T": read_mat(fs, "T"),
    }
    fs.release()
    return cal


def load_patterns(data_dir, prefix):
    images = []
    for i in range(1, 45):
        path = data_dir / f"{prefix}_im{i}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        images.append(img)
    return images


def gray_to_binary(gray):
    binary = gray.copy()
    shift = 1
    while True:
        shifted = binary >> shift
        if not np.any(shifted):
            break
        binary ^= shifted
        shift <<= 1
    return binary


def decode_axis(images, start_pair, bits, thresh):
    gray = np.zeros(images[0].shape, dtype=np.uint32)
    valid = np.ones(images[0].shape, dtype=bool)

    for bit_idx in range(bits):
        normal = images[start_pair + bit_idx * 2].astype(np.int16)
        inverse = images[start_pair + bit_idx * 2 + 1].astype(np.int16)
        diff = normal - inverse

        valid &= np.abs(diff) >= thresh
        bit = diff > 0
        shift = bits - bit_idx - 1
        gray |= bit.astype(np.uint32) << shift

    return gray_to_binary(gray).astype(np.int32), valid


def decode_projector_codes(images, proj_width, proj_height, white_thresh, black_thresh):
    col_bits = int(np.ceil(np.log2(proj_width)))
    row_bits = int(np.ceil(np.log2(proj_height)))
    expected = 2 * (col_bits + row_bits) + 2
    if len(images) != expected:
        raise ValueError(f"Expected {expected} images for {proj_width}x{proj_height}, got {len(images)}")

    white = images[-2].astype(np.int16)
    black = images[-1].astype(np.int16)
    lit = (white - black) >= black_thresh

    proj_x, valid_x = decode_axis(images, 0, col_bits, white_thresh)
    proj_y, valid_y = decode_axis(images, col_bits * 2, row_bits, white_thresh)
    valid = lit & valid_x & valid_y
    valid &= (proj_x >= 0) & (proj_x < proj_width) & (proj_y >= 0) & (proj_y < proj_height)

    proj_x[~valid] = -1
    proj_y[~valid] = -1
    return proj_x, proj_y, valid


def rectify_images(images_l, images_r, cal):
    h, w = images_l[0].shape
    image_size = (w, h)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cal["mtx_l"], cal["dist_l"], cal["mtx_r"], cal["dist_r"], image_size, cal["R"], cal["T"],
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        cal["mtx_l"], cal["dist_l"], R1, P1, image_size, cv2.CV_32FC1
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        cal["mtx_r"], cal["dist_r"], R2, P2, image_size, cv2.CV_32FC1
    )
    rect_l = [cv2.remap(img, map1_l, map2_l, cv2.INTER_NEAREST) for img in images_l]
    rect_r = [cv2.remap(img, map1_r, map2_r, cv2.INTER_NEAREST) for img in images_r]
    return rect_l, rect_r, Q


def disparity_from_opencv_graycode(rect_l, rect_r, proj_width, proj_height, white_thresh, black_thresh):
    if not hasattr(cv2, "structured_light"):
        raise RuntimeError("cv2.structured_light is unavailable; install opencv-contrib-python-headless")

    graycode = cv2.structured_light.GrayCodePattern.create(proj_width, proj_height)
    graycode.setWhiteThreshold(int(round(white_thresh)))
    graycode.setBlackThreshold(int(round(black_thresh)))

    n_patterns = int(graycode.getNumberOfPatternImages())
    if len(rect_l) < n_patterns + 2 or len(rect_r) < n_patterns + 2:
        raise ValueError(f"Need {n_patterns + 2} images per camera, got {len(rect_l)} and {len(rect_r)}")

    captures = [rect_l[:n_patterns], rect_r[:n_patterns]]
    white_images = [rect_l[n_patterns], rect_r[n_patterns]]
    black_images = [rect_l[n_patterns + 1], rect_r[n_patterns + 1]]

    decoded, disparity = graycode.decode(
        captures,
        None,
        black_images,
        white_images,
        cv2.structured_light.DECODE_3D_UNDERWORLD,
    )
    if not decoded:
        raise RuntimeError("OpenCV GrayCodePattern.decode returned false")

    disparity = disparity.astype(np.float32)
    disparity[disparity == 0] = np.nan
    return disparity


def disparity_from_projector_codes(px_l, py_l, valid_l, px_r, py_r, valid_r, proj_width):
    h, w = px_l.shape
    x_coords = np.arange(w, dtype=np.float32)
    disparity = np.full((h, w), np.nan, dtype=np.float32)

    key_l = py_l.astype(np.int64) * proj_width + px_l.astype(np.int64)
    key_r = py_r.astype(np.int64) * proj_width + px_r.astype(np.int64)

    for row in range(h):
        right_valid = valid_r[row]
        if np.count_nonzero(right_valid) < 2:
            continue

        rk = key_r[row, right_valid]
        rx = x_coords[right_valid]
        order = np.argsort(rk, kind="mergesort")
        rk = rk[order]
        rx = rx[order]

        unique_keys, first = np.unique(rk, return_index=True)
        unique_x = rx[first]

        left_valid = valid_l[row]
        lk = key_l[row, left_valid]
        pos = np.searchsorted(unique_keys, lk)
        ok = (pos < unique_keys.size) & (unique_keys[pos.clip(max=max(unique_keys.size - 1, 0))] == lk)
        if not np.any(ok):
            continue

        left_x = x_coords[left_valid]
        disp_values = left_x[ok] - unique_x[pos[ok]]
        disp_row = disparity[row]
        disp_row[np.where(left_valid)[0][ok]] = disp_values

    finite = np.isfinite(disparity)
    if np.any(finite):
        median = np.nanmedian(disparity)
        if median < 0:
            keep = finite & (disparity < 0)
        else:
            keep = finite & (disparity > 0)
        disparity[~keep] = np.nan

    return disparity


def grouped_points_by_key(proj_x, proj_y, valid, proj_width):
    ys, xs = np.where(valid)
    keys = proj_y[ys, xs].astype(np.int64) * proj_width + proj_x[ys, xs].astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    xs = xs[order].astype(np.float64)
    ys = ys[order].astype(np.float64)

    unique, first, counts = np.unique(keys, return_index=True, return_counts=True)
    sum_x = np.add.reduceat(xs, first)
    sum_y = np.add.reduceat(ys, first)
    points = np.column_stack([sum_x / counts, sum_y / counts]).astype(np.float32)
    return unique, points


def triangulate_projector_matches(px_l, py_l, valid_l, px_r, py_r, valid_r, cal, proj_width, color_image):
    keys_l, pts_l = grouped_points_by_key(px_l, py_l, valid_l, proj_width)
    keys_r, pts_r = grouped_points_by_key(px_r, py_r, valid_r, proj_width)

    common, idx_l, idx_r = np.intersect1d(keys_l, keys_r, assume_unique=True, return_indices=True)
    if common.size == 0:
        return np.empty((0, 3), np.float32), np.empty((0,), np.uint8)

    matched_l = pts_l[idx_l]
    matched_r = pts_r[idx_r]

    und_l = cv2.undistortPoints(matched_l.reshape(-1, 1, 2), cal["mtx_l"], cal["dist_l"]).reshape(-1, 2)
    und_r = cv2.undistortPoints(matched_r.reshape(-1, 1, 2), cal["mtx_r"], cal["dist_r"]).reshape(-1, 2)

    P_l = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    P_r = np.hstack([cal["R"], cal["T"]]).astype(np.float64)
    points_h = cv2.triangulatePoints(P_l, P_r, und_l.T, und_r.T).T
    points = points_h[:, :3] / points_h[:, 3:4]

    h, w = color_image.shape[:2]
    xi = np.rint(matched_l[:, 0]).astype(np.int32).clip(0, w - 1)
    yi = np.rint(matched_l[:, 1]).astype(np.int32).clip(0, h - 1)
    colors = color_image[yi, xi]
    return points.astype(np.float32), colors


def write_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt, color in zip(points, colors):
            if np.ndim(color) == 0:
                red = green = blue = int(color)
            else:
                blue, green, red = [int(c) for c in color[:3]]
            f.write(
                f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                f"{red} {green} {blue}\n"
            )


def save_debug(data_dir, px_l, py_l, valid_l, px_r, py_r, valid_r, disparity):
    debug = data_dir / "debug_gray"
    debug.mkdir(exist_ok=True)
    cv2.imwrite(str(debug / "left_valid.png"), valid_l.astype(np.uint8) * 255)
    cv2.imwrite(str(debug / "right_valid.png"), valid_r.astype(np.uint8) * 255)
    cv2.imwrite(str(debug / "left_proj_x.png"), ((px_l.clip(0) % 1280) / 1280 * 255).astype(np.uint8))
    cv2.imwrite(str(debug / "left_proj_y.png"), ((py_l.clip(0) % 800) / 800 * 255).astype(np.uint8))
    disp_vis = np.nan_to_num(disparity, nan=0.0)
    if np.any(np.isfinite(disparity)):
        lo, hi = np.nanpercentile(disparity, [1, 99])
        disp_vis = (disp_vis - lo) / max(hi - lo, 1e-6) * 255
    cv2.imwrite(str(debug / "disparity.png"), disp_vis.clip(0, 255).astype(np.uint8))


def save_disparity_debug(data_dir, disparity):
    debug = data_dir / "debug_gray"
    debug.mkdir(exist_ok=True)
    disp_vis = np.nan_to_num(disparity, nan=0.0)
    if np.any(np.isfinite(disparity)):
        lo, hi = np.nanpercentile(disparity, [1, 99])
        disp_vis = (disp_vis - lo) / max(hi - lo, 1e-6) * 255
    cv2.imwrite(str(debug / "disparity.png"), disp_vis.clip(0, 255).astype(np.uint8))


def main():
    args = parse_args()
    data_dir = Path(args.data)
    calib = Path(args.calib) if args.calib else data_dir / "calibrationParameters.yml"

    print("Loading sample images...")
    images_l = load_patterns(data_dir, "pattern_cam1")
    images_r = load_patterns(data_dir, "pattern_cam2")
    cal = load_calibration(calib)

    if args.decoder == "triangulate":
        print("Decoding projector coordinates in raw camera views...")
        px_l, py_l, valid_l = decode_projector_codes(
            images_l, args.proj_width, args.proj_height, args.white_thresh, args.black_thresh
        )
        px_r, py_r, valid_r = decode_projector_codes(
            images_r, args.proj_width, args.proj_height, args.white_thresh, args.black_thresh
        )
        print(f"  left valid:  {valid_l.mean() * 100:.1f}%")
        print(f"  right valid: {valid_r.mean() * 100:.1f}%")

        print("Triangulating matching projector IDs...")
        pts, colors = triangulate_projector_matches(
            px_l, py_l, valid_l, px_r, py_r, valid_r, cal, args.proj_width, cv2.imread(str(data_dir / "pattern_cam1_im43.jpg"))
        )
        z = pts[:, 2] if len(pts) else np.array([])
        keep = np.isfinite(pts).all(axis=1) & (np.abs(z) < args.max_depth) if len(pts) else np.array([], dtype=bool)
        pts = pts[keep]
        colors = colors[keep]
        if len(pts):
            print("  depth percentiles:", np.percentile(pts[:, 2], [1, 5, 50, 95, 99]))
        print(f"  writing {len(pts)} points")
        write_ply(args.out, pts, colors)
        (data_dir / "debug_gray").mkdir(exist_ok=True)
        cv2.imwrite(str(data_dir / "debug_gray" / "left_valid.png"), valid_l.astype(np.uint8) * 255)
        cv2.imwrite(str(data_dir / "debug_gray" / "right_valid.png"), valid_r.astype(np.uint8) * 255)
        print(f"Done: {args.out}")
        return

    print("Rectifying captured Gray-code frames...")
    rect_l, rect_r, Q = rectify_images(images_l, images_r, cal)

    if args.decoder == "opencv-disparity":
        print("Decoding disparity with cv2.structured_light.GrayCodePattern...")
        disparity = disparity_from_opencv_graycode(
            rect_l, rect_r, args.proj_width, args.proj_height, args.white_thresh, args.black_thresh
        )
        save_disparity_debug(data_dir, disparity)
    else:
        print("Decoding projector coordinates with local Gray-code decoder...")
        px_l, py_l, valid_l = decode_projector_codes(
            rect_l, args.proj_width, args.proj_height, args.white_thresh, args.black_thresh
        )
        px_r, py_r, valid_r = decode_projector_codes(
            rect_r, args.proj_width, args.proj_height, args.white_thresh, args.black_thresh
        )
        print(f"  left valid:  {valid_l.mean() * 100:.1f}%")
        print(f"  right valid: {valid_r.mean() * 100:.1f}%")

        print("Matching decoded projector pixels across rectified stereo rows...")
        disparity = disparity_from_projector_codes(px_l, py_l, valid_l, px_r, py_r, valid_r, args.proj_width)
        save_debug(data_dir, px_l, py_l, valid_l, px_r, py_r, valid_r, disparity)

    valid_disp = np.isfinite(disparity)
    print(f"  valid disparity: {valid_disp.sum()} / {disparity.size} ({valid_disp.mean() * 100:.1f}%)")

    print("Reprojecting disparity to 3D...")
    points = cv2.reprojectImageTo3D(np.nan_to_num(disparity, nan=0.0).astype(np.float32), Q)
    pts = points[valid_disp]
    colors = rect_l[-2][valid_disp]
    z = pts[:, 2]
    keep = np.isfinite(z) & (np.abs(z) < args.max_depth)
    pts = pts[keep]
    colors = colors[keep]

    if len(pts):
        print("  depth percentiles:", np.percentile(pts[:, 2], [1, 5, 50, 95, 99]))
    print(f"  writing {len(pts)} points")

    write_ply(args.out, pts, colors)
    print(f"Done: {args.out}")


if __name__ == "__main__":
    main()
