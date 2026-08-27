"""
Fresh OpenCV structured-light reference reconstruction.

This follows OpenCV's Gray-code point-cloud tutorial closely:
  1. load the OpenCV sample captures and calibration file,
  2. stereo-rectify with the calibration,
  3. remap the captured pattern images using the tutorial's camera/map order,
  4. decode disparity with cv2.structured_light.GrayCodePattern,
  5. reproject disparity with Q and write a PLY.

Use this as the known-good baseline before adapting the pipeline to local
Basler/DLP captures.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sample_data/opencv_structured_light")
    parser.add_argument("--calib", default=None)
    parser.add_argument("--out", default="sample_data/opencv_structured_light/cloud_reference.ply")
    parser.add_argument("--proj-width", type=int, default=1280)
    parser.add_argument("--proj-height", type=int, default=800)
    parser.add_argument("--white-thresh", type=int, default=None)
    parser.add_argument("--black-thresh", type=int, default=None)
    parser.add_argument("--min-disparity", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=1e5)
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
        "cam1_intrinsics": read_mat(fs, "cam1_intrinsics"),
        "cam1_distorsion": read_mat(fs, "cam1_distorsion"),
        "cam2_intrinsics": read_mat(fs, "cam2_intrinsics"),
        "cam2_distorsion": read_mat(fs, "cam2_distorsion"),
        "R": read_mat(fs, "R"),
        "T": read_mat(fs, "T"),
    }
    fs.release()
    return cal


def load_sequence(data_dir, prefix, count):
    images = []
    for idx in range(1, count + 1):
        path = data_dir / f"{prefix}_im{idx}.jpg"
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


def save_disparity_debug(data_dir, disparity, mask):
    debug_dir = data_dir / "debug_reference"
    debug_dir.mkdir(exist_ok=True)

    valid = disparity[mask]
    vis = np.zeros(disparity.shape, dtype=np.uint8)
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        scaled = (disparity - lo) / max(hi - lo, 1e-6) * 255
        vis = np.where(mask, scaled, 0).clip(0, 255).astype(np.uint8)

    cv2.imwrite(str(debug_dir / "disparity.png"), vis)
    cv2.imwrite(str(debug_dir / "mask.png"), mask.astype(np.uint8) * 255)


def main():
    args = parse_args()
    if not hasattr(cv2, "structured_light"):
        raise RuntimeError("cv2.structured_light is missing. Install opencv-contrib-python-headless.")

    data_dir = Path(args.data)
    calib_path = Path(args.calib) if args.calib else data_dir / "calibrationParameters.yml"

    graycode = cv2.structured_light.GrayCodePattern.create(args.proj_width, args.proj_height)
    if args.white_thresh is not None:
        graycode.setWhiteThreshold(args.white_thresh)
    if args.black_thresh is not None:
        graycode.setBlackThreshold(args.black_thresh)

    pattern_count = int(graycode.getNumberOfPatternImages())
    total_count = pattern_count + 2
    print(f"Loading {total_count} images per camera...")
    cam1 = load_sequence(data_dir, "pattern_cam1", total_count)
    cam2 = load_sequence(data_dir, "pattern_cam2", total_count)
    cal = load_calibration(calib_path)

    image_size = (cam1[0].shape[1], cam1[0].shape[0])
    print(f"Image size: {image_size[0]}x{image_size[1]}")

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cal["cam1_intrinsics"],
        cal["cam1_distorsion"],
        cal["cam2_intrinsics"],
        cal["cam2_distorsion"],
        image_size,
        cal["R"],
        cal["T"],
        flags=0,
        alpha=-1,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        cal["cam1_intrinsics"], cal["cam1_distorsion"], R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        cal["cam2_intrinsics"], cal["cam2_distorsion"], R2, P2, image_size, cv2.CV_32FC1
    )

    print("Rectifying images using the OpenCV tutorial map order...")
    captured_pattern = [
        remap_sequence(cam1[:pattern_count], map2x, map2y),
        remap_sequence(cam2[:pattern_count], map1x, map1y),
    ]
    white_images = [
        cv2.remap(cam1[pattern_count], map2x, map2y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(cam2[pattern_count], map1x, map1y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
    ]
    black_images = [
        cv2.remap(cam1[pattern_count + 1], map2x, map2y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
        cv2.remap(cam2[pattern_count + 1], map1x, map1y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT),
    ]

    print("Decoding with cv2.structured_light.GrayCodePattern...")
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

    mask = np.isfinite(disparity) & (disparity >= args.min_disparity)
    print(f"Valid disparity: {int(mask.sum())} / {mask.size} ({mask.mean() * 100:.1f}%)")

    print("Reprojecting to 3D...")
    pointcloud = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    points = pointcloud[mask]
    colors = white_images[0][mask]

    z = points[:, 2]
    keep = np.isfinite(points).all(axis=1) & (np.abs(z) < args.max_depth)
    points = points[keep]
    colors = colors[keep]

    if len(points):
        print("Depth percentiles:", np.percentile(points[:, 2], [1, 5, 50, 95, 99]))
    print(f"Writing {len(points)} points to {args.out}")

    write_ply(args.out, points, colors)
    save_disparity_debug(data_dir, disparity, mask)
    print(f"Debug images: {data_dir / 'debug_reference'}")


if __name__ == "__main__":
    main()
