"""
Fuse multiple structured-light capture directories taken at different exposures.

Each input directory must contain left/left_00.png ... and right/right_00.png ...
for the same static scene and same projector pattern sequence. The output is a
normal caps-style directory that reconstruct_local_gray.py can read.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, nargs="+",
                        help="Capture directories, e.g. caps_150 caps_250 caps_500")
    parser.add_argument("--exposures", required=True, nargs="+", type=float,
                        help="Exposure time in us for each input directory, same order as --inputs")
    parser.add_argument("--out", default="caps_hdr")
    parser.add_argument("--patterns", type=int, default=44)
    parser.add_argument("--saturation", type=int, default=245,
                        help="Ignore an exposure for a pixel if it is at or above this value")
    parser.add_argument("--reference-exposure", type=float, default=None,
                        help="Scale fused images to this exposure; default is the shortest exposure")
    return parser.parse_args()


def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def fuse_images(images, exposures, reference_exposure, saturation):
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    exp = np.asarray(exposures, dtype=np.float32)[:, None, None]

    # Convert each image to a common exposure scale, then choose the longest
    # non-saturated exposure for each pixel. Longer exposures carry more signal
    # in dark/low-contrast regions.
    scaled = stack * (reference_exposure / exp)
    usable = stack < saturation
    rank = np.where(usable, exp, -1.0)
    best = np.argmax(rank, axis=0)
    fused = np.take_along_axis(scaled, best[None, :, :], axis=0)[0]

    # If every exposure saturated at a pixel, keep the shortest exposure.
    all_saturated = ~usable.any(axis=0)
    if np.any(all_saturated):
        fused[all_saturated] = scaled[0][all_saturated]

    return np.clip(fused, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    if len(args.inputs) != len(args.exposures):
        raise ValueError("--inputs and --exposures must have the same length")

    inputs = [Path(p) for p in args.inputs]
    reference_exposure = args.reference_exposure or min(args.exposures)
    out = Path(args.out)
    for side in ("left", "right"):
        (out / side).mkdir(parents=True, exist_ok=True)

    print("HDR fuse inputs:")
    for path, exposure in zip(inputs, args.exposures):
        print(f"  {path}  exposure={exposure:g}us")
    print(f"Output: {out}")
    print(f"Reference exposure: {reference_exposure:g}us")
    print(f"Saturation cutoff: {args.saturation}")

    for side in ("left", "right"):
        for idx in range(args.patterns):
            prefix = side
            name = f"{prefix}_{idx:02d}.png"
            images = [load_gray(path / side / name) for path in inputs]
            fused = fuse_images(images, args.exposures, reference_exposure, args.saturation)
            cv2.imwrite(str(out / side / name), fused)
        print(f"Fused {side}: {args.patterns} images")


if __name__ == "__main__":
    main()
