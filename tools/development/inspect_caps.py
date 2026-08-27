"""
Print quick brightness/change stats for captured scan images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="caps")
    parser.add_argument("--count", type=int, default=44)
    args = parser.parse_args()

    for side in ("left", "right"):
        print(side)
        prev = None
        existing = 0
        for idx in range(args.count):
            path = Path(args.caps) / side / f"{side}_{idx:02d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"  {idx:02d}: missing")
                continue
            existing += 1
            diff = ""
            if prev is not None:
                mad = np.mean(np.abs(image.astype(np.float32) - prev.astype(np.float32)))
                diff = f" diffprev={mad:7.2f}"
            print(
                f"  {idx:02d}: mean={image.mean():7.2f} std={image.std():7.2f} "
                f"min={int(image.min()):3d} max={int(image.max()):3d}{diff}"
            )
            prev = image
        print(f"  files: {existing}/{args.count}\n")


if __name__ == "__main__":
    main()
