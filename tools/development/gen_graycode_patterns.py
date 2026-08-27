"""
Generate OpenCV Gray-code structured-light BMPs for the DLP4500.

The filenames are prefixed with 00..43 so alphabetical order is the intended
flash slot order:
  00..41: OpenCV GrayCodePattern images
  42:     all-white image
  43:     all-black image
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=912, help="Projector width in pixels")
    parser.add_argument("--height", type=int, default=1140, help="Projector height in pixels")
    parser.add_argument("--code-width", type=int, default=None,
                        help="Effective Gray-code width. Patterns are nearest-neighbor scaled to projector width.")
    parser.add_argument("--code-height", type=int, default=None,
                        help="Effective Gray-code height. Patterns are nearest-neighbor scaled to projector height.")
    parser.add_argument("--out-dir", default="imgs/graycode_912x1140")
    return parser.parse_args()


def as_bgr_u8(image):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(cv2, "structured_light"):
        raise RuntimeError("cv2.structured_light is missing. Install opencv-contrib-python-headless.")

    code_width = args.code_width or args.width
    code_height = args.code_height or args.height
    graycode = cv2.structured_light.GrayCodePattern.create(code_width, code_height)
    ok, patterns = graycode.generate()
    if not ok:
        raise RuntimeError("GrayCodePattern.generate failed")

    expected = int(graycode.getNumberOfPatternImages())
    if len(patterns) != expected:
        raise RuntimeError(f"Expected {expected} generated patterns, got {len(patterns)}")

    manifest_lines = [
        f"projector_width: {args.width}",
        f"projector_height: {args.height}",
        f"code_width: {code_width}",
        f"code_height: {code_height}",
        f"graycode_patterns: {expected}",
        "slot_order:",
    ]

    for idx, pattern in enumerate(patterns):
        if pattern.shape[1] != args.width or pattern.shape[0] != args.height:
            pattern = cv2.resize(pattern, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        filename = f"{idx:02d}_graycode.bmp"
        cv2.imwrite(str(out_dir / filename), as_bgr_u8(pattern))
        manifest_lines.append(f"  {idx:02d}: {filename}")

    white_idx = expected
    black_idx = expected + 1
    white = np.full((args.height, args.width), 255, dtype=np.uint8)
    black = np.zeros((args.height, args.width), dtype=np.uint8)

    white_name = f"{white_idx:02d}_white.bmp"
    black_name = f"{black_idx:02d}_black.bmp"
    cv2.imwrite(str(out_dir / white_name), as_bgr_u8(white))
    cv2.imwrite(str(out_dir / black_name), as_bgr_u8(black))
    manifest_lines.append(f"  {white_idx:02d}: {white_name}")
    manifest_lines.append(f"  {black_idx:02d}: {black_name}")

    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")

    print(f"Wrote {expected + 2} BMPs to {out_dir}")
    print(f"Flash them in alphabetical order, slots 0..{expected + 1}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
