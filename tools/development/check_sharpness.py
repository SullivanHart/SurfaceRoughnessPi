"""
Report image sharpness for captured stereo images.

Sharpness is Laplacian variance: larger values generally mean sharper focus.
"""

import argparse
import csv
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="caps", help="Root containing left/ and right/ folders")
    parser.add_argument("--glob", default="*.png", help="Image glob inside each side folder")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    return parser.parse_args()


def classify(value):
    if value < 100:
        return "blurry"
    if value < 500:
        return "usable"
    if value < 1000:
        return "good"
    return "very_good"


def image_stats(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return {
        "path": str(path),
        "sharpness": sharpness,
        "quality": classify(sharpness),
        "mean": float(image.mean()),
        "min": int(image.min()),
        "max": int(image.max()),
    }


def main():
    args = parse_args()
    root = Path(args.root)
    rows = []

    for side in ("left", "right"):
        side_dir = root / side
        if not side_dir.exists():
            print(f"Missing folder: {side_dir}")
            continue
        for path in sorted(side_dir.glob(args.glob)):
            stats = image_stats(path)
            stats["side"] = side
            stats["file"] = path.name
            rows.append(stats)

    if not rows:
        print("No images found.")
        return

    print(f"{'side':<6} {'file':<16} {'sharpness':>10} {'quality':<10} {'mean':>8} {'min':>4} {'max':>4}")
    print("-" * 68)
    for row in rows:
        print(
            f"{row['side']:<6} {row['file']:<16} {row['sharpness']:10.1f} "
            f"{row['quality']:<10} {row['mean']:8.1f} {row['min']:4d} {row['max']:4d}"
        )

    print("\nSummary:")
    for side in ("left", "right"):
        vals = [r["sharpness"] for r in rows if r["side"] == side]
        if vals:
            print(f"  {side}: min={min(vals):.1f}  avg={sum(vals) / len(vals):.1f}  max={max(vals):.1f}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["side", "file", "sharpness", "quality", "mean", "min", "max", "path"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
