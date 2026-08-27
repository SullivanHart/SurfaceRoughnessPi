"""
Fit a plane to a PLY point cloud and report residual curvature by X/Y bins.
"""

import argparse
import numpy as np


def load_ascii_ply(path):
    with open(path) as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a PLY file")
        count = None
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line == "end_header":
                break
        data = np.loadtxt(f, max_rows=count, usecols=(0, 1, 2), dtype=np.float64)
    return data


def fit_plane(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    residual = centered @ normal
    return centroid, normal, residual


def bin_report(points, residual, axis, bins):
    coord = points[:, axis]
    edges = np.linspace(np.percentile(coord, 1), np.percentile(coord, 99), bins + 1)
    print(f"\nResidual by {'XYZ'[axis]} bins:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (coord >= lo) & (coord < hi)
        if mask.sum() < 10:
            continue
        print(
            f"  {lo:9.3f}..{hi:9.3f}: "
            f"n={mask.sum():7d} mean={residual[mask].mean():9.4f} "
            f"median={np.median(residual[mask]):9.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--bins", type=int, default=9)
    args = parser.parse_args()

    pts = load_ascii_ply(args.ply)
    if len(pts) < 3:
        raise ValueError("Need at least 3 points")
    centroid, normal, residual = fit_plane(pts)
    print(f"points: {len(pts)}")
    print(f"centroid: {centroid}")
    print(f"normal: {normal}")
    print(
        "residual mm: "
        f"mean={residual.mean():.6f} std={residual.std():.6f} "
        f"p05={np.percentile(residual, 5):.6f} "
        f"p50={np.percentile(residual, 50):.6f} "
        f"p95={np.percentile(residual, 95):.6f}"
    )
    bin_report(pts, residual, 0, args.bins)
    bin_report(pts, residual, 1, args.bins)


if __name__ == "__main__":
    main()
