"""
Surface roughness smoke-test tool for scanner PLY files.

This ports the useful parts of the Cloud-Viewer roughness flow into a local
Python CLI:
  - load ASCII PLY point cloud
  - crop to a measurement ROI
  - remove form by fitting a plane
  - resample to an XY grid in the fitted-plane coordinate system
  - optionally apply Gaussian short/long wavelength filters
  - report Sa, Sq, and variogram-style Svr

Input point coordinates are assumed to be in millimeters. Reported roughness
values are in micrometers.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_range(text):
    if text is None:
        return None
    lo, hi = text.split(":", 1)
    return float(lo), float(hi)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="ASCII PLY point cloud")
    parser.add_argument("--grid-mm", type=float, default=0.30,
                        help="Grid pitch in mm after form removal")
    parser.add_argument("--x-range", type=parse_range, default=None,
                        help="Crop fitted-plane X range in mm, e.g. -20:20")
    parser.add_argument("--y-range", type=parse_range, default=None,
                        help="Crop fitted-plane Y range in mm, e.g. -15:15")
    parser.add_argument("--z-range", type=parse_range, default=None,
                        help="Crop residual Z range in mm after plane removal")
    parser.add_argument("--percentile-crop", type=float, default=1.0,
                        help="Drop this percent from each XY edge before gridding; 0 disables")
    parser.add_argument("--min-points-per-cell", type=int, default=1)
    parser.add_argument("--max-hole-passes", type=int, default=8,
                        help="Neighbor-fill passes for missing grid cells")
    parser.add_argument("--long-cutoff-mm", type=float, default=0.0,
                        help="Remove wavelengths longer than this by subtracting a Gaussian low-pass")
    parser.add_argument("--short-cutoff-mm", type=float, default=0.0,
                        help="Suppress wavelengths shorter than this with a Gaussian low-pass")
    parser.add_argument("--svr-points", type=int, default=5,
                        help="Number of variogram distance bins")
    parser.add_argument("--svr-span-mm", type=float, default=1.0,
                        help="Variogram bin span in mm")
    parser.add_argument("--save-grid", default=None,
                        help="Optional .npz output containing grid arrays")
    parser.add_argument("--metrics-out", default=None,
                        help="Optional JSON output containing roughness metrics")
    return parser.parse_args()


def load_ascii_ply(path):
    with open(path) as f:
        if f.readline().strip() != "ply":
            raise ValueError("Not a PLY file")
        count = None
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line == "end_header":
                break
        if count is None:
            raise ValueError("PLY vertex count not found")
        data = np.loadtxt(f, max_rows=count, usecols=(0, 1, 2), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, 3)
    return data[np.isfinite(data).all(axis=1)]


def fit_plane_basis(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    x_axis = vh[0]
    x_axis -= normal * np.dot(x_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    coords = np.column_stack((centered @ x_axis, centered @ y_axis, centered @ normal))
    return centroid, normal, x_axis, y_axis, coords


def crop_points(coords, args):
    mask = np.ones(len(coords), dtype=bool)
    if args.percentile_crop > 0:
        p = args.percentile_crop
        xlo, xhi = np.percentile(coords[:, 0], [p, 100 - p])
        ylo, yhi = np.percentile(coords[:, 1], [p, 100 - p])
        mask &= (coords[:, 0] >= xlo) & (coords[:, 0] <= xhi)
        mask &= (coords[:, 1] >= ylo) & (coords[:, 1] <= yhi)
    if args.x_range:
        mask &= (coords[:, 0] >= args.x_range[0]) & (coords[:, 0] <= args.x_range[1])
    if args.y_range:
        mask &= (coords[:, 1] >= args.y_range[0]) & (coords[:, 1] <= args.y_range[1])
    if args.z_range:
        mask &= (coords[:, 2] >= args.z_range[0]) & (coords[:, 2] <= args.z_range[1])
    return coords[mask], mask


def grid_residuals(coords, grid_mm, min_points_per_cell):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    width = int(np.floor((x1 - x0) / grid_mm)) + 1
    height = int(np.floor((y1 - y0) / grid_mm)) + 1
    ix = np.clip(((x - x0) / grid_mm).astype(np.int32), 0, width - 1)
    iy = np.clip(((y - y0) / grid_mm).astype(np.int32), 0, height - 1)

    sums = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)
    np.add.at(sums, (iy, ix), z)
    np.add.at(counts, (iy, ix), 1)
    grid = np.full((height, width), np.nan, dtype=np.float64)
    valid = counts >= min_points_per_cell
    grid[valid] = sums[valid] / counts[valid]
    return grid, valid, (x0, y0, grid_mm)


def fill_holes_neighbor_mean(grid, valid, max_passes):
    filled = grid.copy()
    mask = valid.copy()
    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0
    for _ in range(max_passes):
        missing = ~mask
        if not missing.any():
            break
        values = np.where(mask, filled, 0.0)
        neighbor_sum = cv2.filter2D(values, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        neighbor_count = cv2.filter2D(mask.astype(np.float64), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        can_fill = missing & (neighbor_count > 0)
        if not can_fill.any():
            break
        filled[can_fill] = neighbor_sum[can_fill] / neighbor_count[can_fill]
        mask[can_fill] = True
    return filled, mask


def gaussian_sigma_from_cutoff(cutoff_mm, grid_mm):
    # Cloud-Viewer's ISO-style kernel uses 0.7309 as a cutoff scaling constant.
    # This maps the same cutoff concept to OpenCV's Gaussian sigma in cells.
    return max((0.7309 * cutoff_mm) / (np.sqrt(2 * np.pi) * grid_mm), 0.5)


def nan_aware_gaussian(grid, valid, sigma):
    values = np.where(valid, grid, 0.0).astype(np.float64)
    weights = valid.astype(np.float64)
    blurred_values = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    return blurred_values / np.maximum(blurred_weights, 1e-9)


def apply_filters(grid, valid, grid_mm, short_cutoff_mm, long_cutoff_mm):
    filtered = grid.copy()
    if short_cutoff_mm > 0:
        sigma = gaussian_sigma_from_cutoff(short_cutoff_mm, grid_mm)
        filtered = nan_aware_gaussian(filtered, valid, sigma)
    if long_cutoff_mm > 0:
        sigma = gaussian_sigma_from_cutoff(long_cutoff_mm, grid_mm)
        lowpass = nan_aware_gaussian(filtered, valid, sigma)
        filtered = filtered - lowpass
    return filtered


def sa_sq(grid, valid):
    z_um = grid[valid] * 1000.0
    sa = float(np.mean(np.abs(z_um)))
    sq = float(np.sqrt(np.mean(z_um ** 2)))
    return sa, sq


def svr_grid(grid, valid, grid_mm, points_on_variogram, span_mm):
    max_radius = points_on_variogram * span_mm
    max_cells = int(np.ceil(max_radius / grid_mm))
    sums = np.zeros(points_on_variogram, dtype=np.float64)
    counts = np.zeros(points_on_variogram, dtype=np.int64)

    for dy in range(-max_cells, max_cells + 1):
        for dx in range(-max_cells, max_cells + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.hypot(dx * grid_mm, dy * grid_mm)
            if dist <= 0 or dist > max_radius:
                continue
            bin_idx = int(np.floor(dist / span_mm))
            if bin_idx < 0 or bin_idx >= points_on_variogram:
                continue

            y_src = slice(max(0, -dy), min(grid.shape[0], grid.shape[0] - dy))
            y_dst = slice(max(0, dy), min(grid.shape[0], grid.shape[0] + dy))
            x_src = slice(max(0, -dx), min(grid.shape[1], grid.shape[1] - dx))
            x_dst = slice(max(0, dx), min(grid.shape[1], grid.shape[1] + dx))
            valid_pair = valid[y_src, x_src] & valid[y_dst, x_dst]
            if not valid_pair.any():
                continue
            dz_um = (grid[y_src, x_src][valid_pair] - grid[y_dst, x_dst][valid_pair]) * 1000.0
            sums[bin_idx] += np.sum(dz_um ** 2)
            counts[bin_idx] += len(dz_um)

    var = np.full(points_on_variogram, np.nan, dtype=np.float64)
    ok = counts > 0
    var[ok] = np.sqrt(sums[ok] / (2.0 * counts[ok]))
    svr = float(np.nanmean(var))
    return svr, var, counts


def main():
    args = parse_args()
    points = load_ascii_ply(args.ply)
    if len(points) < 3:
        raise ValueError("Need at least 3 valid points")

    centroid, normal, x_axis, y_axis, coords = fit_plane_basis(points)
    cropped, crop_mask = crop_points(coords, args)
    grid, valid_raw, grid_info = grid_residuals(cropped, args.grid_mm, args.min_points_per_cell)
    filled, valid_filled = fill_holes_neighbor_mean(grid, valid_raw, args.max_hole_passes)
    filtered = apply_filters(
        filled,
        valid_filled,
        args.grid_mm,
        args.short_cutoff_mm,
        args.long_cutoff_mm,
    )

    # Re-zero after filtering so Sa/Sq are about roughness, not residual offset.
    filtered = filtered - np.nanmean(filtered[valid_filled])
    sa, sq = sa_sq(filtered, valid_filled)
    svr, variogram_bins, variogram_counts = svr_grid(
        filtered,
        valid_filled,
        args.grid_mm,
        args.svr_points,
        args.svr_span_mm,
    )

    print(f"input points: {len(points)}")
    print(f"cropped points: {len(cropped)}")
    print(f"plane centroid mm: {centroid}")
    print(f"plane normal: {normal}")
    print(
        "raw residual mm: "
        f"std={coords[:, 2].std():.6f} "
        f"p05={np.percentile(coords[:, 2], 5):.6f} "
        f"p95={np.percentile(coords[:, 2], 95):.6f}"
    )
    print(f"grid: {grid.shape[1]} x {grid.shape[0]} cells, pitch={args.grid_mm:.4f} mm")
    print(
        f"grid coverage: raw={valid_raw.mean() * 100:.1f}% "
        f"filled={valid_filled.mean() * 100:.1f}%"
    )
    print(f"filters: short_cutoff={args.short_cutoff_mm:g} mm long_cutoff={args.long_cutoff_mm:g} mm")
    print("\nRoughness:")
    print(f"  Sa  = {sa:.3f} um")
    print(f"  Sq  = {sq:.3f} um")
    print(f"  Svr = {svr:.3f} um")
    print("\nVariogram bins:")
    for idx, value in enumerate(variogram_bins):
        lo = idx * args.svr_span_mm
        hi = (idx + 1) * args.svr_span_mm
        if np.isfinite(value):
            print(f"  {lo:.3f}-{hi:.3f} mm: {value:.3f} um  pairs={variogram_counts[idx]}")
        else:
            print(f"  {lo:.3f}-{hi:.3f} mm: no pairs")

    if args.save_grid:
        path = Path(args.save_grid)
        np.savez(
            path,
            grid_raw=grid,
            grid_filled=filled,
            grid_filtered=filtered,
            valid_raw=valid_raw,
            valid_filled=valid_filled,
            grid_origin=np.array(grid_info),
            plane_centroid=centroid,
            plane_normal=normal,
            plane_x_axis=x_axis,
            plane_y_axis=y_axis,
        )
        print(f"\nWrote grid data: {path}")

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "sa_um": sa,
            "sq_um": sq,
            "svr_um": svr,
            "points": int(len(points)),
            "cropped_points": int(len(cropped)),
            "grid_width": int(grid.shape[1]),
            "grid_height": int(grid.shape[0]),
            "grid_pitch_mm": float(args.grid_mm),
            "grid_coverage_raw_percent": float(valid_raw.mean() * 100.0),
            "grid_coverage_filled_percent": float(valid_filled.mean() * 100.0),
            "short_cutoff_mm": float(args.short_cutoff_mm),
            "long_cutoff_mm": float(args.long_cutoff_mm),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
