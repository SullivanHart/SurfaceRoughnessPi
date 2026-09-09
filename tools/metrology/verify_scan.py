from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

from svr_roughness import RoughnessConfig, analyze_points, load_points
from svr_roughness.algorithm import apply_dual_pass_gaussian_filter, pca_align_plane
from svr_roughness.result import RoughnessResult


class DeviationStats(NamedTuple):
    mean_um: float
    median_um: float
    std_um: float
    rms_um: float
    p05_um: float
    p25_um: float
    p75_um: float
    p95_um: float


def fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane through 3D points using SVD."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    normal = vh[2]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def align_to_z(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate points so dominant plane normal points along +Z."""
    centroid, normal = fit_plane_svd(pts)
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = float(np.dot(normal, z_axis))
    if s < 1e-8:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s**2))
    rot_pts = (pts - centroid) @ R.T
    return rot_pts, R, centroid


def voxel_downsample_fast(pts: np.ndarray, leaf_size: float) -> np.ndarray:
    """Quick grid voxel downsampling using integer voxel keys."""
    if len(pts) == 0:
        return pts
    voxel_coords = np.floor(pts / leaf_size).astype(np.int64)
    packed = np.ascontiguousarray(voxel_coords).view(
        np.dtype((np.void, voxel_coords.dtype.itemsize * 3))
    )
    _, idx = np.unique(packed, return_index=True)
    return pts[idx]


def make_elevation_feature_map(
    pts: np.ndarray, pitch: float = 0.5
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Rasterize points to a high-pass filtered elevation map for 2D cross-correlation."""
    import cv2

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    nx = int(np.ceil((x_max - x_min) / pitch)) + 1
    ny = int(np.ceil((y_max - y_min) / pitch)) + 1
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    ix = np.clip(((pts[:, 0] - x_min) / pitch).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y_min) / pitch).astype(int), 0, ny - 1)

    flat_idx = iy * nx + ix
    order = np.argsort(flat_idx)
    sorted_idx = flat_idx[order]
    sorted_z = pts[:, 2][order]
    unq, start = np.unique(sorted_idx, return_index=True)
    counts = np.diff(np.append(start, len(sorted_idx)))
    sums = np.add.reduceat(sorted_z, start)
    u_iy, u_ix = np.unravel_index(unq, (ny, nx))
    grid[u_iy, u_ix] = sums / counts

    mask = np.isfinite(grid)
    mean_z = np.nanmean(grid)
    feat = np.where(mask, grid - mean_z, 0.0).astype(np.float32)
    # High-pass filter: subtract broad spatial trend to accentuate surface asperities
    blur = cv2.GaussianBlur(feat, (25, 25), 5.0)
    feat = feat - blur
    feat[~mask] = 0.0
    return feat, mask, (float(x_min), float(y_min))


def raster_synchronized(
    pts: np.ndarray,
    x_min: float,
    y_min: float,
    nx: int,
    ny: int,
    pitch_mm: float = 0.2,
    short_cutoff_mm: float = 1.0,
    long_cutoff_mm: float = 25.0,
) -> np.ndarray:
    """Rasterize points onto a shared coordinate grid and apply dual-pass Gaussian filter."""
    ix = np.clip(np.floor((pts[:, 0] - x_min) / pitch_mm).astype(int), 0, nx - 1)
    iy = np.clip(np.floor((pts[:, 1] - y_min) / pitch_mm).astype(int), 0, ny - 1)
    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    counts = np.zeros((ny, nx), dtype=int)
    sums = np.zeros((ny, nx), dtype=float)
    np.add.at(sums, (iy, ix), pts[:, 2])
    np.add.at(counts, (iy, ix), 1)
    mask = counts > 0
    grid[mask] = sums[mask] / counts[mask]
    mean_val = np.nanmean(grid)
    filled = np.where(mask, grid, mean_val)
    pitch_m = pitch_mm * 0.001
    s_m = short_cutoff_mm * 0.001
    l_m = long_cutoff_mm * 0.001
    filt_m = apply_dual_pass_gaussian_filter(filled * 0.001, pitch_m, s_m, l_m)
    out_um = filt_m * 1e6
    out_um[~mask] = np.nan
    return out_um


def coarse_align_2d(
    ref_pts: np.ndarray, cap_pts: np.ndarray, pitch: float = 0.5, num_angles: int = 180
) -> tuple[bool, bool, float, tuple[float, float], float]:
    """Finds best 2D rotation, flip, translation, and elevation polarity (Z) via normalized cross-correlation.

    Returns:
        (invert_z, flip, angle_deg, (tx_mm, ty_mm), score)
    """
    import cv2

    feat_ref, _, orig_ref = make_elevation_feature_map(ref_pts, pitch)
    feat_cap, _, orig_cap = make_elevation_feature_map(cap_pts, pitch)

    best_score = -1.0
    best_match = (False, False, 0.0, (0.0, 0.0), -1.0)

    angles = np.linspace(0, 360, num_angles, endpoint=False)
    h_c, w_c = feat_cap.shape
    h_r, w_r = feat_ref.shape

    for flip in (False, True):
        for angle in angles:
            M = cv2.getRotationMatrix2D((w_c / 2.0, h_c / 2.0), angle, 1.0)
            rot_c = cv2.warpAffine(feat_cap, M, (w_c, h_c))
            if flip:
                rot_c = np.fliplr(rot_c)

            if rot_c.shape[0] > h_r or rot_c.shape[1] > w_r:
                continue

            res = cv2.matchTemplate(feat_ref, rot_c, cv2.TM_CCOEFF_NORMED)
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(res)

            # Upright Z match (+Z = peaks)
            if max_v > best_score:
                best_score = max_v
                tx = orig_ref[0] + (max_loc[0] + w_c / 2.0) * pitch
                ty = orig_ref[1] + (max_loc[1] + h_c / 2.0) * pitch
                best_match = (False, flip, float(angle), (tx, ty), float(max_v))

            # Inverted Z match (-Z = peaks; valleys match peaks)
            if -min_v > best_score:
                best_score = -min_v
                tx = orig_ref[0] + (min_loc[0] + w_c / 2.0) * pitch
                ty = orig_ref[1] + (min_loc[1] + h_c / 2.0) * pitch
                best_match = (True, flip, float(angle), (tx, ty), float(-min_v))

    return best_match


def run_icp_3d(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    max_dist: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rigid 3D Iterative Closest Point (ICP) registration using KDTree and Kabsch SVD."""
    tree = cKDTree(target_pts)
    curr_pts = source_pts.copy()
    total_R = np.eye(3)
    total_t = np.zeros(3)
    prev_error = float("inf")

    for _ in range(max_iter):
        dists, idxs = tree.query(curr_pts, k=1)
        valid = dists < max_dist
        if valid.sum() < 50:
            break

        src_valid = curr_pts[valid]
        tgt_valid = target_pts[idxs[valid]]

        curr_error = float(np.mean(dists[valid]))
        if abs(prev_error - curr_error) < tolerance:
            break
        prev_error = curr_error

        c_s = np.mean(src_valid, axis=0)
        c_t = np.mean(tgt_valid, axis=0)
        H = (src_valid - c_s).T @ (tgt_valid - c_t)
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1, :] *= -1
            R_step = Vt.T @ U.T
        t_step = c_t - R_step @ c_s

        curr_pts = (curr_pts @ R_step.T) + t_step
        total_R = R_step @ total_R
        total_t = R_step @ total_t + t_step

    return curr_pts, total_R, total_t, prev_error


def crop_reference_to_captured(
    ref_pts: np.ndarray, cap_pts: np.ndarray, margin_mm: float = 0.8
) -> np.ndarray:
    """Crop the reference scan strictly within the 2D bounding footprint of the captured scan."""
    cap_xy_tree = cKDTree(cap_pts[:, :2])
    dists_2d, _ = cap_xy_tree.query(ref_pts[:, :2], k=1)
    crop_mask = dists_2d <= margin_mm
    return ref_pts[crop_mask]


def compute_surface_deviations(
    cap_pts: np.ndarray, ref_pts: np.ndarray, max_dist_mm: float = 2.0
) -> DeviationStats:
    """Computes point-to-point surface deviation statistics in micrometers."""
    tree = cKDTree(ref_pts)
    dists, idxs = tree.query(cap_pts, k=1)
    valid = dists < max_dist_mm

    signed_dz = (cap_pts[valid, 2] - ref_pts[idxs[valid], 2]) * 1000.0

    mean_um = float(np.mean(signed_dz))
    median_um = float(np.median(signed_dz))
    std_um = float(np.std(signed_dz))
    rms_um = float(np.sqrt(np.mean(signed_dz**2)))
    p05, p25, p75, p95 = np.percentile(signed_dz, [5, 25, 75, 95])

    return DeviationStats(
        mean_um=mean_um,
        median_um=median_um,
        std_um=std_um,
        rms_um=rms_um,
        p05_um=float(p05),
        p25_um=float(p25),
        p75_um=float(p75),
        p95_um=float(p95),
    )


def save_ply_binary(filepath: Path, points: np.ndarray) -> None:
    """Save Nx3 float points into binary little-endian PLY."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    pts = np.asarray(points, dtype=np.float32)
    with filepath.open("wb") as handle:
        handle.write(header)
        handle.write(pts.tobytes())


def generate_verification_plot(
    res_cap: RoughnessResult,
    res_ref: RoughnessResult,
    dev_stats: DeviationStats,
    out_png: Path,
    g_cap: np.ndarray | None = None,
    g_ref: np.ndarray | None = None,
    elevation_corr: float | None = None,
) -> None:
    """Generate a high-resolution 4-panel visual verification summary plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: Could not import matplotlib ({e}); skipping verification plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 1. Captured Elevation Grid
    ax1 = axes[0, 0]
    if g_cap is not None and g_ref is not None:
        cap_z_um = g_cap
        ref_z_um = g_ref
        all_vals = np.concatenate([cap_z_um[np.isfinite(cap_z_um)], ref_z_um[np.isfinite(ref_z_um)]])
        if len(all_vals) > 0:
            vmax = float(np.nanpercentile(np.abs(all_vals), 99.5))
            vmin = -vmax
        else:
            vmin, vmax = -400.0, 400.0
    else:
        cap_z_um = res_cap.grid.filtered * 1000.0
        ref_z_um = res_ref.grid.filtered * 1000.0
        vmin, vmax = None, None

    im1 = ax1.imshow(cap_z_um, cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)
    corr_str = f", r = {elevation_corr:+.3f}" if elevation_corr is not None else ""
    ax1.set_title(f"Captured Surface (S_vr = {res_cap.svr_um:.1f} µm, NF = {res_cap.noise_floor_um:.1f} µm{corr_str})")
    ax1.set_xlabel("Grid X (0.2 mm cells)")
    ax1.set_ylabel("Grid Y (0.2 mm cells)")
    plt.colorbar(im1, ax=ax1, label="Roughness Elevation (µm)")

    # 2. Reference Cropped Elevation Grid
    ax2 = axes[0, 1]
    im2 = ax2.imshow(ref_z_um, cmap="turbo", origin="lower", vmin=vmin, vmax=vmax)
    ax2.set_title(f"Reference Cropped (S_vr = {res_ref.svr_um:.1f} µm, NF = {res_ref.noise_floor_um:.1f} µm)")
    ax2.set_xlabel("Grid X (0.2 mm cells)")
    ax2.set_ylabel("Grid Y (0.2 mm cells)")
    plt.colorbar(im2, ax=ax2, label="Roughness Elevation (µm)")

    # 3. Variogram Curve Overlay
    ax3 = axes[1, 0]
    distances = np.arange(1, len(res_cap.variogram_bins_um) + 1) * 0.5
    ax3.plot(distances, res_cap.variogram_bins_um, "r-o", label="Captured Scan", linewidth=2)
    ax3.plot(distances, res_ref.variogram_bins_um, "b--s", label="Reference Cropped", linewidth=2)
    ax3.set_title(f"ASTM WK92969 Variogram (ΔS_vr = {res_cap.svr_um - res_ref.svr_um:+.1f} µm)")
    ax3.set_xlabel("Evaluation Distance (mm)")
    ax3.set_ylabel("Roughness γ(d) (µm)")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Surface Deviation Statistics
    ax4 = axes[1, 1]
    metrics = [
        f"ASTM S_vr (Cap):    {res_cap.svr_um:.2f} µm",
        f"ASTM S_vr (Ref):    {res_ref.svr_um:.2f} µm",
        f"Delta S_vr:         {res_cap.svr_um - res_ref.svr_um:+.2f} µm ({((res_cap.svr_um - res_ref.svr_um)/res_ref.svr_um)*100:+.1f}%)",
        "",
        f"Sa (Cap / Ref):     {res_cap.sa_um:.1f} / {res_ref.sa_um:.1f} µm",
        f"Sq (Cap / Ref):     {res_cap.sq_um:.1f} / {res_ref.sq_um:.1f} µm",
        f"Noise Floor (Cap):  {res_cap.noise_floor_um:.2f} µm",
        f"Noise Floor (Ref):  {res_ref.noise_floor_um:.2f} µm",
        f"Height Correlation: {elevation_corr:+.3f}" if elevation_corr is not None else "",
        "",
        f"Point-to-Point Deviation:",
        f"  Mean Error:       {dev_stats.mean_um:+.2f} µm",
        f"  Median Error:     {dev_stats.median_um:+.2f} µm",
        f"  RMS Error:        {dev_stats.rms_um:.2f} µm",
        f"  Std Deviation:    {dev_stats.std_um:.2f} µm",
        f"  90% Interval:     [{dev_stats.p05_um:.1f}, {dev_stats.p95_um:.1f}] µm",
    ]
    metrics = [m for m in metrics if m != ""]
    ax4.text(
        0.05,
        0.95,
        "\n".join(metrics),
        transform=ax4.transAxes,
        fontsize=11,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )
    ax4.axis("off")
    ax4.set_title("Metrology Agreement & Deviation Summary")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


def verify_samples(
    captured_path: str | Path,
    reference_path: str | Path,
    out_dir: str | Path | None = None,
    grid_mm: float = 0.2,
    short_cutoff_mm: float = 1.0,
    long_cutoff_mm: float = 25.0,
) -> dict:
    """Execute complete automated registration, cropping, and dual ASTM roughness verification."""
    cap_file = Path(captured_path)
    ref_file = Path(reference_path)

    print(f"Loading captured point cloud: {cap_file}...")
    cap_raw = load_points(cap_file)
    print(f"Loading reference point cloud: {ref_file}...")
    ref_raw = load_points(ref_file)

    print(f"Loaded: Captured={len(cap_raw):,} pts, Reference={len(ref_raw):,} pts")

    # 1. Level planes to +Z
    print("Leveling planes to +Z...")
    ref_aligned, _, _ = align_to_z(ref_raw)
    cap_aligned, _, _ = align_to_z(cap_raw)

    # 2. Downsample for registration
    ref_down = voxel_downsample_fast(ref_aligned, 0.5)
    cap_down = voxel_downsample_fast(cap_aligned, 0.5)

    # 3. Coarse 2D Alignment (search orientation, polarity & translation)
    print("Performing multi-scale 2D orientation, polarity & translation search...")
    invert_z, flip, angle, loc, score = coarse_align_2d(ref_down, cap_down, pitch=0.5, num_angles=180)
    print(f"Coarse 2D Match: invert_z={invert_z}, flip={flip}, angle={angle:.1f} deg, translation=({loc[0]:.1f}, {loc[1]:.1f}) mm, score={score:.3f}")

    if invert_z:
        print("Detected inverted surface elevation polarity: orienting +Z to peaks...")
        cap_aligned[:, 2] = -cap_aligned[:, 2]
        cap_down[:, 2] = -cap_down[:, 2]

    # Apply coarse transform
    cap_rough = cap_down.copy()
    if flip:
        cap_rough[:, 0] = -cap_rough[:, 0]
    rad = np.radians(angle)
    R_2d = np.array([
        [np.cos(rad), -np.sin(rad), 0.0],
        [np.sin(rad),  np.cos(rad), 0.0],
        [0.0,          0.0,         1.0]
    ])
    cap_rough = cap_rough @ R_2d.T
    t_2d = np.array([
        loc[0] - np.mean(cap_rough[:, 0]),
        loc[1] - np.mean(cap_rough[:, 1]),
        np.median(ref_down[:, 2]) - np.median(cap_rough[:, 2])
    ])
    cap_rough += t_2d

    # 4. Fine 3D ICP Registration
    print("Refining alignment with 3D Iterative Closest Point (ICP)...")
    cap_reg_down, R_icp, t_icp, icp_err = run_icp_3d(cap_rough, ref_down, max_iter=50, max_dist=2.5)
    print(f"ICP converged: point-to-point residual = {icp_err * 1000.0:.1f} um")

    # Transform full resolution captured points
    cap_full = cap_aligned.copy()
    if flip:
        cap_full[:, 0] = -cap_full[:, 0]
    cap_full = (cap_full @ R_2d.T + t_2d) @ R_icp.T + t_icp

    # 5. Crop Reference Point Cloud to Captured Footprint
    print("Cropping reference sample to captured spatial borders...")
    ref_cropped = crop_reference_to_captured(ref_aligned, cap_full, margin_mm=0.8)
    print(f"Reference points cropped from {len(ref_raw):,} to {len(ref_cropped):,} points")

    # 6. Point-to-Point Surface Deviation
    dev_stats = compute_surface_deviations(cap_full, ref_cropped)

    # 7. Shared PCA coordinate alignment for synchronized visualization
    c_mid = ref_cropped.mean(axis=0)
    _, _, eig_shared = pca_align_plane((ref_cropped - c_mid) * 0.001)
    cap_pca = (cap_full - c_mid) @ eig_shared
    ref_pca = (ref_cropped - c_mid) @ eig_shared

    # Synchronized rasterization bounds
    x_min = min(cap_pca[:, 0].min(), ref_pca[:, 0].min())
    x_max = max(cap_pca[:, 0].max(), ref_pca[:, 0].max())
    y_min = min(cap_pca[:, 1].min(), ref_pca[:, 1].min())
    y_max = max(cap_pca[:, 1].max(), ref_pca[:, 1].max())
    nx = int(np.ceil((x_max - x_min) / grid_mm)) + 1
    ny = int(np.ceil((y_max - y_min) / grid_mm)) + 1

    g_cap = raster_synchronized(
        cap_pca, x_min, y_min, nx, ny, pitch_mm=grid_mm,
        short_cutoff_mm=short_cutoff_mm, long_cutoff_mm=long_cutoff_mm,
    )
    g_ref = raster_synchronized(
        ref_pca, x_min, y_min, nx, ny, pitch_mm=grid_mm,
        short_cutoff_mm=short_cutoff_mm, long_cutoff_mm=long_cutoff_mm,
    )
    valid_overlap = np.isfinite(g_cap) & np.isfinite(g_ref)
    elevation_corr = float(np.corrcoef(g_cap[valid_overlap], g_ref[valid_overlap])[0, 1]) if np.any(valid_overlap) else 0.0
    print(f"Synchronized Spatial Pearson Correlation: {elevation_corr:+.4f} across {np.sum(valid_overlap):,} cells")

    # 8. Dual ASTM WK92969 Roughness Analysis
    print("Running dual ASTM WK92969 analysis on both identical patches...")
    cfg = RoughnessConfig(grid_mm=grid_mm, short_cutoff_mm=short_cutoff_mm, long_cutoff_mm=long_cutoff_mm)

    res_cap = analyze_points(cap_full, cfg)
    res_ref = analyze_points(ref_cropped, cfg)

    delta_svr = res_cap.svr_um - res_ref.svr_um
    pct_agreement = ((res_ref.svr_um - abs(delta_svr)) / res_ref.svr_um) * 100.0 if res_ref.svr_um > 0 else 0.0

    # Print Formatted Report
    print("\n" + "=" * 70)
    print("       ASTM WK92969 IDENTICAL-SAMPLE VERIFICATION REPORT")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Captured Scan':<16} | {'Reference Cropped':<16} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'S_vr (ASTM Roughness)':<25} | {res_cap.svr_um:13.2f} um | {res_ref.svr_um:13.2f} um | {delta_svr:+7.2f} um")
    print(f"{'Sa (Arithmetical Mean)':<25} | {res_cap.sa_um:13.2f} um | {res_ref.sa_um:13.2f} um | {res_cap.sa_um - res_ref.sa_um:+7.2f} um")
    print(f"{'Sq (Root Mean Square)':<25} | {res_cap.sq_um:13.2f} um | {res_ref.sq_um:13.2f} um | {res_cap.sq_um - res_ref.sq_um:+7.2f} um")
    print(f"{'Dynamic Noise Floor':<25} | {res_cap.noise_floor_um:13.2f} um | {res_ref.noise_floor_um:13.2f} um | {res_cap.noise_floor_um - res_ref.noise_floor_um:+7.2f} um")
    print(f"{'Raw S_vr (Uncorrected)':<25} | {res_cap.svr_raw_um:13.2f} um | {res_ref.svr_raw_um:13.2f} um | {res_cap.svr_raw_um - res_ref.svr_raw_um:+7.2f} um")
    print(f"{'Valid Points':<25} | {res_cap.points:13,d}    | {res_ref.points:13,d}    |")
    print("-" * 70)
    print(f"Metrology Agreement: {pct_agreement:.1f}%")
    print(f"Surface Deviation:   Mean={dev_stats.mean_um:+.1f} um, RMS={dev_stats.rms_um:.1f} um, Std={dev_stats.std_um:.1f} um")
    print(f"Spatial Correlation: {elevation_corr:+.3f} across {np.sum(valid_overlap):,} shared cells")
    print("-" * 70)
    print("Variogram Bins (Evaluation Length 0.5 to 5.0 mm):")
    for b_idx in range(len(res_cap.variogram_bins_um)):
        d_lo = b_idx * 0.5
        d_hi = (b_idx + 1) * 0.5
        v_cap = res_cap.variogram_bins_um[b_idx]
        v_ref = res_ref.variogram_bins_um[b_idx]
        diff = v_cap - v_ref
        pct = (diff / v_ref) * 100 if v_ref > 0 else 0.0
        print(f"  Bin {b_idx} ({d_lo:.1f}-{d_hi:.1f} mm): Cap={v_cap:6.2f} um | Ref={v_ref:6.2f} um | Delta={diff:+6.2f} um ({pct:+5.1f}%)")
    print("=" * 70 + "\n")

    summary_dict = {
        "captured": {
            "path": str(cap_file),
            "points": int(res_cap.points),
            "svr_um": float(res_cap.svr_um),
            "sa_um": float(res_cap.sa_um),
            "sq_um": float(res_cap.sq_um),
            "noise_floor_um": float(res_cap.noise_floor_um),
            "variogram_bins_um": [float(v) for v in res_cap.variogram_bins_um],
        },
        "reference_cropped": {
            "path": str(ref_file),
            "points": int(res_ref.points),
            "svr_um": float(res_ref.svr_um),
            "sa_um": float(res_ref.sa_um),
            "sq_um": float(res_ref.sq_um),
            "noise_floor_um": float(res_ref.noise_floor_um),
            "variogram_bins_um": [float(v) for v in res_ref.variogram_bins_um],
        },
        "delta": {
            "svr_um": float(delta_svr),
            "agreement_pct": float(pct_agreement),
            "height_correlation": float(elevation_corr),
            "deviation_mean_um": float(dev_stats.mean_um),
            "deviation_rms_um": float(dev_stats.rms_um),
            "deviation_std_um": float(dev_stats.std_um),
        },
    }

    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        save_ply_binary(od / "captured_aligned.ply", cap_full)
        save_ply_binary(od / "reference_cropped.ply", ref_cropped)
        (od / "verification_report.json").write_text(json.dumps(summary_dict, indent=2))
        plot_path = od / "verification_comparison.png"
        generate_verification_plot(
            res_cap, res_ref, dev_stats, plot_path,
            g_cap=g_cap, g_ref=g_ref, elevation_corr=elevation_corr,
        )
        if plot_path.exists():
            print(f"Saved verification plot: {plot_path}")
            # Also copy to top-level output/ directory if od is an output/ subdirectory
            if od.parent.name == "output":
                import shutil
                try:
                    shutil.copyfile(plot_path, od.parent / "verification_comparison.png")
                    print(f"Also copied plot to: {od.parent / 'verification_comparison.png'}")
                except Exception:
                    pass
        print(f"Exported aligned point clouds, JSON, and comparison plot to: {od}")

    return summary_dict


BASE_DIR = Path(__file__).resolve().parents[2]
REFERENCE_DIR = BASE_DIR / "data" / "reference"
DEFAULT_CAPTURED = BASE_DIR / "output" / "pointclouds" / "latest.ply"

SCRATA_SAMPLES = {
    "a1": "SCRATA_A1.pcd",
    "a2": "SCRATA_A2.pcd",
    "a3": "SCRATA_A3.pcd",
    "a4": "SCRATA_A4.pcd",
    "1": "SCRATA_A1.pcd",
    "2": "SCRATA_A2.pcd",
    "3": "SCRATA_A3.pcd",
    "4": "SCRATA_A4.pcd",
    "scrata_a1": "SCRATA_A1.pcd",
    "scrata_a2": "SCRATA_A2.pcd",
    "scrata_a3": "SCRATA_A3.pcd",
    "scrata_a4": "SCRATA_A4.pcd",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify portable scanner accuracy against reference point cloud by aligning, cropping, and dual ASTM analysis."
    )
    parser.add_argument(
        "arg1",
        nargs="?",
        default=None,
        help="SCRATA sample (A1, A2, A3, A4) or path to captured point cloud (.ply, .pcd). Defaults to A4 if omitted.",
    )
    parser.add_argument(
        "arg2",
        nargs="?",
        default=None,
        help="SCRATA sample (A1, A2, A3, A4) or path to reference point cloud (.pcd, .ply).",
    )
    parser.add_argument(
        "--sample",
        "-s",
        choices=["A1", "A2", "A3", "A4", "a1", "a2", "a3", "a4"],
        default=None,
        help="Reference SCRATA sample name (A1-A4). Defaults to A4.",
    )
    parser.add_argument(
        "--captured",
        "-c",
        default=None,
        help="Path to captured point cloud (defaults to output/pointclouds/latest.ply).",
    )
    parser.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Path to reference sample point cloud (defaults to data/reference/SCRATA_<sample>.pcd).",
    )
    parser.add_argument("--out-dir", "-o", default=None, help="Directory to save aligned PLYs, JSON, and comparison plot")
    parser.add_argument("--grid-mm", type=float, default=0.2, help="Grid cell size in mm (default: 0.2)")
    parser.add_argument("--short-cutoff-mm", type=float, default=1.0, help="Short cutoff lambda_s in mm (default: 1.0)")
    parser.add_argument("--long-cutoff-mm", type=float, default=25.0, help="Long cutoff lambda_c in mm (default: 25.0)")

    args = parser.parse_args()

    # 1. Resolve Reference Sample / Path
    ref_path: Path | None = None
    sample_key: str = "a4"

    if args.reference:
        ref_path = Path(args.reference)
    elif args.sample:
        sample_key = args.sample.lower()
        ref_path = REFERENCE_DIR / SCRATA_SAMPLES[sample_key]
    elif args.arg2:
        if args.arg2.lower() in SCRATA_SAMPLES:
            sample_key = args.arg2.lower()
            ref_path = REFERENCE_DIR / SCRATA_SAMPLES[sample_key]
        else:
            ref_path = Path(args.arg2)
    elif args.arg1 and args.arg1.lower() in SCRATA_SAMPLES:
        sample_key = args.arg1.lower()
        ref_path = REFERENCE_DIR / SCRATA_SAMPLES[sample_key]
    else:
        sample_key = "a4"
        ref_path = REFERENCE_DIR / SCRATA_SAMPLES[sample_key]

    # 2. Resolve Captured Scan Path
    cap_path: Path | None = None
    if args.captured:
        cap_path = Path(args.captured)
    elif args.arg1 and args.arg1.lower() not in SCRATA_SAMPLES:
        cap_path = Path(args.arg1)
    else:
        cap_path = DEFAULT_CAPTURED

    if not cap_path.exists():
        # Fallback to check example_files if latest.ply doesn't exist yet
        alt_example = BASE_DIR.parent / "svr-roughness" / "example_files" / f"scanner_SCRATA_{sample_key.upper()}_v3.ply"
        if alt_example.exists():
            cap_path = alt_example
        else:
            raise FileNotFoundError(
                f"Captured point cloud not found at '{cap_path}'. "
                f"Run a scan first or specify captured scan path explicitly."
            )

    if not ref_path.exists():
        # Also check SurfInspect/TestFiles as fallback if reference dir is missing
        alt_ref = BASE_DIR.parent / "SurfInspect" / "TestFiles" / SCRATA_SAMPLES.get(sample_key, "SCRATA_A4.pcd")
        if alt_ref.exists():
            ref_path = alt_ref
        else:
            raise FileNotFoundError(f"Reference point cloud not found at '{ref_path}'.")

    # Default out_dir if not specified
    out_dir = args.out_dir
    if not out_dir:
        sample_tag = sample_key.lower() if sample_key in SCRATA_SAMPLES else "sample"
        out_dir = BASE_DIR / "output" / f"verification_{sample_tag}"

    print(f"Captured Scan:    {cap_path}")
    print(f"Reference Scan:   {ref_path}")
    print(f"Output Directory: {out_dir}")

    verify_samples(
        cap_path,
        ref_path,
        out_dir=out_dir,
        grid_mm=args.grid_mm,
        short_cutoff_mm=args.short_cutoff_mm,
        long_cutoff_mm=args.long_cutoff_mm,
    )


if __name__ == "__main__":
    main()
