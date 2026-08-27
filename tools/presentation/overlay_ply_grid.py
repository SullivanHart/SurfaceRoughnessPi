"""
Overlay a measurement grid and/or roughness height map onto an ASCII PLY.

The scale grid is computed in the best-fit plane of the cloud. If a roughness
.npz grid from roughness_from_ply.py is supplied, each point is projected into
that saved fitted-plane coordinate system and recolored by the corresponding
plane-removed height residual.
"""

import argparse
from pathlib import Path

import numpy as np


def parse_rgb(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected R,G,B")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise argparse.ArgumentTypeError("RGB values must be 0-255")
    return rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="Input ASCII PLY")
    parser.add_argument("--out", default=None, help="Output PLY")
    parser.add_argument("--grid-mm", type=float, default=5.0, help="Grid pitch in fitted-plane millimeters")
    parser.add_argument("--line-width-mm", type=float, default=0.25, help="Grid line half-width in millimeters")
    parser.add_argument("--show-grid-lines", action="store_true", help="Draw scale grid lines on top of the colors")
    parser.add_argument("--major-every", type=int, default=5, help="Make every Nth grid line a major line")
    parser.add_argument("--grid-color", type=parse_rgb, default=(0, 255, 255), help="Minor grid color as R,G,B")
    parser.add_argument("--major-color", type=parse_rgb, default=(255, 64, 64), help="Major grid color as R,G,B")
    parser.add_argument("--origin", choices=("center", "min"), default="center",
                        help="Grid origin in fitted-plane coordinates")
    parser.add_argument("--darken", type=float, default=0.55,
                        help="Darken non-grid colors by this multiplier; 1 keeps original brightness")
    parser.add_argument("--roughness-grid", default=None,
                        help=".npz grid written by roughness_from_ply.py --save-grid")
    parser.add_argument("--roughness-field", choices=("grid_filtered", "grid_filled", "grid_raw"),
                        default="grid_filtered")
    parser.add_argument("--roughness-valid", choices=("valid_filled", "valid_raw"),
                        default="valid_filled")
    parser.add_argument("--roughness-clip-um", type=float, default=0.0,
                        help="Residual color half-range in micrometers; 0 uses robust percentiles")
    parser.add_argument("--missing-color", type=parse_rgb, default=(45, 45, 45),
                        help="Color for points outside the roughness grid")
    return parser.parse_args()


def read_ascii_ply(path):
    header = []
    vertex_count = None
    properties = []
    with open(path) as f:
        first = f.readline().rstrip("\n")
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        header.append(first)
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            stripped = line.rstrip("\n")
            header.append(stripped)
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property ") and vertex_count is not None:
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY vertex count not found")
        data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, properties, data


def ensure_rgb_properties(header, properties, data):
    if data.shape[1] < 3:
        raise ValueError("PLY needs at least x y z columns")
    lower = [p.lower() for p in properties]
    has_rgb = all(name in lower for name in ("red", "green", "blue"))
    if has_rgb:
        return header, properties, data

    insert_at = header.index("end_header")
    rgb_header = [
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    header = header[:insert_at] + rgb_header + header[insert_at:]
    properties = properties + ["red", "green", "blue"]
    gray = np.full((len(data), 3), 180.0, dtype=np.float64)
    data = np.column_stack((data[:, :3], gray, data[:, 3:]))
    return header, properties, data


def property_indices(properties):
    lower = [p.lower() for p in properties]
    return {
        "x": lower.index("x"),
        "y": lower.index("y"),
        "z": lower.index("z"),
        "red": lower.index("red"),
        "green": lower.index("green"),
        "blue": lower.index("blue"),
    }


def fit_plane_coords(points):
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
    return np.column_stack((centered @ x_axis, centered @ y_axis, centered @ normal))


def project_with_saved_basis(points, grid_data):
    centroid = grid_data["plane_centroid"].astype(np.float64)
    x_axis = grid_data["plane_x_axis"].astype(np.float64)
    y_axis = grid_data["plane_y_axis"].astype(np.float64)
    normal = grid_data["plane_normal"].astype(np.float64)
    centered = points - centroid
    return np.column_stack((centered @ x_axis, centered @ y_axis, centered @ normal))


def residual_colors(residual_mm, clip_um):
    residual_um = residual_mm * 1000.0
    if clip_um <= 0:
        finite = np.isfinite(residual_um)
        if finite.any():
            clip_um = float(np.nanpercentile(np.abs(residual_um[finite]), 98))
        else:
            clip_um = 1.0
    clip_um = max(float(clip_um), 1e-6)
    t = np.clip((residual_um + clip_um) / (2.0 * clip_um), 0.0, 1.0)
    stops = np.array([
        [49, 54, 149],
        [69, 117, 180],
        [224, 243, 248],
        [255, 255, 191],
        [253, 174, 97],
        [215, 48, 39],
    ], dtype=np.float64)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(np.int32), 0, len(stops) - 2)
    frac = (x - i)[:, None]
    colors = stops[i] * (1.0 - frac) + stops[i + 1] * frac
    return colors.clip(0, 255), clip_um


def apply_roughness_colors(data, idx, points, args):
    grid_data = np.load(args.roughness_grid)
    grid = grid_data[args.roughness_field].astype(np.float64)
    valid = grid_data[args.roughness_valid].astype(bool)
    x0, y0, pitch = grid_data["grid_origin"].astype(np.float64)

    coords = project_with_saved_basis(points, grid_data)
    ix = np.floor((coords[:, 0] - x0) / pitch).astype(np.int64)
    iy = np.floor((coords[:, 1] - y0) / pitch).astype(np.int64)
    in_bounds = (ix >= 0) & (iy >= 0) & (ix < grid.shape[1]) & (iy < grid.shape[0])
    good = np.zeros(len(points), dtype=bool)
    good[in_bounds] = valid[iy[in_bounds], ix[in_bounds]] & np.isfinite(grid[iy[in_bounds], ix[in_bounds]])

    residual = np.full(len(points), np.nan, dtype=np.float64)
    residual[good] = grid[iy[good], ix[good]]

    colors, clip_um = residual_colors(residual[good], args.roughness_clip_um)
    rgb_indices = [idx["red"], idx["green"], idx["blue"]]
    data[:, rgb_indices] = np.array(args.missing_color, dtype=np.float64)
    data[np.ix_(good, rgb_indices)] = colors

    print(f"roughness grid: {args.roughness_grid}")
    print(f"roughness field: {args.roughness_field}")
    print(f"roughness mapped points: {int(good.sum())}/{len(points)}")
    print(f"roughness color scale: +/-{clip_um:.1f} um")
    return coords[:, 0], coords[:, 1]


def distance_to_grid(values, origin, pitch):
    shifted = values - origin
    nearest = np.round(shifted / pitch) * pitch
    return np.abs(shifted - nearest)


def grid_index(values, origin, pitch):
    return np.round((values - origin) / pitch).astype(np.int64)


def write_ascii_ply(path, header, properties, data):
    idx = property_indices(properties)
    rgb_cols = {idx["red"], idx["green"], idx["blue"]}
    with open(path, "w") as f:
        for line in header:
            f.write(line + "\n")
        for row in data:
            parts = []
            for col, value in enumerate(row):
                if col in rgb_cols:
                    parts.append(str(int(np.clip(round(value), 0, 255))))
                else:
                    parts.append(f"{float(value):.6f}")
            f.write(" ".join(parts) + "\n")


def main():
    args = parse_args()
    if args.grid_mm <= 0:
        raise ValueError("--grid-mm must be positive")
    if args.line_width_mm <= 0:
        raise ValueError("--line-width-mm must be positive")

    in_path = Path(args.ply)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_grid.ply")
    header, properties, data = read_ascii_ply(in_path)
    header, properties, data = ensure_rgb_properties(header, properties, data)
    idx = property_indices(properties)

    points = data[:, [idx["x"], idx["y"], idx["z"]]]
    finite = np.isfinite(points).all(axis=1)
    x = np.full(len(data), np.nan, dtype=np.float64)
    y = np.full(len(data), np.nan, dtype=np.float64)
    if args.roughness_grid:
        rough_x, rough_y = apply_roughness_colors(data, idx, points, args)
        x[finite] = rough_x[finite]
        y[finite] = rough_y[finite]
    else:
        coords = fit_plane_coords(points[finite])
        x[finite] = coords[:, 0]
        y[finite] = coords[:, 1]

    if args.origin == "center":
        origin_x = 0.0
        origin_y = 0.0
    else:
        origin_x = np.nanmin(x)
        origin_y = np.nanmin(y)

    rgb_indices = [idx["red"], idx["green"], idx["blue"]]
    if args.darken != 1.0 and not args.roughness_grid:
        data[:, rgb_indices] = np.clip(data[:, rgb_indices] * args.darken, 0, 255)

    minor_count = 0
    major_count = 0
    if args.show_grid_lines:
        near_x = distance_to_grid(x, origin_x, args.grid_mm) <= args.line_width_mm
        near_y = distance_to_grid(y, origin_y, args.grid_mm) <= args.line_width_mm
        grid_mask = finite & (near_x | near_y)

        ix = grid_index(x, origin_x, args.grid_mm)
        iy = grid_index(y, origin_y, args.grid_mm)
        major_mask = grid_mask & (
            (np.mod(np.abs(ix), max(args.major_every, 1)) == 0)
            | (np.mod(np.abs(iy), max(args.major_every, 1)) == 0)
        )
        minor_mask = grid_mask & ~major_mask
        data[np.ix_(minor_mask, rgb_indices)] = np.array(args.grid_color, dtype=np.float64)
        data[np.ix_(major_mask, rgb_indices)] = np.array(args.major_color, dtype=np.float64)
        minor_count = int(minor_mask.sum())
        major_count = int(major_mask.sum())

    write_ascii_ply(out_path, header, properties, data)
    print(f"input points: {len(data)}")
    print(f"grid pitch: {args.grid_mm:g} mm")
    print(f"minor grid points: {minor_count}")
    print(f"major grid points: {major_count}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
