"""
Render an ASCII PLY point cloud as a rotating presentation GIF.

This avoids interactive OpenGL viewers and only depends on numpy + Pillow.
It is intended for scanner output PLY files written by reconstruct_local_gray.py.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_size(text):
    if "x" in text:
        w, h = text.lower().split("x", 1)
        return int(w), int(h)
    size = int(text)
    return size, size


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--out", default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--size", type=parse_size, default=(900, 700),
                        help="Output size, e.g. 900x700 or 800")
    parser.add_argument("--duration-ms", type=int, default=90)
    parser.add_argument("--max-points", type=int, default=400000,
                        help="Downsample large clouds for faster GIF generation")
    parser.add_argument("--elevation", type=float, default=28.0,
                        help="Camera elevation angle in degrees")
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--color", choices=("height", "ply", "gray"), default="height")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_ascii_ply(path):
    properties = []
    vertex_count = None
    with open(path) as f:
        if f.readline().strip() != "ply":
            raise ValueError("Not a PLY file")
        in_vertex = False
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            elif in_vertex and line.startswith("property"):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY vertex count not found")
        data = np.loadtxt(f, max_rows=vertex_count, dtype=np.float32)

    if data.ndim == 1:
        data = data.reshape(1, -1)
    idx = {name: i for i, name in enumerate(properties)}
    points = data[:, [idx["x"], idx["y"], idx["z"]]].astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    data = data[finite]

    rgb = None
    if all(name in idx for name in ("red", "green", "blue")):
        rgb = data[:, [idx["red"], idx["green"], idx["blue"]]].clip(0, 255).astype(np.uint8)
    return points, rgb


def height_colors(points):
    z = points[:, 2]
    lo, hi = np.percentile(z, [2, 98])
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)

    stops = np.array([
        [35, 32, 95],
        [33, 145, 140],
        [218, 224, 88],
        [244, 109, 67],
    ], dtype=np.float32)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(np.int32), 0, len(stops) - 2)
    frac = (x - i)[:, None]
    colors = stops[i] * (1 - frac) + stops[i + 1] * frac
    return colors.clip(0, 255).astype(np.uint8)


def rotation_matrix(elevation_deg, azimuth_deg):
    elev = np.deg2rad(elevation_deg)
    az = np.deg2rad(azimuth_deg)
    rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(elev), -np.sin(elev)],
        [0, np.sin(elev), np.cos(elev)],
    ], dtype=np.float32)
    return rx @ rz


def render(points, colors, width, height, elevation, azimuth, point_size, background):
    bg = 255 if background == "white" else 0
    image = np.full((height, width, 3), bg, dtype=np.uint8)
    zbuf = np.full((height, width), -np.inf, dtype=np.float32)

    cam = points @ rotation_matrix(elevation, azimuth).T
    radius = np.percentile(np.linalg.norm(points[:, :3], axis=1), 99)
    scale = 0.44 * min(width, height) / max(radius, 1e-6)
    sx = (cam[:, 0] * scale + width / 2).astype(np.int32)
    sy = (-cam[:, 1] * scale + height / 2).astype(np.int32)
    depth = cam[:, 2]

    offsets = [(0, 0)]
    if point_size >= 2:
        r = point_size // 2
        offsets = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    for dx, dy in offsets:
        x = sx + dx
        y = sy + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not inside.any():
            continue
        x_i = x[inside]
        y_i = y[inside]
        d_i = depth[inside]
        c_i = colors[inside]
        lin = y_i * width + x_i
        order = np.lexsort((d_i, lin))
        lin_sorted = lin[order]
        unique, last = np.unique(lin_sorted, return_index=True)
        last = np.r_[last[1:] - 1, len(order) - 1]
        chosen = order[last]
        x_c = x_i[chosen]
        y_c = y_i[chosen]
        d_c = d_i[chosen]
        update = d_c > zbuf[y_c, x_c]
        image[y_c[update], x_c[update]] = c_i[chosen][update]
        zbuf[y_c[update], x_c[update]] = d_c[update]

    return Image.fromarray(image)


def main():
    args = parse_args()
    width, height = args.size
    points, ply_rgb = load_ascii_ply(args.ply)
    points = points - points.mean(axis=0)

    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        if ply_rgb is not None:
            ply_rgb = ply_rgb[keep]

    if args.color == "height":
        colors = height_colors(points)
    elif args.color == "ply" and ply_rgb is not None:
        colors = ply_rgb
    else:
        gray = height_colors(points).mean(axis=1).astype(np.uint8)
        colors = np.repeat(gray[:, None], 3, axis=1)

    out = Path(args.out) if args.out else Path(args.ply).with_suffix(".gif")
    print(f"Rendering {len(points)} points to {out} ({args.frames} frames)")

    frames = []
    for idx in range(args.frames):
        azimuth = 360.0 * idx / args.frames
        frames.append(render(
            points,
            colors,
            width,
            height,
            args.elevation,
            azimuth,
            args.point_size,
            args.background,
        ))
        if (idx + 1) % 10 == 0 or idx + 1 == args.frames:
            print(f"  frame {idx + 1}/{args.frames}")

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
