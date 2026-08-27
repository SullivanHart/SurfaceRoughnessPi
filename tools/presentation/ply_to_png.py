"""
Render an ASCII PLY point cloud as a static presentation PNG.

Uses the lightweight renderer from ply_to_gif.py, so it only needs numpy + Pillow.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from ply_to_gif import height_colors, load_ascii_ply, parse_size, render


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply")
    parser.add_argument("--out", default=None)
    parser.add_argument("--size", type=parse_size, default=(1200, 900),
                        help="Output size, e.g. 1200x900 or 1000")
    parser.add_argument("--max-points", type=int, default=700000)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--point-size", type=int, default=3)
    parser.add_argument("--color", choices=("height", "residual", "ply", "gray"), default="residual")
    parser.add_argument("--background", choices=("white", "black"), default="black")
    parser.add_argument("--residual-clip-um", type=float, default=350.0,
                        help="Residual color scale half-range in micrometers")
    parser.add_argument("--flatten", action="store_true",
                        help="Render plane-removed residual surface with Z exaggerated")
    parser.add_argument("--z-exaggeration", type=float, default=25.0,
                        help="Z scale multiplier when --flatten is used")
    parser.add_argument("--label", default=None)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


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
    return coords


def residual_colors(residual_mm, clip_um):
    residual_um = residual_mm * 1000.0
    t = np.clip((residual_um + clip_um) / max(2 * clip_um, 1e-6), 0, 1)
    stops = np.array([
        [49, 54, 149],
        [69, 117, 180],
        [224, 243, 248],
        [255, 255, 191],
        [253, 174, 97],
        [215, 48, 39],
    ], dtype=np.float32)
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(np.int32), 0, len(stops) - 2)
    frac = (x - i)[:, None]
    colors = stops[i] * (1 - frac) + stops[i + 1] * frac
    return colors.clip(0, 255).astype(np.uint8)


def add_label(image, text, background):
    if not text:
        return image
    draw = ImageDraw.Draw(image)
    fill = (255, 255, 255) if background == "black" else (0, 0, 0)
    shadow = (0, 0, 0) if background == "black" else (255, 255, 255)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font = None
    x, y = 24, 22
    draw.text((x + 2, y + 2), text, fill=shadow, font=font)
    draw.text((x, y), text, fill=fill, font=font)
    return image


def main():
    args = parse_args()
    width, height = args.size
    points, ply_rgb = load_ascii_ply(args.ply)
    plane_coords = fit_plane_basis(points)
    render_points = points - points.mean(axis=0)

    if args.flatten:
        render_points = plane_coords.copy()
        render_points[:, 2] *= args.z_exaggeration

    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(points), size=args.max_points, replace=False)
        points = points[keep]
        render_points = render_points[keep]
        plane_coords = plane_coords[keep]
        if ply_rgb is not None:
            ply_rgb = ply_rgb[keep]

    if args.color == "residual":
        colors = residual_colors(plane_coords[:, 2], args.residual_clip_um)
    elif args.color == "height":
        colors = height_colors(render_points)
    elif args.color == "ply" and ply_rgb is not None:
        colors = ply_rgb
    else:
        gray = height_colors(points).mean(axis=1).astype(np.uint8)
        colors = np.repeat(gray[:, None], 3, axis=1)

    out = Path(args.out) if args.out else Path(args.ply).with_suffix(".png")
    print(f"Rendering {len(points)} points to {out}")
    image = render(
        render_points,
        colors,
        width,
        height,
        args.elevation,
        args.azimuth,
        args.point_size,
        args.background,
    )
    if args.color == "residual" and args.label is None:
        args.label = f"Plane residual color scale: +/-{args.residual_clip_um:g} um"
    image = add_label(image, args.label, args.background)
    image.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
