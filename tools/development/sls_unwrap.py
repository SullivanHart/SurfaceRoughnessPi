"""
Decode and inspect the 16-image sinusoidal structured-light captures.

This is intentionally reconstruction-free: it only verifies that the projected
column coordinate unwraps cleanly in each camera. Use it before stereo matching.

Expected image order:
  0- 3: period=14 horizontal phase shifts 0, pi/2, pi, 3pi/2
  4- 7: period=14 vertical phase shifts 0, pi/2, pi, 3pi/2
  8-11: period=48 horizontal phase shifts 0, pi/2, pi, 3pi/2
 12-15: period=48 vertical phase shifts 0, pi/2, pi, 3pi/2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="./caps", help="Directory containing left/ and right/")
    parser.add_argument("--left-dir", default=None, help="Override left capture directory")
    parser.add_argument("--right-dir", default=None, help="Override right capture directory")
    parser.add_argument("--glob", default="*.png", help="Image glob used inside each camera directory")
    parser.add_argument("--out-dir", default="./debug_unwrap", help="Output directory for maps and stats")
    parser.add_argument("--hi-period", type=float, default=14.0)
    parser.add_argument("--lo-period", type=float, default=48.0)
    parser.add_argument("--mod-thresh", type=float, default=20.0)
    parser.add_argument("--smooth", type=int, default=7, help="Odd circular-mean filter size for low phase")
    return parser.parse_args()


def load_gray_sequence(directory, glob):
    paths = sorted(Path(directory).glob(glob))
    if len(paths) < 16:
        raise ValueError(f"{directory} has {len(paths)} images matching {glob}; need at least 16")

    images = []
    for path in paths[:16]:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        images.append(img.astype(np.float32))

    shape = images[0].shape
    bad = [str(paths[i]) for i, img in enumerate(images) if img.shape != shape]
    if bad:
        raise ValueError(f"All images must share shape {shape}; mismatched: {bad[:3]}")

    return paths[:16], images


def decode_phase(I0, I1, I2, I3):
    return np.arctan2(I3 - I1, I0 - I2)


def phase_modulation(I0, I1, I2, I3):
    return np.sqrt((I3 - I1) ** 2 + (I0 - I2) ** 2) / 2


def circular_smooth(phi, size):
    if size <= 1:
        return phi
    return np.arctan2(
        uniform_filter(np.sin(phi).astype(np.float32), size=size),
        uniform_filter(np.cos(phi).astype(np.float32), size=size),
    )


def abs_projector_coordinate(phi_hi, phi_lo_s, imgs_hi, imgs_lo, hi_period, lo_period, mod_thresh):
    mod_hi = phase_modulation(*imgs_hi)
    mod_lo = phase_modulation(*imgs_lo)
    valid_mask = (mod_hi >= mod_thresh) & (mod_lo >= mod_thresh)
    full_period = int(round(np.lcm(int(round(hi_period)), int(round(lo_period)))))
    abs_col = np.full(phi_hi.shape, np.nan, dtype=np.float32)

    # This CRT shortcut is valid for the default 14/48 periods:
    # c = a + 48t, c = b mod 14, and 48 == 6 mod 14.
    if int(round(hi_period)) != 14 or int(round(lo_period)) != 48:
        raise ValueError("Only the current 14/48 period pair is supported for CRT anchoring")

    segment_count = 0
    for row in range(phi_hi.shape[0]):
        x_valid = np.where(valid_mask[row])[0]
        if x_valid.size < 2:
            continue
        for segment in np.split(x_valid, np.where(np.diff(x_valid) > 1)[0] + 1):
            if segment.size < 2:
                continue

            phi_v = phi_hi[row, segment]
            phi_lo_v = phi_lo_s[row, segment]
            mh = mod_hi[row, segment]
            ml = mod_lo[row, segment]

            phi_abs = np.unwrap(phi_v)
            seed = int(np.argmax(mh * ml))
            a = (phi_lo_v[seed] * (lo_period / (2 * np.pi))) % lo_period
            b = (phi_v[seed] * (hi_period / (2 * np.pi))) % hi_period
            k = int(round((b - a) / 2))
            t = (5 * k) % 7
            c_seed = (a + lo_period * t) % full_period

            abs_col[row, segment] = (
                (phi_abs - phi_abs[seed]) * (hi_period / (2 * np.pi)) + c_seed
            ).astype(np.float32)
            segment_count += 1

    return abs_col, valid_mask, mod_hi, mod_lo, full_period, segment_count


def write_u8(path, image):
    cv2.imwrite(str(path), image.clip(0, 255).astype(np.uint8))


def save_debug(name, out_dir, imgs, args):
    phi_hi = decode_phase(*imgs[0:4])
    phi_lo = decode_phase(*imgs[8:12])
    phi_lo_s = circular_smooth(phi_lo, args.smooth)
    abs_col, valid, mod_hi, mod_lo, full_period, segments = abs_projector_coordinate(
        phi_hi, phi_lo_s, imgs[0:4], imgs[8:12],
        args.hi_period, args.lo_period, args.mod_thresh,
    )

    cam_dir = out_dir / name
    cam_dir.mkdir(parents=True, exist_ok=True)
    np.save(cam_dir / "abs_col.npy", abs_col)
    np.save(cam_dir / "valid_mask.npy", valid)

    write_u8(cam_dir / "wrapped_hi.png", (phi_hi + np.pi) / (2 * np.pi) * 255)
    write_u8(cam_dir / "wrapped_lo.png", (phi_lo + np.pi) / (2 * np.pi) * 255)
    write_u8(cam_dir / "mod_hi.png", mod_hi)
    write_u8(cam_dir / "mod_lo.png", mod_lo)
    write_u8(cam_dir / "valid_mask.png", valid.astype(np.uint8) * 255)

    abs_vis = np.zeros_like(abs_col, dtype=np.float32)
    finite = np.isfinite(abs_col)
    abs_vis[finite] = (abs_col[finite] % full_period) / full_period * 255
    write_u8(cam_dir / "abs_col_mod.png", abs_vis)

    finite_count = int(finite.sum())
    stats = {
        "shape": f"{abs_col.shape[1]}x{abs_col.shape[0]}",
        "finite_pixels": finite_count,
        "finite_fraction": finite_count / abs_col.size,
        "valid_mask_fraction": float(valid.mean()),
        "segments": segments,
        "mod_hi_p05_p50_p95": np.percentile(mod_hi, [5, 50, 95]).tolist(),
        "mod_lo_p05_p50_p95": np.percentile(mod_lo, [5, 50, 95]).tolist(),
    }
    if finite_count:
        stats["abs_col_p01_p50_p99"] = np.percentile(abs_col[finite], [1, 50, 99]).tolist()

    with open(cam_dir / "stats.txt", "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    return stats


def main():
    args = parse_args()
    caps = Path(args.caps)
    left_dir = Path(args.left_dir) if args.left_dir else caps / "left"
    right_dir = Path(args.right_dir) if args.right_dir else caps / "right"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_paths, left_imgs = load_gray_sequence(left_dir, args.glob)
    right_paths, right_imgs = load_gray_sequence(right_dir, args.glob)

    print("Loaded left:")
    for p in left_paths:
        print(f"  {p}")
    print("Loaded right:")
    for p in right_paths:
        print(f"  {p}")

    left_stats = save_debug("left", out_dir, left_imgs, args)
    right_stats = save_debug("right", out_dir, right_imgs, args)

    print("\nUnwrap stats:")
    for label, stats in [("left", left_stats), ("right", right_stats)]:
        print(f"  {label}: finite={stats['finite_pixels']} ({stats['finite_fraction'] * 100:.1f}%), "
              f"valid_mask={stats['valid_mask_fraction'] * 100:.1f}%, segments={stats['segments']}")
    print(f"\nWrote debug maps to {out_dir}")


if __name__ == "__main__":
    main()
