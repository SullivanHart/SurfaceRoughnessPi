"""
Generate Hybrid 8-Step Sinusoidal Phase-Shift + Coarse Gray-Code Patterns for DLP350/DLP4500.

Sequence Layout:
  00..07: 8-step horizontal sinusoidal phase shifts (Period P = 16 pixels)
          Harmonic-cancelling: cancels 2nd, 3rd, 4th, 5th, 6th, and 7th non-linear harmonics.
  08..19: 6-bit complementary Gray-code pairs (positive + inverse) to resolve fringe order.
  20:     All-white reference (255)
  21:     All-black reference (0)

Total: 22 patterns (half the size of the 40-pattern pure Gray-code set, with continuous sub-pixel phase).
"""

import argparse
from pathlib import Path
import json
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hybrid phase-shift + Gray-code patterns.")
    parser.add_argument("--width", type=int, default=912, help="Projector native width in pixels")
    parser.add_argument("--height", type=int, default=1140, help="Projector native height in pixels")
    parser.add_argument("--code-width", type=int, default=456,
                        help="Effective pattern width (e.g. 456 scaled to 912)")
    parser.add_argument("--code-height", type=int, default=570,
                        help="Effective pattern height (e.g. 570 scaled to 1140)")
    parser.add_argument("--period", type=int, default=16,
                        help="Sinusoidal fringe period in pixels (default: 16)")
    parser.add_argument("--num-phases", type=int, default=8,
                        help="Number of phase shift steps (default: 8 for harmonic suppression)")
    parser.add_argument("--out-dir", default="imgs/hybrid_phase_gray",
                        help="Output directory for generated BMP files")
    return parser.parse_args()


def binary_to_gray(n: int) -> int:
    return n ^ (n >> 1)


def as_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = args.code_width
    h = args.code_height
    P = args.period
    N = args.num_phases

    # Number of fringe periods across width
    num_periods = int(np.ceil(w / P))
    # Number of Gray-code bits needed to index each fringe uniquely
    gray_bits = int(np.ceil(np.log2(num_periods)))

    print(f"Configuration:")
    print(f"  Pattern size: {w} x {h} (scaled to {args.width} x {args.height})")
    print(f"  Sinusoidal period P: {P} pixels ({num_periods} periods across width)")
    print(f"  Phase shift steps N: {N}")
    print(f"  Coarse Gray-code bits: {gray_bits} ({2 * gray_bits} images for positive+inverse)")
    print(f"  Total patterns: {N + 2 * gray_bits + 2}")

    slot = 0
    manifest = {
        "projector_width": args.width,
        "projector_height": args.height,
        "code_width": w,
        "code_height": h,
        "period": P,
        "num_phases": N,
        "gray_bits": gray_bits,
        "num_periods": num_periods,
        "slots": [],
    }

    x_coords = np.arange(w, dtype=np.float64)

    # 1. Sinusoidal Phase-Shift Patterns (00..N-1)
    for k in range(N):
        delta = 2.0 * np.pi * k / N
        # Scale to [8, 247] to prevent clipping
        profile = 127.5 + 118.0 * np.cos(2.0 * np.pi * x_coords / P + delta)
        pattern = np.tile(profile.astype(np.uint8), (h, 1))

        if (w, h) != (args.width, args.height):
            pattern = cv2.resize(pattern, (args.width, args.height), interpolation=cv2.INTER_NEAREST)

        filename = f"{slot:02d}_phase_{k}_of_{N}.bmp"
        cv2.imwrite(str(out_dir / filename), as_bgr_u8(pattern))
        manifest["slots"].append({
            "index": slot,
            "type": "phase",
            "phase_step": k,
            "filename": filename,
        })
        slot += 1

    # 2. Coarse Gray-Code Patterns (positive + inverse pairs)
    fringe_indices = np.floor(x_coords / P).astype(np.int32)
    gray_words = np.array([binary_to_gray(int(k)) for k in fringe_indices], dtype=np.int32)

    for bit in range(gray_bits - 1, -1, -1):
        bit_val = ((gray_words >> bit) & 1).astype(np.uint8) * 255
        pattern_pos = np.tile(bit_val, (h, 1))
        pattern_inv = 255 - pattern_pos

        for is_inv, pat, label in [(False, pattern_pos, "pos"), (True, pattern_inv, "inv")]:
            if (w, h) != (args.width, args.height):
                pat = cv2.resize(pat, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            filename = f"{slot:02d}_gray_bit_{bit}_{label}.bmp"
            cv2.imwrite(str(out_dir / filename), as_bgr_u8(pat))
            manifest["slots"].append({
                "index": slot,
                "type": "gray",
                "bit": bit,
                "is_inverse": is_inv,
                "filename": filename,
            })
            slot += 1

    # 3. White & Black Reference Frames
    white = np.full((args.height, args.width), 255, dtype=np.uint8)
    black = np.zeros((args.height, args.width), dtype=np.uint8)

    white_name = f"{slot:02d}_white.bmp"
    cv2.imwrite(str(out_dir / white_name), as_bgr_u8(white))
    manifest["slots"].append({"index": slot, "type": "white", "filename": white_name})
    slot += 1

    black_name = f"{slot:02d}_black.bmp"
    cv2.imwrite(str(out_dir / black_name), as_bgr_u8(black))
    manifest["slots"].append({"index": slot, "type": "black", "filename": black_name})
    slot += 1

    # Save manifest files
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    manifest_txt = [
        f"projector_width: {args.width}",
        f"projector_height: {args.height}",
        f"code_width: {w}",
        f"code_height: {h}",
        f"period_pixels: {P}",
        f"phase_steps: {N}",
        f"gray_bits: {gray_bits}",
        f"total_patterns: {slot}",
        "slot_order:",
    ]
    for s in manifest["slots"]:
        manifest_txt.append(f"  {s['index']:02d}: {s['filename']}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest_txt) + "\n")

    print(f"\nSuccessfully generated {slot} pattern images in {out_dir}")
    print(f"Flash them in slot order 00..{slot - 1}")
    print(f"Manifest written to: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

