"""
Download OpenCV's structured-light sample stereo dataset.

The dataset contains two camera views of a Gray-code projection sequence plus
OpenCV stereo calibration parameters. It is useful as a known-good 3D
reconstruction smoke test before integrating local camera/projector capture.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/"
    "testdata/cv/structured_light/data"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sample_data/opencv_structured_light")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["calibrationParameters.yml"]
    names += [f"pattern_cam1_im{i}.jpg" for i in range(1, 45)]
    names += [f"pattern_cam2_im{i}.jpg" for i in range(1, 45)]

    for name in names:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"exists  {dst}")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"fetch   {url}")
        urlretrieve(url, dst)

    print(f"\nDownloaded {len(names)} files to {out_dir}")


if __name__ == "__main__":
    main()
