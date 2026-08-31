# Portable Surface Scanner

This directory is organized so the Pi can run the scanner from one stable entry point:

```bash
./run_scan.sh
```

`scan.py` waits for the serial trigger, captures the flashed DLP4500 pattern sequence, reconstructs a point cloud, and runs the surface roughness calculation through `svr_roughness.analyze_file()` from the pip-installable `svr-roughness` package.

## Runtime Layout

- `scan.py` - main closed-Pi scanner service.
- `run_scan.sh` - shell wrapper that runs `scan.py` from the project root.
- `projector/` - DLP4500 control source plus runtime binaries in `projector/bin/`.
- `arduino/` - ESP/handheld controller firmware.
- `config/` - machine-specific calibration files.
- `data/captures/latest/` - latest captured left/right pattern images.
- `data/calibration/` - accepted calibration images and calibration debug output.
- `data/focus/` - manual focus/alignment captures.
- `output/pointclouds/` - reconstructed `.ply` files.
- `output/roughness/` - roughness grids and saved analysis data.
- `output/presentation/` - presentation images/GIFs.
- `tools/runtime/` - reconstruction code called by `scan.py`.
- `tools/calibration/` - user-run calibration and calibration verification tools.
- `tools/development/` - diagnostics and development helpers.
- `tools/presentation/` - point-cloud rendering tools for slides/reports.
- `imgs/graycode_456x570_scaled_to_912x1140/` - current DLP4500 flash image set.

The normal runtime source is `scan.py`, `projector/`, `arduino/`, `tools/runtime/`, and the installed `svr-roughness` package. Everything under `tools/development/` or `tools/presentation/` is optional support tooling.

## Current Scan Defaults

The default runtime assumes the DLP4500 flash contains the 40-pattern 456 x 570 Gray-code set:

- patterns `00`-`37`: Gray-code/inverse pairs
- pattern `38`: white
- pattern `39`: black

Default reconstruction uses `config/calibration.npz`, `CALIB_ZERO_DISPARITY`, positive disparity, and threshold values that matched the latest good scan.

## Common Commands

Run a normal scan and analysis:

```bash
./run_scan.sh
```

Capture only, without reconstruction or roughness:

```bash
./run_scan.sh --no-postprocess
```

Run reconstruction manually on the latest capture:

```bash
venv/bin/python tools/runtime/reconstruct_local_gray.py
```

Install or refresh the shared roughness library during Pi setup:

```bash
venv/bin/python -m pip install ../svr-roughness
```

`scan.py` imports `svr_roughness.analyze_file()` directly during post-processing; roughness is no longer executed through a subprocess.

Recalibrate with the 10 x 7 ChArUco board:

```bash
venv/bin/python tools/calibration/calibrate_charuco.py \
  --brightness 34 \
  --exposure 250 \
  --squares-x 10 \
  --squares-y 7 \
  --square-size-mm 10 \
  --marker-size-mm 7 \
  --min-frames 15 \
  --max-frames 60
```

Verify calibration:

```bash
venv/bin/python tools/calibration/verify_charuco_calibration.py
```

## ESP Calibration Flow

The closed-Pi calibration path is in `scan.py`.

- ESP sends `CALIB_ON`: enter calibration mode.
- User places the 10 mm checkerboard in view.
- ESP sends `CALIB_TRIGGER`: capture one valid stereo calibration frame.
- ESP sends `CALIB_OFF`: compute/save calibration if enough frames were accepted.
- After calibration succeeds, `scan.py` writes `config/calibration.npz`.

Default plain-checkerboard settings are `--checkerboard 10x6 --square-size-mm 10`.

## LCD Result Flow

After a successful scan, `scan.py` sends status messages to the ESP:

- `SCANNING`
- `RECONSTRUCTING`
- `ROUGHNESS`
- `RESULT SA=<um> SQ=<um> SVR=<um>`
- `SCAN_DONE`

The LCD displays the final `RESULT` values until the next scan or calibration action.
