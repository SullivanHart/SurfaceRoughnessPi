# Tools

Tools are grouped by whether they are part of normal scanner operation.

## Runtime

`tools/runtime/` is called by `scan.py` after every successful scan.

- `reconstruct_local_gray.py` - decodes the captured Gray-code sequence and writes the point cloud.

## Calibration

`tools/calibration/` is for user-run calibration and verification.

- `calibrate_charuco.py` - standalone ChArUco calibration flow.
- `calibrate.py` - standalone plain-checkerboard calibration flow.
- `verify_charuco_calibration.py` and `verify_calibration.py` - calibration checks.
- board generation/inspection scripts are kept here because they support calibration setup.

The closed-Pi path also supports calibration through `scan.py`: the ESP sends `CALIBRATE` to enter calibration mode, `TRIGGER` to accept frames, and `CALIBRATE` again to exit without saving.

## Development

`tools/development/` contains diagnostics and experiments used while tuning the scanner. These are not required for a normal closed-Pi scan.

Examples: focus/sharpness checks, plane analysis, sample-data reconstruction, HDR capture fusion, and pattern generation.

## Presentation

`tools/presentation/` renders point clouds for reports and slides.
