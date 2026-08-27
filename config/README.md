# Config

This directory holds machine-specific scanner configuration.

Expected runtime file:

- `calibration.npz` - current stereo calibration used by `scan.py`.

Calibration files are local artifacts. Regenerate them with:

```bash
venv/bin/python tools/calibration/calibrate_charuco.py --squares-x 10 --squares-y 7 --square-size-mm 10 --marker-size-mm 7
```

The closed-Pi runtime can also write this file from `scan.py` when the ESP sends the calibration toggle command.
