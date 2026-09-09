"""
flash_projector.py — Flash firmware binary onto LightCrafter 4500 (DLPC350).

Usage:
    python tools/development/flash_projector.py [firmware_path]
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECTOR_DIR = BASE_DIR / "projector"
BINARY = PROJECTOR_DIR / "bin" / "flash_firmware"
DEFAULT_FIRMWARE = BASE_DIR / "firmware" / "DLPR350PROM_current.bin"

def main():
    firmware = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_FIRMWARE

    if not firmware.exists():
        print(f"ERROR: Firmware file not found: {firmware}")
        sys.exit(1)

    # Build binary if needed
    if not BINARY.exists():
        print("Building flash_firmware utility...")
        build_dir = PROJECTOR_DIR / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True)
        subprocess.run(["cmake", "--build", ".", "--target", "flash_firmware"], cwd=build_dir, check=True)

    print(f"Flashing firmware: {firmware}")
    cmd = [str(BINARY), str(firmware)]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
