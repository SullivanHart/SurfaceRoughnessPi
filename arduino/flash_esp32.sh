#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${1:-/dev/ttyUSB0}"

echo "=================================================="
echo " Flashing ESP32 LCD Controller Firmware"
echo "=================================================="

# Check if scanner.service is holding the serial port
STOPPED_SERVICE=false
if command -v systemctl &>/dev/null && systemctl is-active --quiet scanner.service 2>/dev/null; then
    echo "Stopping scanner.service to release ${PORT}..."
    sudo systemctl stop scanner.service
    STOPPED_SERVICE=true
fi

# Ensure service is restored on exit if we stopped it
restore_service() {
    if [ "$STOPPED_SERVICE" = true ]; then
        echo "Restarting scanner.service..."
        sudo systemctl start scanner.service || true
    fi
}
trap restore_service EXIT

# Find PlatformIO executable
PIO_BIN=""
if command -v pio &>/dev/null; then
    PIO_BIN="pio"
elif command -v platformio &>/dev/null; then
    PIO_BIN="platformio"
elif [ -f "${HOME}/.platformio/penv/bin/pio" ]; then
    PIO_BIN="${HOME}/.platformio/penv/bin/pio"
elif [ -f "${HOME}/.local/bin/pio" ]; then
    PIO_BIN="${HOME}/.local/bin/pio"
elif [ -f "${ROOT_DIR}/venv/bin/pio" ]; then
    PIO_BIN="${ROOT_DIR}/venv/bin/pio"
fi

if [ -z "${PIO_BIN}" ]; then
    echo "PlatformIO not found. Installing into virtual environment..."
    if [ -f "${ROOT_DIR}/venv/bin/python" ]; then
        "${ROOT_DIR}/venv/bin/python" -m pip install platformio
        PIO_BIN="${ROOT_DIR}/venv/bin/pio"
    elif command -v pip3 &>/dev/null; then
        pip3 install --user platformio
        PIO_BIN="${HOME}/.local/bin/pio"
    else
        echo "ERROR: Could not find or install PlatformIO."
        echo "Please install it with: pip install platformio"
        exit 1
    fi
fi

echo "Using PlatformIO: ${PIO_BIN}"
echo "Target port: ${PORT}"

# Compile and upload
"${PIO_BIN}" run -d "${SCRIPT_DIR}" -t upload --upload-port "${PORT}"

echo "=================================================="
echo " ESP32 firmware flash completed successfully!"
echo "=================================================="

