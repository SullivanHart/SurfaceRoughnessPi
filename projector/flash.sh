#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FIRMWARE="${1:-${ROOT_DIR}/firmware/DLPR350PROM_current.bin}"

if [ ! -f "${FIRMWARE}" ]; then
    echo "ERROR: Firmware binary not found at '${FIRMWARE}'"
    echo "Usage: ./projector/flash.sh [path/to/firmware.bin]"
    exit 1
fi

echo "=================================================="
echo " Preparing DLPC350 Flasher"
echo "=================================================="

# Compile flash_firmware if binary does not exist
if [ ! -f "${SCRIPT_DIR}/bin/flash_firmware" ]; then
    echo "Compiling flash_firmware..."
    mkdir -p "${SCRIPT_DIR}/build"
    cd "${SCRIPT_DIR}/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target flash_firmware
    cd "${ROOT_DIR}"
fi

echo "Target Firmware: ${FIRMWARE}"
exec "${SCRIPT_DIR}/bin/flash_firmware" "${FIRMWARE}"
