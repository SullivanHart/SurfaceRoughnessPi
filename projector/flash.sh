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

# Locate or compile flash_firmware
BIN=""
if [ -f "${SCRIPT_DIR}/bin/flash_firmware" ]; then
    BIN="${SCRIPT_DIR}/bin/flash_firmware"
elif [ -f "${SCRIPT_DIR}/build/bin/flash_firmware" ]; then
    BIN="${SCRIPT_DIR}/build/bin/flash_firmware"
elif [ -f "${SCRIPT_DIR}/build/flash_firmware" ]; then
    BIN="${SCRIPT_DIR}/build/flash_firmware"
else
    echo "Compiling flash_firmware..."
    mkdir -p "${SCRIPT_DIR}/build"
    cd "${SCRIPT_DIR}/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target flash_firmware
    cd "${ROOT_DIR}"
    if [ -f "${SCRIPT_DIR}/build/bin/flash_firmware" ]; then
        BIN="${SCRIPT_DIR}/build/bin/flash_firmware"
    elif [ -f "${SCRIPT_DIR}/build/flash_firmware" ]; then
        BIN="${SCRIPT_DIR}/build/flash_firmware"
    elif [ -f "${SCRIPT_DIR}/bin/flash_firmware" ]; then
        BIN="${SCRIPT_DIR}/bin/flash_firmware"
    fi
fi

if [ -z "${BIN}" ] || [ ! -f "${BIN}" ]; then
    echo "ERROR: Could not find compiled flash_firmware binary!"
    exit 1
fi

# Mirror into projector/bin for consistency
mkdir -p "${SCRIPT_DIR}/bin"
cp -f "${BIN}" "${SCRIPT_DIR}/bin/flash_firmware" 2>/dev/null || true

echo "Target Firmware: ${FIRMWARE}"
exec "${BIN}" "${FIRMWARE}"
