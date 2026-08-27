cmake_minimum_required(VERSION 3.10)
project(projector_ctl CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── Dependencies ──────────────────────────────────────────────────────────────
# Install on the Pi with:
#   sudo apt install libhidapi-dev libusb-1.0-0-dev libudev-dev cmake build-essential

# Use the system hidapi (avoids bundling hid.c and its relative-path issues).
# hidapi-hidraw is preferred on Linux; falls back to plain hidapi if not found.
find_library(HIDAPI_LIB NAMES hidapi-hidraw hidapi REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
# Only the two DLPC350 translation units are needed — no OpenCV, no full SDK.
add_executable(projector_ctl
    projector_ctl.cpp
    sdk/dlpc350_api.cpp
    sdk/dlpc350_usb.cpp
)

target_include_directories(projector_ctl PRIVATE sdk/include)
target_link_libraries(projector_ctl ${HIDAPI_LIB})

add_executable(set_brightness
    set_brightness.cpp
    sdk/dlpc350_api.cpp
    sdk/dlpc350_usb.cpp
)

target_include_directories(set_brightness PRIVATE sdk/include)
target_link_libraries(set_brightness ${HIDAPI_LIB})

add_executable(calib_test
    calib_test.cpp
    sdk/dlpc350_api.cpp
    sdk/dlpc350_usb.cpp
)

target_include_directories(calib_test PRIVATE sdk/include)
target_link_libraries(calib_test ${HIDAPI_LIB})

add_executable(projector_diag
    projector_diag.cpp
    sdk/dlpc350_api.cpp
    sdk/dlpc350_usb.cpp
)

target_include_directories(projector_diag PRIVATE sdk/include)
target_link_libraries(projector_diag ${HIDAPI_LIB})

add_executable(seq_test
    seq_test.cpp
    sdk/dlpc350_api.cpp
    sdk/dlpc350_usb.cpp
)

target_include_directories(seq_test PRIVATE sdk/include)
target_link_libraries(seq_test ${HIDAPI_LIB})
