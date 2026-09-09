from pypylon import pylon
import serial
from serial.threaded import LineReader, ReaderThread
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import argparse
import json
import subprocess
import RPi.GPIO as GPIO

BASE_DIR = Path(__file__).resolve().parent
TOOLS_DIR = BASE_DIR / "tools"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

sys.path.insert(0, str(BASE_DIR / "projector"))
import projector as proj


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


parser = argparse.ArgumentParser()
parser.add_argument("--brightness", type=int, default=60, metavar="0-255")
parser.add_argument("--exposure", type=int, default=550, metavar="US")
parser.add_argument("--min-mod", type=float, default=1.2,
                    help="Minimum phase modulation threshold for phase reconstruction (default: 1.2)")
parser.add_argument("--min-contrast", type=float, default=2.0,
                    help="Minimum white-black intensity contrast for phase reconstruction (default: 2.0)")
parser.add_argument("--proj-pattern-ms", type=int, default=150,
                    help="Projector display exposure per pattern in milliseconds (default: 150ms -> 2.9s scan)")
parser.add_argument("--patterns", type=int, default=40, help="Number of scan patterns to capture")
parser.add_argument("--out-dir", default="data/captures/latest", help="Directory to write captured left/ and right/ images")
parser.add_argument("--calib", default="config/calibration.npz")
parser.add_argument("--recon-out", default="output/pointclouds/latest.ply")
parser.add_argument("--proj-width", type=int, default=456)
parser.add_argument("--proj-height", type=int, default=570)
parser.add_argument("--flash-proj-width", type=int, default=456,
                    help="Effective projector width used when generating the flashed pattern set")
parser.add_argument("--flash-proj-height", type=int, default=570,
                    help="Effective projector height used when generating the flashed pattern set")
parser.add_argument("--white-thresh", type=int, default=1)
parser.add_argument("--black-thresh", type=int, default=1)
parser.add_argument("--min-disparity", type=float, default=0.1)
parser.add_argument("--max-disparity", type=float, default=2500.0)
parser.add_argument("--min-component-area", type=int, default=1000,
                    help="Reject connected mask components smaller than this (0 disables)")
parser.add_argument("--median-filter", type=int, default=5,
                    help="Odd kernel size for disparity outlier filtering (0 disables; keep 0 for tilted surfaces)")
parser.add_argument("--max-median-diff", type=float, default=1.5,
                    help="Reject pixels whose disparity differs from local median by more than this")
parser.add_argument("--disparity-filter", choices=("bilateral", "median", "none"), default="median",
                    help="Sub-pixel disparity edge-preserving smoothing filter to lower point cloud noise floor (default: median)")
parser.add_argument("--disparity-filter-radius", type=int, default=3,
                    help="Diameter of pixel neighborhood for disparity smoothing (default: 3)")
parser.add_argument("--disparity-filter-sigma-color", type=float, default=0.30,
                    help="Filter sigma in disparity space in pixels (default: 0.30 px)")
parser.add_argument("--disparity-filter-sigma-space", type=float, default=1.5,
                    help="Filter sigma in coordinate space in pixels (default: 1.5 px)")
parser.add_argument("--burst-count", type=int, default=1,
                    help="Number of pattern bursts to average temporally for noise reduction (default: 1)")
parser.add_argument("--settle-delay", type=float, default=1.5,
                    help="Seconds to wait after trigger to let mechanical vibration settle before burst")
parser.add_argument("--plane-filter-mm", type=float, default=2.0,
                    help="Keep points within this distance of a robust fitted plane in 3D (0 disables)")
parser.add_argument("--roughness-out-ply", default="output/pointclouds/latest_roughness.ply",
                    help="Path to write colorized roughness PLY")
parser.add_argument("--preview-out", default="output/pointclouds/latest_preview.png",
                    help="Path to write live 2D preview image")
parser.add_argument("--no-zero-disparity-rectify", action="store_true")
parser.add_argument("--no-postprocess", action="store_true",
                    help="Only capture images; skip reconstruction and roughness analysis")
parser.add_argument("--roughness-grid-mm", type=float, default=0.20)
parser.add_argument("--roughness-short-cutoff-mm", type=float, default=1.0)
parser.add_argument("--roughness-long-cutoff-mm", type=float, default=25.0)
parser.add_argument("--roughness-save-grid", default="output/roughness/latest_grid.npz")
parser.add_argument("--roughness-metrics-out", default="output/roughness/latest_metrics.json")
parser.add_argument("--inter-pattern-delay", type=float, default=0.02,
                    help="Seconds to wait after each triggered pattern capture")
parser.add_argument("--debug-reconstruction", action="store_true",
                    help="Write reconstruction debug images and bit contrast diagnostics")
parser.add_argument("--legacy16", action="store_true", help="Use the old 16-pattern sinusoidal projector sequence")
parser.add_argument("--recon-mode", choices=("phase", "gray"), default="phase",
                    help="Reconstruction method: 'phase' for subpixel phase-stereo (<10 um noise), or 'gray' for legacy Gray-code")
parser.add_argument("--phase-period", type=int, default=16, help="Fringe period for phase reconstruction (default: 16)")
parser.add_argument("--phase-steps", type=int, default=8, help="Phase shift steps (default: 8)")
parser.add_argument("--gray-bits", type=int, default=5, help="Coarse Gray-code bits (default: 5)")
parser.add_argument("--checkerboard", default="10x6",
                    help="Calibration checkerboard inner corners as COLSxROWS")
parser.add_argument("--square-size-mm", type=float, default=10.0)
parser.add_argument("--calib-min-frames", type=int, default=10)
parser.add_argument("--calib-max-frames", type=int, default=20)
parser.add_argument("--serial-port", default="/dev/ttyUSB0")
parser.add_argument("--device-retry-sec", type=float, default=3.0)
parser.add_argument("--max-calib-rms", type=float, default=2.0,
                    help="Reject new calibration if stereo RMS is above this")
parser.add_argument("--max-calib-y-error", type=float, default=1.0,
                    help="Reject new calibration if mean rectified Y error is above this")
parser.add_argument("--min-baseline-mm", type=float, default=20.0)
parser.add_argument("--max-baseline-mm", type=float, default=200.0)
args = parser.parse_args()


def parse_checkerboard(value):
    cols, rows = value.lower().split("x", 1)
    return int(cols), int(rows)

# -------------------------------
# Configuration
# -------------------------------
CAPTURE_ROOT = project_path(args.out_dir)
CAPTURE_DIR_LEFT = CAPTURE_ROOT / "left"
CAPTURE_DIR_RIGHT = CAPTURE_ROOT / "right"

CALIB_DIR_LEFT = DATA_DIR / "calibration" / "left"
CALIB_DIR_RIGHT = DATA_DIR / "calibration" / "right"
CALIB_OUT = project_path(args.calib)

CHECKERBOARD = parse_checkerboard(args.checkerboard)
SQUARE_SIZE_MM = args.square_size_mm
MIN_CALIB_FRAMES = args.calib_min_frames
MAX_CALIB_FRAMES = args.calib_max_frames

SERIAL_PORT = args.serial_port
SERIAL_BAUD = 115200

PROJECTOR_TRIGGER_PIN = 17

# Updated timing settings
PULSE_US = 100                 # short clean trigger pulse
INTER_PATTERN_DELAY_S = args.inter_pattern_delay   # allow projector pattern settle
FRAME_TIMEOUT_MS = 3000        # scan captures: 3s (17 patterns, tight timing)
CALIB_FRAME_TIMEOUT_MS = 1000  # calibration captures: 1s (single frame, allows fast CALIBRATE processing)
LINE_LOW_TIMEOUT_S = 1.0

CAPTURE_DIR_LEFT.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR_RIGHT.mkdir(parents=True, exist_ok=True)
CALIB_DIR_LEFT.mkdir(parents=True, exist_ok=True)
CALIB_DIR_RIGHT.mkdir(parents=True, exist_ok=True)
CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)

camera_left = None
camera_right = None
converter_left = None
converter_right = None
ser = None
thread = None

# -------------------------------
# Helpers
# -------------------------------
def clear_directory(path: Path):
    for item in path.iterdir():
        if item.is_file():
            item.unlink()


def sleep_retry(reason):
    print(f"{reason}; retrying in {args.device_retry_sec:g}s")
    time.sleep(args.device_retry_sec)


def now_s():
    return time.monotonic()


def fire_trigger(pulse_us=PULSE_US):
    GPIO.output(PROJECTOR_TRIGGER_PIN, GPIO.HIGH)
    time.sleep(pulse_us / 1_000_000.0)
    GPIO.output(PROJECTOR_TRIGGER_PIN, GPIO.LOW)


def get_line1_status(cam):
    try:
        cam.LineSelector.SetValue("Line1")
        return bool(cam.LineStatus.GetValue())
    except Exception as e:
        print("Line status error:", e)
        return False


def dump_camera_state(cam, name=""):
    try:
        print(
            f"[{name}] "
            f"TriggerMode={cam.TriggerMode.GetValue()}, "
            f"TriggerSource={cam.TriggerSource.GetValue()}, "
            f"TriggerActivation={cam.TriggerActivation.GetValue()}, "
            f"AcquisitionMode={cam.AcquisitionMode.GetValue()}, "
            f"Line1={get_line1_status(cam)}"
        )
    except Exception as e:
        print(f"[{name}] Could not dump state: {e}")


def flush_stale_frames(cam, name=""):
    flushed = 0
    while True:
        try:
            if not cam.GetGrabResultWaitObject().Wait(0):
                break

            stale = cam.RetrieveResult(
                1,
                pylon.TimeoutHandling_Return
            )

            if stale:
                stale.Release()
                flushed += 1
            else:
                break

        except Exception:
            break

    print(f"[{name}] Flushed {flushed} stale frame(s)")


def wait_for_trigger_lines_low(timeout_s=LINE_LOW_TIMEOUT_S):
    if camera_left is None or camera_right is None:
        raise RuntimeError("Cameras are not initialized")
    start = time.monotonic()

    while time.monotonic() - start < timeout_s:
        left_low = not get_line1_status(camera_left)
        right_low = not get_line1_status(camera_right)

        if left_low and right_low:
            return True

        time.sleep(0.001)

    return False


def run_command(label, command):
    print(f"\n=== {label} ===")
    print(" ".join(str(part) for part in command))
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def run_roughness_analysis():
    try:
        from svr_roughness import RoughnessConfig, analyze_file, format_report
    except ImportError as exc:
        raise RuntimeError(
            "svr-roughness is not installed. Run: "
            "venv/bin/python -m pip install ../svr-roughness"
        ) from exc

    print("\n=== Surface roughness analysis ===")
    ply_path = project_path(args.recon_out)
    print(
        "svr_roughness.analyze_file "
        f"{ply_path} "
        f"--grid-mm {args.roughness_grid_mm} "
        f"--short-cutoff-mm {args.roughness_short_cutoff_mm} "
        f"--long-cutoff-mm {args.roughness_long_cutoff_mm}"
    )
    result = analyze_file(
        ply_path,
        RoughnessConfig(
            grid_mm=args.roughness_grid_mm,
            short_cutoff_mm=args.roughness_short_cutoff_mm,
            long_cutoff_mm=args.roughness_long_cutoff_mm,
        ),
    )
    print(format_report(result))

    if args.roughness_save_grid:
        grid_path = project_path(args.roughness_save_grid)
        result.save_grid_npz(grid_path)
        print(f"\nWrote grid data: {grid_path}")
    if args.roughness_metrics_out:
        metrics_path = project_path(args.roughness_metrics_out)
        result.save_metrics_json(metrics_path)
        print(f"Wrote metrics: {metrics_path}")


def ensure_projector_scan():
    if args.legacy16:
        proj.start_scan16()
    else:
        proj.start_scan(NUM_PATTERNS, exposure_ms=args.proj_pattern_ms)
    proj.set_brightness(args.brightness)


def wait_for_projector_scan():
    send_status("WAIT_PROJECTOR")
    while True:
        try:
            ensure_projector_scan()
            print(f"Projector: scan sequence ready  brightness={args.brightness}  cam_exposure={args.exposure}us  proj_pattern={args.proj_pattern_ms}ms")
            return
        except Exception as e:
            send_status("WAIT_PROJECTOR")
            sleep_retry(f"Projector unavailable or not ready: {e}")


def setup_camera(cam):
    cam.ExposureAuto.SetValue("Off")
    cam.ExposureTime.SetValue(float(args.exposure))

    cam.GainAuto.SetValue("Off")
    cam.Gain.SetValue(0.0)

    cam.TriggerSelector.SetValue("FrameStart")
    cam.TriggerMode.SetValue("On")
    cam.TriggerSource.SetValue("Line1")
    cam.TriggerActivation.SetValue("RisingEdge")

    cam.AcquisitionMode.SetValue("Continuous")
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)


def make_converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def wait_for_cameras():
    global camera_left, camera_right, converter_left, converter_right
    factory = pylon.TlFactory.GetInstance()
    send_status("WAIT_CAMERAS")
    while True:
        try:
            devices = factory.EnumerateDevices()
            if len(devices) < 2:
                raise RuntimeError(f"Need 2 cameras connected, found {len(devices)}")

            camera_left = pylon.InstantCamera(factory.CreateDevice(devices[1]))
            camera_right = pylon.InstantCamera(factory.CreateDevice(devices[0]))
            camera_left.Open()
            camera_right.Open()
            setup_camera(camera_left)
            setup_camera(camera_right)
            converter_left = make_converter()
            converter_right = make_converter()
            print("Left:", camera_left.GetDeviceInfo().GetModelName())
            print("Right:", camera_right.GetDeviceInfo().GetModelName())
            dump_camera_state(camera_left, "LEFT startup")
            dump_camera_state(camera_right, "RIGHT startup")
            return
        except Exception as e:
            close_cameras()
            send_status("WAIT_CAMERAS")
            sleep_retry(f"Cameras unavailable or not ready: {e}")


def wait_for_serial():
    global ser
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
            time.sleep(2)
            print(f"Serial ready: {SERIAL_PORT}")
            send_status("SYSTEM_BOOT")
            return
        except Exception as e:
            close_serial()
            sleep_retry(f"Serial unavailable: {SERIAL_PORT}: {e}")


def close_cameras():
    global camera_left, camera_right
    for cam in (camera_left, camera_right):
        if cam is None:
            continue
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            if cam.IsOpen():
                cam.Close()
        except Exception:
            pass
    camera_left = None
    camera_right = None


def close_serial():
    global ser, thread
    try:
        if thread is not None:
            thread.close()
    except Exception:
        pass
    try:
        if ser is not None and ser.is_open:
            ser.close()
    except Exception:
        pass
    thread = None
    ser = None


def cleanup_runtime():
    close_serial()
    close_cameras()
    try:
        proj.stop()
    except Exception:
        pass


def graycode_patterns(width, height):
    if not hasattr(cv2, "structured_light"):
        raise RuntimeError("cv2.structured_light is missing. Install opencv-contrib-python-headless.")
    graycode = cv2.structured_light.GrayCodePattern.create(width, height)
    return list(graycode.generate()[1])


def find_pattern_mapping(flash_width, flash_height, decode_width, decode_height):
    flash_patterns = graycode_patterns(flash_width, flash_height)
    decode_patterns = graycode_patterns(decode_width, decode_height)
    used = set()
    mapping = []
    for decode_idx, decode_pattern in enumerate(decode_patterns):
        scaled = cv2.resize(
            decode_pattern,
            (flash_width, flash_height),
            interpolation=cv2.INTER_NEAREST,
        )
        match = None
        for flash_idx, flash_pattern in enumerate(flash_patterns):
            if flash_idx in used:
                continue
            if flash_pattern.shape == scaled.shape and np.array_equal(flash_pattern, scaled):
                match = flash_idx
                break
        if match is None:
            raise ValueError(
                "Cannot map flashed Gray-code patterns to requested decode resolution: "
                f"decode pattern {decode_idx} has no exact match. "
                "Use a coarser resolution that drops low bits, e.g. 456x285, 228x570, or 228x285."
            )
        used.add(match)
        mapping.append(match)

    white_slot = len(flash_patterns)
    black_slot = len(flash_patterns) + 1
    mapping.extend([white_slot, black_slot])
    return mapping, len(flash_patterns) + 2


def make_capture_plan():
    if args.legacy16:
        return {idx: idx for idx in range(16)}, 16, 16

    if args.recon_mode == "phase":
        phase_count = args.phase_steps + 2 * args.gray_bits + 2
        capture_by_slot = {idx: idx for idx in range(phase_count)}
        return capture_by_slot, phase_count, phase_count

    mapping, flash_total = find_pattern_mapping(
        args.flash_proj_width,
        args.flash_proj_height,
        args.proj_width,
        args.proj_height,
    )
    if args.patterns != flash_total:
        print(
            f"WARNING: --patterns {args.patterns} does not match flashed Gray-code count "
            f"{flash_total} for {args.flash_proj_width}x{args.flash_proj_height}; using {flash_total}."
        )
    capture_by_slot = {slot: save_idx for save_idx, slot in enumerate(mapping)}
    return capture_by_slot, flash_total, len(mapping)


CAPTURE_BY_SLOT, PROJECTOR_SLOT_COUNT, EXPECTED_CAPTURE_COUNT = make_capture_plan()
NUM_PATTERNS = PROJECTOR_SLOT_COUNT
print(
    f"Capture plan: mode={args.recon_mode} flash={args.flash_proj_width}x{args.flash_proj_height} "
    f"decode={args.proj_width}x{args.proj_height} "
    f"projector_slots={PROJECTOR_SLOT_COUNT} saved_frames={EXPECTED_CAPTURE_COUNT} "
    f"burst_count={args.burst_count}"
)


def send_result(metrics_path):
    try:
        metrics = json.loads(Path(metrics_path).read_text())
        sa = float(metrics["sa_um"])
        sq = float(metrics["sq_um"])
        svr = float(metrics["svr_um"])
        noise = float(metrics.get("noise_floor_um", 0.0))
        raw_svr = float(metrics.get("svr_raw_um", svr))
        if noise > 0:
            print(f"Roughness Result: Sa={sa:.1f} um, Sq={sq:.1f} um, Svr={svr:.1f} um (noise floor: {noise:.1f} um, raw Svr: {raw_svr:.1f} um)")
            send_status(f"RESULT SA={sa:.1f} SQ={sq:.1f} SVR={svr:.1f} NF={noise:.1f}")
        else:
            print(f"Roughness Result: Sa={sa:.1f} um, Sq={sq:.1f} um, Svr={svr:.1f} um")
            send_status(f"RESULT SA={sa:.1f} SQ={sq:.1f} SVR={svr:.1f}")
    except Exception as e:
        print(f"Could not send roughness result: {e}")


def run_postprocess():
    if args.no_postprocess:
        print("Post-processing disabled (--no-postprocess)")
        return

    if args.recon_mode == "phase":
        recon_cmd = [
            sys.executable,
            str(TOOLS_DIR / "runtime" / "reconstruct_phase_stereo.py"),
            "--caps",
            str(CAPTURE_ROOT),
            "--calib",
            str(project_path(args.calib)),
            "--out",
            str(project_path(args.recon_out)),
            "--period",
            str(args.phase_period),
            "--num-phases",
            str(args.phase_steps),
            "--gray-bits",
            str(args.gray_bits),
            "--min-mod",
            str(args.min_mod),
            "--min-contrast",
            str(args.min_contrast),
            "--min-disparity",
            str(args.min_disparity),
            "--plane-filter-mm",
            str(args.plane_filter_mm),
            "--min-component-area",
            str(args.min_component_area),
            "--median-filter",
            str(args.median_filter),
            "--max-median-diff",
            str(args.max_median_diff),
            "--disparity-filter",
            str(args.disparity_filter),
            "--disparity-filter-radius",
            str(args.disparity_filter_radius),
            "--disparity-filter-sigma-color",
            str(args.disparity_filter_sigma_color),
            "--disparity-filter-sigma-space",
            str(args.disparity_filter_sigma_space),
            "--disparity-sign",
            "auto",
        ]
        if not args.no_zero_disparity_rectify:
            recon_cmd.append("--zero-disparity-rectify")
        if not args.debug_reconstruction:
            recon_cmd.append("--no-debug")
    else:
        recon_cmd = [
            sys.executable,
            str(TOOLS_DIR / "runtime" / "reconstruct_local_gray.py"),
            "--caps",
            str(CAPTURE_ROOT),
            "--calib",
            str(project_path(args.calib)),
            "--out",
            str(project_path(args.recon_out)),
            "--proj-width",
            str(args.proj_width),
            "--proj-height",
            str(args.proj_height),
            "--min-component-area",
            str(args.min_component_area),
            "--median-filter",
            str(args.median_filter),
            "--max-median-diff",
            str(args.max_median_diff),
            "--plane-filter-mm",
            str(args.plane_filter_mm),
            "--white-thresh",
            str(args.white_thresh),
            "--black-thresh",
            str(args.black_thresh),
            "--disparity-sign",
            "positive",
            "--min-disparity",
            str(args.min_disparity),
        ]
        if not args.no_zero_disparity_rectify:
            recon_cmd.append("--zero-disparity-rectify")
        if not args.debug_reconstruction:
            recon_cmd.extend(["--no-debug", "--no-bit-debug"])

    send_status("RECONSTRUCTING")
    run_command("Reconstructing point cloud", recon_cmd)

    send_status("ROUGHNESS")
    run_roughness_analysis()
    if args.roughness_metrics_out:
        send_result(project_path(args.roughness_metrics_out))

    grid_file = project_path(args.roughness_save_grid)
    if grid_file.exists():
        try:
            overlay_cmd = [
                sys.executable,
                str(TOOLS_DIR / "presentation" / "overlay_ply_grid.py"),
                str(project_path(args.recon_out)),
                "--roughness-grid",
                str(grid_file),
                "--show-grid-lines",
                "--out",
                str(project_path(args.roughness_out_ply)),
            ]
            run_command("Generating colorized roughness point cloud", overlay_cmd)

            render_cmd = [
                sys.executable,
                str(TOOLS_DIR / "presentation" / "ply_to_png.py"),
                str(project_path(args.roughness_out_ply)),
                "--color",
                "ply",
                "--out",
                str(project_path(args.preview_out)),
            ]
            run_command("Generating live preview image", render_cmd)
        except Exception as e:
            print(f"Warning: could not generate roughness visual overlay / preview: {e}")


def detect_checkerboard(gray):
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    try:
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags)
        if found:
            return True, corners
    except Exception:
        pass

    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
    if found:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def compute_rectified_y_error(imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, image_size, R, T):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, flags=0, alpha=-1
    )
    frame_means = []
    all_dy = []
    for corners_l, corners_r in zip(imgpoints_l, imgpoints_r):
        und_l = cv2.undistortPoints(corners_l, mtx_l, dist_l, R=R1, P=P1).reshape(-1, 2)
        und_r = cv2.undistortPoints(corners_r, mtx_r, dist_r, R=R2, P=P2).reshape(-1, 2)
        dy = np.abs(und_l[:, 1] - und_r[:, 1])
        frame_means.append(float(np.mean(dy)))
        all_dy.extend(float(v) for v in dy)
    if not all_dy:
        return {"mean": float("inf"), "p95": float("inf"), "max_frame_mean": float("inf")}
    return {
        "mean": float(np.mean(all_dy)),
        "p95": float(np.percentile(all_dy, 95)),
        "max_frame_mean": float(max(frame_means)),
    }


def calibration_is_acceptable(rms, y_error, T):
    baseline = float(np.linalg.norm(T))
    checks = [
        (np.isfinite(rms) and rms <= args.max_calib_rms, f"stereo RMS {rms:.3f}px <= {args.max_calib_rms:.3f}px"),
        (
            np.isfinite(y_error["mean"]) and y_error["mean"] <= args.max_calib_y_error,
            f"mean rectified Y {y_error['mean']:.3f}px <= {args.max_calib_y_error:.3f}px",
        ),
        (
            args.min_baseline_mm <= baseline <= args.max_baseline_mm,
            f"baseline {baseline:.3f}mm in [{args.min_baseline_mm:.3f}, {args.max_baseline_mm:.3f}]mm",
        ),
    ]
    for ok, message in checks:
        print(("PASS " if ok else "FAIL ") + message)
    return all(ok for ok, _ in checks)


# -------------------------------
# GPIO Setup
# -------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(PROJECTOR_TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)

# -------------------------------
# Scan routine
# -------------------------------
scan_in_progress = False
scan_count = 0
last_scan_end = 0.0
SCAN_COOLDOWN_S = 1.0

calibration_mode = False
calib_frame_count = 0

def run_scan():
    global scan_in_progress, scan_count, last_scan_end

    if scan_in_progress:
        print("Scan ignored: already running")
        send_status("SCANNING")
        return

    if now_s() - last_scan_end < SCAN_COOLDOWN_S:
        print("Scan ignored: cooldown")
        return

    if camera_left is None or camera_right is None or not camera_left.IsOpen() or not camera_right.IsOpen() or camera_left.IsCameraDeviceRemoved() or camera_right.IsCameraDeviceRemoved():
        print("Scan aborted: camera missing or disconnected")
        send_status("WAIT_CAMERAS")
        raise RuntimeError("Camera missing or disconnected at scan start")

    send_status("SCANNING")
    scan_in_progress = True
    scan_count += 1

    try:
        ensure_projector_scan()
        print(f"\n=== Starting scan {scan_count} ===")

        clear_directory(CAPTURE_DIR_LEFT)
        clear_directory(CAPTURE_DIR_RIGHT)

        flush_stale_frames(camera_left, "LEFT")
        flush_stale_frames(camera_right, "RIGHT")

        saved_count = 0
        consecutive_failures = 0
        buffered_pairs = []
        accum_left = {}
        accum_right = {}
        accum_count = {}

        if args.settle_delay > 0:
            print(f"Holding steady: settling for {args.settle_delay:g}s...")
            time.sleep(args.settle_delay)

        t_burst_start = time.time()

        for burst_idx in range(args.burst_count):
            if args.burst_count > 1:
                print(f"\n--- Burst {burst_idx + 1}/{args.burst_count} ---")
                if burst_idx > 0:
                    time.sleep(0.05)

            for slot_idx in range(PROJECTOR_SLOT_COUNT):
                if camera_left.IsCameraDeviceRemoved() or camera_right.IsCameraDeviceRemoved():
                    send_status("CAMERA_ERR")
                    raise RuntimeError(f"Camera disconnected during scan at slot {slot_idx:02d}")

                save_idx = CAPTURE_BY_SLOT.get(slot_idx)
                if save_idx is None:
                    print(f"\nSlot {slot_idx:02d} (skip)")
                else:
                    burst_tag = f" [burst {burst_idx + 1}/{args.burst_count}]" if args.burst_count > 1 else ""
                    print(f"\nSlot {slot_idx:02d} -> saved frame {save_idx:02d}{burst_tag}")

                if not wait_for_trigger_lines_low():
                    print("  WARNING: trigger line stuck HIGH")

                print(
                    "  Before pulse:",
                    "L=", get_line1_status(camera_left),
                    "R=", get_line1_status(camera_right)
                )

                fire_trigger()

                print("  Trigger fired")

                try:
                    grab_left = camera_left.RetrieveResult(
                        FRAME_TIMEOUT_MS,
                        pylon.TimeoutHandling_ThrowException
                    )

                    grab_right = camera_right.RetrieveResult(
                        FRAME_TIMEOUT_MS,
                        pylon.TimeoutHandling_ThrowException
                    )

                except Exception as e:
                    consecutive_failures += 1
                    print(f"  TIMEOUT slot {slot_idx:02d} (failure {consecutive_failures}/3): {e}")
                    print(
                        "  After timeout:",
                        "L=", get_line1_status(camera_left),
                        "R=", get_line1_status(camera_right)
                    )
                    if camera_left.IsCameraDeviceRemoved() or camera_right.IsCameraDeviceRemoved() or consecutive_failures >= 3:
                        send_status("CAMERA_ERR")
                        raise RuntimeError(f"Aborting scan due to camera disconnection / repeated timeouts at slot {slot_idx:02d}") from e

                    time.sleep(INTER_PATTERN_DELAY_S)
                    continue

                if grab_left.GrabSucceeded() and grab_right.GrabSucceeded():
                    consecutive_failures = 0
                    if save_idx is not None:
                        img_left = converter_left.Convert(grab_left).GetArray()
                        img_right = converter_right.Convert(grab_right).GetArray()

                        if args.burst_count == 1:
                            buffered_pairs.append((save_idx, img_left, img_right))
                        else:
                            if save_idx not in accum_left:
                                accum_left[save_idx] = img_left.astype(np.uint16)
                                accum_right[save_idx] = img_right.astype(np.uint16)
                                accum_count[save_idx] = 1
                            else:
                                accum_left[save_idx] += img_left.astype(np.uint16)
                                accum_right[save_idx] += img_right.astype(np.uint16)
                                accum_count[save_idx] += 1

                        print(f"  Grabbed pair {save_idx:02d} (RAM buffer)")
                        if burst_idx == 0:
                            saved_count += 1
                    else:
                        print(f"  Skipped slot {slot_idx:02d}")

                    try:
                        print(
                            "  timestamps:",
                            grab_left.TimeStamp,
                            grab_right.TimeStamp
                        )
                    except Exception:
                        pass
                else:
                    consecutive_failures += 1
                    print(f"  Grab failed slot {slot_idx:02d} (failure {consecutive_failures}/3)")
                    if consecutive_failures >= 3:
                        send_status("CAMERA_ERR")
                        raise RuntimeError(f"Aborting scan due to consecutive grab failures at slot {slot_idx:02d}")

                grab_left.Release()
                grab_right.Release()

                time.sleep(INTER_PATTERN_DELAY_S)

        t_burst_end = time.time()
        burst_info = f" across {args.burst_count} bursts" if args.burst_count > 1 else ""
        print(f"\nOptical capture burst completed in {t_burst_end - t_burst_start:.2f}s ({saved_count}/{EXPECTED_CAPTURE_COUNT} frames in RAM{burst_info})")

        if args.burst_count > 1 and accum_left:
            print(f"Averaging {len(accum_left)} image pairs across {args.burst_count} bursts and flushing to disk...")
            t_flush_start = time.time()
            for s_idx in sorted(accum_left.keys()):
                cnt = accum_count[s_idx]
                avg_l = np.clip(np.round(accum_left[s_idx] / cnt), 0, 255).astype(np.uint8)
                avg_r = np.clip(np.round(accum_right[s_idx] / cnt), 0, 255).astype(np.uint8)
                cv2.imwrite(str(CAPTURE_DIR_LEFT / f"left_{s_idx:02d}.png"), avg_l)
                cv2.imwrite(str(CAPTURE_DIR_RIGHT / f"right_{s_idx:02d}.png"), avg_r)
            print(f"Disk write completed in {time.time() - t_flush_start:.2f}s")
        elif buffered_pairs:
            print(f"Flushing {len(buffered_pairs)} image pairs to disk...")
            t_flush_start = time.time()
            for s_idx, i_left, i_right in buffered_pairs:
                cv2.imwrite(str(CAPTURE_DIR_LEFT / f"left_{s_idx:02d}.png"), i_left)
                cv2.imwrite(str(CAPTURE_DIR_RIGHT / f"right_{s_idx:02d}.png"), i_right)
            print(f"Disk write completed in {time.time() - t_flush_start:.2f}s")

        print(f"\nScan complete: {saved_count}/{EXPECTED_CAPTURE_COUNT} saved ({PROJECTOR_SLOT_COUNT} projector slots)")
        if saved_count == EXPECTED_CAPTURE_COUNT:
            try:
                run_postprocess()
                send_status("ANALYSIS_DONE")
            except Exception as e:
                print(f"Post-processing failed: {e}")
                send_status("ANALYSIS_FAIL")
        else:
            print("Skipping post-processing because the scan is incomplete.")
            send_status("SCAN_INCOMPLETE")

    finally:
        scan_in_progress = False
        last_scan_end = now_s()
        send_status("SCAN_DONE")


# -------------------------------
# Calibration mode
# -------------------------------

def send_status(msg: str) -> None:
    if ser is None or not ser.is_open:
        print(f"Serial unavailable; status not sent: {msg}")
        return
    try:
        ser.write((msg + "\n").encode())
    except Exception as e:
        print(f"Serial write failed: {e}")


def enter_calibration_mode():
    global calibration_mode, calib_frame_count

    if calibration_mode:
        print("Already in calibration mode")
        return

    if scan_in_progress:
        print("Cannot enter calibration mode: scan in progress")
        return

    print("\n=== Entering calibration mode ===")
    clear_directory(CALIB_DIR_LEFT)
    clear_directory(CALIB_DIR_RIGHT)
    calib_frame_count = 0

    try:
        proj.start_calibration(args.brightness)
    except Exception as e:
        print(f"Projector switch failed: {e}")
        send_status("CALIB_FAIL")
        return

    # Projector's internal DISP_STOP may leave TRIG_OUT in an uncertain state.
    # Settle, then flush any frames the cameras accumulated during the switch.
    time.sleep(0.5)
    flush_stale_frames(camera_left, "LEFT")
    flush_stale_frames(camera_right, "RIGHT")

    calibration_mode = True
    print(f"Projector showing checkerboard. CALIB_TRIGGER to capture (need {MIN_CALIB_FRAMES}-{MAX_CALIB_FRAMES}), CALIB_OFF to compute/save.")
    send_status("CALIB_MODE")


def capture_calibration_frame():
    global calib_frame_count

    if scan_in_progress:
        print("Capture ignored: scan in progress")
        return

    print(f"\nCalib frame {calib_frame_count + 1}/{MAX_CALIB_FRAMES}")

    flush_stale_frames(camera_left, "LEFT")
    flush_stale_frames(camera_right, "RIGHT")

    lines_low = wait_for_trigger_lines_low()
    print(f"  Line1 before pulse: L={get_line1_status(camera_left)} R={get_line1_status(camera_right)} (wait_low={'OK' if lines_low else 'TIMEOUT'})")

    fire_trigger()
    print(f"  Line1 after  pulse: L={get_line1_status(camera_left)} R={get_line1_status(camera_right)}")

    try:
        grab_left = camera_left.RetrieveResult(CALIB_FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
        grab_right = camera_right.RetrieveResult(CALIB_FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
    except Exception as e:
        print(f"  TIMEOUT — camera did not trigger within {CALIB_FRAME_TIMEOUT_MS}ms: {e}")
        print(f"  Line1 after timeout: L={get_line1_status(camera_left)} R={get_line1_status(camera_right)}")
        print("  → Projector may not be firing TRIG_OUT in calibration sequence")
        send_status("FRAME ERR")
        return

    print(f"  GrabSucceeded: L={grab_left.GrabSucceeded()} R={grab_right.GrabSucceeded()}")

    if not (grab_left.GrabSucceeded() and grab_right.GrabSucceeded()):
        print(f"  Grab failed — L error: {grab_left.GetErrorDescription()}  R error: {grab_right.GetErrorDescription()}")
        grab_left.Release()
        grab_right.Release()
        send_status("FRAME ERR")
        return

    img_left = converter_left.Convert(grab_left).GetArray()
    img_right = converter_right.Convert(grab_right).GetArray()
    grab_left.Release()
    grab_right.Release()

    print(f"  Image shape: L={img_left.shape} R={img_right.shape}")
    print(f"  Brightness:  L_mean={img_left.mean():.1f} R_mean={img_right.mean():.1f}")

    gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    ret_l, _ = detect_checkerboard(gray_l)
    ret_r, _ = detect_checkerboard(gray_r)

    print(f"  Corners ({CHECKERBOARD}): L={ret_l} R={ret_r}")

    if not (ret_l and ret_r):
        print("  → No corners found — misalignment, bad exposure, or wrong CHECKERBOARD size")
        send_status("FRAME BAD")
        return

    file_l = CALIB_DIR_LEFT / f"calib_{calib_frame_count:02d}.png"
    file_r = CALIB_DIR_RIGHT / f"calib_{calib_frame_count:02d}.png"
    cv2.imwrite(str(file_l), img_left)
    cv2.imwrite(str(file_r), img_right)
    calib_frame_count += 1
    print(f"  Saved ({calib_frame_count}/{MAX_CALIB_FRAMES})")
    send_status(f"FRAME {calib_frame_count}/{MAX_CALIB_FRAMES}")

    if calib_frame_count >= MAX_CALIB_FRAMES:
        finalize_calibration()


def finalize_calibration():
    global calibration_mode, calib_frame_count

    if calib_frame_count < MIN_CALIB_FRAMES:
        print(f"Need at least {MIN_CALIB_FRAMES} frames (have {calib_frame_count}). Exiting without saving.")
        send_status("CALIB_FAIL")
        exit_calibration_mode()
        return

    print(f"\n=== Computing calibration from {calib_frame_count} frames ===")
    send_status("CALIB_COMPUTING")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    img_shape = None

    for lf, rf in zip(sorted(CALIB_DIR_LEFT.glob("*.png")), sorted(CALIB_DIR_RIGHT.glob("*.png"))):
        img_l = cv2.imread(str(lf))
        img_r = cv2.imread(str(rf))
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        ret_l, corners_l = detect_checkerboard(gray_l)
        ret_r, corners_r = detect_checkerboard(gray_r)
        if ret_l and ret_r:
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            img_shape = gray_l.shape[::-1]

    if len(objpoints) < MIN_CALIB_FRAMES:
        print(f"Only {len(objpoints)} usable frames after re-check. Aborting.")
        send_status("CALIB_FAIL")
        exit_calibration_mode()
        return

    print(f"Calibrating on {len(objpoints)} valid frames...")
    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_shape, None, None)
    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_shape, None, None)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r, img_shape
    )

    print(f"Stereo RMS reprojection error: {rms:.4f} px")
    print(f"Translation T (mm): {T.ravel()}")
    y_error = compute_rectified_y_error(
        imgpoints_l,
        imgpoints_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        img_shape,
        R,
        T,
    )
    print(
        "Rectified Y error: "
        f"mean={y_error['mean']:.4f}px "
        f"p95={y_error['p95']:.4f}px "
        f"max_frame_mean={y_error['max_frame_mean']:.4f}px"
    )
    if not calibration_is_acceptable(rms, y_error, T):
        print(f"New calibration rejected; keeping existing {CALIB_OUT}")
        send_status("CALIB_REJECTED")
        exit_calibration_mode()
        return

    temp_out = CALIB_OUT.with_suffix(".tmp.npz")
    np.savez(
        str(temp_out),
        mtxL=mtx_l,
        distL=dist_l,
        mtxR=mtx_r,
        distR=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        rms=rms,
        rectified_y_error_mean=y_error["mean"],
        rectified_y_error_p95=y_error["p95"],
        rectified_y_error_max_frame_mean=y_error["max_frame_mean"],
        checkerboard=np.array(CHECKERBOARD),
        square_size_mm=np.array(SQUARE_SIZE_MM),
        image_size=np.array(img_shape),
    )
    temp_out.replace(CALIB_OUT)
    print(f"Calibration saved to {CALIB_OUT}")
    send_status("CALIB_DONE")

    exit_calibration_mode()


def exit_calibration_mode():
    global calibration_mode, calib_frame_count

    calibration_mode = False
    calib_frame_count = 0

    try:
        if args.legacy16:
            proj.start_scan16()
        else:
            proj.start_scan(NUM_PATTERNS)
    except Exception as e:
        print(f"Projector switch back to scan failed: {e}")

    print("=== Returned to scan mode ===")
    send_status("SCAN_MODE")


# -------------------------------
# Serial handler
# -------------------------------
class TriggerHandler(LineReader):
    def handle_line(self, line):
        line = line.strip()
        print(f"Serial: {line}")

        if line in ("TRIGGER", "CALIB_TRIGGER", "CALIB_ON", "CALIBRATE"):
            if (
                camera_left is None
                or camera_right is None
                or not camera_left.IsOpen()
                or not camera_right.IsOpen()
                or camera_left.IsCameraDeviceRemoved()
                or camera_right.IsCameraDeviceRemoved()
            ):
                print("Serial trigger ignored: cameras not ready")
                send_status("WAIT_CAMERAS")
                return

        if calibration_mode:
            if line in ("TRIGGER", "CALIB_TRIGGER"):
                capture_calibration_frame()
            elif line in ("CALIB_OFF", "CALIBRATE"):
                finalize_calibration()
        else:
            if line == "TRIGGER":
                run_scan()
            elif line in ("CALIB_ON", "CALIBRATE"):
                enter_calibration_mode()


def run_runtime_once():
    global thread
    wait_for_serial()
    thread = ReaderThread(ser, TriggerHandler)
    thread.start()
    send_status("SYSTEM_BOOT")
    wait_for_cameras()
    wait_for_projector_scan()
    send_status("SCAN_MODE")
    print("Waiting for serial trigger...")
    while True:
        time.sleep(1)
        if ser is None or not ser.is_open:
            raise RuntimeError("Serial port closed")
        if camera_left is None or camera_right is None:
            raise RuntimeError("Camera handle missing")
        if not camera_left.IsOpen() or not camera_right.IsOpen():
            raise RuntimeError("Camera closed")
        if camera_left.IsCameraDeviceRemoved() or camera_right.IsCameraDeviceRemoved():
            raise RuntimeError("Camera device unplugged")


# -------------------------------
# Main supervisor
# -------------------------------
try:
    while True:
        try:
            run_runtime_once()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Runtime error: {e}")
            send_status("SYSTEM_RESTART")
            cleanup_runtime()
            time.sleep(args.device_retry_sec)
            print("Restarting scanner runtime...")
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    try:
        send_status("SCAN_MODE")
        time.sleep(0.1)
    except Exception:
        pass
    cleanup_runtime()
    GPIO.cleanup()
    print("Clean shutdown.")
