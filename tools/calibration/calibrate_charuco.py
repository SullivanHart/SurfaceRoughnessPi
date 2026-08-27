"""
Stereo calibration using a ChArUco board.

ChArUco marker IDs remove the corner-order ambiguity that plain checkerboards
can have in stereo calibration.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import RPi.GPIO as GPIO
import serial
from pypylon import pylon
from serial.threaded import LineReader, ReaderThread

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "projector"))
import projector as proj


parser = argparse.ArgumentParser()
parser.add_argument("--brightness", type=int, default=34)
parser.add_argument("--exposure", type=int, default=250)
parser.add_argument("--squares-x", type=int, default=8)
parser.add_argument("--squares-y", type=int, default=5)
parser.add_argument("--square-size-mm", type=float, default=10.0)
parser.add_argument("--marker-size-mm", type=float, default=7.0)
parser.add_argument("--min-frames", type=int, default=10)
parser.add_argument("--max-frames", type=int, default=60)
parser.add_argument("--eval-every", type=int, default=5,
                    help="Recompute calibration after this many newly accepted frames")
parser.add_argument("--target-stereo-rms", type=float, default=2.0,
                    help="Stop early when stereo RMS is at or below this value")
parser.add_argument("--target-y-error", type=float, default=0.8,
                    help="Stop early when mean rectified vertical error is at or below this value")
parser.add_argument("--target-max-y-error", type=float, default=1.5,
                    help="Stop early when worst-frame mean rectified vertical error is at or below this value")
parser.add_argument("--min-corners", type=int, default=40)
parser.add_argument("--min-sharpness", type=float, default=100.0,
                    help="Reject captures if either camera Laplacian sharpness is below this value")
parser.add_argument("--min-coverage-x", type=float, default=0.55,
                    help="Minimum fraction of board width covered by common detected ChArUco corners")
parser.add_argument("--min-coverage-y", type=float, default=0.55,
                    help="Minimum fraction of board height covered by common detected ChArUco corners")
parser.add_argument("--out", default="config/calibration.npz",
                    help="Calibration file to write")
parser.add_argument("--image-dir", default="data/calibration",
                    help="Directory for accepted calibration image pairs")
parser.add_argument("--debug-dir", default="data/calibration/debug",
                    help="Directory for latest calibration debug images")
args = parser.parse_args()

def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


CALIB_ROOT = project_path(args.image_dir)
CALIB_DIR_LEFT = CALIB_ROOT / "left"
CALIB_DIR_RIGHT = CALIB_ROOT / "right"
DEBUG_DIR = project_path(args.debug_dir)
CALIB_OUT = project_path(args.out)
for d in (CALIB_DIR_LEFT, CALIB_DIR_RIGHT, DEBUG_DIR):
    d.mkdir(parents=True, exist_ok=True)
CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
TRIGGER_PIN = 17
FRAME_TIMEOUT_MS = 3000

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(
    (args.squares_x, args.squares_y),
    args.square_size_mm,
    args.marker_size_mm,
    dictionary,
)
detector = cv2.aruco.CharucoDetector(board)
board_points = board.getChessboardCorners()
board_width_mm = (args.squares_x - 1) * args.square_size_mm
board_height_mm = (args.squares_y - 1) * args.square_size_mm


def clear_directory(path):
    for p in path.iterdir():
        if p.is_file():
            p.unlink()


def fire_trigger():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(100 / 1_000_000.0)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)


def flush_stale_frames(cam, name=""):
    flushed = 0
    while True:
        try:
            if not cam.GetGrabResultWaitObject().Wait(0):
                break
            stale = cam.RetrieveResult(1, pylon.TimeoutHandling_Return)
            if stale:
                stale.Release()
                flushed += 1
            else:
                break
        except Exception:
            break
    if flushed:
        print(f"[{name}] Flushed {flushed} stale frame(s)")


def detect_charuco(gray):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None:
        return None, None, marker_corners, marker_ids
    return charuco_corners.astype(np.float32), charuco_ids.reshape(-1).astype(np.int32), marker_corners, marker_ids


def draw_debug(image, charuco_corners, charuco_ids, marker_corners, marker_ids):
    out = image.copy()
    if marker_ids is not None and len(marker_ids):
        cv2.aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    if charuco_corners is not None and charuco_ids is not None and len(charuco_ids):
        cv2.aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)
    return out


def detection_roi(gray, charuco_corners, marker_corners, pad=20):
    points = []
    if charuco_corners is not None and len(charuco_corners):
        points.append(charuco_corners.reshape(-1, 2))
    if marker_corners is not None and len(marker_corners):
        points.append(np.concatenate([c.reshape(-1, 2) for c in marker_corners], axis=0))
    if not points:
        return gray, (0, 0, gray.shape[1], gray.shape[0])

    pts = np.concatenate(points, axis=0)
    h, w = gray.shape[:2]
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + pad)
    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


def send_status(msg):
    try:
        ser.write((msg + "\n").encode())
    except Exception as e:
        print(f"Serial write failed: {e}")


def common_corner_quality(common_ids):
    obj = board_points[common_ids]
    if len(obj) == 0:
        return 0.0, 0.0
    coverage_x = (float(obj[:, 0].max() - obj[:, 0].min()) / board_width_mm) if board_width_mm else 0.0
    coverage_y = (float(obj[:, 1].max() - obj[:, 1].min()) / board_height_mm) if board_height_mm else 0.0
    return coverage_x, coverage_y


GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)

factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()
if len(devices) < 2:
    raise RuntimeError("Need 2 cameras connected")

camera_left = pylon.InstantCamera(factory.CreateDevice(devices[1]))
camera_right = pylon.InstantCamera(factory.CreateDevice(devices[0]))
camera_left.Open()
camera_right.Open()
print("Left: ", camera_left.GetDeviceInfo().GetModelName())
print("Right:", camera_right.GetDeviceInfo().GetModelName())


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


setup_camera(camera_left)
setup_camera(camera_right)

converter_left = pylon.ImageFormatConverter()
converter_left.OutputPixelFormat = pylon.PixelType_BGR8packed
converter_left.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
converter_right = pylon.ImageFormatConverter()
converter_right.OutputPixelFormat = pylon.PixelType_BGR8packed
converter_right.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

try:
    proj.start_calibration(brightness=args.brightness)
    print(f"Projector: calibration pattern  brightness={args.brightness}  exposure={args.exposure}us")
except Exception as e:
    print(f"Projector startup failed: {e}")
    GPIO.cleanup()
    camera_left.Close()
    camera_right.Close()
    sys.exit(1)

time.sleep(2.0)
flush_stale_frames(camera_left, "LEFT")
flush_stale_frames(camera_right, "RIGHT")

stored_objpoints = []
stored_pts_l = []
stored_pts_r = []
active = True
frame_count = 0
image_size = None
best_calibration = None
best_score = None
last_eval_frame = 0


def compute_rectified_y_error(calib):
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        calib["mtx_l"],
        calib["dist_l"],
        calib["mtx_r"],
        calib["dist_r"],
        calib["image_size"],
        calib["R"],
        calib["T"],
        alpha=0,
    )
    frame_means = []
    all_errors = []
    for pts_l, pts_r in zip(stored_pts_l, stored_pts_r):
        rect_l = cv2.undistortPoints(pts_l, calib["mtx_l"], calib["dist_l"], R=R1, P=P1)
        rect_r = cv2.undistortPoints(pts_r, calib["mtx_r"], calib["dist_r"], R=R2, P=P2)
        errors = np.abs(rect_l[:, 0, 1] - rect_r[:, 0, 1])
        frame_means.append(float(errors.mean()))
        all_errors.extend(errors.tolist())
    return {
        "mean": float(np.mean(all_errors)),
        "p95": float(np.percentile(all_errors, 95)),
        "max_frame_mean": float(np.max(frame_means)),
    }


def compute_calibration():
    print(f"\n=== Evaluating ChArUco stereo calibration from {frame_count} frames ===")
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(stored_objpoints, stored_pts_l, image_size, None, None)
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(stored_objpoints, stored_pts_r, image_size, None, None)
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        stored_objpoints,
        stored_pts_l,
        stored_pts_r,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    calib = {
        "ret_l": float(ret_l),
        "ret_r": float(ret_r),
        "rms": float(rms),
        "mtx_l": mtx_l,
        "dist_l": dist_l,
        "mtx_r": mtx_r,
        "dist_r": dist_r,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "image_size": image_size,
        "frames": frame_count,
    }
    calib["y_error"] = compute_rectified_y_error(calib)
    return calib


def save_calibration(calib, reason):
    np.savez(
        str(CALIB_OUT),
        mtxL=calib["mtx_l"],
        distL=calib["dist_l"],
        mtxR=calib["mtx_r"],
        distR=calib["dist_r"],
        R=calib["R"],
        T=calib["T"],
        E=calib["E"],
        F=calib["F"],
        rms=calib["rms"],
        rms_left=calib["ret_l"],
        rms_right=calib["ret_r"],
        rectified_y_error_mean=calib["y_error"]["mean"],
        rectified_y_error_p95=calib["y_error"]["p95"],
        rectified_y_error_max_frame_mean=calib["y_error"]["max_frame_mean"],
        calibration_frames=calib["frames"],
        calibration_type=np.array("charuco"),
        charuco_squares=np.array([args.squares_x, args.squares_y]),
        square_size_mm=np.array(args.square_size_mm),
        marker_size_mm=np.array(args.marker_size_mm),
        image_size=np.array(calib["image_size"]),
    )
    print(f"Saved {CALIB_OUT} ({reason})")


def evaluate_calibration(final=False):
    global active, best_calibration, best_score, last_eval_frame
    if frame_count < min(args.min_frames, args.max_frames):
        return False
    if not final and frame_count - last_eval_frame < max(1, args.eval_every):
        return False
    last_eval_frame = frame_count

    calib = compute_calibration()
    y = calib["y_error"]
    print(f"Single-camera RMS: L={calib['ret_l']:.4f}px R={calib['ret_r']:.4f}px")
    print(f"Stereo RMS reprojection error: {calib['rms']:.4f} px")
    print(
        "Rectified Y error: "
        f"mean={y['mean']:.4f}px p95={y['p95']:.4f}px max_frame_mean={y['max_frame_mean']:.4f}px"
    )
    print(f"Translation T (mm): {calib['T'].ravel()}")

    score = (y["mean"], y["max_frame_mean"], calib["rms"])
    if best_score is None or score < best_score:
        best_score = score
        best_calibration = calib
        save_calibration(calib, "new best")
    else:
        print(
            "Keeping previous best: "
            f"mean_y={best_score[0]:.4f}px max_frame_mean={best_score[1]:.4f}px rms={best_score[2]:.4f}px"
        )

    target_met = (
        calib["rms"] <= args.target_stereo_rms
        and y["mean"] <= args.target_y_error
        and y["max_frame_mean"] <= args.target_max_y_error
    )
    if target_met:
        print("Calibration targets met; stopping early.")
        send_status("CALIB_DONE")
        active = False
        return True
    if final:
        if best_calibration is not None and best_calibration is not calib:
            save_calibration(best_calibration, "best after max frames")
        print("Reached max frames before all targets were met; saved best calibration.")
        send_status("CALIB_DONE")
        active = False
    return False


def capture_frame():
    global frame_count, image_size
    if not active:
        return
    print(f"\nCapturing frame {frame_count + 1}/{args.max_frames}")
    flush_stale_frames(camera_left, "LEFT")
    flush_stale_frames(camera_right, "RIGHT")
    fire_trigger()

    try:
        grab_l = camera_left.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
        grab_r = camera_right.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
    except Exception as e:
        print(f"  TIMEOUT: {e}")
        send_status("FRAME ERR")
        return

    if not (grab_l.GrabSucceeded() and grab_r.GrabSucceeded()):
        print("  Grab failed")
        grab_l.Release()
        grab_r.Release()
        send_status("FRAME ERR")
        return

    img_l = converter_left.Convert(grab_l).GetArray()
    img_r = converter_right.Convert(grab_r).GetArray()
    grab_l.Release()
    grab_r.Release()
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray_l.shape[::-1]

    corners_l, ids_l, marker_corners_l, marker_ids_l = detect_charuco(gray_l)
    corners_r, ids_r, marker_corners_r, marker_ids_r = detect_charuco(gray_r)
    n_l = 0 if ids_l is None else len(ids_l)
    n_r = 0 if ids_r is None else len(ids_r)
    print(f"  ChArUco corners: L={n_l} R={n_r}")

    crop_l, roi_l = detection_roi(gray_l, corners_l, marker_corners_l)
    crop_r, roi_r = detection_roi(gray_r, corners_r, marker_corners_r)
    sharp_l = cv2.Laplacian(crop_l, cv2.CV_64F).var()
    sharp_r = cv2.Laplacian(crop_r, cv2.CV_64F).var()
    print(f"  Board brightness: L_mean={crop_l.mean():.1f} R_mean={crop_r.mean():.1f}")
    print(f"  Board sharpness:  L={sharp_l:.1f} R={sharp_r:.1f}")
    print(f"  Board ROI: L={roi_l} R={roi_r}")

    cv2.imwrite(str(DEBUG_DIR / "charuco_left.png"), draw_debug(img_l, corners_l, ids_l, marker_corners_l, marker_ids_l))
    cv2.imwrite(str(DEBUG_DIR / "charuco_right.png"), draw_debug(img_r, corners_r, ids_r, marker_corners_r, marker_ids_r))
    cv2.imwrite(str(DEBUG_DIR / "gray_left.png"), gray_l)
    cv2.imwrite(str(DEBUG_DIR / "gray_right.png"), gray_r)
    cv2.imwrite(str(DEBUG_DIR / "crop_left.png"), crop_l)
    cv2.imwrite(str(DEBUG_DIR / "crop_right.png"), crop_r)

    if ids_l is None or ids_r is None:
        print("  -> No ChArUco corners found")
        send_status("FRAME BAD")
        return
    if sharp_l < args.min_sharpness or sharp_r < args.min_sharpness:
        print(f"  -> Need board sharpness at least {args.min_sharpness:.1f} in both cameras")
        send_status("FRAME BAD")
        return

    common, idx_l, idx_r = np.intersect1d(ids_l, ids_r, return_indices=True)
    coverage_x, coverage_y = common_corner_quality(common)
    print(f"  Common corners: {len(common)}  coverage: x={coverage_x:.2f} y={coverage_y:.2f}")
    if len(common) < args.min_corners:
        print(f"  -> Need at least {args.min_corners} common corners")
        send_status("FRAME BAD")
        return
    if coverage_x < args.min_coverage_x or coverage_y < args.min_coverage_y:
        print(
            f"  -> Need coverage at least x={args.min_coverage_x:.2f} "
            f"y={args.min_coverage_y:.2f}"
        )
        send_status("FRAME BAD")
        return

    objp = board_points[common].astype(np.float32)
    pts_l = corners_l[idx_l].astype(np.float32)
    pts_r = corners_r[idx_r].astype(np.float32)

    stored_objpoints.append(objp)
    stored_pts_l.append(pts_l)
    stored_pts_r.append(pts_r)

    idx = frame_count
    cv2.imwrite(str(CALIB_DIR_LEFT / f"calib_{idx:02d}.png"), img_l)
    cv2.imwrite(str(CALIB_DIR_RIGHT / f"calib_{idx:02d}.png"), img_r)

    frame_count += 1
    print(f"  Saved ({frame_count}/{args.max_frames})")
    send_status(f"FRAME {frame_count}/{args.max_frames}")
    if frame_count >= args.max_frames:
        finalize_calibration()
    else:
        evaluate_calibration()


def finalize_calibration():
    global active
    if frame_count < min(args.min_frames, args.max_frames):
        print(f"Only {frame_count} frames; need at least {min(args.min_frames, args.max_frames)}")
        return
    evaluate_calibration(final=True)


def discard_and_exit():
    global active
    print("\n=== Discarding calibration, returning to scan mode ===")
    send_status("SCAN_MODE")
    active = False


class CalibHandler(LineReader):
    def handle_line(self, line):
        line = line.strip()
        print(f"Serial: {line!r}")
        if line == "TRIGGER":
            capture_frame()
        elif line == "CALIBRATE":
            discard_and_exit()


clear_directory(CALIB_DIR_LEFT)
clear_directory(CALIB_DIR_RIGHT)
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
time.sleep(2)
thread = ReaderThread(ser, CalibHandler)
thread.start()
send_status("CALIB_MODE")
print(
    f"ChArUco calibration ready. Need at least {min(args.min_frames, args.max_frames)} frames; "
    f"evaluating every {args.eval_every} accepted frames up to {args.max_frames}."
)
print(
    "Targets: "
    f"stereo RMS <= {args.target_stereo_rms:.2f}px, "
    f"mean rectified Y <= {args.target_y_error:.2f}px, "
    f"worst-frame mean Y <= {args.target_max_y_error:.2f}px."
)

try:
    while active:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    try:
        ser.write(b"SCAN_MODE\n")
        time.sleep(0.1)
    except Exception:
        pass
    try:
        thread.close()
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass
    if camera_left.IsGrabbing():
        camera_left.StopGrabbing()
    if camera_right.IsGrabbing():
        camera_right.StopGrabbing()
    camera_left.Close()
    camera_right.Close()
    try:
        proj.start_scan()
    except Exception:
        pass
    GPIO.cleanup()
    print("Clean shutdown.")
