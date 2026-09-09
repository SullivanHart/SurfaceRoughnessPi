"""
Standalone stereo calibration script.

Usage: python3 calibrate.py

Hold a physical checkerboard in view of both cameras.
- TRIGGER press  → fire projector TRIG_IN → cameras capture via Line1
- TRIGGER hold 5s → discard and exit (sends SCAN_MODE to Arduino)
- Automatically saves calibration.npz when MAX_CALIB_FRAMES valid frames collected.
"""

from pypylon import pylon
import serial
from serial.threaded import LineReader, ReaderThread
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import argparse
import RPi.GPIO as GPIO

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "projector"))
import projector as proj

parser = argparse.ArgumentParser(description="Stereo calibration capture")
parser.add_argument("--brightness", type=int, default=None,
                    metavar="0-255",
                    help="LED brightness to set after projector starts (default: leave as-is)")
parser.add_argument("--exposure", type=int, default=1000, metavar="US")
parser.add_argument("--square-size-mm", type=float, default=10.0)
parser.add_argument("--checkerboard", default="11x8",
                    help="Inner-corner count as COLSxROWS, e.g. 11x8 for a 12x9-square board")
parser.add_argument("--max-frames", type=int, default=30)
parser.add_argument("--min-frames", type=int, default=10)
parser.add_argument("--fallback-detectors", action="store_true",
                    help="Try slower inverted/classic/transposed detectors if the primary detector fails")
args = parser.parse_args()


def parse_checkerboard(value: str) -> tuple[int, int]:
    try:
        cols, rows = value.lower().split("x", 1)
        cols, rows = int(cols), int(rows)
    except Exception as exc:
        raise argparse.ArgumentTypeError("checkerboard must look like COLSxROWS, e.g. 11x8") from exc
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError("checkerboard dimensions must both be >= 2")
    return cols, rows

# -------------------------------
# Configuration
# -------------------------------
CALIB_DIR_LEFT  = BASE_DIR / "data" / "calibration" / "left"
CALIB_DIR_RIGHT = BASE_DIR / "data" / "calibration" / "right"
DEBUG_DIR       = BASE_DIR / "data" / "calibration" / "debug"   # last captured pair, always overwritten
CALIB_OUT       = BASE_DIR / "config" / "calibration.npz"

CHECKERBOARD = parse_checkerboard(args.checkerboard)
SQUARE_SIZE_MM = args.square_size_mm
MIN_CALIB_FRAMES = min(args.min_frames, args.max_frames)
MAX_CALIB_FRAMES = args.max_frames

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200

TRIGGER_PIN      = 17   # GPIO pin (used by scan.py; kept for GPIO init)
EXPOSURE_US      = args.exposure
FRAME_TIMEOUT_MS = 3000

CALIB_DIR_LEFT.mkdir(parents=True, exist_ok=True)
CALIB_DIR_RIGHT.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# GPIO
# -------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)


def fire_trigger():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(100 / 1_000_000.0)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)


# -------------------------------
# Camera setup
# -------------------------------
factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()

if len(devices) < 2:
    raise RuntimeError("Need 2 cameras connected")

camera_left  = pylon.InstantCamera(factory.CreateDevice(devices[1]))
camera_right = pylon.InstantCamera(factory.CreateDevice(devices[0]))

camera_left.Open()
camera_right.Open()

print("Left: ", camera_left.GetDeviceInfo().GetModelName())
print("Right:", camera_right.GetDeviceInfo().GetModelName())


def setup_camera(cam):
    cam.ExposureAuto.SetValue("Off")
    cam.ExposureTime.SetValue(float(EXPOSURE_US))
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

converter_left  = pylon.ImageFormatConverter()
converter_left.OutputPixelFormat  = pylon.PixelType_BGR8packed
converter_left.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

converter_right  = pylon.ImageFormatConverter()
converter_right.OutputPixelFormat  = pylon.PixelType_BGR8packed
converter_right.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


# -------------------------------
# Projector startup
# -------------------------------
try:
    brightness_arg = args.brightness  # passed directly to start_calibration
    proj.start_calibration(brightness=brightness_arg)
    print("Projector: calibration sequence ready (continuous, no GPIO trigger needed)")
except Exception as e:
    print(f"Projector startup failed: {e}")
    GPIO.cleanup()
    camera_left.Close()
    camera_right.Close()
    sys.exit(1)

# Let projector settle before first capture
time.sleep(2.0)


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


# Flush any frames accumulated during projector startup settle
flush_stale_frames(camera_left,  "LEFT")
flush_stale_frames(camera_right, "RIGHT")


def clear_directory(path: Path):
    for f in path.iterdir():
        if f.is_file():
            f.unlink()


def send_status(msg: str):
    try:
        ser.write((msg + "\n").encode())
    except Exception as e:
        print(f"Serial write failed: {e}")


# -------------------------------
# State
# -------------------------------
calib_frame_count = 0
active = True       # set False to trigger clean shutdown from serial thread
stored_objpoints = []
stored_pts_l = []
stored_pts_r = []


# -------------------------------
# Calibration logic
# -------------------------------
def capture_frame():
    global calib_frame_count

    print(f"\nCapturing frame {calib_frame_count + 1}/{MAX_CALIB_FRAMES}")

    flush_stale_frames(camera_left,  "LEFT")
    flush_stale_frames(camera_right, "RIGHT")

    fire_trigger()
    # Both cameras trigger on TRIG_OUT_2 (Line1) from the projector.

    try:
        grab_left  = camera_left.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
        grab_right = camera_right.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
    except Exception as e:
        print(f"  TIMEOUT — camera did not respond within {FRAME_TIMEOUT_MS}ms: {e}")
        send_status("FRAME ERR")
        return

    print(f"  GrabSucceeded: L={grab_left.GrabSucceeded()} R={grab_right.GrabSucceeded()}")

    if not (grab_left.GrabSucceeded() and grab_right.GrabSucceeded()):
        print(f"  Grab failed — "
              f"L: {grab_left.GetErrorDescription()}  "
              f"R: {grab_right.GetErrorDescription()}")
        grab_left.Release()
        grab_right.Release()
        send_status("FRAME ERR")
        return

    img_left  = converter_left.Convert(grab_left).GetArray()
    img_right = converter_right.Convert(grab_right).GetArray()
    grab_left.Release()
    grab_right.Release()

    print(f"  Shape: L={img_left.shape} R={img_right.shape}")
    print(f"  Brightness: L_mean={img_left.mean():.1f} R_mean={img_right.mean():.1f}")

    gray_l = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    sharp_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
    sharp_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
    print(f"  Sharpness (Laplacian var): L={sharp_l:.1f} R={sharp_r:.1f}  (>100 is usable, >500 is good)")

    # Try the declared board orientation first, then the transposed orientation
    # in case the board is rotated 90 degrees in the image.  SB with exhaustive
    # search is most robust; classic detection is kept as a fallback.
    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_EXHAUSTIVE |
        cv2.CALIB_CB_ACCURACY
    )
    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_FILTER_QUADS
    )
    attempts = [(f"SB exhaustive {CHECKERBOARD}", CHECKERBOARD, "sb", sb_flags, False)]
    if args.fallback_detectors:
        for size in (CHECKERBOARD, (CHECKERBOARD[1], CHECKERBOARD[0])):
            attempts.extend([
                (f"SB exhaustive inverted {size}", size, "sb", sb_flags, True),
                (f"classic {size}", size, "classic", classic_flags, False),
                (f"classic inverted {size}", size, "classic", classic_flags, True),
            ])

    ret_l = ret_r = False
    corners_l = corners_r = None
    CHECKERBOARD_USED = CHECKERBOARD
    for label, size, detector, flags, invert in attempts:
        test_l = 255 - gray_l if invert else gray_l
        test_r = 255 - gray_r if invert else gray_r
        if detector == "sb":
            rl, cl = cv2.findChessboardCornersSB(test_l, size, flags)
            rr, cr = cv2.findChessboardCornersSB(test_r, size, flags)
        else:
            rl, cl = cv2.findChessboardCorners(test_l, size, flags)
            rr, cr = cv2.findChessboardCorners(test_r, size, flags)
        print(f"  {label}: L={rl} R={rr}")
        if rl and rr:
            ret_l, corners_l = rl, cl
            ret_r, corners_r = rr, cr
            CHECKERBOARD_USED = size
            print(f"  ^^^ MATCH — using {label}")
            break

    # Save annotated debug images so you can see what OpenCV detected
    dbg_l = img_left.copy()
    dbg_r = img_right.copy()
    cv2.drawChessboardCorners(dbg_l, CHECKERBOARD_USED, corners_l, ret_l)
    cv2.drawChessboardCorners(dbg_r, CHECKERBOARD_USED, corners_r, ret_r)
    cv2.imwrite(str(DEBUG_DIR / "debug_left.png"),  dbg_l)
    cv2.imwrite(str(DEBUG_DIR / "debug_right.png"), dbg_r)
    # Also save raw grayscale for inspection
    cv2.imwrite(str(DEBUG_DIR / "gray_left.png"),  gray_l)
    cv2.imwrite(str(DEBUG_DIR / "gray_right.png"), gray_r)
    print(f"  Debug images saved to {DEBUG_DIR}/")

    if not (ret_l and ret_r):
        print("  → No corners found in any configuration")
        send_status("FRAME BAD")
        return

    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

    objp = np.zeros((CHECKERBOARD_USED[0] * CHECKERBOARD_USED[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_USED[0], 0:CHECKERBOARD_USED[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM
    stored_objpoints.append(objp)
    stored_pts_l.append(corners_l)
    stored_pts_r.append(corners_r)

    idx = calib_frame_count
    cv2.imwrite(str(CALIB_DIR_LEFT  / f"calib_{idx:02d}.png"), img_left)
    cv2.imwrite(str(CALIB_DIR_RIGHT / f"calib_{idx:02d}.png"), img_right)

    calib_frame_count += 1
    print(f"  Saved ({calib_frame_count}/{MAX_CALIB_FRAMES})")
    send_status(f"FRAME {calib_frame_count}/{MAX_CALIB_FRAMES}")

    if calib_frame_count >= MAX_CALIB_FRAMES:
        finalize_calibration()


def finalize_calibration():
    global active

    if calib_frame_count < MIN_CALIB_FRAMES:
        print(f"Only {calib_frame_count} frames — need at least {MIN_CALIB_FRAMES}")
        return

    print(f"\n=== Computing stereo calibration from {len(stored_objpoints)} frames ===")
    send_status("CALIB_COMPUTING")

    # Use corners detected in-memory (avoids re-running detector on saved images)
    sample = cv2.imread(str(CALIB_DIR_LEFT / "calib_00.png"), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        img_shape = (1920, 1200)
    else:
        img_shape = sample.shape[::-1]

    print(f"Running calibrateCamera on {len(stored_objpoints)} frames...")
    _, mtx_l, dist_l, _, _ = cv2.calibrateCamera(stored_objpoints, stored_pts_l, img_shape, None, None)
    _, mtx_r, dist_r, _, _ = cv2.calibrateCamera(stored_objpoints, stored_pts_r, img_shape, None, None)

    print("Running stereoCalibrate...")
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        stored_objpoints, stored_pts_l, stored_pts_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        img_shape,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print(f"Stereo RMS reprojection error: {rms:.4f} px")
    print(f"Translation T (mm): {T.ravel()}")
    np.savez(
        str(CALIB_OUT),
        mtxL=mtx_l, distL=dist_l,
        mtxR=mtx_r, distR=dist_r,
        R=R, T=T, E=E, F=F,
        rms=rms,
        checkerboard=np.array(CHECKERBOARD),
        square_size_mm=np.array(SQUARE_SIZE_MM),
        image_size=np.array(img_shape)
    )
    print(f"Saved {CALIB_OUT}")
    send_status("CALIB_DONE")
    active = False


def discard_and_exit():
    global active
    print("\n=== Discarding calibration, returning to scan mode ===")
    send_status("SCAN_MODE")
    active = False


# -------------------------------
# Serial handler
# -------------------------------
class CalibHandler(LineReader):
    def handle_line(self, line):
        line = line.strip()
        print(f"Serial: {line!r}")
        if line in ("TRIGGER", "CALIB_TRIGGER"):
            capture_frame()
        elif line in ("CALIBRATE", "CALIB_OFF"):
            if calib_frame_count >= MIN_CALIB_FRAMES:
                finalize_calibration()
            else:
                discard_and_exit()
        elif line == "CALIB_ON":
            send_status("CALIB_MODE")


# -------------------------------
# Main
# -------------------------------
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
time.sleep(2)   # let Arduino reset after serial open

# Clear any leftover images from a previous session
clear_directory(CALIB_DIR_LEFT)
clear_directory(CALIB_DIR_RIGHT)

thread = ReaderThread(ser, CalibHandler)
thread.start()

send_status("CALIB_MODE")
print(f"Calibration ready — TRIGGER to capture, hold 5s to discard and exit")
print(f"Need {MIN_CALIB_FRAMES}–{MAX_CALIB_FRAMES} valid frames")

try:
    while active:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    # Stop serial thread before touching cameras to avoid segfault
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
        proj.start_scan()   # return projector to scan sequence on exit
    except Exception:
        pass

    GPIO.cleanup()
    print("Clean shutdown.")
