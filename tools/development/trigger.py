"""
trigger.py — Manual capture tool for focus/alignment tuning.

Press TRIGGER on the handheld button to capture both cameras.
Images are saved to ./focus/ with an incrementing index.
Sharpness (Laplacian variance) is printed after each capture.

Usage:
    python trigger.py                       # calibration pattern, default brightness
    python trigger.py --brightness 34       # explicit brightness
    python trigger.py --exposure 3500       # explicit exposure (µs)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import RPi.GPIO as GPIO
from pypylon import pylon

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "projector"))
import projector as proj

parser = argparse.ArgumentParser()
parser.add_argument("--brightness", type=int, default=34, metavar="0-255")
parser.add_argument("--exposure", type=int, default=250, metavar="US")
args = parser.parse_args()

TRIGGER_PIN      = 17
PULSE_US         = 100
FRAME_TIMEOUT_MS = 3000
SAVE_DIR         = BASE_DIR / "data" / "focus"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# GPIO
# -------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)


def fire_trigger():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(PULSE_US / 1_000_000.0)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)


# -------------------------------
# Cameras
# -------------------------------
factory = pylon.TlFactory.GetInstance()
devices = factory.EnumerateDevices()
if len(devices) < 2:
    GPIO.cleanup()
    raise RuntimeError("Need 2 cameras connected")

camera_left  = pylon.InstantCamera(factory.CreateDevice(devices[1]))
camera_right = pylon.InstantCamera(factory.CreateDevice(devices[0]))
camera_left.Open()
camera_right.Open()
print(f"Left:  {camera_left.GetDeviceInfo().GetModelName()}")
print(f"Right: {camera_right.GetDeviceInfo().GetModelName()}")


def setup_camera(cam, exposure_us):
    cam.ExposureAuto.SetValue("Off")
    cam.ExposureTime.SetValue(float(exposure_us))
    cam.GainAuto.SetValue("Off")
    cam.Gain.SetValue(0.0)
    cam.TriggerSelector.SetValue("FrameStart")
    cam.TriggerMode.SetValue("On")
    cam.TriggerSource.SetValue("Line1")
    cam.TriggerActivation.SetValue("RisingEdge")
    cam.AcquisitionMode.SetValue("Continuous")
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)


setup_camera(camera_left,  args.exposure)
setup_camera(camera_right, args.exposure)

conv_left  = pylon.ImageFormatConverter()
conv_left.OutputPixelFormat  = pylon.PixelType_BGR8packed
conv_left.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

conv_right  = pylon.ImageFormatConverter()
conv_right.OutputPixelFormat  = pylon.PixelType_BGR8packed
conv_right.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


def flush(cam):
    while True:
        try:
            if not cam.GetGrabResultWaitObject().Wait(0):
                break
            r = cam.RetrieveResult(1, pylon.TimeoutHandling_Return)
            if r:
                r.Release()
            else:
                break
        except Exception:
            break


# -------------------------------
# Projector
# -------------------------------
try:
    proj.start_calibration(brightness=args.brightness)
    print(f"Projector: calibration pattern  brightness={args.brightness}  exposure={args.exposure}µs")
except Exception as e:
    print(f"Projector startup failed: {e}", file=sys.stderr)
    camera_left.Close()
    camera_right.Close()
    GPIO.cleanup()
    sys.exit(1)

time.sleep(2.0)
flush(camera_left)
flush(camera_right)

# -------------------------------
# Capture
# -------------------------------
def capture():
    global frame_idx

    flush(camera_left)
    flush(camera_right)
    fire_trigger()

    try:
        grab_l = camera_left.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
        grab_r = camera_right.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
    except Exception as e:
        print(f"  TIMEOUT: {e}")
        return

    if not (grab_l.GrabSucceeded() and grab_r.GrabSucceeded()):
        print(f"  Grab failed — L:{grab_l.GetErrorDescription()}  R:{grab_r.GetErrorDescription()}")
        grab_l.Release()
        grab_r.Release()
        return

    img_l = cv2.flip(conv_left.Convert(grab_l).GetArray(), -1)
    img_r = cv2.flip(conv_right.Convert(grab_r).GetArray(), -1)
    grab_l.Release()
    grab_r.Release()

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    sharp_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
    sharp_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()

    path_l = SAVE_DIR / "left.png"
    path_r = SAVE_DIR / "right.png"
    cv2.imwrite(str(path_l), img_l)
    cv2.imwrite(str(path_r), img_r)

    print(f"  L sharpness={sharp_l:.1f}  mean={gray_l.mean():.1f}"
          f"    R sharpness={sharp_r:.1f}  mean={gray_r.mean():.1f}")


print(f"\nReady — press Enter to capture, Ctrl-C to exit")
print(f"Saving to {SAVE_DIR.resolve()}\n")

try:
    while True:
        input("Enter to capture > ")
        capture()
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    if camera_left.IsGrabbing():
        camera_left.StopGrabbing()
    if camera_right.IsGrabbing():
        camera_right.StopGrabbing()
    camera_left.Close()
    camera_right.Close()
    try:
        proj.stop()
    except Exception:
        pass
    GPIO.cleanup()
    print("Done.")

os._exit(0)
