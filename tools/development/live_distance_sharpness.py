"""
Continuously capture stereo frames and print focus/distance quality metrics.

Use this while moving the scanner toward/away from a wall or target. The script
projects the calibration pattern, triggers both cameras, and prints sharpness,
brightness, and saturation until stopped with Ctrl-C.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import RPi.GPIO as GPIO
from pypylon import pylon

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR / "projector"))
import projector as proj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--brightness", type=int, default=34, metavar="0-255")
    parser.add_argument("--exposure", type=int, default=250, metavar="US")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between captures")
    parser.add_argument("--crop", type=float, default=0.6,
                        help="Center crop fraction used for metrics, 1.0 uses whole image")
    parser.add_argument("--save", action="store_true", help="Save latest images to focus/")
    parser.add_argument("--no-projector", action="store_true", help="Do not change projector state")
    return parser.parse_args()


TRIGGER_PIN = 17
PULSE_US = 100
FRAME_TIMEOUT_MS = 3000
SAVE_DIR = BASE_DIR / "data" / "focus"


def fire_trigger():
    GPIO.output(TRIGGER_PIN, GPIO.HIGH)
    time.sleep(PULSE_US / 1_000_000.0)
    GPIO.output(TRIGGER_PIN, GPIO.LOW)


def flush(cam):
    while True:
        try:
            if not cam.GetGrabResultWaitObject().Wait(0):
                break
            result = cam.RetrieveResult(1, pylon.TimeoutHandling_Return)
            if result:
                result.Release()
            else:
                break
        except Exception:
            break


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


def center_crop(gray, fraction):
    fraction = min(max(fraction, 0.05), 1.0)
    if fraction >= 0.999:
        return gray
    h, w = gray.shape[:2]
    cw = int(w * fraction)
    ch = int(h * fraction)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return gray[y0:y0 + ch, x0:x0 + cw]


def metrics(gray, crop_fraction):
    roi = center_crop(gray, crop_fraction)
    saturated = float(np.mean(roi >= 245) * 100.0)
    dark = float(np.mean(roi <= 5) * 100.0)
    return {
        "sharp": float(cv2.Laplacian(roi, cv2.CV_64F).var()),
        "mean": float(roi.mean()),
        "p05": float(np.percentile(roi, 5)),
        "p95": float(np.percentile(roi, 95)),
        "sat": saturated,
        "dark": dark,
    }


def quality_label(sharp):
    if sharp < 100:
        return "soft"
    if sharp < 500:
        return "usable"
    if sharp < 1000:
        return "good"
    return "sharp"


def main():
    args = parse_args()
    if args.save:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)

    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) < 2:
        GPIO.cleanup()
        raise RuntimeError("Need 2 cameras connected")

    camera_left = pylon.InstantCamera(factory.CreateDevice(devices[1]))
    camera_right = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    camera_left.Open()
    camera_right.Open()
    print(f"Left:  {camera_left.GetDeviceInfo().GetModelName()}")
    print(f"Right: {camera_right.GetDeviceInfo().GetModelName()}")

    setup_camera(camera_left, args.exposure)
    setup_camera(camera_right, args.exposure)

    converter_left = pylon.ImageFormatConverter()
    converter_left.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter_left.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    converter_right = pylon.ImageFormatConverter()
    converter_right.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter_right.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    try:
        if not args.no_projector:
            proj.start_calibration(brightness=args.brightness)
            print(f"Projector: calibration pattern  brightness={args.brightness}  exposure={args.exposure}us")
            time.sleep(2.0)

        flush(camera_left)
        flush(camera_right)
        print("Move the scanner/target and maximize sharpness. Ctrl-C to stop.")
        print("Columns use the center crop unless --crop 1.0 is passed.\n")
        print(
            f"{'#':>4}  {'L sharp':>9} {'L q':>6} {'L mean':>7} {'L p05':>6} {'L p95':>6} {'L sat%':>7}  "
            f"{'R sharp':>9} {'R q':>6} {'R mean':>7} {'R p05':>6} {'R p95':>6} {'R sat%':>7}"
        )

        idx = 0
        while True:
            flush(camera_left)
            flush(camera_right)
            fire_trigger()

            grab_l = camera_left.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
            grab_r = camera_right.RetrieveResult(FRAME_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException)
            if not (grab_l.GrabSucceeded() and grab_r.GrabSucceeded()):
                print(f"{idx:4d}  grab failed")
                grab_l.Release()
                grab_r.Release()
                time.sleep(args.interval)
                idx += 1
                continue

            img_l = converter_left.Convert(grab_l).GetArray()
            img_r = converter_right.Convert(grab_r).GetArray()
            grab_l.Release()
            grab_r.Release()

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            m_l = metrics(gray_l, args.crop)
            m_r = metrics(gray_r, args.crop)

            if args.save:
                cv2.imwrite(str(SAVE_DIR / "left_live.png"), img_l)
                cv2.imwrite(str(SAVE_DIR / "right_live.png"), img_r)

            print(
                f"{idx:4d}  "
                f"{m_l['sharp']:9.1f} {quality_label(m_l['sharp']):>6} {m_l['mean']:7.1f} "
                f"{m_l['p05']:6.1f} {m_l['p95']:6.1f} {m_l['sat']:7.2f}  "
                f"{m_r['sharp']:9.1f} {quality_label(m_r['sharp']):>6} {m_r['mean']:7.1f} "
                f"{m_r['p05']:6.1f} {m_r['p95']:6.1f} {m_r['sat']:7.2f}",
                flush=True,
            )
            idx += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if camera_left.IsGrabbing():
            camera_left.StopGrabbing()
        if camera_right.IsGrabbing():
            camera_right.StopGrabbing()
        camera_left.Close()
        camera_right.Close()
        try:
            if not args.no_projector:
                proj.stop()
        except Exception:
            pass
        GPIO.cleanup()
        print("Clean shutdown.")


if __name__ == "__main__":
    main()
