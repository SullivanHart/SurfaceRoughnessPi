"""
Send one clean trigger pulse on GPIO 17.
"""

import argparse
import time

import RPi.GPIO as GPIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", type=int, default=17)
    parser.add_argument("--pulse-us", type=int, default=100)
    args = parser.parse_args()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(args.pin, GPIO.OUT, initial=GPIO.LOW)
    try:
        GPIO.output(args.pin, GPIO.HIGH)
        time.sleep(args.pulse_us / 1_000_000.0)
        GPIO.output(args.pin, GPIO.LOW)
        print(f"Pulsed GPIO {args.pin} for {args.pulse_us} us")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
