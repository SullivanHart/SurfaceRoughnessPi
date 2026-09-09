"""
projector.py — LightCrafter 4500 sequence control for Raspberry Pi.

Wraps projector_ctl (compiled C++ binary) via subprocess.
The binary keeps the USB connection open so the DLPC350 maintains its
sequence state.  Commands are sent over stdin so the daemon never restarts
between scan/calibrate/brightness changes — the DLPC350 state is preserved
and no stray scan patterns appear between operations.
"""

import subprocess
import os
import atexit

_BINARY     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bin", "projector_ctl")
_BRIGHTNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bin", "set_brightness")

# DLPC350 LED driver minimum: values below this produce an unreliable sequence
# (device reverts to firmware scan after first trigger).  Determined empirically
BRIGHTNESS_MIN = 20

# The running projector_ctl daemon (holds USB open for the lifetime of
# the Python process).  None if not yet started.
_proc = None
_mode = None   # "scan" | "calibrate" | None
_leds = None   # LED bitmask active in calibration (1=R 2=G 3=RG 4=B 5=RB 6=GB 7=RGB)


def _kill_daemon() -> None:
    """Terminate the daemon and wait for it to exit."""
    global _proc, _mode, _leds
    _mode = None
    _leds = None
    if _proc is not None and _proc.poll() is None:
        try:
            _proc.stdin.write("quit\n")
            _proc.stdin.flush()
        except Exception:
            pass
        try:
            _proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _proc.kill()
            _proc.wait()
    _proc = None


atexit.register(_kill_daemon)


def _read_ready(proc: subprocess.Popen, context: str) -> None:
    """Read stdout lines from proc until READY or process death."""
    while True:
        line = proc.stdout.readline()
        if not line:
            proc.wait()
            stderr = proc.stderr.read().strip()
            raise RuntimeError(
                f"{context} failed (exit {proc.returncode}):\n{stderr}"
            )
        line = line.strip()
        if line:
            print(line)
        if line == "READY":
            return
        if line.startswith("ERROR"):
            raise RuntimeError(f"{context}: {line}")


def _daemon_alive() -> bool:
    return _proc is not None and _proc.poll() is None


def _launch_daemon(args: list) -> None:
    """Kill any existing daemon, start a new one, wait for READY."""
    global _proc, _mode
    _kill_daemon()
    _proc = subprocess.Popen(
        [_BINARY] + args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _read_ready(_proc, f"projector_ctl {' '.join(args)}")


def _send_command(cmd: str) -> None:
    """Send one command to the running daemon and wait for READY."""
    if not _daemon_alive():
        raise RuntimeError("projector daemon is not running")
    _proc.stdin.write(cmd + "\n")
    _proc.stdin.flush()
    _read_ready(_proc, f"projector_ctl {cmd!r}")


def _run_transient(cmd: str) -> None:
    """Run a transient command (sleep/wake) that exits immediately."""
    result = subprocess.run(
        [_BINARY, cmd],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"projector_ctl {cmd} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )


def stop() -> None:
    """Kill the running projector daemon and release the USB connection."""
    _kill_daemon()


def start_scan(count: int = 44, exposure_ms: int = 200, force: bool = False) -> None:
    """
    Switch to a scan sequence (flash slots 0..count-1).
    Reprograms the LUT if not already in scan mode or if force=True.
    """
    global _mode
    if not 1 <= count <= 256:
        raise ValueError(f"scan count must be 1-256, got {count}")
    mode = f"scan:{count}:{exposure_ms}"
    if not force and _daemon_alive() and _mode == mode:
        return
    if _daemon_alive():
        _send_command(f"scan {count} {exposure_ms}")
    else:
        _launch_daemon(["scan", str(count), str(exposure_ms)])
    _mode = mode


def start_scan16() -> None:
    """
    Switch to the legacy 16-pattern sinusoidal scan sequence (flash slots 0–15).
    """
    global _mode
    if _daemon_alive() and _mode == "scan16":
        return
    _launch_daemon(["scan16"])
    _mode = "scan16"


def start_slot(index: int, brightness: int = None, leds: int = 7) -> None:
    """
    Switch to a single repeated flash slot. Useful for projector flash debugging.
    """
    global _mode, _leds
    if not 0 <= index <= 255:
        raise ValueError(f"slot index must be 0-255, got {index}")
    if brightness is not None:
        if not 0 <= brightness <= 255:
            raise ValueError(f"brightness must be 0-255, got {brightness}")
        if brightness < BRIGHTNESS_MIN:
            raise ValueError(
                f"brightness {brightness} is below the DLPC350 minimum safe value "
                f"({BRIGHTNESS_MIN}); the sequence becomes unreliable below this threshold"
            )
    if not 1 <= leds <= 7:
        raise ValueError(f"leds must be 1-7, got {leds}")

    bri_arg = str(brightness) if brightness is not None else "-1"
    _launch_daemon(["slot", str(index), bri_arg, str(leds)])
    _mode = f"slot:{index}"
    _leds = leds


def start_calibration(brightness: int = None, leds: int = 7) -> None:
    """
    Switch to the single checkerboard calibration pattern (flash slot 16).

    brightness: 0-255 LED current (must be >= BRIGHTNESS_MIN), or None to leave unchanged.
    leds:       LED bitmask 1-7 (1=R 2=G 3=RG 4=B 5=RB 6=GB 7=RGB), default 7 (white).

    If already in calibration mode with the same LED combination, only the LED
    current is updated (no LUT reprogram).  If the LED combination changes, the
    daemon is relaunched so the LUT is reprogrammed with the new led_select value.
    """
    global _mode, _leds
    if brightness is not None:
        if not 0 <= brightness <= 255:
            raise ValueError(f"brightness must be 0-255, got {brightness}")
        if brightness < BRIGHTNESS_MIN:
            raise ValueError(
                f"brightness {brightness} is below the DLPC350 minimum safe value "
                f"({BRIGHTNESS_MIN}); the sequence becomes unreliable below this threshold"
            )
    if not 1 <= leds <= 7:
        raise ValueError(f"leds must be 1-7, got {leds}")

    if _daemon_alive() and _mode == "calibrate" and _leds == leds:
        # Same LED combination — just update brightness via SetLedCurrents (no LUT reprogram).
        if brightness is not None:
            try:
                _send_command(f"brightness {brightness}")
                return
            except RuntimeError:
                # SetLedCurrents failed — DLPC350 in bad state; relaunch from scratch.
                pass

    # Relaunch: new LED combination, not yet in calibrate mode, or brightness cmd failed.
    # Pass brightness as -1 if not specified (daemon leaves LED current unchanged).
    bri_arg = str(brightness) if brightness is not None else "-1"
    _launch_daemon(["calibrate", bri_arg, str(leds)])
    _mode = "calibrate"
    _leds = leds


def set_brightness(level: int) -> None:
    """
    Set LED current on all three channels (0-255). Does not affect the
    running pattern sequence. 255 = full, 0 = off.
    """
    if not 0 <= level <= 255:
        raise ValueError(f"brightness must be 0-255, got {level}")
    if level < BRIGHTNESS_MIN:
        raise ValueError(
            f"brightness {level} is below the DLPC350 minimum safe value ({BRIGHTNESS_MIN})"
        )
    result = subprocess.run(
        [_BRIGHTNESS, str(level)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"set_brightness {level} failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    if result.stdout:
        print(result.stdout.strip())


def sleep() -> None:
    """Stop the pattern sequence and put the projector in standby."""
    _kill_daemon()
    _run_transient("sleep")


def wake() -> None:
    """Exit standby. Must be followed by start_scan() or start_calibration()."""
    _run_transient("wake")


if __name__ == "__main__":
    import sys
    cmds = {
        "scan": start_scan,
        "scan16": start_scan16,
        "calibrate": start_calibration,
        "sleep": sleep,
        "wake": wake,
    }
    if len(sys.argv) >= 2 and sys.argv[1] == "slot":
        slot = int(sys.argv[2]) if len(sys.argv) >= 3 else 42
        brightness = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        leds = int(sys.argv[4]) if len(sys.argv) >= 5 else 7
        start_slot(slot, brightness, leds)
        print(f"projector_ctl slot {slot}: OK")
        sys.exit(0)

    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print("Usage: python projector.py scan | scan16 | slot <idx> [brightness] [leds] | calibrate | sleep | wake",
              file=sys.stderr)
        sys.exit(1)
    cmds[sys.argv[1]]()
    print(f"projector_ctl {sys.argv[1]}: OK")
