/*
 * projector_ctl.cpp
 *
 * Minimal LightCrafter 4500 sequence switcher.
 * Uses the DLPC350 USB API directly — no OpenCV, no high-level SDK.
 *
 * Usage:
 *   projector_ctl scan [count]        — Gray-code structured-light patterns, external trigger
 *   projector_ctl scan16              — legacy 16 sinusoidal patterns, external trigger
 *   projector_ctl slot <idx> [brightness] [leds] — repeat one flash slot, external trigger
 *   projector_ctl white [brightness]  — solid white via internal TPG (video mode)
 *
 * Scan mode uses external-positive-edge trigger on TRIG_IN_1; TRIG_OUT_2 syncs cameras.
 * White mode uses the DLPC350 internal test pattern generator — no trigger chain.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>  // usleep

#include <dlp_platforms/lightcrafter_4500/common.hpp>    // must precede dlpc350_api.hpp
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

// ── Configuration ─────────────────────────────────────────────────────────────
// Match these to the values used when the firmware was originally built.

static const int          NUM_GRAY_PATTERNS = 44;    // legacy Gray-code flash image slots 0–43
static const int          NUM_PHASE_PATTERNS = 20;   // v18 hybrid phase-shift slots 0–19
static const int          NUM_LEGACY_PATTERNS = 16;  // flash image slots 0–15
static const int          CALIB_IMAGE_INDEX = 18;    // white image in slot 18 for v18 20-pattern set
static const unsigned int SCAN_EXPOSURE_US  = 500000; // 500 ms (full 8-bit PWM integration)
static const unsigned int CALIB_EXPOSURE_US = 200000; // 200 ms for calibration
static const int          BITDEPTH          = 8;     // MONO_8BPP
static const int          LED_SELECT        = 7;     // WHITE (R+G+B simultaneous)
static const int          PATTERN_NUMBER    = 0;     // bit-plane 0 for 8bpp (G7–G0)

// Trigger output timing (µs). Matches SDK defaults; adjust if your wiring needs it.
static const unsigned int TRIG_OUT_RISING  = 187;
static const unsigned int TRIG_OUT_FALLING = 187;

// ─────────────────────────────────────────────────────────────────────────────

static const int TRIG_EXT_POS = 1;

static const unsigned int DISP_STOP  = 0;
static const unsigned int DISP_START = 2;

static const int TRIG_MODE_EXT = 3;

static void cleanup_and_die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    DLPC350_USB_Close();
    exit(1);
}

static void signal_handler(int)
{
    // On SIGTERM/SIGINT: close the USB connection cleanly but do NOT call
    // DLPC350_USB_Exit() — the DLPC350 resets its sequence when the HID
    // library is torn down, so we leave that to OS file-descriptor cleanup.
    DLPC350_USB_Close();
    _exit(0);
}

/*
 * Program the DLPC350 to play `count` flash images starting at `first_image`
 * using external-positive-edge trigger.  Used for scan mode.
 */
static void start_sequence(int first_image, int count, bool repeat)
{
    // Stop whatever is currently running
    if (DLPC350_PatternDisplay(DISP_STOP) < 0)
        cleanup_and_die("PatternDisplay(STOP) failed");
    usleep(10000);

    if (DLPC350_SetTrigOutConfig(1, false, TRIG_OUT_RISING, TRIG_OUT_FALLING) < 0)
        cleanup_and_die("SetTrigOutConfig(TRIG_OUT_1) failed");
    usleep(10000);

    if (DLPC350_SetTrigOutConfig(2, false, TRIG_OUT_RISING, 0) < 0)
        cleanup_and_die("SetTrigOutConfig(TRIG_OUT_2) failed");
    usleep(10000);

    DLPC350_ClearExpLut();

    for (int i = 0; i < count; i++) {
        if (DLPC350_AddToExpLut(
                TRIG_EXT_POS,   // external positive edge
                PATTERN_NUMBER,
                BITDEPTH,
                LED_SELECT,
                false,          // invert pattern
                true,           // insert black (required for ext trigger)
                true,           // buffer swap
                false,          // trigger_out_share_prev
                SCAN_EXPOSURE_US,
                SCAN_EXPOSURE_US) < 0)
            cleanup_and_die("AddToExpLut failed");
    }

    if (DLPC350_SetPatternDisplayMode(false) < 0)
        cleanup_and_die("SetPatternDisplayMode(flash) failed");

    if (DLPC350_SetPatternTriggerMode(TRIG_MODE_EXT) < 0)
        cleanup_and_die("SetPatternTriggerMode(3) failed");

    // Image LUT: consecutive flash slots
    unsigned char image_lut[256];
    for (int i = 0; i < count; i++)
        image_lut[i] = (unsigned char)(first_image + i);

    if (DLPC350_SendVarExpImageLut(image_lut, (unsigned int)count) < 0)
        cleanup_and_die("SendVarExpImageLut failed");

    if (DLPC350_SendVarExpPatLut() < 0)
        cleanup_and_die("SendVarExpPatLut failed");

    // numPatsForTrigOut2 == numLutEntries for mono (non-RGB) patterns
    if (DLPC350_SetVarExpPatternConfig(
            (unsigned int)count,
            (unsigned int)count,
            (unsigned int)count,
            repeat) < 0)
        cleanup_and_die("SetVarExpPatternConfig failed");

    // Validate — the hardware checks timing, bit-plane numbers, etc.
    if (DLPC350_StartPatLutValidate() < 0)
        cleanup_and_die("StartPatLutValidate failed");

    usleep(100000); // 100 ms: give the device time to run validation

    bool         ready      = false;
    unsigned int val_status = 0;
    for (int polls = 0; !ready && polls < 50; polls++) {
        usleep(10000);
        if (DLPC350_CheckPatLutValidate(&ready, &val_status) < 0)
            cleanup_and_die("CheckPatLutValidate failed");
    }

    if (val_status != 0) {
        fprintf(stderr, "ERROR: Sequence validation failed (status=0x%02X)\n", val_status);
        if (val_status & 0x01) fprintf(stderr, "  bit0: exposure/period out of range\n");
        if (val_status & 0x02) fprintf(stderr, "  bit1: pattern number invalid\n");
        if (val_status & 0x04) fprintf(stderr, "  bit2: trigger overlaps black vector\n");
        if (val_status & 0x08) fprintf(stderr, "  bit3: black vector missing\n");
        if (val_status & 0x10) fprintf(stderr, "  bit4: exposure/period delta < 230 µs\n");
        DLPC350_USB_Close();
        exit(1);
    }

    usleep(10000);

    if (DLPC350_PatternDisplay(DISP_START) < 0)
        cleanup_and_die("PatternDisplay(START) failed");

    usleep(10000);
}


/*
 * Calibration sequence: single LUT entry, buf_swap=true, pointing to
 * CALIB_IMAGE_INDEX (white image, slot 16).
 *
 * A 1-entry loop with buf_swap=true reliably fires TRIG_OUT_2 on every
 * external trigger.  Multi-entry approaches failed because:
 *   - All buf_swap=true with same slot: DLPC350 suppresses TRIG_OUT when
 *     it detects no image change between consecutive loads.
 *   - buf_swap=false on entries 1+: TRIG_OUT_2 is not fired for those entries.
 * The 1-entry loop avoids both problems.
 *
 * brightness >= 0 sets LED current; brightness < 0 leaves it unchanged.
 * To change brightness without reprogramming the LUT, use set_led_brightness().
 */
static void start_single_slot_sequence(int image_index, int brightness, int led_select)
{
    if (DLPC350_PatternDisplay(DISP_STOP) < 0)
        cleanup_and_die("PatternDisplay(STOP) failed");
    usleep(100000);  // 100 ms: let DMD fully stop before reprogramming

    if (DLPC350_SetTrigOutConfig(1, false, TRIG_OUT_RISING, TRIG_OUT_FALLING) < 0)
        cleanup_and_die("SetTrigOutConfig(TRIG_OUT_1) failed");
    usleep(10000);
    if (DLPC350_SetTrigOutConfig(2, false, TRIG_OUT_RISING, 0) < 0)
        cleanup_and_die("SetTrigOutConfig(TRIG_OUT_2) failed");
    usleep(10000);

    if (brightness >= 0) {
        unsigned char v = (unsigned char)brightness;
        if (DLPC350_SetLedCurrents(v, v, v) < 0)
            cleanup_and_die("SetLedCurrents failed");
        usleep(10000);
        fprintf(stdout, "Brightness set to %d\n", brightness);
    }

    DLPC350_ClearExpLut();
    if (DLPC350_AddToExpLut(
            TRIG_EXT_POS,
            PATTERN_NUMBER,
            BITDEPTH,
            led_select,  // 1=R 2=G 3=RG 4=B 5=RB 6=GB 7=RGB
            false,       // invert
            true,        // insert_black (required for ext trigger)
            true,        // buf_swap=true: load from flash every trigger → TRIG_OUT_2 fires
            false,       // trigger_out_share_prev
            CALIB_EXPOSURE_US,
            CALIB_EXPOSURE_US) < 0)
        cleanup_and_die("AddToExpLut (calibrate) failed");

    if (DLPC350_SetPatternDisplayMode(false) < 0)
        cleanup_and_die("SetPatternDisplayMode(flash) failed");
    if (DLPC350_SetPatternTriggerMode(TRIG_MODE_EXT) < 0)
        cleanup_and_die("SetPatternTriggerMode(3) failed");

    unsigned char image_lut[1] = { (unsigned char)image_index };
    if (DLPC350_SendVarExpImageLut(image_lut, 1) < 0)
        cleanup_and_die("SendVarExpImageLut (calibrate) failed");
    if (DLPC350_SendVarExpPatLut() < 0)
        cleanup_and_die("SendVarExpPatLut (calibrate) failed");
    if (DLPC350_SetVarExpPatternConfig(1, 1, 1, true) < 0)
        cleanup_and_die("SetVarExpPatternConfig (calibrate) failed");

    if (DLPC350_StartPatLutValidate() < 0)
        cleanup_and_die("StartPatLutValidate (calibrate) failed");
    usleep(100000);

    bool ready = false; unsigned int val_status = 0;
    for (int p = 0; !ready && p < 50; p++) {
        usleep(10000);
        if (DLPC350_CheckPatLutValidate(&ready, &val_status) < 0)
            cleanup_and_die("CheckPatLutValidate (calibrate) failed");
    }
    if (val_status != 0) {
        fprintf(stderr, "ERROR: Calibration validation failed (0x%02X)\n", val_status);
        if (val_status & 0x01) fprintf(stderr, "  bit0: exposure/period out of range\n");
        if (val_status & 0x02) fprintf(stderr, "  bit1: pattern number invalid\n");
        if (val_status & 0x04) fprintf(stderr, "  bit2: trigger overlaps black vector\n");
        if (val_status & 0x08) fprintf(stderr, "  bit3: black vector missing\n");
        if (val_status & 0x10) fprintf(stderr, "  bit4: exposure/period delta < 230µs\n");
        DLPC350_USB_Close();
        exit(1);
    }

    usleep(10000);
    if (DLPC350_PatternDisplay(DISP_START) < 0)
        cleanup_and_die("PatternDisplay(START) failed");
    usleep(10000);
}

static void start_calibration_sequence(int brightness, int led_select)
{
    start_single_slot_sequence(CALIB_IMAGE_INDEX, brightness, led_select);
}

int main(int argc, char *argv[])
{
    if (argc < 2 ||
        (strcmp(argv[1], "scan")      != 0 &&
         strcmp(argv[1], "scan16")    != 0 &&
         strcmp(argv[1], "slot")      != 0 &&
         strcmp(argv[1], "calibrate") != 0 &&
         strcmp(argv[1], "sleep")     != 0 &&
         strcmp(argv[1], "wake")      != 0))
    {
        fprintf(stderr, "Usage: projector_ctl scan | scan16 | slot <idx> [brightness 0-255] [leds 1-7] | calibrate [brightness 0-255] | sleep | wake\n");
        return 1;
    }

    const char *cmd = argv[1];

    // Optional args for calibrate: projector_ctl calibrate [brightness] [leds]
    //   brightness: 0-255, or -1 to leave unchanged (default -1)
    //   leds:       1-7 bitmask (1=R 2=G 3=RG 4=B 5=RB 6=GB 7=RGB), default 7
    int brightness = -1;
    int led_select = LED_SELECT;
    int slot_index = CALIB_IMAGE_INDEX;
    int scan_count = NUM_GRAY_PATTERNS;
    if (strcmp(cmd, "slot") == 0) {
        if (argc < 3) {
            fprintf(stderr, "ERROR: slot requires an image index\n");
            return 1;
        }
        slot_index = atoi(argv[2]);
        if (slot_index < 0 || slot_index > 255) {
            fprintf(stderr, "ERROR: slot index must be 0-255 (got %d)\n", slot_index);
            return 1;
        }
        if (argc >= 4) {
            brightness = atoi(argv[3]);
            if (brightness != -1 && (brightness < 0 || brightness > 255)) {
                fprintf(stderr, "ERROR: brightness must be 0-255 (got %d)\n", brightness);
                return 1;
            }
        }
        if (argc >= 5) {
            led_select = atoi(argv[4]);
            if (led_select < 1 || led_select > 7) {
                fprintf(stderr, "ERROR: leds must be 1-7 (got %d)\n", led_select);
                return 1;
            }
        }
    } else if (strcmp(cmd, "scan") == 0) {
        if (argc >= 3) {
            scan_count = atoi(argv[2]);
            if (scan_count < 1 || scan_count > 256) {
                fprintf(stderr, "ERROR: scan count must be 1-256 (got %d)\n", scan_count);
                return 1;
            }
        }
    } else if (strcmp(cmd, "calibrate") == 0) {
        if (argc >= 3) {
            brightness = atoi(argv[2]);
            if (brightness != -1 && (brightness < 0 || brightness > 255)) {
                fprintf(stderr, "ERROR: brightness must be 0-255 (got %d)\n", brightness);
                return 1;
            }
        }
        if (argc >= 4) {
            led_select = atoi(argv[3]);
            if (led_select < 1 || led_select > 7) {
                fprintf(stderr, "ERROR: leds must be 1-7 (got %d)\n", led_select);
                return 1;
            }
        }
    }

    // Connect over USB
    // Install signal handlers before opening USB so they're always active.
    // On SIGTERM/SIGINT we close USB cleanly but skip USB_Exit() — calling
    // hid_exit() causes the DLPC350 to reset its sequence state.
    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);

    if (DLPC350_USB_Init() < 0)
        cleanup_and_die("USB init failed");

    {
        // The DLPC350 re-enumerates on USB after the previous session closes.
        // Retry for up to 5 seconds to let the device come back.
        int opened = -1;
        for (int i = 0; i < 10 && opened < 0; i++) {
            opened = DLPC350_USB_Open();
            if (opened < 0) usleep(500000);  // 500 ms
        }
        if (opened < 0) {
            fprintf(stderr, "ERROR: USB open failed after retries "
                            "(check USB cable and udev rule)\n");
            return 1;
        }
    }

    // After USB (re-)connect the DLPC350 may be running firmware auto-scan.
    // Give it time to settle, then issue a best-effort stop before any mode
    // changes — without this, commands sent while the DMD is mid-cycle are
    // silently ignored and the device stays on the firmware sequence.
    usleep(300000);                   // 300 ms: let device finish any in-progress cycle
    DLPC350_PatternDisplay(DISP_STOP); // best-effort — ignore return value
    usleep(200000);                   // 200 ms: settle after stop

    if (strcmp(cmd, "sleep") == 0) {
        if (DLPC350_PatternDisplay(DISP_STOP) < 0)
            cleanup_and_die("PatternDisplay(STOP) failed");
        usleep(10000);
        if (DLPC350_SetPowerMode(true) < 0)
            cleanup_and_die("SetPowerMode(standby) failed");
        // sleep/wake are transient commands — exit immediately after issuing.
        DLPC350_USB_Close();
        return 0;
    } else if (strcmp(cmd, "wake") == 0) {
        if (DLPC350_SetPowerMode(false) < 0)
            cleanup_and_die("SetPowerMode(normal) failed");
        usleep(50000);
        if (DLPC350_SetMode(true) < 0)
            cleanup_and_die("SetMode(pattern_sequence) failed");
        usleep(10000);
        DLPC350_USB_Close();
        return 0;
    } else {
        // scan or calibrate: program the LUT, then stay alive to keep USB open.
        // The DLPC350 resets its sequence state when the USB connection closes,
        // so we must hold the connection for the duration of the scan/calibration.
        if (DLPC350_SetPowerMode(false) < 0)
            cleanup_and_die("SetPowerMode(normal) failed");
        usleep(50000);
        if (DLPC350_SetMode(true) < 0)
            cleanup_and_die("SetMode(pattern_sequence) failed");
        usleep(50000);
        if (strcmp(cmd, "scan") == 0) {
            start_sequence(0, scan_count, true);
        } else if (strcmp(cmd, "scan16") == 0) {
            start_sequence(0, NUM_LEGACY_PATTERNS, true);
        } else if (strcmp(cmd, "slot") == 0) {
            start_single_slot_sequence(slot_index, brightness, led_select);
        } else {
            start_calibration_sequence(brightness, led_select);
        }
    }

    // Signal Python that the sequence is running and the projector is ready.
    printf("READY\n");
    fflush(stdout);

    // Command loop: accept reprogram commands over stdin so the daemon never
    // needs to restart (and the USB connection — and DLPC350 state — stays up).
    //
    // Protocol: Python writes one command per line; daemon reprograms and
    // replies "READY\n" on success or "ERROR: <msg>\n" on failure.
    //
    // Commands:
    //   calibrate [brightness]   — switch to calibration sequence
    //   slot <idx> [brightness [leds]] — switch to one repeated flash slot
    //   scan [count]             — switch to Gray-code scan sequence
    //   scan16                   — switch to legacy 16-pattern scan sequence
    //   quit                     — clean exit
    //
    char line[256];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        // strip trailing whitespace/newline
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r' || line[len-1] == ' '))
            line[--len] = '\0';

        if (len == 0) continue;

        if (strcmp(line, "quit") == 0) {
            break;
        } else if (strncmp(line, "brightness", 10) == 0) {
            // Change LED current only — no LUT reprogram, no DISP_STOP.
            // Avoids the DLPC350 SDK heap corruption from calling DISP_STOP twice.
            if (line[10] != ' ' || line[11] == '\0') {
                fprintf(stdout, "ERROR: usage: brightness <0-255>\n");
                fflush(stdout);
                continue;
            }
            int bri = atoi(line + 11);
            if (bri < 0 || bri > 255) {
                fprintf(stdout, "ERROR: brightness must be 0-255\n");
                fflush(stdout);
                continue;
            }
            unsigned char v = (unsigned char)bri;
            if (DLPC350_SetLedCurrents(v, v, v) < 0) {
                fprintf(stdout, "ERROR: SetLedCurrents failed\n");
                fflush(stdout);
                continue;
            }
            fprintf(stdout, "Brightness set to %d\n", bri);
            printf("READY\n");
            fflush(stdout);
        } else if (strncmp(line, "slot", 4) == 0) {
            // "slot <idx> [brightness [leds]]"
            if (line[4] != ' ' || line[5] == '\0') {
                fprintf(stdout, "ERROR: usage: slot <idx> [brightness [leds]]\n");
                fflush(stdout);
                continue;
            }
            char *p = line + 5;
            int idx = atoi(p);
            if (idx < 0 || idx > 255) {
                fprintf(stdout, "ERROR: slot index must be 0-255\n");
                fflush(stdout);
                continue;
            }
            int bri = -1;
            int lsel = LED_SELECT;
            p = strchr(p, ' ');
            if (p && *(p + 1)) {
                bri = atoi(p + 1);
                if (bri != -1 && (bri < 0 || bri > 255)) {
                    fprintf(stdout, "ERROR: brightness must be 0-255\n");
                    fflush(stdout);
                    continue;
                }
                p = strchr(p + 1, ' ');
                if (p && *(p + 1)) {
                    lsel = atoi(p + 1);
                    if (lsel < 1 || lsel > 7) {
                        fprintf(stdout, "ERROR: leds must be 1-7\n");
                        fflush(stdout);
                        continue;
                    }
                }
            }
            start_single_slot_sequence(idx, bri, lsel);
            printf("READY\n");
            fflush(stdout);
        } else if (strncmp(line, "calibrate", 9) == 0) {
            // "calibrate [brightness [leds]]"
            int bri  = -1;
            int lsel = LED_SELECT;
            if (line[9] == ' ' && line[10] != '\0') {
                bri = atoi(line + 10);
                if (bri != -1 && (bri < 0 || bri > 255)) {
                    fprintf(stdout, "ERROR: brightness must be 0-255\n");
                    fflush(stdout);
                    continue;
                }
                const char *p = strchr(line + 10, ' ');
                if (p && *(p + 1)) {
                    lsel = atoi(p + 1);
                    if (lsel < 1 || lsel > 7) {
                        fprintf(stdout, "ERROR: leds must be 1-7\n");
                        fflush(stdout);
                        continue;
                    }
                }
            }
            start_calibration_sequence(bri, lsel);
            printf("READY\n");
            fflush(stdout);
        } else if (strncmp(line, "scan", 4) == 0 && (line[4] == '\0' || line[4] == ' ')) {
            int count = NUM_GRAY_PATTERNS;
            if (line[4] == ' ' && line[5] != '\0') {
                count = atoi(line + 5);
                if (count < 1 || count > 256) {
                    fprintf(stdout, "ERROR: scan count must be 1-256\n");
                    fflush(stdout);
                    continue;
                }
            }
            start_sequence(0, count, true);
            printf("READY\n");
            fflush(stdout);
        } else if (strcmp(line, "scan16") == 0) {
            start_sequence(0, NUM_LEGACY_PATTERNS, true);
            printf("READY\n");
            fflush(stdout);
        } else {
            fprintf(stdout, "ERROR: unknown command '%s'\n", line);
            fflush(stdout);
        }
    }

    // stdin closed or "quit" received — exit cleanly
    DLPC350_USB_Close();
    return 0;
}
