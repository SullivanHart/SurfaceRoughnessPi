/*
 * projector_diag.cpp
 *
 * Programs the calibration sequence then continuously polls the DLPC350
 * status, printing any changes.  Run this in one terminal while firing
 * GPIO triggers from Python in another — it will show exactly when and
 * what changes after each trigger.
 *
 * Usage:
 *   projector_diag          — run until Ctrl-C
 *   projector_diag <secs>   — run for N seconds
 *
 * Build: add to CMakeLists.txt (see below), then make -j4 in build/
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <dlp_platforms/lightcrafter_4500/common.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

// ── Match projector_ctl.cpp config ────────────────────────────────────────────
static const int          CALIB_IMAGE_INDEX = 16;
static const int          NUM_ENTRIES       = 16;
static const unsigned int EXPOSURE_US       = 500000;
static const int          BITDEPTH          = 8;
static const int          LED_SELECT        = 7;
static const int          PATTERN_NUMBER    = 0;
static const unsigned int TRIG_OUT_RISING   = 187;
static const unsigned int TRIG_OUT_FALLING  = 187;
static const int          TRIG_EXT_POS      = 1;
static const unsigned int DISP_STOP         = 0;
static const unsigned int DISP_START        = 2;
static const int          TRIG_MODE_EXT     = 3;

static void die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    DLPC350_USB_Close();
    DLPC350_USB_Exit();
    exit(1);
}

// ── Read and print the full projector state ───────────────────────────────────
struct State {
    bool     mode_pattern;        // true=pattern, false=video
    unsigned disp_action;         // 0=stop,1=pause,2=start
    int      trig_mode;
    unsigned num_entries;
    unsigned num_pats_trig2;
    unsigned num_images;
    bool     repeat;
    unsigned char hw, sys, main_s;
    unsigned char led_r, led_g, led_b;
};

static bool read_state(State &s)
{
    bool mode = false;
    if (DLPC350_GetMode(&mode) < 0) return false;
    s.mode_pattern = mode;

    unsigned action = 0;
    if (DLPC350_GetPatternDisplay(&action) < 0) return false;
    s.disp_action = action;

    int tm = 0;
    if (DLPC350_GetPatternTriggerMode(&tm) < 0) return false;
    s.trig_mode = tm;

    unsigned ne = 0, np = 0, ni = 0; bool rep = false;
    DLPC350_GetVarExpPatternConfig(&ne, &np, &ni, &rep);
    s.num_entries = ne;
    s.num_pats_trig2 = np;
    s.num_images = ni;
    s.repeat = rep;

    unsigned char hw = 0, sys = 0, ms = 0;
    DLPC350_GetStatus(&hw, &sys, &ms);
    s.hw = hw; s.sys = sys; s.main_s = ms;

    unsigned char r = 0, g = 0, b = 0;
    DLPC350_GetLedCurrents(&r, &g, &b);
    s.led_r = r; s.led_g = g; s.led_b = b;

    return true;
}

static void print_state(const State &s, const char *label)
{
    const char *disp_str =
        s.disp_action == 0 ? "STOPPED" :
        s.disp_action == 1 ? "PAUSED"  : "RUNNING";

    // MainStatus bits (DLPC350 datasheet Table 2-65):
    // bit0=DMD parked, bit1=seq aborted, bit2=seq active, bit3=seq running,
    // bit4=seq waiting for trigger, bit5=frame buffer freeze
    printf("[%s]\n", label);
    printf("  Mode         : %s\n",        s.mode_pattern ? "PATTERN" : "VIDEO");
    printf("  Display      : %s (action=%u)\n", disp_str, s.disp_action);
    printf("  TriggerMode  : %d\n",        s.trig_mode);
    printf("  LUT entries  : %u  repeat=%s\n", s.num_entries, s.repeat ? "yes" : "no");
    printf("  LEDs         : R=%u G=%u B=%u\n", s.led_r, s.led_g, s.led_b);
    printf("  HW status    : 0x%02X\n",    s.hw);
    printf("  Sys status   : 0x%02X\n",    s.sys);
    printf("  Main status  : 0x%02X", s.main_s);
    if (s.main_s & 0x02) printf("  [SEQ ABORTED]");
    if (s.main_s & 0x04) printf("  [SEQ ACTIVE]");
    if (s.main_s & 0x08) printf("  [SEQ RUNNING]");
    if (s.main_s & 0x10) printf("  [WAITING FOR TRIGGER]");
    printf("\n");
}

static bool states_differ(const State &a, const State &b)
{
    return a.mode_pattern   != b.mode_pattern   ||
           a.disp_action    != b.disp_action    ||
           a.trig_mode      != b.trig_mode      ||
           a.num_entries    != b.num_entries    ||
           a.repeat         != b.repeat         ||
           a.main_s         != b.main_s         ||
           a.led_r          != b.led_r;
}

// ── Read back and verify the programmed LUT ───────────────────────────────────
static void verify_lut()
{
    printf("\n--- LUT readback ---\n");

    unsigned char img_lut[16] = {};
    if (DLPC350_GetvarExpImageLut(img_lut, 1) >= 0)
        printf("  Image LUT[0] = %u (expected %d)\n", img_lut[0], CALIB_IMAGE_INDEX);
    else
        printf("  Image LUT readback failed\n");

    for (int i = 0; i < 4 && i < NUM_ENTRIES; i++) {
        int trig = 0, pat = 0, bd = 0, led = 0, exp = 0, period = 0;
        bool inv = false, blk = false, swap = false, prev = false;
        if (DLPC350_GetVarExpPatLutItem(i, &trig, &pat, &bd, &led,
                                        &inv, &blk, &swap, &prev,
                                        &exp, &period) >= 0)
            printf("  PatLUT[%d]: trig=%d bitdepth=%d led=%d "
                   "insert_black=%d buf_swap=%d exp=%d period=%d\n",
                   i, trig, bd, led, (int)blk, (int)swap, exp, period);
        else
            printf("  PatLUT[%d]: readback failed\n", i);
    }
    printf("---\n\n");
}

// ── Program calibration sequence ──────────────────────────────────────────────
static void program_calibration()
{
    printf("Programming calibration sequence...\n");

    if (DLPC350_PatternDisplay(DISP_STOP) < 0)         die("PatternDisplay(STOP)");
    usleep(10000);
    if (DLPC350_SetTrigOutConfig(1, false, TRIG_OUT_RISING, TRIG_OUT_FALLING) < 0)
        die("SetTrigOutConfig(1)");
    usleep(10000);
    if (DLPC350_SetTrigOutConfig(2, false, TRIG_OUT_RISING, 0) < 0)
        die("SetTrigOutConfig(2)");
    usleep(10000);

    DLPC350_ClearExpLut();
    for (int i = 0; i < NUM_ENTRIES; i++) {
        bool swap = (i == 0);
        if (DLPC350_AddToExpLut(TRIG_EXT_POS, PATTERN_NUMBER, BITDEPTH, LED_SELECT,
                                false, true, swap, false,
                                EXPOSURE_US, EXPOSURE_US) < 0)
            die("AddToExpLut");
    }
    if (DLPC350_SetPatternDisplayMode(false) < 0) die("SetPatternDisplayMode");
    if (DLPC350_SetPatternTriggerMode(TRIG_MODE_EXT) < 0) die("SetPatternTriggerMode");

    unsigned char image_lut[1] = { (unsigned char)CALIB_IMAGE_INDEX };
    if (DLPC350_SendVarExpImageLut(image_lut, 1) < 0) die("SendVarExpImageLut");
    if (DLPC350_SendVarExpPatLut() < 0)               die("SendVarExpPatLut");
    if (DLPC350_SetVarExpPatternConfig(NUM_ENTRIES, NUM_ENTRIES, 1, true) < 0)
        die("SetVarExpPatternConfig");

    if (DLPC350_StartPatLutValidate() < 0) die("StartPatLutValidate");
    usleep(100000);

    bool ready = false; unsigned int val = 0;
    for (int i = 0; !ready && i < 50; i++) {
        usleep(10000);
        DLPC350_CheckPatLutValidate(&ready, &val);
    }

    if (val != 0) {
        fprintf(stderr, "Validation failed: 0x%02X\n", val);
        if (val & 0x01) fprintf(stderr, "  bit0: exposure/period out of range\n");
        if (val & 0x02) fprintf(stderr, "  bit1: pattern number invalid\n");
        if (val & 0x04) fprintf(stderr, "  bit2: trigger overlaps black vector\n");
        if (val & 0x08) fprintf(stderr, "  bit3: black vector missing\n");
        if (val & 0x10) fprintf(stderr, "  bit4: exposure/period delta < 230µs\n");
        DLPC350_USB_Close();
        DLPC350_USB_Exit();
        exit(1);
    }

    printf("Validation OK.\n");
    verify_lut();

    usleep(10000);
    if (DLPC350_PatternDisplay(DISP_START) < 0) die("PatternDisplay(START)");
    usleep(10000);
    printf("Sequence started.\n\n");
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    int run_secs = 0;
    if (argc >= 2) run_secs = atoi(argv[1]);

    if (DLPC350_USB_Init() < 0)  die("USB init");
    if (DLPC350_USB_Open() < 0) {
        fprintf(stderr, "ERROR: USB open failed\n");
        DLPC350_USB_Exit();
        return 1;
    }

    if (DLPC350_SetPowerMode(false) < 0) die("SetPowerMode(normal)");
    usleep(10000);
    if (DLPC350_SetMode(true) < 0) die("SetMode(pattern)");
    usleep(10000);

    program_calibration();

    State prev = {}, cur = {};
    read_state(prev);
    print_state(prev, "initial");

    printf("Polling every 200ms. Fire GPIO triggers from another terminal.\n");
    printf("Any state change will be printed immediately.\n\n");

    long long polls = 0;
    long long deadline_polls = run_secs > 0 ? (run_secs * 1000LL / 200) : -1;

    int consecutive_failures = 0;

    while (deadline_polls < 0 || polls < deadline_polls) {
        usleep(200000);
        polls++;

        if (!read_state(cur)) {
            consecutive_failures++;
            if (consecutive_failures == 1)
                printf("[~%.1fs] USB read failed — attempting reconnect...\n", polls * 0.2);

            // Try to reconnect
            DLPC350_USB_Close();
            usleep(500000);
            if (DLPC350_USB_Open() == 0) {
                printf("[~%.1fs] USB reconnected\n", polls * 0.2);
                consecutive_failures = 0;
                // Re-read state after reconnect to see what the device reports
                if (read_state(cur)) {
                    char label[64];
                    snprintf(label, sizeof(label), "STATE AFTER RECONNECT (~%.1fs)", polls * 0.2);
                    print_state(cur, label);
                    prev = cur;
                }
            } else if (consecutive_failures % 5 == 0) {
                printf("[~%.1fs] Still disconnected (%d failures)\n",
                       polls * 0.2, consecutive_failures);
            }
            continue;
        }

        consecutive_failures = 0;

        if (states_differ(prev, cur)) {
            char label[64];
            snprintf(label, sizeof(label), "CHANGE at ~%.1fs", polls * 0.2);
            print_state(cur, label);
            prev = cur;
        }
    }

    printf("\nStopping sequence.\n");
    DLPC350_PatternDisplay(DISP_STOP);
    DLPC350_USB_Close();
    DLPC350_USB_Exit();
    return 0;
}
