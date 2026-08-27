/*
 * seq_test.cpp — Incremental calibration sequence tester.
 *
 * Starts with the simplest possible sequence and works up to find where
 * the DLPC350 stops showing the checkerboard on every trigger.
 *
 * Usage:
 *   seq_test <N> [all_swap]
 *
 *   N         number of LUT entries (1, 2, 4, 8, or 16)
 *   all_swap  if present, every entry uses buffer_swap=true
 *             (default: only entry 0 uses buffer_swap=true)
 *
 * Keep this running and fire GPIO triggers with trigger.py.
 * The projector should show the checkerboard on EVERY trigger.
 *
 * Progression to find the failure point:
 *   ./seq_test 1          — simplest: 1 entry, buf_swap=true
 *   ./seq_test 2          — 2 entries: entry 0 swap=true, entry 1 swap=false
 *   ./seq_test 4          — 4 entries
 *   ./seq_test 16         — full calibration sequence
 *   ./seq_test 16 all_swap — all swap=true (originally broken: DLPC350 skips TRIG_OUT)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#include <dlp_platforms/lightcrafter_4500/common.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

static const int          CALIB_IMAGE_INDEX = 16;
static const unsigned int EXPOSURE_US       = 500000;
static const int          BITDEPTH          = 8;
static const int          LED_SELECT        = 7;
static const int          PATTERN_NUMBER    = 0;
static const unsigned int TRIG_OUT_RISING   = 187;
static const unsigned int TRIG_OUT_FALLING  = 187;
static const int          TRIG_EXT_POS      = 1;
static const int          TRIG_MODE_EXT     = 3;

static void die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    DLPC350_USB_Close();
    exit(1);
}

static void signal_handler(int) { DLPC350_USB_Close(); _exit(0); }

int main(int argc, char *argv[])
{
    int  n_entries = 1;
    bool all_swap  = false;

    if (argc >= 2) n_entries = atoi(argv[1]);
    if (argc >= 3 && strcmp(argv[2], "all_swap") == 0) all_swap = true;

    if (n_entries < 1 || n_entries > 256) {
        fprintf(stderr, "Usage: seq_test <N> [all_swap]  (N = 1..256)\n");
        return 1;
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);

    if (DLPC350_USB_Init() < 0) die("USB init");
    {
        int opened = -1;
        for (int i = 0; i < 10 && opened < 0; i++) {
            opened = DLPC350_USB_Open();
            if (opened < 0) { fprintf(stderr, "  retrying USB open...\n"); usleep(500000); }
        }
        if (opened < 0) die("USB open failed after retries");
    }

    printf("Connected.\n");

    DLPC350_PatternDisplay(0);   // stop
    usleep(50000);
    DLPC350_SetPowerMode(false);
    usleep(10000);
    DLPC350_SetMode(true);       // pattern sequence mode
    usleep(10000);

    DLPC350_SetTrigOutConfig(1, false, TRIG_OUT_RISING, TRIG_OUT_FALLING);
    usleep(10000);
    DLPC350_SetTrigOutConfig(2, false, TRIG_OUT_RISING, 0);
    usleep(10000);

    printf("Building %d-entry LUT  [%s]\n",
           n_entries, all_swap ? "all buf_swap=true" : "entry 0 swap=true, rest swap=false");

    DLPC350_ClearExpLut();
    for (int i = 0; i < n_entries; i++) {
        bool swap = all_swap ? true : (i == 0);
        int rc = DLPC350_AddToExpLut(
            TRIG_EXT_POS,
            PATTERN_NUMBER,
            BITDEPTH,
            LED_SELECT,
            false,       // invert
            true,        // insert_black (required for ext trigger)
            swap,
            false,       // trig_out_share_prev
            EXPOSURE_US,
            EXPOSURE_US);
        if (rc < 0) die("AddToExpLut");
        printf("  entry %2d: buf_swap=%d\n", i, (int)swap);
    }

    DLPC350_SetPatternDisplayMode(false);
    DLPC350_SetPatternTriggerMode(TRIG_MODE_EXT);

    // Image LUT: all entries point to slot CALIB_IMAGE_INDEX (checkerboard).
    // Only need one image LUT slot regardless of n_entries, because every
    // PatLUT entry points to image index 0 in the image LUT, which maps to
    // flash slot CALIB_IMAGE_INDEX.
    unsigned char image_lut[1] = { (unsigned char)CALIB_IMAGE_INDEX };
    if (DLPC350_SendVarExpImageLut(image_lut, 1) < 0) die("SendVarExpImageLut");
    if (DLPC350_SendVarExpPatLut()               < 0) die("SendVarExpPatLut");
    // numImages = number of buf_swap=true entries in the pattern LUT.
    // This tells the DLPC350 how many entries are in the image LUT and when
    // to wrap the image pointer back to index 0 on each repeat loop.
    int num_images = all_swap ? n_entries : 1;
    if (DLPC350_SetVarExpPatternConfig(
            (unsigned int)n_entries,
            (unsigned int)n_entries,
            (unsigned int)num_images,
            true) < 0) die("SetVarExpPatternConfig");

    printf("Validating...\n");
    if (DLPC350_StartPatLutValidate() < 0) die("StartPatLutValidate");
    usleep(200000);

    bool         ready  = false;
    unsigned int status = 0;
    for (int p = 0; !ready && p < 50; p++) {
        usleep(10000);
        DLPC350_CheckPatLutValidate(&ready, &status);
    }

    if (status != 0) {
        fprintf(stderr, "Validation FAILED (0x%02X)\n", status);
        if (status & 0x01) fprintf(stderr, "  bit0: exposure/period out of range\n");
        if (status & 0x02) fprintf(stderr, "  bit1: pattern number invalid\n");
        if (status & 0x04) fprintf(stderr, "  bit2: trigger overlaps black vector\n");
        if (status & 0x08) fprintf(stderr, "  bit3: black vector missing\n");
        if (status & 0x10) fprintf(stderr, "  bit4: exposure/period delta < 230µs\n");
        DLPC350_USB_Close();
        return 1;
    }
    printf("Validation OK.\n");

    if (DLPC350_PatternDisplay(2) < 0) die("PatternDisplay(START)");  // start
    usleep(10000);

    printf("\nSequence running. Fire GPIO triggers with trigger.py.\n");
    printf("Projector should show checkerboard on EVERY trigger.\n");
    printf("Ctrl-C to stop.\n\n");
    fflush(stdout);

    while (1) usleep(5000000);
    return 0;
}
