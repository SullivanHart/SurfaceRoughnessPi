/*
 * calib_test.cpp
 *
 * Minimal calibration sequence test.
 * Programs 16 LUT entries all pointing to flash slot 16 (checkerboard),
 * using the same external-trigger setup as the working scan sequence.
 *
 * This avoids the single-entry repeat loop which appears to not cycle
 * correctly on the DLPC350.
 *
 * Usage:
 *   calib_test          — start and keep running (Ctrl-C to exit)
 *   calib_test <secs>   — run for N seconds then stop
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <dlp_platforms/lightcrafter_4500/common.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

static const int          CALIB_IMAGE_INDEX = 16;
static const int          NUM_ENTRIES       = 16;   // same count as scan; all point to slot 16
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

int main(int argc, char *argv[])
{
    int run_secs = 0;
    if (argc >= 2)
        run_secs = atoi(argv[1]);

    if (DLPC350_USB_Init() < 0)
        die("USB init failed");

    if (DLPC350_USB_Open() < 0) {
        fprintf(stderr, "ERROR: USB open failed\n");
        DLPC350_USB_Exit();
        return 1;
    }

    printf("Connected to DLPC350\n");

    // Stop current sequence
    if (DLPC350_PatternDisplay(DISP_STOP) < 0)
        die("PatternDisplay(STOP) failed");
    usleep(10000);

    // Trigger output config (same as scan)
    if (DLPC350_SetTrigOutConfig(1, false, TRIG_OUT_RISING, TRIG_OUT_FALLING) < 0)
        die("SetTrigOutConfig(1) failed");
    usleep(10000);
    if (DLPC350_SetTrigOutConfig(2, false, TRIG_OUT_RISING, 0) < 0)
        die("SetTrigOutConfig(2) failed");
    usleep(10000);

    // Build 16-entry LUT.
    // Entry 0: buffer_swap=true  → loads slot 16 (checkerboard) into the DMD buffer.
    // Entries 1-15: buffer_swap=false → reuse the same buffer; TRIG_OUT still fires.
    // This avoids the DLPC350 skipping TRIG_OUT when it detects no image change.
    DLPC350_ClearExpLut();
    for (int i = 0; i < NUM_ENTRIES; i++) {
        bool swap = (i == 0);   // only load the image on the first entry
        if (DLPC350_AddToExpLut(
                TRIG_EXT_POS,
                PATTERN_NUMBER,
                BITDEPTH,
                LED_SELECT,
                false,   // invert
                true,    // insert_black (required for ext trigger)
                swap,    // buffer_swap: true only on first entry
                false,   // trigger_out_share_prev
                EXPOSURE_US,
                EXPOSURE_US) < 0)
            die("AddToExpLut failed");
    }

    if (DLPC350_SetPatternDisplayMode(false) < 0)
        die("SetPatternDisplayMode(flash) failed");
    if (DLPC350_SetPatternTriggerMode(TRIG_MODE_EXT) < 0)
        die("SetPatternTriggerMode(3) failed");

    // Image LUT: only 1 entry needed — only entry 0 does a buffer_swap
    unsigned char image_lut[1] = { (unsigned char)CALIB_IMAGE_INDEX };
    if (DLPC350_SendVarExpImageLut(image_lut, 1) < 0)
        die("SendVarExpImageLut failed");
    if (DLPC350_SendVarExpPatLut() < 0)
        die("SendVarExpPatLut failed");
    if (DLPC350_SetVarExpPatternConfig(NUM_ENTRIES, NUM_ENTRIES, 1, true) < 0)
        die("SetVarExpPatternConfig failed");

    // Validate
    if (DLPC350_StartPatLutValidate() < 0)
        die("StartPatLutValidate failed");
    usleep(100000);

    bool         ready  = false;
    unsigned int status = 0;
    for (int i = 0; !ready && i < 50; i++) {
        usleep(10000);
        DLPC350_CheckPatLutValidate(&ready, &status);
    }

    if (status != 0) {
        fprintf(stderr, "Validation failed (0x%02X)\n", status);
        if (status & 0x01) fprintf(stderr, "  bit0: exposure/period out of range\n");
        if (status & 0x02) fprintf(stderr, "  bit1: pattern number invalid\n");
        if (status & 0x04) fprintf(stderr, "  bit2: trigger overlaps black vector\n");
        if (status & 0x08) fprintf(stderr, "  bit3: black vector missing\n");
        if (status & 0x10) fprintf(stderr, "  bit4: exposure/period delta < 230µs\n");
        DLPC350_USB_Close();
        DLPC350_USB_Exit();
        return 1;
    }

    usleep(10000);
    if (DLPC350_PatternDisplay(DISP_START) < 0)
        die("PatternDisplay(START) failed");
    usleep(10000);

    printf("Calibration sequence running (%d entries, all slot %d)\n",
           NUM_ENTRIES, CALIB_IMAGE_INDEX);
    printf("Each GPIO trigger on pin 17 advances one entry (all show checkerboard).\n");

    if (run_secs > 0) {
        printf("Running for %d seconds...\n", run_secs);
        sleep(run_secs);
    } else {
        printf("Press Ctrl-C to stop.\n");
        while (1) sleep(1);
    }

    DLPC350_PatternDisplay(DISP_STOP);
    DLPC350_USB_Close();
    DLPC350_USB_Exit();
    printf("Done.\n");
    return 0;
}
