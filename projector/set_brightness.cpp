/*
 * set_brightness.cpp
 *
 * Set the LightCrafter 4500 LED current (brightness) without changing
 * the running pattern sequence.
 *
 * Usage:
 *   set_brightness <0-255>
 *
 * Sets R, G, B currents to the same value (white output at that level).
 * 255 = full brightness, 0 = LEDs off.
 * The setting persists until the device is power-cycled or overwritten.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <dlp_platforms/lightcrafter_4500/common.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

static void cleanup_and_die(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    DLPC350_USB_Close();
    DLPC350_USB_Exit();
    exit(1);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: set_brightness <0-255>\n");
        return 1;
    }

    int level = atoi(argv[1]);
    if (level < 0 || level > 255) {
        fprintf(stderr, "ERROR: brightness must be 0-255 (got %d)\n", level);
        return 1;
    }

    unsigned char v = (unsigned char)level;

    if (DLPC350_USB_Init() < 0)
        cleanup_and_die("USB init failed");

    if (DLPC350_USB_Open() < 0) {
        fprintf(stderr, "ERROR: USB open failed — device not found\n");
        DLPC350_USB_Exit();
        return 1;
    }

    // Read back current values so we can confirm the write
    unsigned char r_before, g_before, b_before;
    DLPC350_GetLedCurrents(&r_before, &g_before, &b_before);
    printf("Before: R=%u G=%u B=%u\n", r_before, g_before, b_before);

    if (DLPC350_SetLedCurrents(v, v, v) < 0)
        cleanup_and_die("SetLedCurrents failed");

    usleep(10000);

    unsigned char r_after, g_after, b_after;
    DLPC350_GetLedCurrents(&r_after, &g_after, &b_after);
    printf("After:  R=%u G=%u B=%u\n", r_after, g_after, b_after);

    DLPC350_USB_Close();
    DLPC350_USB_Exit();
    return 0;
}
