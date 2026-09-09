/*
 * flash_firmware.cpp
 *
 * Command-line utility to flash DLPC350 / LightCrafter 4500 firmware binaries
 * over USB directly on Linux / Raspberry Pi.
 *
 * Implements the exact Texas Instruments bootloader flash sequence from the
 * official DLPLCR4500 GUI:
 *   1. Enter programming mode
 *   2. Re-enumerate and verify USB connection
 *   3. Query flash manufacturer and device ID
 *   4. Protect sector 0 (128 KB bootloader) and erase application/pattern sectors
 *   5. Stream firmware payload via DLPC350_UploadData
 *   6. Calculate and verify flash hardware checksum
 *   7. Exit programming mode (reboot into application mode)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <sys/stat.h>

#include <dlp_platforms/lightcrafter_4500/common.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_api.hpp>
#include <dlp_platforms/lightcrafter_4500/dlpc350_usb.hpp>

static const uint32_t SECTOR_SIZE = 128 * 1024; // 128 KB sectors (0x20000)
static const uint32_t BOOTLOADER_SIZE = 128 * 1024; // Sector 0 is bootloader (protected)

static int wait_for_usb_connection(int max_wait_seconds)
{
    for (int i = 0; i < max_wait_seconds * 2; i++) {
        DLPC350_USB_Close();
        usleep(500000); // 500 ms
        if (DLPC350_USB_Open() == 0) {
            return 0;
        }
    }
    return -1;
}

int main(int argc, char *argv[])
{
    const char *firmware_path = "firmware/DLPR350PROM_current.bin";
    if (argc >= 2) {
        firmware_path = argv[1];
    }

    fprintf(stdout, "==================================================\n");
    fprintf(stdout, " DLPC350 LightCrafter 4500 Firmware Flasher (CLI) \n");
    fprintf(stdout, "==================================================\n");
    fprintf(stdout, "Firmware binary: %s\n", firmware_path);

    // 1. Open and validate firmware file
    FILE *fp = fopen(firmware_path, "rb");
    if (!fp) {
        // Try relative to project root
        char alt_path[512];
        snprintf(alt_path, sizeof(alt_path), "../%s", firmware_path);
        fp = fopen(alt_path, "rb");
        if (!fp) {
            fprintf(stderr, "ERROR: Cannot open firmware file '%s'\n", firmware_path);
            return 1;
        }
        firmware_path = alt_path;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size <= (long)BOOTLOADER_SIZE) {
        fprintf(stderr, "ERROR: Firmware size (%ld bytes) is too small (must be > 128 KB)\n", file_size);
        fclose(fp);
        return 1;
    }

    uint8_t *file_buf = (uint8_t *)malloc(file_size);
    if (!file_buf) {
        fprintf(stderr, "ERROR: Memory allocation failed (%ld bytes)\n", file_size);
        fclose(fp);
        return 1;
    }

    if (fread(file_buf, 1, file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "ERROR: Failed to read complete firmware file\n");
        free(file_buf);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    // Compute expected 32-bit checksum for the programmed application/pattern payload
    uint32_t expected_checksum = 0;
    for (long i = BOOTLOADER_SIZE; i < file_size; i++) {
        expected_checksum += file_buf[i];
    }
    fprintf(stdout, "Firmware size:    %ld bytes (%.2f MB)\n", file_size, file_size / (1024.0 * 1024.0));
    fprintf(stdout, "Expected checksum: 0x%08X\n", expected_checksum);

    // 2. Initialize USB
    if (DLPC350_USB_Init() < 0) {
        fprintf(stderr, "ERROR: Failed to initialize USB HID library\n");
        free(file_buf);
        return 1;
    }

    fprintf(stdout, "\nConnecting to DLPC350 over USB...\n");
    if (wait_for_usb_connection(5) < 0) {
        fprintf(stderr, "ERROR: Cannot connect to DLPC350 (VID 0x0451, PID 0x6401)\n");
        fprintf(stderr, "Please check the USB cable and udev permissions.\n");
        free(file_buf);
        return 1;
    }
    fprintf(stdout, "Connected to DLPC350.\n");

    // 3. Enter Programming Mode if not already in it
    uint16_t test_man_id = 0;
    if (DLPC350_GetFlashManID(&test_man_id) < 0 || test_man_id == 0) {
        fprintf(stdout, "Switching DLPC350 into bootloader programming mode...\n");
        DLPC350_EnterProgrammingMode();
        DLPC350_USB_Close();

        fprintf(stdout, "Waiting for bootloader USB re-enumeration (3s)...\n");
        sleep(3); // Allow bootloader to reset USB and re-enumerate

        if (wait_for_usb_connection(15) < 0) {
            fprintf(stderr, "ERROR: Failed to reconnect to bootloader after mode switch\n");
            free(file_buf);
            return 1;
        }
    } else {
        fprintf(stdout, "DLPC350 is already in bootloader programming mode.\n");
    }

    // 4. Query Flash Hardware IDs with retry
    uint16_t man_id = 0;
    unsigned long long dev_id = 0;
    bool id_ok = false;
    for (int retry = 0; retry < 10; retry++) {
        if (DLPC350_GetFlashManID(&man_id) == 0 && DLPC350_GetFlashDevID(&dev_id) == 0 && man_id != 0) {
            id_ok = true;
            break;
        }
        usleep(500000); // 500 ms
    }

    if (!id_ok) {
        fprintf(stderr, "WARNING: Could not query flash IDs (proceeding with standard flash type)\n");
    } else {
        fprintf(stdout, "Flash Manufacturer ID: 0x%04X, Device ID: 0x%04llX\n", man_id, dev_id & 0xFFFF);
    }

    DLPC350_SetFlashType(0);

    // 5. Erase application and pattern sectors (Skipping Sector 0 to protect Bootloader)
    uint32_t start_sector = BOOTLOADER_SIZE / SECTOR_SIZE; // Sector 1 (0x20000)
    uint32_t last_sector = (file_size - 1) / SECTOR_SIZE;

    fprintf(stdout, "\n--- Step 1/3: Erasing Flash Sectors ---\n");
    fprintf(stdout, "Preserving Sector 0 (Bootloader @ 0x00000000)\n");
    fprintf(stdout, "Erasing Sectors %u through %u (@ 0x%08X to 0x%08X)...\n",
            start_sector, last_sector, start_sector * SECTOR_SIZE, last_sector * SECTOR_SIZE);

    for (uint32_t sec = start_sector; sec <= last_sector; sec++) {
        uint32_t addr = sec * SECTOR_SIZE;
        fprintf(stdout, "\rErasing sector %2u / %2u [@ 0x%08X] ... ",
                sec - start_sector + 1, last_sector - start_sector + 1, addr);
        fflush(stdout);

        if (DLPC350_SetFlashAddr(addr) < 0 || DLPC350_FlashSectorErase() < 0) {
            fprintf(stderr, "\nERROR: Flash sector erase failed at address 0x%08X\n", addr);
            DLPC350_USB_Close();
            free(file_buf);
            return 1;
        }
        DLPC350_WaitForFlashReady();
    }
    fprintf(stdout, "Done!\n");

    // 6. Program Firmware Image (Payload after 128 KB bootloader)
    fprintf(stdout, "\n--- Step 2/3: Programming Firmware ---\n");
    uint32_t upload_len = file_size - BOOTLOADER_SIZE;
    uint32_t total_to_upload = upload_len;
    uint32_t current_offset = BOOTLOADER_SIZE;

    if (DLPC350_SetFlashAddr(BOOTLOADER_SIZE) < 0 || DLPC350_SetUploadSize(upload_len) < 0) {
        fprintf(stderr, "ERROR: Failed to initialize download size and address\n");
        DLPC350_USB_Close();
        free(file_buf);
        return 1;
    }

    int last_percent = -1;
    while (upload_len > 0) {
        int bytes_sent = DLPC350_UploadData(file_buf + current_offset, upload_len);
        if (bytes_sent <= 0) {
            fprintf(stderr, "\nERROR: UploadData failed at offset 0x%08X\n", current_offset);
            DLPC350_USB_Close();
            free(file_buf);
            return 1;
        }

        upload_len -= bytes_sent;
        current_offset += bytes_sent;

        int percent = (int)(((total_to_upload - upload_len) * 100ULL) / total_to_upload);
        if (percent != last_percent) {
            last_percent = percent;
            fprintf(stdout, "\rProgramming: [%-25s] %3d%% (0x%08X / 0x%08X)",
                    std::string(percent / 4, '=').c_str(), percent, current_offset, (uint32_t)file_size);
            fflush(stdout);
        }
    }
    fprintf(stdout, "\nProgramming complete!\n");

    // 7. Verify Checksum
    fprintf(stdout, "\n--- Step 3/3: Verifying Hardware Checksum ---\n");
    uint32_t flash_checksum = 0;
    uint32_t temp_chksum = 0;

    for (uint32_t sec = start_sector; sec <= last_sector; sec++) {
        uint32_t addr = sec * SECTOR_SIZE;
        uint32_t chunk = SECTOR_SIZE;
        if (addr + chunk > (uint32_t)file_size) {
            chunk = file_size - addr;
        }

        if (DLPC350_SetFlashAddr(addr) < 0 || DLPC350_SetUploadSize(chunk) < 0 ||
            DLPC350_CalculateFlashChecksum() < 0) {
            fprintf(stderr, "ERROR: Checksum computation request failed at sector %u\n", sec);
            DLPC350_USB_Close();
            free(file_buf);
            return 1;
        }
        DLPC350_WaitForFlashReady();

        if (DLPC350_GetFlashChecksum(&temp_chksum) < 0) {
            fprintf(stderr, "ERROR: Failed to retrieve checksum for sector %u\n", sec);
            DLPC350_USB_Close();
            free(file_buf);
            return 1;
        }
        flash_checksum += temp_chksum;
    }

    fprintf(stdout, "Calculated on device: 0x%08X\n", flash_checksum);
    fprintf(stdout, "Expected from binary: 0x%08X\n", expected_checksum);

    if (flash_checksum != expected_checksum) {
        fprintf(stderr, "ERROR: Checksum verification FAILED!\n");
        DLPC350_USB_Close();
        free(file_buf);
        return 1;
    }
    fprintf(stdout, ">> Checksum VERIFIED SUCCESSFULLY! <<\n");

    // 8. Exit Programming Mode and Reboot
    fprintf(stdout, "\nExiting programming mode and restarting projector...\n");
    DLPC350_ExitProgrammingMode();
    DLPC350_USB_Close();
    free(file_buf);

    fprintf(stdout, "==================================================\n");
    fprintf(stdout, " SUCCESS: LightCrafter 4500 Flashed Successfully! \n");
    fprintf(stdout, "==================================================\n");

    return 0;
}
