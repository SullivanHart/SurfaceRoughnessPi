# DLPC350 Firmware Images

## Current Version: v18 (DLPR350PROM_v4.4.0_new18.bin)

This firmware binary configures the Texas Instruments DLPC350 DMD controller with the 20-pattern hybrid phase-shifting + Gray-code sequence for sub-pixel structured-light 3D scanning.

### Pattern Slot Mapping (20 slots, 0 to 19):
- **Slots 0–7**: 8-step sinusoidal phase shift (P = 16 px, delta = 2*pi*k / 8).
- **Slots 8–17**: 5-bit Gray code positive and inverted pairs (2^5 * 16 = 512 >= 456 px):
  - Slot 8: Bit 4 Pos (08_gray_bit_4_pos.bmp)
  - Slot 9: Bit 4 Inv (09_gray_bit_4_inv.bmp)
  - Slot 10: Bit 3 Pos (10_gray_bit_3_pos.bmp)
  - Slot 11: Bit 3 Inv (11_gray_bit_3_inv.bmp)
  - Slot 12: Bit 2 Pos (12_gray_bit_2_pos.bmp)
  - Slot 13: Bit 2 Inv (13_gray_bit_2_inv.bmp)
  - Slot 14: Bit 1 Pos (14_gray_bit_1_pos.bmp)
  - Slot 15: Bit 1 Inv (15_gray_bit_1_inv.bmp)
  - Slot 16: Bit 0 Pos (16_gray_bit_0_pos.bmp)
  - Slot 17: Bit 0 Inv (17_gray_bit_0_inv.bmp)
- **Slot 18**: Solid White reference (18_white.bmp)
- **Slot 19**: Solid Black reference (19_black.bmp)

### Usage with Scanner Service:
`ash
./run_scan.sh --recon-mode phase
`
Or directly:
`ash
python scan.py --recon-mode phase
`
Reconstruction uses 	ools/runtime/reconstruct_phase_stereo.py.
