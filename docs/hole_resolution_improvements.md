# Technical Roadmap: Resolving Optical Occlusion Holes in Structured Light Scans

## 1. Physics & Optical Root Cause of Hole Formation

In stereo structured-light 3D scanning of sand-cast surfaces (e.g., SCRATA comparators A1–A4), holes are not software glitches or random dropout:
- **Three-Ray Triangulation Requirement**: A valid 3D point $(X, Y, Z)$ requires three unoccluded optical lines of sight:
  1. **Projector ray** (illuminating the point with sinusoidal fringes)
  2. **Left Camera ray** (observing the pattern)
  3. **Right Camera ray** (observing the pattern on the same epipolar line)
- **Deep Cavity Geometry**: Sand-cast plates have steep-walled pits and cavities ($100\ \mu\text{m}$ to $850\ \mu\text{m}$ deep) surrounded by protruding sand grains.
- **Occlusion Mechanism**: With a physical stereo baseline angle of $20^\circ - 25^\circ$, any steep wall or overhanging grain rim blocks the line of sight for at least one of the sensors (or casts a shadow from the projector). If any single ray is blocked, zero phase photons reach the sensor, resulting in `NaN`.

---

## 2. Bookmarked Improvement Strategies

### Improvement 1: Multi-Angle Scan Fusion (Dual-Orientation Scanning)
- **Priority**: High (Industry Gold Standard)
- **Mechanism**:
  - The stereo baseline is strictly horizontal ($X$-axis). All line-of-sight shadows occur behind vertical walls (shadows cast along $X$).
  - Capturing Scan 1 at $0^\circ$ and Scan 2 after rotating the plate (or scanner) by $90^\circ$ makes the baseline perpendicular to the original shadows.
  - The two scans are registered using rigid 3D ICP (already implemented in `tools/metrology/verify_scan.py`) and fused via voxel nearest-neighbor blending.
- **Expected Gain**: Recovers $>80\%$ of all remaining cavity holes with 100% genuine measured surface points.

### Improvement 2: Projector-Camera Monocular Triangulation Fallback
- **Priority**: High
- **Mechanism**:
  - Currently, `reconstruct_phase_stereo.py` strictly requires **Left Camera + Right Camera** stereo matching. If the Right Camera is occluded in a pit, the point is discarded—even if the Left Camera and Projector have clear line of sight.
  - By calibrating the DLP projector as a calibrated optical device (`K_proj`, `R_proj`, `T_proj`), points occluded in one camera can fall back to **Left-to-Projector** or **Right-to-Projector** triangulation.
- **Expected Gain**: Points only require 2 optical rays instead of 3, significantly shrinking blind spots.

### Improvement 3: Dual-Exposure / Multi-Exposure HDR Capture
- **Priority**: Medium
- **Mechanism**:
  - Deep cavity bottoms reflect substantially less light than exposed plateau peaks. In single-exposure captures ($550\ \mu\text{s}$), pits often suffer low modulation amplitude ($M < 1.2$), triggering SNR rejection.
  - Capture two bursts per pattern: standard exposure ($500\ \mu\text{s}$) for peaks and long exposure ($1500\ \mu\text{s}$) for dark pits.
  - Merge into a single high-dynamic-range irradiance map before phase unwrapping.
- **Expected Gain**: Boosts fringe modulation inside dark crevices above the noise floor without saturating high peaks.

### Improvement 4: Continuous Epipolar Spline Inpainting
- **Priority**: Medium (Algorithmic / Post-Processing)
- **Mechanism**:
  - In `reconstruct_phase_stereo.py`, interpolate small micro-gaps ($\le 3 - 5\text{ px}$) along rectified epipolar scanlines using monotonic cubic Hermite splines or Navier-Stokes disparity inpainting prior to 3D reprojection.
  - Bridges micro-shadows based strictly on local rim slope without flattening or introducing planar distortion.

---

## 3. Metrological Impact on ASTM WK92969 Roughness
Even with 6.5% line-of-sight shadowing in deep cavities, the current scanner achieves:
- **$S_{vr}$ (ASTM Roughness)**: $119.45\ \mu\text{m}$ vs $122.61\ \mu\text{m}$ (**97.4% agreement**)
- **$S_a$ (Arithmetical Mean)**: $126.03\ \mu\text{m}$ vs $125.10\ \mu\text{m}$ (**99.3% agreement**)
- **$S_q$ (RMS Roughness)**: $160.74\ \mu\text{m}$ vs $161.12\ \mu\text{m}$ (**99.8% agreement**)
- **Global Surface Deviation**: $-7.8\ \mu\text{m}$ mean error.

This is because the variogram $\gamma(d)$ computes statistical distance pairs across valid points without zero-bias, and the 2.5D Gaussian filtration standard (ISO 16610-61 / SurfInspect) uses 8-pass iterative neighbor propagation across micro-holes.
