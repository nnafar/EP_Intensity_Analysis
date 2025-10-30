# Enhanced GUV Analysis with Membrane Detection

## Overview

This enhanced version improves membrane detection accuracy by using **radial intensity profile analysis** instead of relying solely on the fixed radius from the CSV file. This approach is adapted from the more sophisticated `skeleton.py` pipeline.

## Key Improvements

### 1. **Accurate Membrane Detection**
- **Before**: Used fixed radius from CSV to create circular masks
- **After**: Calculates radial intensity profiles and detects membrane peaks
- **Result**: Masks now align with the actual membrane location visible in the image

### 2. **Peak Detection Algorithm**
The enhanced method:
- Creates 360 radial profiles from the GUV center
- Averages them to get a smooth radial intensity profile
- Uses `scipy.signal.find_peaks()` to detect the membrane peak
- Calculates membrane borders using **width at half maximum**
- Scores multiple peaks based on:
  - Distance from expected radius
  - Peak height
  - Peak prominence
  - Position (not too early/late)

### 3. **Quality Metrics**
Each GUV detection includes:
- Detected inner and outer radii
- Quality flags (e.g., "multiple_peaks", "wide_peak", "signal_inside_vesicle")
- Detection success/failure status
- All metrics saved to a CSV log file

### 4. **Enhanced Visualization**
- Mask visualizations for problematic detections
- Radial profile plots showing detected borders
- Comparison with CSV estimate

## Files

### Core Files (Enhanced)
- **`guv_analysis_utils_enhanced.py`** - Enhanced utilities with membrane detection
- **`run_analysis_enhanced.py`** - Main analysis script using detection
- **`config.py`** - Configuration (same as before)

### Original Files (Reference)
- `guv_analysis_utils.py` - Original utilities
- `run_analysis.py` - Original analysis script
- `skeleton.py` - Reference implementation
- `main.py` - Reference main script

## How to Use

### Quick Start

1. **Keep your existing `config.py`** - No changes needed!

2. **Run the enhanced analysis**:
   ```python
   python run_analysis_enhanced.py
   ```

3. **Check the outputs**:
   - Mask visualizations in `OUTPUT_IMAGE_FOLDER`
   - Detection quality log: `{EXPERIMENT_BASE_NAME}_detection_quality.csv`
   - Radial profile plots for problematic detections

### Configuration

All settings remain in `config.py`. The enhanced version uses the same parameters plus automatically detects the membrane.

### Output Files

The enhanced version creates:

1. **Detection Quality Log** (`*_detection_quality.csv`):
   ```csv
   guv_id,estimated_radius,detected_inner,detected_outer,failed,comments
   1,45,38,47,False,OK
   2,52,40,55,False,multiple_peaks
   3,48,0,0,True,no_membrane_peak_detected
   ```

2. **Mask Visualizations** (PNG files):
   - Purple: Inner GUV area (measurement region)
   - Red: Membrane region
   - Blue: Background ring

3. **Radial Profile Plots** (for problematic detections):
   - Shows the intensity profile
   - Marks detected borders
   - Compares with CSV estimate

## Understanding the Detection

### What the Algorithm Does

1. **Radial Profiles**: Creates 360 lines radiating from the center
2. **Averaging**: Averages all profiles to get smooth signal
3. **Peak Finding**: Finds intensity peaks (membrane signal)
4. **Border Calculation**: Uses FWHM (Full Width at Half Maximum)

### Visual Example

```
Intensity
    ^
    |     /\
    |    /  \        <- Membrane peak
    |   /    \
    |  /      \___   <- Background
    | /            
    |/_______________> Radius
      ^  ^  ^
      |  |  |
    Inner Peak Outer
    border    border
```

### Quality Flags

Common quality flags and their meanings:

- **`OK`**: Clean detection, no issues
- **`multiple_peaks`**: Multiple membrane-like peaks detected, chose best one
- **`wide_peak`**: Membrane appears thicker than expected
- **`signal_inside_vesicle`**: Unexpected signal inside the vesicle
- **`signal_outside_vesicle`**: Signal clusters outside membrane
- **`no_membrane_peak_detected`**: Failed to find membrane (uses fallback)
- **`width_calc_failed`**: FWHM calculation failed (uses approximate borders)

### When Detection Fails

If detection fails:
1. The algorithm falls back to using 80% of the CSV radius estimate
2. A warning is printed
3. The GUV is flagged in the quality log
4. Visualization and profile plots are automatically generated

## Advantages Over Fixed-Radius Approach

| Aspect | Fixed Radius (Old) | Membrane Detection (New) |
|--------|-------------------|-------------------------|
| Accuracy | Depends on CSV accuracy | Uses actual image data |
| Irregular GUVs | Poor handling | Adapts to actual shape |
| Quality Control | None | Automated quality flags |
| Debugging | Manual inspection needed | Automatic diagnostics |
| Reproducibility | Good | Excellent |

## Example Results

### Before (Fixed Radius)
```
GUV 1: Using radius 45px from CSV
  - Inner mask: 30px (45-15 for membrane thickness)
  - No verification of actual membrane location
```

### After (Detection)
```
GUV 1: Estimated radius 45px from CSV
  - Detecting membrane from radial intensity profile...
  - Detected membrane: inner=38px, outer=47px (estimate was 45px)
  - Quality flags: OK
```

## Troubleshooting

### Problem: Many failed detections

**Possible causes:**
- Poor membrane channel contrast
- Radius estimate very wrong (10x scaling issue)
- Membrane not circular

**Solutions:**
1. Check your CSV radius values
2. Verify the 10x correction is appropriate
3. Examine the radial profile plots
4. Adjust the detection parameters (see Advanced Configuration)

### Problem: All detections flagged with warnings

**Possible causes:**
- Low signal-to-noise ratio
- Irregular vesicles
- Contamination or debris

**Solutions:**
1. Review the quality flags
2. Examine mask visualizations
3. Consider filtering out low-quality GUVs from analysis

## Advanced Configuration

You can tune the detection by modifying `create_guv_masks_with_detection()` parameters:

```python
inner_mask, membrane_mask, background_mask, detection_info = utils.create_guv_masks_with_detection(
    roi_guide_frame, 
    (xc, yc), 
    radius_estimate=guv_radius_estimate,
    bg_buffer=cfg.BG_BUFFER_PIXELS,
    bg_width=cfg.BG_RING_WIDTH_PIXELS,
    num_angles=360,      # More angles = smoother profile
    length_excess=1.5    # How far beyond radius to sample
)
```

## Comparison with skeleton.py Approach

This implementation uses the **same core algorithm** as `skeleton.py`:

| Feature | skeleton.py | Enhanced Version |
|---------|------------|------------------|
| Radial profiles | ✓ | ✓ |
| Peak detection | ✓ | ✓ |
| FWHM borders | ✓ | ✓ |
| Quality scoring | ✓ | ✓ |
| Multi-channel | ✓ | Not yet |
| Template matching | ✓ | Not needed (single-frame ROI) |

The key difference is that this version is adapted for **time-lapse dye uptake analysis** where:
- ROI detection happens once (first frame of C1)
- Measurement happens over time (C3 stack)

## Next Steps

### Recommended Workflow

1. **Run enhanced analysis** on your data
2. **Review quality log** - identify problematic GUVs
3. **Examine visualizations** for failed/flagged detections
4. **Filter data** - exclude low-quality detections if needed
5. **Proceed with fitting** - use the improved masks for kinetic analysis

### Future Enhancements

Potential improvements:
- [ ] Adaptive parameter tuning based on image statistics
- [ ] Multi-channel membrane detection (use both C1 and C3)
- [ ] Ellipse fitting for non-circular GUVs
- [ ] Machine learning-based quality classification
- [ ] Real-time preview during analysis

## Questions?

The enhanced detection should work "out of the box" with your existing configuration. If you encounter issues:

1. Check the detection quality log
2. Review the radial profile plots
3. Compare visualizations between old and new methods
4. Adjust the radius estimate if needed (check the 10x correction)

## References

This implementation adapts the membrane detection approach from:
- `skeleton.py` by @kkourkoulou
- DisGUVery detection framework
- Peak detection via scipy.signal


# Choice of Model
## Model 1: 4-Parameter (Single Exponential Rise with Linear Drift)

$$I(t) = I_{offset} + A \cdot \left(1- e^{-t/\tau}\right) + D \cdot t$$

This is a phenomenological model used when you believe the resealing/uptake can be described by **one dominant kinetic process** ($\tau$), but you need to correct for measurement artifacts like photobleaching. 

- $I_{offset}$ (Baseline Intensity): The initial fluorescence background intensity at $t=0$.

- $A$ (Amplitude): The total magnitude of the fluorescence increase due to dye entry after the pulse.

- $tau$ (Single Time Constant): Represents the average characteristic time of the combined dye influx/pore resealing process. It treats the resealing as a single-step event.

- $D$ (Linear Drift Term): This term accounts for any slow, steady change in intensity over time, like photobleaching (if $D < 0$) or slow focus drift. It is an artifact correction, not a physical resealing process.

## Model 2: 5-Parameter (Double Exponential Resealing)

$$I(t) = A_{f} - A_{1} \ e^{-t/\tau_{1}} - A_{2} \ e^{-t/\tau_{1}}$$

This is the standard model in electroporation literature (like the paper you cited) because it accounts for the **multi-stage nature of pore resealing**, which is rarely a single event.

- $A_{f}$ (Final Intensity): The steady-state (plateau) intensity reached when the process is complete.

- $A_{1}$ (Fast Amplitude): The fraction of the total signal associated with the fast resealing/uptake component.

- $tau_{1}$ (Fast Time Constant): Represents the time scale for the quick process, typically related to the rapid closure of small, numerous pores.

- $A_{2}$ (Slow Amplitude): The fraction of the total signal associated with the slow resealing/uptake component.

- $tau_{2}$ (Slow Time Constant): Represents the time scale for the slow process, often related to the reorganization or closure of large, stable pores (macropores) or membrane repair.

## Key Conceptual Difference

The choice between the two models depends on what you hypothesize drives the long-term changes in your signal:
- **Choose Model 1 (4-PARAM) if**: You believe the dye uptake is kinetically simple (one $\tau$), and the long-term drift you see in your plot is likely a linear measurement artifact (e.g., photobleaching). You use the $D \cdot t$ term to strip away the artifact and find the true $\tau$ of the primary event.

- **Choose Model 2 (5-PARAM) if**: You believe the dye uptake and subsequent membrane resealing is inherently complex and multi-staged. The $\tau_1$ and $\tau_2$ values provide distinct physical insights into the different types of pores or repair mechanisms involved in the resealing process. If there is also photobleaching, you would typically correct the raw data for linear drift before applying this model.
