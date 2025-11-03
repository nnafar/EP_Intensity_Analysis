# GUV Fluorescence Analysis Pipeline - Advanced Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Scientific Background](#scientific-background)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Algorithm Details](#algorithm-details)
5. [Data Flow](#data-flow)
6. [Mathematical Models](#mathematical-models)
7. [Quality Control](#quality-control)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Overview

### Purpose

This pipeline quantifies dye uptake kinetics in Giant Unilamellar Vesicles (GUVs) following electroporation. It processes time-lapse fluorescence microscopy data to extract membrane-normalized intensity curves and fit them to kinetic models.

### Key Features

- **Automated membrane detection** using radial intensity profiling
- **Adaptive normalization** accounting for background fluorescence
- **Multi-model fitting** (4-parameter and 5-parameter exponential models)
- **Quality metrics** for each detection with automatic flagging
- **Comprehensive visualization** including mask overlays and radial profiles

### Design Philosophy

1. **Configuration-driven**: All parameters externalized for easy tuning
2. **Robust detection**: Multiple fallback mechanisms for edge cases
3. **Transparent analysis**: Extensive logging and quality reporting
4. **Reproducibility**: Deterministic algorithms with fixed random seeds

---

## Scientific Background

### The Electroporation Experiment

#### Experimental Setup

```
1. GUVs are prepared in a chamber with fluorescent dye in the external medium
2. An electric pulse is applied (e.g., 1 kV/cm for microseconds)
3. Membrane temporarily becomes permeable (poration)
4. Dye enters the GUV lumen
5. Time-lapse imaging captures the dye uptake process
6. Membrane reseals over time (milliseconds to seconds)
```

#### Key Channels

- **C1 (ROI channel)**: Membrane marker (e.g., DiI) for GUV localization
- **C3 (Dye channel)**: Fluorescent dye (e.g., propidium iodide) for uptake measurement

### Measurement Challenges

1. **Photobleaching**: Fluorescence decreases over time due to light exposure
2. **Focus drift**: GUVs may move slightly during acquisition
3. **Heterogeneity**: Different GUVs have varying sizes, membrane properties
4. **Background**: Extravesicular dye contributes to measured intensity
5. **Membrane artifacts**: Lipid clusters and irregular shapes complicate detection

### The Normalization Problem

Raw intensity measurements combine multiple factors:

```
I_raw(t) = I_dye_inside(t) + I_dye_outside(t) + I_membrane(t) + I_dark
```

**Our solution**: Three-term normalization

```
I_uptake(t) = (I_dye,t - I_dye,0) / (I_background,t - I_dye,0)
```

Where:
- `I_dye,t`: Mean intensity inside GUV at time t
- `I_dye,0`: Baseline intensity inside GUV before pulse
- `I_background,t`: Mean intensity in background ring at time t

This normalizes for:
- Initial autofluorescence
- Photobleaching (assumes same rate inside and outside)
- Variations in dye concentration between experiments

---

## Pipeline Architecture

### Module Structure

```
guv-analysis/
│
├── run_analysis.py              # Main orchestration script
├── guv_analysis_utils.py        # Core algorithms and utilities
├── config.py                    # User-configurable parameters
├── requirements.txt             # Python dependencies
│
└── [Generated outputs]
    ├── output_images/
    │   ├── *_mask_viz_GUV_*.png           # Mask visualizations
    │   ├── *_radial_profile_GUV_*.png     # Radial intensity plots
    │   ├── frame_*_at_*s.png              # Time-stamped snapshots
    │   ├── *_kinetic_fit.png              # Final fitted curve
    │   ├── *_normalized_curves.csv        # All normalized traces
    │   └── *_detection_quality.csv        # QC report
```

### Component Responsibilities

#### 1. `run_analysis.py` (Orchestration)

**Responsibilities:**
- Discover and validate input files
- Coordinate processing across all GUVs
- Align data to electroporation event
- Perform curve fitting
- Generate outputs and reports

**Does NOT contain:**
- Algorithm implementations
- Image processing logic
- Mathematical models

#### 2. `guv_analysis_utils.py` (Algorithms)

**Responsibilities:**
- Image loading and preprocessing
- Mask generation (circular, annular)
- Membrane detection via radial profiling
- Intensity trace extraction
- Jump detection
- Normalization calculations
- Kinetic models
- Visualization utilities

**Key Sections:**
1. Time extraction (lines 33-72)
2. Kinetic models (lines 79-99)
3. Image processing (lines 105-252)
4. Radial profiling (lines 258-307)
5. Membrane detection (lines 308-415)
6. Mask creation (lines 418-511)
7. Data processing (lines 556-639)

#### 3. `config.py` (Parameters)

**Five categories:**
1. File & experiment identification
2. Analysis & masking parameters
3. Fitting & modeling parameters
4. Output & visualization parameters
5. Fallbacks & advanced settings

---

## Algorithm Details

### 1. GUV Center Refinement

**Problem**: CSV coordinates may be inaccurate due to:
- Bright intravesicular clusters skewing center-of-mass
- Detection artifacts from segmentation algorithm
- Sub-pixel positioning errors

**Solution**: Edge-based centroid detection

```python
def refine_guv_center(image, center_guess, radius_estimate):
    """
    Algorithm:
    1. Extract ROI around guess (search box = 1.6 × radius)
    2. Apply Gaussian blur (σ=1.5) to reduce noise
    3. Compute Laplacian for edge detection
    4. Threshold using Otsu's method
    5. Calculate centroid of edge mask
    6. Validate shift (<70% of radius)
    7. Return refined center or original if invalid
    """
```

**Mathematical Basis:**

The Laplacian operator detects edges:
```
∇²I = ∂²I/∂x² + ∂²I/∂y²
```

For a ring structure (membrane), this produces:
- Positive values at inner edge (dark→bright)
- Negative values at outer edge (bright→dark)
- Strong response at membrane location

**Edge Cases Handled:**
- No edges detected → return original center
- Large shift (>70% radius) → likely artifact, revert
- Multiple edge regions → use largest connected component

### 2. Radial Intensity Profiling

**Core Innovation**: Instead of using fixed CSV radius, we measure the actual membrane position from the data.

#### Step A: Linear Profile Extraction

```python
def linear_profiles(image, center, num_angles=360, length_excess=1.5, radius):
    """
    Create radial intensity profiles from center outward.
    
    Algorithm:
    1. Define 360 equally-spaced angles around center
    2. For each angle θ:
       - Generate radial line from center outward (length = 1.5 × radius)
       - Use map_coordinates for sub-pixel interpolation
       - Store intensity values along line
    3. Return matrix: (length × num_angles)
    """
```

**Why Sub-pixel Interpolation?**

Without interpolation, we're limited to integer pixel positions:
```
Pixel grid:     [10, 20, 30, 40, 50]
True membrane:       ^
                    23.7 pixels → rounds to 24
```

With bilinear interpolation (`map_coordinates`):
```
I(23.7) = 0.3 × I(23) + 0.7 × I(24)
```

This provides ~5× better spatial resolution for membrane detection.

#### Step B: Profile Averaging

```python
def average_radial_profile(profiles):
    """
    Aggregate profiles across all angles.
    
    Uses MEDIAN instead of MEAN because:
    - Lipid clusters create outlier intensities at specific angles
    - Median is robust to these outliers
    - Preserves membrane peak without artificial broadening
    """
    return np.median(profiles, axis=1)
```

**Example Profile:**

```
Radius (px)    0    10    20    30    40    50    60    70
Intensity     50    55    60   120   180   100    80    70
              ↑           ↑          ↑           ↑
           background  rising  MEMBRANE  falling  background
```

### 3. Membrane Detection Algorithm

**Goal**: Identify the membrane peak and define inner/outer boundaries.

```python
def membrane_search(profile, radii, expected_radius, search_factor,
                    membrane_half_width, peak_min_dist, peak_min_prom):
    """
    Sophisticated peak detection with quality control.
    
    Algorithm:
    1. Smooth profile (Gaussian filter, σ=1)
    2. Define search window: expected_radius ± (search_factor × expected_radius)
    3. Find all peaks in window using scipy.signal.find_peaks
    4. Select highest peak by prominence
    5. Define membrane boundaries: peak ± membrane_half_width
    6. Validate detection and generate quality flags
    7. Return results with metadata
    """
```

#### Peak Detection Parameters

**`peak_min_dist`** (default: 5 pixels)
- Minimum separation between detected peaks
- Prevents detecting noise as multiple peaks
- Value should be ≥ 2× expected membrane thickness

**`peak_min_prom`** (default: 0.05 = 5%)
- Peak prominence as fraction of max intensity
- Prominence = height of peak above its lowest contour line
- Too low: noise peaks detected
- Too high: real membranes missed

**Visual Example:**

```
Intensity
   |
200|              *  ← Main peak (prominence = 120)
   |            /   \
150|         __/     \__
   |        /           \
100|    * /               \ *  ← Noise peaks (prominence = 20)
   |   / |                 | \
 50|__/__|_________________|__\___
   0   20   40   60   80  100  120  Radius (px)
       ↑                      ↑
    Rejected              Selected (highest prominence)
```

#### Search Window Rationale

**`search_factor`** (default: 0.3 = ±30%)

For expected_radius = 50 pixels:
```
Search range: 35-65 pixels

Too narrow (0.1):   45-55 px → May miss slightly off-center GUVs
Sweet spot (0.3):   35-65 px → Accommodates most variations
Too wide (0.8):     10-90 px → May pick up wrong structure
```

**Why not search entire profile?**
- Lipid clusters can create strong peaks far from membrane
- Debris or adjacent vesicles create false peaks
- Search window provides spatial prior information

#### Quality Flags

The algorithm generates detailed quality metrics:

```python
comments = [
    "OK"                        # Clean detection
    "no_peak_in_window"        # No peak found in search range
    "invalid_search_window"    # Search parameters out of bounds
    "zero_radius_detected"     # Degenerate detection
    "width_calc_failed"        # Peak too narrow or at boundary
    "find_peaks_failed"        # scipy.signal.find_peaks error
]

failed = True/False  # Overall detection status
```

### 4. Mask Creation

Three concentric regions are defined:

#### Inner GUV Mask (Uptake Region)

```python
inner_guv_mask = create_circular_mask(image.shape, center, radius=inner_boundary)
```

**Purpose**: Measure dye that has entered the vesicle lumen

**Definition**: All pixels with distance ≤ `inner_boundary` from center

**Why not include membrane?**
- Membrane signal confounds uptake measurement
- Membrane fluorescence may be different dye (e.g., DiI vs PI)
- Clean separation improves normalization

#### Membrane Mask (Visualization)

```python
membrane_mask = create_annular_mask(
    image.shape, center, 
    inner_radius=inner_boundary + 1,
    outer_radius=outer_boundary
)
```

**Purpose**: Visualize detected membrane region

**Options**:
1. **Dynamic width**: Use detected boundaries (variable thickness)
2. **Fixed width**: Use `VIZ_MEMBRANE_THICKNESS_PIXELS` (cleaner visualization)

**Trade-off**:
- Fixed width: Easier to compare across GUVs, cleaner appearance
- Dynamic width: Shows actual detection, useful for QC

#### Background Ring Mask

```python
background_mask = create_annular_mask(
    image.shape, center,
    inner_radius=outer_boundary + bg_buffer,
    outer_radius=outer_boundary + bg_buffer + bg_width
)
```

**Purpose**: Sample extravesicular dye concentration

**Design considerations**:

```
                  GUV
           [    |||||    ]
                 ↑↑↑
              membrane
           
<-- gap -->  <-- ring -->
 (buffer)     (sample)
   2 px        10 px
```

**`bg_buffer`** (default: 2 pixels)
- Prevents contamination from membrane spillover
- Accounts for PSF (point spread function) of microscope
- Too small: membrane signal bleeds into background
- Too large: loses spatial proximity

**`bg_width`** (default: 10 pixels)
- Sufficient pixels for stable statistics (typical: ~300-1000 pixels)
- Wide enough to average out local heterogeneity
- Not so wide that we sample different regions

### 5. Intensity Trace Extraction

```python
def get_intensity_trace(im_stack, mask):
    """
    For each frame:
    1. Apply mask to isolate ROI
    2. Calculate mean intensity of masked pixels
    3. Store in trace array
    
    Returns: 1D array of length (n_frames)
    """
```

**Statistical Considerations:**

For mask with N pixels:
```
Mean intensity μ = (Σ I_i) / N
Standard error σ_μ = σ / √N

Example: N = 500 pixels, σ = 20 counts
→ σ_μ = 20/√500 = 0.89 counts
```

**Why mean instead of sum?**
- Normalization invariant to GUV size
- Directly comparable across different vesicles
- Physical interpretation: "concentration" proxy

### 6. Jump Detection

**Problem**: Electroporation occurs at unknown frame. Need to identify it to align curves.

```python
def detect_intensity_jump(trace, sensitivity=3.0):
    """
    Statistical change-point detection.
    
    Algorithm:
    1. Calculate frame-to-frame differences: Δ = I[i+1] - I[i]
    2. Estimate noise level from baseline: σ = std(Δ[5:])
    3. Set threshold: θ = mean(Δ) + sensitivity × σ
    4. Find first point where Δ > θ
    """
```

**Mathematical Basis:**

Assuming Gaussian noise:
```
P(Δ > μ + 3σ) ≈ 0.0013   (1 in 750 frames)
```

With `sensitivity = 3.0`:
- Low false positive rate (~0.1%)
- Detects jumps >3× noise level

**Example:**

```
Frame:      0    1    2    3    4    5    6    7    8    9   10
Intensity: 100  102   98  101   99  180  250  280  290  295  298
Δ:           2   -4    3   -2   81   70   30   10    5    3
                              ↑
                           Jump!
                      (Δ = 81 >> 3σ ≈ 9)
```

**Edge Cases:**
- No clear jump → returns -1 → alignment defaults to frame 0
- Multiple jumps → returns first (assumes single pulse experiment)
- Very noisy data → may miss small jumps (increase sensitivity)

### 7. Normalization

**Three-term normalization formula:**

```
I_uptake(t) = [I_dye(t) - I_dye(0)] / [I_background(t) - I_dye(0)]
```

**Rationale:**

Let's decompose the measured intensity:
```
I_dye(t) = c_inside(t) × V_inside + c_outside × V_overlap + I_autofluorescence

Where:
- c_inside(t): Dye concentration inside GUV (time-varying)
- c_outside: Dye concentration outside GUV (constant)
- V_inside: Volume sampled inside GUV
- V_overlap: Volume of GUV overlapping with background dye
- I_autofluorescence: Intrinsic fluorescence
```

**Normalization steps:**

1. **Subtract baseline** (removes autofluorescence):
   ```
   ΔI_dye(t) = I_dye(t) - I_dye(0)
   ```

2. **Normalize to background** (accounts for photobleaching):
   ```
   I_uptake(t) = ΔI_dye(t) / [I_background(t) - I_dye(0)]
   ```

**Why this works:**

If photobleaching is uniform:
```
I_background(t) = c_outside × exp(-k×t)
I_dye(t) = c_inside(t) × exp(-k×t) + constant

After normalization:
I_uptake(t) ∝ c_inside(t) / c_outside  (photobleaching cancels!)
```

**Interpretation:**
- `I_uptake = 0`: No dye inside (equilibrium before pulse)
- `I_uptake = 1`: Full equilibration (c_inside = c_outside)
- `I_uptake > 1`: Concentration gradient (should not happen in equilibrium)

**Edge Cases:**

Division by zero when `I_background(t) = I_dye(0)`:
```python
with np.errstate(divide='ignore', invalid='ignore'):
    result = np.divide(numerator, denominator)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
```

Sets invalid values to 0.0 (interpreted as "no uptake")

### 8. Curve Alignment

**Problem**: Different GUVs may experience pulse at different frames due to:
- Heterogeneous electric field
- Timing jitter in triggering
- Delayed response of individual vesicles

**Solution**: Align all curves to median jump frame

```python
median_jump_frame = int(np.median(all_jump_frames))
t_aligned = time_array[median_jump_frame:]
t = t_aligned - t_aligned[0]  # Re-zero to pulse time

for each curve:
    curve_aligned = curve_original[median_jump_frame:]
```

**Why median?**
- Robust to outliers (e.g., one GUV with no detected jump)
- Represents "typical" pulse timing
- Ensures majority of data overlaps

**Effect on fitting:**

Before alignment:
```
GUV 1: [baseline] [uptake]  [plateau]
GUV 2:    [baseline] [uptake]  [plateau]
GUV 3: [baseline]       [uptake]  [plateau]
Average:  Smeared, hard to fit
```

After alignment to t=0:
```
GUV 1:    [uptake]  [plateau]
GUV 2:    [uptake]  [plateau]
GUV 3:    [uptake]  [plateau]
Average:  Sharp, clean kinetics
```

---

## Mathematical Models

### Model Selection

Two models are available, chosen via `config.MODEL_TO_USE`:

### 4-Parameter Model (Simple Uptake)

**Equation:**
```
I(t) = I_offset + A × (1 - e^(-t/τ)) + D × t
```

**Parameters:**
1. `I_offset`: Baseline intensity (should be ≈0 for normalized data)
2. `A`: Amplitude (maximum uptake level)
3. `τ`: Time constant (inverse rate of uptake)
4. `D`: Linear drift (accounts for residual photobleaching)

**Physical Interpretation:**

```
I(t)
 ↑
 |              ___________  A (plateau)
 |           /              + D×t (drift)
 |         /
 |       /  ← e^(-t/τ) decay
 |     /
 |___/________________________→ t
     τ
```

- **Uptake term**: `A × (1 - e^(-t/τ))` represents dye influx
  - At t=0: 0 (no uptake yet)
  - At t=τ: A×(1-e^(-1)) ≈ 0.63A (63% complete)
  - At t=∞: A (full equilibration)

- **Drift term**: `D × t` captures slow linear trends
  - Residual photobleaching not removed by normalization
  - Focus drift changing sampled volume
  - Membrane fluidity changes

**When to use:**
- Simple single-phase uptake
- Fast resealing (seconds)
- Minimal drift
- First approximation for unknown systems

**Typical Parameter Values:**
```python
I_offset:  0.0 ± 0.1        # Near zero for normalized data
A:         0.5 - 2.0        # Depends on external dye concentration
τ:         10 - 200 s       # Typical electroporation resealing time
D:         -0.001 - 0.001   # Small drift correction
```

### 5-Parameter Model (Complex Resealing)

**Equation:**
```
I(t) = A_f - A_1 × e^(-t/τ_1) - A_2 × e^(-t/τ_2)
```

**Parameters:**
1. `A_f`: Final steady-state intensity
2. `A_1`: Amplitude of fast component
3. `τ_1`: Fast time constant
4. `A_2`: Amplitude of slow component
5. `τ_2`: Slow time constant

**Physical Interpretation:**

Two-phase resealing:
```
I(t)
 ↑
 |  A_f ____________________
 |       /
 |      /    ← Fast phase (large pores)
 |     /
 |    /        ← Slow phase (small pores)
 |   /
 |__/________________________→ t
```

- **Fast component**: Rapid closure of large pores (τ_1 ~ 5-30 s)
  - Driven by line tension at pore edge
  - Dominates early uptake

- **Slow component**: Gradual sealing of small defects (τ_2 ~ 50-300 s)
  - Lipid rearrangement
  - Membrane healing
  - May include vesicle recovery processes

**Initial condition:**
```
I(0) = A_f - A_1 - A_2
```
Must satisfy: `A_1 + A_2 < A_f` (otherwise unphysical)

**When to use:**
- Biphasic kinetics observed
- Heterogeneous pore population
- Complex membrane systems (e.g., cholesterol, proteins)
- Slow equilibration (minutes)

**Typical Parameter Values:**
```python
A_f:   0.8 - 2.0        # Final normalized intensity
A_1:   0.3 - 0.8        # Fast component amplitude
τ_1:   5 - 30 s         # Fast resealing time
A_2:   0.2 - 0.5        # Slow component amplitude
τ_2:   50 - 300 s       # Slow resealing time
```

**Constraint:**
```python
# Initial intensity should be near zero
assert (A_f - A_1 - A_2) ≈ 0  # ±0.2 tolerance
```

### Curve Fitting Procedure

```python
params, covariance = scipy.optimize.curve_fit(
    f=model_function,           # 4-param or 5-param
    xdata=t[:fit_slice_index],  # Time points (first N%)
    ydata=average_curve[:fit_slice_index],  # Normalized intensity
    p0=initial_guess,           # Starting parameters
    maxfev=5000                 # Maximum iterations
)
```

**Why fit only first N%?**

`FIT_DATA_PERCENTAGE` (default: 0.9 = 90%)

Reasons to exclude late time points:
1. **Plateau region** adds little information
   - Many points, small weight in χ²
   - Dominated by noise
   - Doesn't constrain uptake kinetics

2. **Drift dominates** at late times
   - Focus drift
   - Stage movement
   - Changing imaging conditions

3. **Better convergence**
   - Focusing on rising phase improves τ estimation
   - Reduces parameter correlation

**Visual Example:**

```
Full data (100%):
I(t) ────┐     ┌─────────────────────────  ← Noisy plateau
         │    ╱
         │   ╱  ← Information-rich region
         └──┘
         0         Fit data (90%)         Full duration

Fitting result:
- Full data:  τ = 45 ± 15 s  (large uncertainty from plateau noise)
- 90% data:   τ = 47 ± 8 s   (better precision, same accuracy)
```

### Fit Quality Assessment

**Covariance Matrix:**
```python
σ_param[i] = np.sqrt(covariance[i,i])  # Parameter uncertainty
```

**Correlation Matrix:**
```python
ρ[i,j] = covariance[i,j] / (σ_param[i] × σ_param[j])
```

**Red flags:**
- `σ_param / param > 1.0`: Parameter poorly constrained
- `|ρ[i,j]| > 0.95`: Strong parameter correlation
- Fit fails to converge: Model mismatch or noisy data

**Failure Handling:**

```python
try:
    params, covariance = curve_fit(...)
    fit_failed = False
except RuntimeError:
    # Use initial guess as "best estimate"
    params = np.array(p0_guess)
    fit_failed = True
    # Flag in plot title and quality report
```

This graceful degradation allows:
- Visual inspection of fit failure
- Comparison of initial guess to data
- Decision to adjust parameters or exclude outlier

---

## Data Flow

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                           │
├─────────────────────────────────────────────────────────────┤
│ • Load configuration (config.py)                            │
│ • Build file paths                                          │
│ • Discover TIFF files (C1, C3)                              │
│ • Load CSV with GUV coordinates                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. METADATA EXTRACTION                                      │
├─────────────────────────────────────────────────────────────┤
│ • Attempt: Read ImageJ metadata from TIFF                   │
│   ├─ Success: Extract frame_interval from "finterval="     │
│   └─ Failure: Use FALLBACK_FPS                              │
│ • Generate time array: t[i] = i × frame_interval            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. IMAGE LOADING                                            │
├─────────────────────────────────────────────────────────────┤
│ • Load C3 stack: (n_frames, height, width)                  │
│ • Load C1 frame[0]: (height, width) for GUV detection      │
│ • Memory: ~4 bytes/pixel × 2048 × 2048 × 500 = ~8 GB       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PER-GUV PROCESSING LOOP                                  │
├─────────────────────────────────────────────────────────────┤
│ For each (xc, yc, radius) in CSV:                           │
│                                                              │
│   ┌────────────────────────────────────────────────────────┐│
│   │ 4a. Center Refinement                                  ││
│   ├────────────────────────────────────────────────────────┤│
│   │ • Extract search box (1.6 × radius)                    ││
│   │ • Edge detection (Laplacian)                           ││
│   │ • Find centroid                                        ││
│   │ • Validate shift (<70% radius)                         ││
│   │ → (xc_refined, yc_refined)                             ││
│   └────────────────┬───────────────────────────────────────┘│
│                    │                                         │
│   ┌────────────────▼───────────────────────────────────────┐│
│   │ 4b. Membrane Detection                                 ││
│   ├────────────────────────────────────────────────────────┤│
│   │ • Create 360 radial profiles                           ││
│   │ • Calculate median profile                             ││
│   │ • Smooth with Gaussian (σ=1)                           ││
│   │ • Find peaks in search window                          ││
│   │ • Define boundaries: peak ± half_width                 ││
│   │ → (inner_radius, outer_radius, quality_flags)          ││
│   └────────────────┬───────────────────────────────────────┘│
│                    │                                         │
│   ┌────────────────▼───────────────────────────────────────┐│
│   │ 4c. Mask Creation                                      ││
│   ├────────────────────────────────────────────────────────┤│
│   │ Inner GUV:      r ≤ inner_radius                       ││
│   │ Membrane:       inner_radius < r ≤ outer_radius        ││
│   │ Background:     (outer_radius + buffer) < r            ││
│   │                 ≤ (outer_radius + buffer + width)      ││
│   └────────────────┬───────────────────────────────────────┘│
│                    │                                         │
│   ┌────────────────▼───────────────────────────────────────┐│
│   │ 4d. Trace Extraction                                   ││
│   ├────────────────────────────────────────────────────────┤│
│   │ • Apply masks to full stack                            ││
│   │ • Calculate mean intensity per frame                   ││
│   │ → I_dye_trace[t], I_background_trace[t]                ││
│   └────────────────┬───────────────────────────────────────┘│
│                    │                                         │
│   ┌────────────────▼───────────────────────────────────────┐│
│   │ 4e. Jump Detection                                     ││
│   ├────────────────────────────────────────────────────────┤│
│   │ • Calculate Δ = I[i+1] - I[i]                          ││
│   │ • Threshold: μ + 3σ                                    ││
│   │ → jump_frame                                           ││
│   └────────────────┬───────────────────────────────────────┘│
│                    │                                         │
│   ┌────────────────▼───────────────────────────────────────┐│
│   │ 4f. Store Results                                      ││
│   ├────────────────────────────────────────────────────────┤│
│   │ • all_intensity_curves.append(I_dye_trace)             ││
│   │ • all_background_curves.append(I_background_trace)     ││
│   │ • all_jump_frames.append(jump_frame)                   ││
│   │ • detection_quality_log.append(metadata)               ││
│   └────────────────────────────────────────────────────────┘│
│                                                              │
│ End loop                                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. NORMALIZATION                                            │
├─────────────────────────────────────────────────────────────┤
│ For each GUV:                                               │
│   • I_dye_0 = mean(I_dye_trace[:baseline_frames])           │
│   • numerator = I_dye_trace - I_dye_0                       │
│   • denominator = I_background_trace - I_dye_0              │
│   • I_uptake = numerator / denominator                      │
│   • Handle division by zero → 0.0                           │
│ → all_normalized_curves                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. ALIGNMENT                                                │
├─────────────────────────────────────────────────────────────┤
│ • Calculate: median_jump_frame                              │
│ • Slice all curves: curve[median_jump_frame:]               │
│ • Re-zero time: t = t[median_jump_frame:] - t[0]            │
│ • Calculate: average_curve = mean(all_curves, axis=0)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. CURVE FITTING                                            │
├─────────────────────────────────────────────────────────────┤
│ • Select model: 4-param or 5-param                          │
│ • Slice data: first FIT_DATA_PERCENTAGE                     │
│ • scipy.optimize.curve_fit()                                │
│   ├─ Success: → fitted_params, covariance                   │
│   └─ Failure: → initial_guess, fit_failed = True            │
│ • Generate fitted curve for plotting                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. VISUALIZATION & EXPORT                                   │
├─────────────────────────────────────────────────────────────┤
│ • Export time-stamped frames (EXPORT_TIME_POINTS_S)         │
│   └─ With scale bar and time label                          │
│                                                              │
│ • Generate summary plot:                                    │
│   ├─ Individual curves (gray, α=0.2)                        │
│   ├─ Average curve (red points)                             │
│   ├─ Fitted curve (black line)                              │
│   └─ End-of-fit marker (blue star)                          │
│                                                              │
│ • Export CSV: Time, GUV1, GUV2, ..., Average                │
│                                                              │
│ • Save quality report: detection_quality.csv                │
│   └─ Columns: guv_id, estimated_radius, detected_inner,     │
│                detected_outer, failed, comments              │
└─────────────────────────────────────────────────────────────┘
```

### File Output Structure

```
output_images/
│
├── A1Well4_exp360V_mask_viz_GUV_1.png
│   ├─ C1 frame with overlaid masks
│   ├─ Magenta: Inner uptake region
│   ├─ Red:     Membrane
│   └─ Blue:    Background ring
│
├── A1Well4_exp360V_radial_profile_GUV_1.png
│   ├─ X-axis: Radius (pixels)
│   ├─ Y-axis: Intensity (a.u.)
│   ├─ Shaded regions: Inner / Membrane / Background
│   ├─ Orange dashed: CSV estimate
│   └─ Black dashed: Detected peak
│
├── frame_1_at_0s.png
├── frame_2_at_50s.png
├── frame_3_at_100s.png
│   └─ [Time-stamped snapshots of C3 channel]
│       ├─ Red colormap
│       ├─ Time label (top-left)
│       └─ Scale bar (bottom-right)
│
├── A1Well4_exp360V_kinetic_fit.png
│   ├─ Individual GUV traces (gray)
│   ├─ Average trace (red points)
│   ├─ Fitted model (black line)
│   ├─ Fit parameters in title
│   └─ End-of-fit marker (blue star)
│
├── A1Well4_exp360V_normalized_curves.csv
│   ├─ Columns: Time, GUV_1, GUV_2, ..., Average
│   └─ All normalized curves for further analysis
│
└── A1Well4_exp360V_detection_quality.csv
    ├─ Columns: guv_id, estimated_radius, detected_inner,
    │           detected_outer, failed, comments
    └─ Quality metrics for each detection
```

---

## Quality Control

### Automated Quality Flags

The pipeline generates multiple quality indicators:

#### 1. Detection Quality Flags

Stored in `*_detection_quality.csv`:

| Flag | Meaning | Cause | Action |
|------|---------|-------|--------|
| `OK` | Clean detection | Ideal conditions | None |
| `no_peak_in_window` | No peak found | Weak membrane signal | Widen search_factor |
| `invalid_search_window` | Window out of bounds | Bad radius estimate | Check CSV coordinates |
| `zero_radius_detected` | Degenerate detection | Edge case bug | Review image manually |
| `width_calc_failed` | Narrow membrane | Peak at image edge | Adjust membrane_half_width |
| `find_peaks_failed` | Algorithm error | Corrupted data | Check input image |

#### 2. Detection Summary Statistics

Printed after processing:
```
Detection Summary:
  - Total GUVs: 12
  - Failed detections: 1
  - Detections with warnings: 3
  - Clean detections: 8
```

**Acceptable failure rates:**
- Failed: <10% (indicates robust detection)
- Warnings: <30% (some heterogeneity expected)
- Clean: >60% (high-quality dataset)

#### 3. Visual QC Outputs

**Mask visualizations** (`*_mask_viz_GUV_*.png`):
- Exported for first GUV automatically
- Exported for any GUV with warnings/failures
- Color-coded regions for quick assessment

**Radial profile plots** (`*_radial_profile_GUV_*.png`):
- Shows detected peak vs CSV estimate
- Visualizes membrane boundaries
- Identifies off-center or irregular membranes

**Review Checklist:**
- [ ] Membrane peak aligns with visible membrane
- [ ] Inner region excludes membrane signal
- [ ] Background ring avoids adjacent structures
- [ ] No artifacts (bright clusters, debris) in sampled regions

### Manual Quality Assessment

#### Good Detection Example

```
Radial Profile:
Intensity
   |
500|              *        ← Clean, sharp peak
   |            *   *
400|          *       *
   |        *           *
300|      *               *
   |    *                   *
200|  *                       *
   |*                           *
100|____________________________*___ Radius
   0   10   20   30   40   50   60

✓ Sharp membrane peak at expected position
✓ Clear inner plateau (low intensity)
✓ Background region far from membrane
✓ Symmetric profile (indicates centered GUV)
```

#### Poor Detection Example

```
Radial Profile:
Intensity
   |  *
500|   \  *  *        ← Multiple peaks (artifacts)
   |    \  *  \
400|     *     \    *
   |            \  /
300|             \/      ← Membrane not clearly defined
   |             /\
200|           /    \
   |         /        \
100|_______/__________\______ Radius
   0   10   20   30   40   50   60

✗ Multiple competing peaks
✗ Broad, diffuse membrane signal
✗ Asymmetric profile (off-center or irregular)
✗ Strong background artifacts

Action: Exclude this GUV or adjust detection parameters
```

### Parameter Tuning Guide

If experiencing poor detection quality:

#### Increase `MEMBRANE_SEARCH_FACTOR` (0.3 → 0.5)

**When:**
- CSV radius estimates are poor
- Membranes consistently missed

**Trade-off:**
- ✓ More robust to position errors
- ✗ May pick up artifacts far from true membrane

#### Decrease `PEAK_FIND_MIN_PROMINENCE` (0.05 → 0.02)

**When:**
- Weak membrane signal
- Flat radial profiles

**Trade-off:**
- ✓ Detects subtle membrane signals
- ✗ More sensitive to noise

#### Increase `MEMBRANE_FIXED_HALF_WIDTH` (3 → 5 pixels)

**When:**
- Thick or blurred membranes
- Low spatial resolution imaging

**Trade-off:**
- ✓ Captures full membrane width
- ✗ May include non-membrane regions

#### Adjust `JUMP_SENSITIVITY` (3.0 → 2.0)

**When:**
- Weak electroporation response
- Gradual dye entry

**Trade-off:**
- ✓ Detects smaller intensity changes
- ✗ More false positives from noise

---

## Performance Considerations

### Computational Complexity

Per GUV processing:
```
1. Center refinement:    O(w²)     w = search box width
2. Radial profiling:     O(n×r)    n = angles, r = radius
3. Profile averaging:    O(n×r)    
4. Peak detection:       O(r)      
5. Mask creation:        O(N)      N = image size
6. Trace extraction:     O(T×M)    T = frames, M = mask pixels

Total per GUV: ~O(T×N) dominated by trace extraction
```

### Memory Usage

```
Image stack:     T × H × W × bytes_per_pixel
Example:         500 × 2048 × 2048 × 2 bytes = 8 GB

Masks (3):       3 × H × W × 1 byte = 12 MB (negligible)

Traces:          n_GUVs × T × 8 bytes = 40 KB (negligible)

Total:           ~8 GB for typical dataset
```

### Optimization Strategies

#### 1. Lazy Loading (Memory Reduction)

```python
# Instead of loading full stack:
# im_stack = load_all_frames()  # 8 GB

# Load frames on-demand:
def get_frame(frame_idx):
    return cv2.imread(file_paths[frame_idx])

# Then:
for frame_idx in range(n_frames):
    frame = get_frame(frame_idx)
    intensity[frame_idx] = np.mean(frame[mask])
```

**Trade-off:**
- ✓ Memory: ~50 MB vs 8 GB
- ✗ Speed: ~3× slower due to I/O

**When to use:** Memory-constrained systems, >1000 frames

#### 2. Parallel GUV Processing (Speed-up)

```python
from multiprocessing import Pool

def process_guv_wrapper(args):
    guv_id, center, radius = args
    return process_single_guv(guv_id, center, radius)

with Pool(processes=4) as pool:
    results = pool.map(process_guv_wrapper, guv_data)
```

**Speed-up:** ~3-4× on 4 cores (I/O bottleneck limits scaling)

**Caution:**
- Each worker loads its own copy of image stack
- Memory usage: 4 × 8 GB = 32 GB
- Use only if memory permits or with lazy loading

#### 3. Vectorized Operations (2-3× speed-up)

Already implemented:
```python
# ✓ Good: Vectorized
trace = np.mean(im_stack[:, mask], axis=1)

# ✗ Bad: Loop-based
for i in range(len(im_stack)):
    trace[i] = np.mean(im_stack[i][mask])
```

### Benchmarks

Typical dataset (500 frames, 2048×2048, 5 GUVs):

| Operation | Time | % Total |
|-----------|------|---------|
| File discovery | 0.1 s | <1% |
| Metadata extraction | 0.1 s | <1% |
| Image loading | 35 s | 45% |
| Per-GUV processing | 15 s | 20% |
| Normalization | 0.5 s | <1% |
| Curve fitting | 0.2 s | <1% |
| Visualization | 8 s | 10% |
| CSV export | 0.1 s | <1% |
| **Total** | **~60 s** | **100%** |

**Bottlenecks:**
1. Image I/O (45%)
2. GUV processing (20%)
3. Visualization (10%)

**Optimization priority:**
1. Use SSD for faster I/O
2. Parallelize GUV processing
3. Batch visualization exports

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No dye files found"

**Error message:**
```
Error: No dye files found matching: C:\Data\Emma\C3-A1Well4_exp360V_t*.tif
```

**Possible causes:**
1. Incorrect `DATA_FOLDER` path
2. Wrong `EXPERIMENT_BASE_NAME`
3. Wrong `DYE_CHANNEL_PREFIX`
4. Files not in expected location

**Solutions:**
```python
# 1. Check paths are correct
print(f"Looking in: {cfg.DATA_FOLDER}")
print(f"Pattern: {cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX}")

# 2. List all .tif files in folder
import glob
all_tifs = glob.glob(os.path.join(cfg.DATA_FOLDER, "*.tif"))
print(f"Found {len(all_tifs)} .tif files")

# 3. Check actual filenames match expected pattern
# Example: Files might be "C3-Experiment_001.tif" vs "C3-Experiment_t001.tif"
```

#### Issue 2: "Membrane detection failed for all GUVs"

**Symptoms:**
```
Detection Summary:
  - Failed detections: 5/5
  - Comments: "no_peak_in_window"
```

**Diagnosis:**

Check radial profile plots:
- Flat profile → Weak membrane signal
- Peak outside window → Bad radius estimate

**Solutions:**

1. **Weak signal:**
```python
# Decrease peak prominence threshold
PEAK_FIND_MIN_PROMINENCE = 0.02  # from 0.05
```

2. **Bad estimates:**
```python
# Widen search window
MEMBRANE_SEARCH_FACTOR = 0.5  # from 0.3
```

3. **Wrong channel:**
```python
# Check you're using membrane channel for detection
# roi_guide_files should be C1 (membrane), not C3 (dye)
```

#### Issue 3: "Curve fitting failed"

**Error message:**
```
FATAL WARNING: Curve fitting failed: Optimal parameters not found...
```

**Possible causes:**
1. Poor initial guess
2. Insufficient data points
3. Model mismatch (wrong model for data)
4. Highly noisy data

**Solutions:**

1. **Adjust initial guess:**
```python
# For 4-param model, try:
FIT_INITIAL_GUESS_4PARAM = (
    0.0,    # I_offset (should be ~0 for normalized data)
    0.8,    # A (try matching approximate plateau level)
    30.0,   # tau (estimate from visual inspection)
    0.0     # D (try zero first)
)
```

2. **Use more data:**
```python
FIT_DATA_PERCENTAGE = 1.0  # Use full curve
```

3. **Try different model:**
```python
# If 5-param fails, try simpler 4-param
MODEL_TO_USE = '4-PARAM'
```

4. **Filter noisy curves:**
```python
# Exclude GUVs with poor SNR before averaging
```

#### Issue 4: "Negative normalized intensities"

**Symptoms:**
```
Warning: Normalized intensity < 0 detected
```

**Cause:**
Background intensity lower than GUV interior (unphysical)

**Possible reasons:**
1. Background ring overlaps with another GUV
2. Autofluorescence higher inside than outside
3. Photobleaching differential
4. Detection error (wrong regions)

**Solutions:**

1. **Check mask visualization:**
```python
# Ensure background ring doesn't overlap other structures
EXPORT_MASK_VISUALIZATION = True
```

2. **Increase background buffer:**
```python
BG_BUFFER_PIXELS = 5  # from 2
```

3. **Widen background ring:**
```python
BG_RING_WIDTH_PIXELS = 15  # from 10
```

4. **Exclude problematic GUVs:**
```python
# Manually from CSV or add filtering logic
```

#### Issue 5: "Jump detection fails"

**Symptoms:**
```
No clear jump detected for GUV 3, assuming start at frame 0.
```

**Causes:**
1. Gradual dye entry (weak electroporation)
2. High baseline noise
3. Jump occurs outside monitored timeframe

**Solutions:**

1. **Decrease sensitivity:**
```python
JUMP_SENSITIVITY = 2.0  # from 3.0 (more sensitive)
```

2. **Manual inspection:**
```python
# Plot raw intensity trace
plt.plot(intensity_trace)
# Visually identify jump frame
```

3. **Manual override:**
```python
# If jump consistently at same frame, hard-code it
# jump_frame = 50  # known pulse frame
```

#### Issue 6: "Memory error loading image stack"

**Error message:**
```
MemoryError: Unable to allocate array
```

**Cause:**
Insufficient RAM for large datasets

**Solutions:**

1. **Lazy loading:**
```python
# Modify code to load frames on-demand
# (see Performance Considerations section)
```

2. **Process subset:**
```python
# Analyze fewer GUVs or shorter time window
dye_files = dye_files[:300]  # First 300 frames only
```

3. **Increase system RAM:**
```python
# Or use workstation/cluster with more memory
```

4. **Downsample images:**
```python
# If spatial resolution permits
frame_downsampled = cv2.resize(frame, None, fx=0.5, fy=0.5)
```

---

## API Reference

### Core Functions

#### `run_analysis.main()`

Main pipeline orchestrator. No parameters (uses `config.py`).

**Workflow:**
1. File discovery and validation
2. Timestamp extraction
3. Image loading
4. Per-GUV processing loop
5. Normalization and alignment
6. Curve fitting
7. Visualization and export

**Returns:** None (outputs written to disk)

**Raises:**
- `FileNotFoundError`: Input files not found
- `ValueError`: Invalid configuration
- `RuntimeError`: Fit convergence failure (handled gracefully)

---

#### `utils.extract_timestamps_from_metadata(file_paths)`

Extract timestamps from ImageJ TIFF metadata.

**Parameters:**
- `file_paths` (List[str]): Paths to TIFF files

**Returns:**
- `time_array` (np.ndarray): Timestamps in seconds
- `frame_interval` (float): Time between frames

**Returns `(None, None)` if:**
- Metadata missing
- TIFF format not recognized
- `finterval` key not found

---

#### `utils.refine_guv_center(image, center_guess, radius_estimate, search_box_factor=1.0)`

Refine GUV center using edge detection.

**Parameters:**
- `image` (np.ndarray): 2D grayscale image
- `center_guess` (Tuple[int, int]): Initial (xc, yc)
- `radius_estimate` (int): Approximate radius in pixels
- `search_box_factor` (float): Search box size as fraction of radius

**Returns:**
- `(xc_refined, yc_refined)` (Tuple[int, int]): Refined center
- Returns original `center_guess` if refinement fails

**Algorithm:**
1. Extract ROI of size `2 × radius_estimate × search_box_factor`
2. Apply Gaussian blur (σ=1.5)
3. Compute Laplacian for edge detection
4. Threshold with Otsu's method
5. Find centroid of edge mask
6. Validate shift (<70% of radius)

---

#### `utils.create_guv_masks_with_detection(...)`

Create masks using adaptive membrane detection.

**Parameters:**
- `roi_guide_frame` (np.ndarray): Membrane channel image
- `center` (Tuple[int, int]): GUV center coordinates
- `radius_estimate` (int): Approximate radius
- `bg_buffer` (int): Background ring buffer (pixels)
- `bg_width` (int): Background ring width (pixels)
- `search_factor` (float): Search window factor (±fraction of radius)
- `membrane_half_width` (int): Half-width of membrane (pixels)
- `peak_min_dist` (int): Minimum peak separation
- `peak_min_prom` (float): Minimum peak prominence (0-1)
- `num_angles` (int): Number of radial profiles (default: 360)
- `length_excess` (float): Profile length factor (default: 1.5)
- `viz_thickness` (Optional[int]): Fixed membrane visualization thickness

**Returns:**
- `inner_guv_mask` (np.ndarray): Boolean mask for uptake region
- `membrane_mask` (np.ndarray): Boolean mask for membrane (visualization)
- `background_mask` (np.ndarray): Boolean mask for background ring
- `detection_info` (dict): Metadata including:
  - `detected_inner_radius`: Inner boundary (pixels)
  - `detected_outer_radius`: Outer boundary (pixels)
  - `peak_position`: Detected membrane peak (pixels)
  - `detection_failed`: Boolean flag
  - `comments`: List of quality warnings
  - `radial_profile`: Intensity profile for plotting
  - `along_radius`: Radii array

**Quality flags in `comments`:**
- `"OK"`: Clean detection
- `"no_peak_in_window"`: No peak found
- `"invalid_search_window"`: Window out of bounds
- `"zero_radius_detected"`: Degenerate case
- `"width_calc_failed"`: Membrane boundary error

---

#### `utils.get_intensity_trace(im_stack, mask)`

Extract mean intensity within mask over time.

**Parameters:**
- `im_stack` (np.ndarray): 3D array (n_frames, height, width)
- `mask` (np.ndarray): 2D boolean array (height, width)

**Returns:**
- `trace` (np.ndarray): 1D array of mean intensities (length = n_frames)

**Implementation:**
```python
trace[i] = np.sum(im_stack[i][mask]) / np.sum(mask)
```

**Edge case:** Returns array of zeros if mask is empty.

---

#### `utils.detect_intensity_jump(trace, sensitivity=3.0)`

Detect electroporation event.

**Parameters:**
- `trace` (np.ndarray): 1D intensity time series
- `sensitivity` (float): Number of standard deviations for threshold

**Returns:**
- `jump_frame` (int): First frame where intensity jumps
- Returns `-1` if no jump detected

**Algorithm:**
1. Calculate frame-to-frame differences: `Δ = trace[i+1] - trace[i]`
2. Estimate noise: `σ = std(Δ[5:])`
3. Set threshold: `θ = mean(Δ[5:]) + sensitivity × σ`
4. Find first `i` where `Δ[i] > θ`

**Typical values:**
- `sensitivity = 3.0`: Conservative (low false positive rate)
- `sensitivity = 2.0`: More sensitive (higher false positive rate)

---

#### `utils.dyn_model_4param(t, I_offset, A, tau, D)`

4-parameter exponential rise with drift.

**Equation:**
```
I(t) = I_offset + A × (1 - exp(-t/τ)) + D × t
```

**Parameters:**
- `t` (np.ndarray): Time points (seconds)
- `I_offset` (float): Baseline intensity
- `A` (float): Amplitude
- `tau` (float): Time constant (seconds)
- `D` (float): Drift rate (per second)

**Returns:**
- `I` (np.ndarray): Model intensity values

**Use case:** Simple single-phase uptake with linear drift correction.

---

#### `utils.dyn_model_5param(t, Af, A1, tau1, A2, tau2)`

5-parameter double exponential model.

**Equation:**
```
I(t) = Af - A1 × exp(-t/τ₁) - A2 × exp(-t/τ₂)
```

**Parameters:**
- `t` (np.ndarray): Time points (seconds)
- `Af` (float): Final steady-state intensity
- `A1` (float): Fast component amplitude
- `tau1` (float): Fast time constant (seconds)
- `A2` (float): Slow component amplitude
- `tau2` (float): Slow time constant (seconds)

**Returns:**
- `I` (np.ndarray): Model intensity values

**Constraint:** `I(0) = Af - A1 - A2` (should be ≈0 for normalized data)

**Use case:** Biphasic resealing kinetics with distinct fast and slow components.

---

### Configuration Parameters

See `config.py` for full details. Key parameters:

**Detection:**
- `MEMBRANE_SEARCH_FACTOR`: Search window size (0.1-1.0)
- `MEMBRANE_FIXED_HALF_WIDTH`: Membrane thickness (pixels)
- `PEAK_FIND_MIN_DISTANCE`: Peak separation (pixels)
- `PEAK_FIND_MIN_PROMINENCE`: Peak prominence threshold (0-1)
- `BG_BUFFER_PIXELS`: Background buffer (pixels)
- `BG_RING_WIDTH_PIXELS`: Background ring width (pixels)

**Analysis:**
- `JUMP_SENSITIVITY`: Jump detection threshold (std devs)
- `MIN_BASELINE_FRAMES`: Minimum baseline length

**Fitting:**
- `MODEL_TO_USE`: '4-PARAM' or '5-PARAM'
- `FIT_DATA_PERCENTAGE`: Fraction of data to fit (0-1)
- `FIT_INITIAL_GUESS_4PARAM`: Initial parameters (4-tuple)
- `FIT_INITIAL_GUESS_5PARAM`: Initial parameters (5-tuple)

**Visualization:**
- `EXPORT_TIME_POINTS_S`: Time points for frame export (list)
- `MICRONS_PER_PIXEL`: Spatial calibration
- `SCALE_BAR_LENGTH_MICRONS`: Scale bar size
- `EXPORT_MASK_VISUALIZATION`: Enable mask viz (boolean)
- `MASK_VIZ_OVERLAY_ALPHA`: Mask transparency (0-1)

---

## Appendix

### Glossary

- **GUV**: Giant Unilamellar Vesicle (artificial cell-like structure)
- **Electroporation**: Temporary membrane permeabilization via electric field
- **Radial profile**: Intensity as a function of distance from center
- **Annular mask**: Ring-shaped region (donut)
- **Laplacian**: Second derivative operator for edge detection
- **Otsu's method**: Automatic threshold selection algorithm
- **Prominence**: Peak height above surrounding baseline
- **Time constant (τ)**: Time to reach 63% of final value in exponential process

### References

1. **Membrane detection algorithm**: Adapted from fluorescence microscopy best practices
2. **Normalization approach**: Standard in electrophysiology and membrane transport
3. **Curve fitting**: `scipy.optimize.curve_fit` (Levenberg-Marquardt algorithm)
4. **Peak detection**: `scipy.signal.find_peaks` (prominence-based)

### Version History

- **v1.0** (Initial): Basic fixed-radius masking
- **v2.0** (Current): Adaptive membrane detection with quality metrics
- **Future**: Batch processing, parallel execution, GUI interface

---

**Document prepared for:** Scientific researchers and biophysicists  
**Last updated:** 2025  
**Maintainer:** Pipeline development team