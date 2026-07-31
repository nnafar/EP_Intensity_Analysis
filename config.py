# -*- coding: utf-8 -*-
"""
GUV ANALYSIS CONFIGURATION

Workflow
--------
1. Run  `python interactive_select.py`  to draw circles on the first ROI
   frame.  This saves  CIRCLES_JSON_PATH  (e.g. "guv_circles.json").
2. Run  `python run_analysis.py`.  If CIRCLES_JSON_PATH is missing the
   selector is launched automatically.
"""

import numpy as np
import os
import re

# -----------------------------------------------------------------------------
# --- 1. FILE & EXPERIMENT IDENTIFICATION ---
# -----------------------------------------------------------------------------

INPUT_FORMAT = 'ND2' 

DATA_FOLDER  = r"D:\EP\260422_InvE_Empty_SRB_400V"
EXPERIMENT_BASE_NAME  = "DOPC_Empty_Experiment1-400V-500us-frame5"

# --- NEW DIRECTORY ROUTING ---
# Define the master parent directory where all analyses will be stored
PARENT_OUTPUT_FOLDER = r"D:\EP\Outputs"

# Extract the 6-digit YYMMDD date from the DATA_FOLDER name
_data_dir_name = os.path.basename(os.path.normpath(DATA_FOLDER))
_match = re.search(r'^(\d{6})', _data_dir_name)
_yymmdd = _match.group(1) if _match else "000000"

# Construct the specific output subfolder: Parent\YYMMDD_EXPERIMENT_BASE_NAME
OUTPUT_IMAGE_FOLDER = os.path.join(PARENT_OUTPUT_FOLDER, f"{_yymmdd}_{EXPERIMENT_BASE_NAME}")

# Route the circle definitions JSON into the new output subfolder
CIRCLES_JSON_PATH = os.path.join(OUTPUT_IMAGE_FOLDER, f"{EXPERIMENT_BASE_NAME}_guv_circles.json")

# Required if INPUT_FORMAT = 'ND2'
ND2_FILE_NAME = f"{EXPERIMENT_BASE_NAME}.nd2"

# ND2 dimensions (typically T, C, Y, X)
# 488=Membrane, 561=Dye, 640=Actin
ND2_CHANNEL_IDX_ACTIN    = 0
ND2_CHANNEL_IDX_DYE      = 1
ND2_CHANNEL_IDX_MEMBRANE = 2

# TIFF Prefixes (typically T, C, Y, X)
ROI_CHANNEL_PREFIX    = "C1_"    # Guide / membrane channel (used for tracking)
ACTIN_CHANNEL_PREFIX  = "C2_"    # Actin channel (for future implementation)
DYE_CHANNEL_PREFIX    = "C3_"    # Dye / measurement channel

# The new frame index pattern
TIF_SUFFIX            = "-f*.tif"

# Variable frame rate schedule: (start_frame_index, end_frame_index, interval_seconds)
# Set to `None` to attempt metadata extraction.
FRAME_INTERVAL_SCHEDULE = [
    (0,  5,  1.0),   # Pre-pulse: 5 frames, 1 s interval
    (5,  22, 0.1),   # Pulse: 17 frames, 100 ms (0.1 s) interval
    (22, None, 5.0)  # Post-pulse: remaining frames, 5 s interval
]

# -----------------------------------------------------------------------------
# --- 2. ANALYSIS & MASKING PARAMETERS ---
# -----------------------------------------------------------------------------

# 'LABELED'   – bright ring (fluorescent membrane)
# 'UNLABELED' – dark ring  (brightfield / phase contrast)
MEMBRANE_DETECTION_MODE = 'LABELED'

# Search window for radius detection: look ± this fraction of current radius.
MEMBRANE_SEARCH_FACTOR    = 0.35

# Half-width of the membrane mask in pixels (mask = peak ± this).
MEMBRANE_FIXED_HALF_WIDTH = 3

# scipy.signal.find_peaks parameters
PEAK_FIND_MIN_DISTANCE   = 5
PEAK_FIND_MIN_PROMINENCE = 0.05

# Background annulus geometry
BG_BUFFER_PIXELS     = 1    # gap (px) between outer membrane edge and bg ring start
BG_RING_WIDTH_PIXELS = 4    # width (px) of background sampling ring

# Background neighbor-exclusion
# When a second GUV's body (inner + membrane) overlaps this GUV's background
# annulus, those pixels are excluded from the background estimate before
# computing the median/std. BG_EXCLUSION_PADDING_PX is added on top of
# MEMBRANE_FIXED_HALF_WIDTH when defining the neighbor's excluded footprint,
# as a safety margin against a slightly under-tracked neighbor edge.
BG_EXCLUSION_PADDING_PX = 2   # px

# Minimum number of background pixels that must remain after neighbor
# exclusion. If fewer remain (e.g. a very crowded FOV), the exclusion falls
# back to the full (unexcluded) annulus for that frame rather than
# extracting background statistics from a near-empty sample.
BG_EXCLUSION_MIN_PIXELS = 20

# How many consecutive frames to keep excluding a tracked neighbor's last
# known position after its ellipse goes missing (transient tracking loss,
# not yet declared ruptured). Bridges brief tracking gaps without
# indefinitely excluding a stale position once the neighbor has actually
# ruptured/dispersed or drifted out of frame.
BG_NEIGHBOR_HOLD_FRAMES = 5

# MAD-based sigma-clip threshold applied to the background pixels AFTER
# position-based neighbor exclusion. This is the only defense against a
# neighboring vesicle the user never circled (no tracked position exists
# to exclude it by) — pixels more than this many scaled-MADs from the
# annulus median are dropped before computing background stats. Set to 0
# to disable.
BG_SIGMA_CLIP = 3.0

# Percentile of the (neighbor-excluded, sigma-clipped) annulus pixels taken
# as THE background value for each frame. This is the I_bg,t that enters the
# normalisation denominator, the dead-GUV separation gate, and the actin
# background trace.
#
# NOTE: 50.0 is exactly the median (the pipeline's original behaviour); the
# active setting is 15.0. Changing this number is not a robustness tweak — it
# moves I_bg,t, which moves every normalised curve and therefore every
# released fraction. A low percentile biases the estimate toward the dim end
# of the annulus, guarding against residual contamination from an untracked
# neighbour or bright debris that survived both the position exclusion and
# the MAD clip. But on a genuinely clean annulus it is no longer estimating
# the bath — it is estimating the dim tail of the bath's noise distribution.
#
# How that bias propagates (normalisation is (I_bg,t - I_dye,t)/(I_bg,t -
# I_dye,0), so I_bg appears in BOTH numerator and denominator and the bias
# does NOT simply cancel): a downward bias b in I_bg leaves the PRE-PULSE
# value pinned at exactly 1 — numerator and denominator are equal there, so
# that end is bias-immune — but shifts the fully-permeabilised end by about
# -b / |I_bg,0 - I_dye,0|, i.e. the bias expressed in units of pre-pulse
# contrast. For roughly Gaussian annulus noise the 15th percentile sits
# ~1.04 sigma below the mean, so the released fraction is inflated by
# approximately sigma_bg / contrast.
#
# Two things to check on real data before trusting this setting:
#   1. bg_percentile_raw vs bg_median_raw in
#      {experiment}_background_diagnostics.csv — their difference IS b. The
#      run log now prints the mean offset directly.
#   2. Fully-permeabilised GUVs in the intensity plots. With an unbiased
#      background they asymptote at 0; a downward-biased one drives them
#      slightly NEGATIVE, and released_pct in the endpoint CSV can exceed
#      100%. A small consistent undershoot is the signature of this bias
#      rather than of a real measurement.
BG_PERCENTILE = 15.0

# -----------------------------------------------------------------------------
# --- 3. FITTING & MODELING ---
# -----------------------------------------------------------------------------

# Dye Efflux: ['EFFLUX-1EXP' or 'EFFLUX-2EXP'] 
# Dye Influx: ['INFLUX-1EXP' or 'INFLUX-2EXP']
MODEL_TO_USE        = 'EFFLUX-1EXP' 
FIT_DATA_PERCENTAGE = 0.9

# Model-free endpoint figure
# Adds a per-GUV multipanel plot (and CSV) of the normalised dye intensity
# BEFORE the pulse vs. at the END of the movie. Intended for datasets where
# the post-pulse acquisition is too coarse to resolve the efflux transient,
# so a fitted tau is set by the sampling interval rather than by membrane
# permeability. The endpoint contrast makes no assumption about the shape of
# the trajectory between the two time points.
EXPORT_PREPOST_INTENSITY_PLOT = True

# Frames averaged for the pre-pulse baseline level (window ends at the frame
# immediately before the detected pulse frame — same window the dye
# normalisation uses, so I_pre comes out at ~1 by construction).
PREPOST_N_PRE_FRAMES = 5

# Frames averaged at the END of each GUV's own trace for the final level.
# Taken per-GUV from its last FINITE post-pulse frames, so a vesicle whose
# tracking ended early reports its own true endpoint (with t_final_s in the
# CSV recording when that was) rather than being padded or dropped. Set to 1
# to use the single last frame; >1 averages down endpoint noise at the cost
# of reaching slightly further back in time.
PREPOST_N_FINAL_FRAMES = 3

# Single-vesicle crops accompanying the endpoint figure: membrane + dye
# (+ actin when ANALYZE_ACTIN_CHANNEL is on) at the last pre-pulse frame and
# at each GUV's own final frame. Written to the dye/intensity/ folder.
EXPORT_GUV_CROPS = True

# Half-width of the crop box in units of each GUV's PRE-PULSE radius.
# 2.0 = box spans 4 radii, leaving roughly one radius of surroundings on
# each side. The box is sized from the pre-pulse radius and held constant
# for the final frame, so deflation shows up as the vesicle shrinking
# inside a fixed field rather than being re-zoomed to fill it.
CROP_FACTOR = 2.0

# Also write each crop as a standalone 8-bit PNG into
# dye/intensity/crops/ (one file per GUV per channel per timepoint), for
# assembling figures by hand. The montage PNG is written regardless.
CROP_SAVE_INDIVIDUAL = True

# Use ONE crop box size for every GUV in the montage, sized from the largest
# pre-pulse radius in the experiment. Matplotlib stretches each panel to fill
# its axes, so per-GUV box sizes would make a 10 um and a 25 um vesicle look
# identical; a single box keeps the whole montage at one pixel scale. Set to
# False for tighter framing of small vesicles, at the cost of losing
# between-GUV size comparison by eye.
CROP_UNIFORM_BOX = True

# Display colours for the black-to-colour lookup applied to each channel.
CROP_CHANNEL_COLORS = {
    'membrane': '#00FF66',
    'dye':      '#FF3355',
    'actin':    '#33CCFF',
}

# Pulse-frame detection
# Set to an integer to hard-pin the pulse frame and skip auto-detection.
# Set to None to let the pipeline detect it from the dye signal drop/rise.
PULSE_FRAME_OVERRIDE = None

# Gaussian σ (frames) used to smooth the mean trace before differentiation.
# Increase if the signal is noisy; decrease if pulses are very abrupt.
PULSE_DETECT_SMOOTH_SIGMA = 2.0

# Dead-GUV filter: a GUV is excluded from dye-channel analysis if its
# pre-pulse interior signal is not separated from background by at least
# this many multiples of the background's own RAW per-pixel std:
#   |I_dye,0 - I_bg,0|  <  MIN_PREPULSE_SEPARATION_SIGMA * bg_std0
# This is a plain amplitude check, NOT a statistical significance test —
# it does not scale with the number of background pixels (that SEM-scaled
# version was tried and over-excluded everything; see background
# diagnostics discussion). bg_std0 is already exported per-GUV in
# {experiment}_background_diagnostics.csv (bg_std_raw), so this threshold
# can be sanity-checked against real values from your own data. Raise it
# if too many marginal-but-real GUVs are being kept; lower it if clearly
# loaded GUVs are being excluded.
MIN_PREPULSE_SEPARATION_SIGMA = 2.0

# Maximum number of frames to evaluate for the pulse within the fast-acquisition window.
# Set to None to search the entire fast window.
PULSE_SEARCH_MAX_FRAMES = 5

# -----------------------------------------------------------------------------
# --- 4. OUTPUT & VISUALIZATION ---
# -----------------------------------------------------------------------------

# Output sub-folders (created automatically at runtime inside the new directory)
FOLDER_TRACKING = os.path.join(OUTPUT_IMAGE_FOLDER, "tracking")   # track PNGs/AVIs, metrics, CSVs
FOLDER_MASKS    = os.path.join(OUTPUT_IMAGE_FOLDER, "masks")      # mask overlay AVI
FOLDER_DYE      = os.path.join(OUTPUT_IMAGE_FOLDER, "dye")        # dye-channel root: diagnostics + snapshots

# The dye folder is split so that model-dependent and model-free products
# never sit side by side. Anything under fitting/ inherits the assumptions of
# MODEL_TO_USE (and is only meaningful when the post-pulse sampling actually
# resolves the transient); anything under intensity/ is a direct measurement
# that stands on its own. Diagnostics that belong to neither (pulse-frame
# detection, background diagnostics, dye snapshots) stay in the dye/ root.
FOLDER_DYE_FITTING   = os.path.join(FOLDER_DYE, "fitting")     # fit grids, fit params, model comparison, boxplots
FOLDER_DYE_INTENSITY = os.path.join(FOLDER_DYE, "intensity")   # normalised traces, pre-vs-final endpoint
FOLDER_ACTIN    = os.path.join(OUTPUT_IMAGE_FOLDER, "actin")      # actin cortex plots, traces, profiles

EXPORT_TIME_POINTS_S     = [0, 50, 100, 200, 300]
EXPORT_DEBUG_PLOTS       = True

MICRONS_PER_PIXEL        = 0.11 # Plan Apo λ 60x Oil
SCALE_BAR_LENGTH_MICRONS = 10

# Percentiles for contrast boosting (Lower p_low or lower p_high = more contrast)
CONTRAST_P_LOW  = 0.5  # Ignore bottom 0.5% of pixels
CONTRAST_P_HIGH = 98.0 # Ignore top 2.0% of pixels

EXPORT_MASK_VISUALIZATION     = False
MASK_VIZ_OVERLAY_ALPHA        = 0.7
VIZ_MEMBRANE_THICKNESS_PIXELS = None

# Consolidated video export:
EXPORT_CONSOLIDATED_TRACK_VIDEO = True
EXPORT_CONSOLIDATED_MASK_VIDEO  = True
VIDEO_EXPORT_FPS                = 10.0


# -----------------------------------------------------------------------------
# --- 5. TRACKING PARAMETERS ---
# -----------------------------------------------------------------------------

# Add this flag to disable dye analysis
TRACKING_ONLY_MODE = False

# Centre grid search
# Search radius = previous_radius × TRACKING_SEARCH_WINDOW_FACTOR.
# Increase if GUVs move more than ~70 % of their radius between frames.
TRACKING_SEARCH_WINDOW_FACTOR = 0.7 

# Grid step = previous_radius × TRACKING_GRID_STEP_FACTOR.
# Smaller = finer / slower. 0.12–0.20 is a good balance.
TRACKING_GRID_STEP_FACTOR = 0.20 

# Radius change clamp 
# Maximum fractional radius change allowed per frame.
# 0.20 = up to 20 % shrink or grow per frame.
TRACKING_MAX_RADIUS_CHANGE_FACTOR = 0.20

# Rupture detection
# GUV is declared ruptured when its ring-quality score < this threshold.
# Typical good-frame scores are 10–30; set this to ~20–30 % of that.
RUPTURE_SCORE_THRESHOLD = 4.0

# Number of consecutive frames below RUPTURE_SCORE_THRESHOLD required
# to declare rupture (reduces false positives from single bad frames).
RUPTURE_DETECTION_CONSECUTIVE_FAILS = 3

# GUV fate refinement: distinguishes gradual SHRINKAGE (deflation) from
# abrupt RUPTURE. A GUV whose radius has declined by at least this fraction
# from its own pre-pulse baseline is classified SHRUNK regardless of how
# tracking ended — including overriding a raw RUPTURED tag, since a vesicle
# that deflates below the ring-detector's minimum resolvable size will also
# trip the tracker's rupture-score exit, but that's a shrinkage artifact of
# detection, not a membrane burst. 0.15 = 15% radius loss.
SHRINKAGE_FRACTION_THRESHOLD = 0.15

# Number of the LAST valid tracked frames (per GUV) averaged to get its
# terminal radius for the shrinkage check. Smooths single-frame ring
# detection noise right at the endpoint rather than keying the whole
# fate classification off one potentially noisy frame.
SHRINKAGE_TERMINAL_N_FRAMES = 3

# Tracking outputs
# CSV: per-frame (guv_id, frame, x, y, radius, ring_score)
EXPORT_TRACKING_DATA = True

# PNG: centre trajectory overlaid on the last valid ROI frame
EXPORT_TRACK_VISUALIZATION = True


# -----------------------------------------------------------------------------
# --- 7. ACTIN CORTEX ANALYSIS ---
# -----------------------------------------------------------------------------

# Enable actin cortex analysis on the C2 channel
ANALYZE_ACTIN_CHANNEL = False

# Which mask to use for the cortex signal.
# 'membrane' = the ring mask (peak ± MEMBRANE_FIXED_HALF_WIDTH).
# 'inner'    = the inner lumen mask (gives lumenal actin only).
# Both are always extracted; this controls the *cortex* definition.
ACTIN_CORTEX_MASK = 'membrane'

# Half-width (pixels) for the cortex ring mask in C2.
# Increase slightly if the cortex ring is thicker than the membrane label.
ACTIN_CORTEX_HALF_WIDTH = 5   # px  (~0.44 µm at 0.11 µm/px)

# Minimum peak prominence for cortex detection.
# The peak-to-baseline amplitude must exceed this fraction of the total
# radial profile range (max − min) to be accepted as a real cortex peak.
# Raise this value if flat-profile GUVs (no cortex) are being falsely detected;
# lower it if faint-but-real cortices are being missed.  Default = 0.10 (10 %).
ACTIN_PEAK_MIN_PROMINENCE = 0.10

# Number of angular samples used for the cortex angular profile and Gini index.
# Higher values give a finer angular map but increase per-frame compute time.
# 72 = 5° resolution (matches the tracking grid search default).
ACTIN_N_ANGLES = 72

# Padding (pixels) added outward to each side of the membrane FWHM border before
# it is used as the search window for the actin peak.  Matches the ±3 px padding
# in skeleton.py's membrane_detection().  Increase if the actin peak is being
# clipped at the membrane edge; decrease (or set to 0) for tighter confinement.
ACTIN_MEMBRANE_BORDER_PADDING = 3

# Export a per-GUV CSV and plot of cortex / lumen actin over time
EXPORT_ACTIN_TRACES = True

# Smoothing sigma (frames) for the actin ratio plot; set 0 to disable
ACTIN_PLOT_SMOOTH_SIGMA = 1.5

# Electrode axis orientation, in degrees, in the IMAGE FRAME (0 = along +x /
# horizontal, matching the angular sampling convention used for
# 'angular_profiles'). The electrode-facing poles are drawn at this angle
# and this angle + 180 on the actin angular kymograph. This is the
# CODE-frame angle used internally for all pole/equator masking and the
# polarization index — do not change this to relabel the plot axis; see
# ANGLE_DISPLAY_OFFSET_DEG below for that.
ELECTRODE_ANGLE_DEG = 0.0

# Display-only rotation applied to the kymograph angle axis at plot time:
#   display_angle = (code_angle + ANGLE_DISPLAY_OFFSET_DEG) mod 360
# Does NOT affect pole/equator masking, the polarization index, or any other
# computation — those stay in code-frame (0 deg = directly right of center,
# 90 deg = directly below, matching standard image row/column axes).
# Default 90 deg matches the lab convention: 0 deg at the top of the vesicle
# (a reference point on the equator, not a pole), sweeping clockwise so that
# 90 deg = cathode-facing pole (right) and 270 deg = anode-facing pole
# (left) — i.e. code 270 -> display 0, code 0 -> display 90, code 180 ->
# display 270. Must be a multiple of (360 / ACTIN_N_ANGLES) so the axis
# roll lands exactly on a sampled angle.
ANGLE_DISPLAY_OFFSET_DEG = 90.0

# Generate per-GUV actin angular kymographs (angle vs. time post-pulse,
# cortex intensity as color) plus one population-average kymograph per
# experiment, to check whether cortex breakdown after electroporation is
# directional (concentrated at the electrode-facing poles) or uniform
# (all around the vesicle). Requires ANALYZE_ACTIN_CHANNEL = True.
EXPORT_ACTIN_KYMOGRAPH = True

# Half-width (degrees) of the angular window averaged around each pole and
# each equator direction for the quantitative pole-vs-equator trace and
# polarization index. 22.5 deg means the 4 windows (2 poles + 2 equator
# points) exactly tile the full 360 deg with no gaps/overlap.
POLE_EQUATOR_HALF_WIDTH_DEG = 22.5

# -----------------------------------------------------------------------------
# --- 6. ADVANCED ---
# -----------------------------------------------------------------------------

N_WORKERS    = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
FALLBACK_FPS = 1.0 # second(s)