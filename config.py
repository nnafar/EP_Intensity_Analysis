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

# -----------------------------------------------------------------------------
# --- 1. FILE & EXPERIMENT IDENTIFICATION ---
# -----------------------------------------------------------------------------

# Set this to the specific folder for the current run
DATA_FOLDER           = r"M:\tnw\bn\gk\NN\2_Data-Analysis\BulkElectroporation\260331_DOPC_BranchedCortex_Experiment1-400V-500us-frame7"

# The shared base name across all channels for the image files
EXPERIMENT_BASE_NAME  = "DOPC_BranchedCortex_Experiment1-400V-500us-frame7"

# Channel prefixes updated to use underscores
ROI_CHANNEL_PREFIX    = "C1_"    # Guide / membrane channel (used for tracking)
ACTIN_CHANNEL_PREFIX  = "C2_"    # Actin channel (for future implementation)
DYE_CHANNEL_PREFIX    = "C3_"    # Dye / measurement channel

# The new frame index pattern
TIF_SUFFIX            = "-f*.tif"

# Path where the interactively-drawn circle definitions are saved/loaded.
CIRCLES_JSON_PATH = os.path.join(DATA_FOLDER, f"{EXPERIMENT_BASE_NAME}_guv_circles.json")

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

# -----------------------------------------------------------------------------
# --- 3. FITTING & MODELING ---
# -----------------------------------------------------------------------------

# Dye Efflux: ['EFFLUX-1EXP' or 'EFFLUX-2EXP'] 
# Dye Influx: ['INFLUX-1EXP' or 'INFLUX-2EXP']
MODEL_TO_USE        = 'EFFLUX-1EXP' 
FIT_DATA_PERCENTAGE = 0.9

# Pulse-frame detection
# Set to an integer to hard-pin the pulse frame and skip auto-detection.
# Set to None to let the pipeline detect it from the dye signal drop/rise.
PULSE_FRAME_OVERRIDE = None

# Gaussian σ (frames) used to smooth the mean trace before differentiation.
# Increase if the signal is noisy; decrease if pulses are very abrupt.
PULSE_DETECT_SMOOTH_SIGMA = 2.0

# -----------------------------------------------------------------------------
# --- 4. OUTPUT & VISUALIZATION ---
# -----------------------------------------------------------------------------

OUTPUT_IMAGE_FOLDER      = os.path.join(DATA_FOLDER, "output_images")

# Output sub-folders (created automatically at runtime)
FOLDER_TRACKING = os.path.join(OUTPUT_IMAGE_FOLDER, "tracking")   # track PNGs/AVIs, metrics, CSVs
FOLDER_MASKS    = os.path.join(OUTPUT_IMAGE_FOLDER, "masks")      # mask overlay AVI
FOLDER_DYE      = os.path.join(OUTPUT_IMAGE_FOLDER, "dye")        # fit plots, frame exports, CSVs
FOLDER_SCORES   = os.path.join(OUTPUT_IMAGE_FOLDER, "scores")     # per-GUV ring-score plots

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

# Tracking outputs
# CSV: per-frame (guv_id, frame, x, y, radius, ring_score)
EXPORT_TRACKING_DATA = True

# PNG: centre trajectory overlaid on the last valid ROI frame
EXPORT_TRACK_VISUALIZATION = True


# -----------------------------------------------------------------------------
# --- 7. ACTIN CORTEX ANALYSIS ---
# -----------------------------------------------------------------------------

# Enable actin cortex analysis on the C2 channel
ANALYZE_ACTIN_CHANNEL = True

# Which mask to use for the cortex signal.
# 'membrane' = the ring mask (peak ± MEMBRANE_FIXED_HALF_WIDTH).
# 'inner'    = the inner lumen mask (gives lumenal actin only).
# Both are always extracted; this controls the *cortex* definition.
ACTIN_CORTEX_MASK = 'membrane'

# Half-width (pixels) for the cortex ring mask in C2.
# Increase slightly if the cortex ring is thicker than the membrane label.
ACTIN_CORTEX_HALF_WIDTH = 4   # px  (~0.44 µm at 0.11 µm/px)

# Export a per-GUV CSV and plot of cortex / lumen actin over time
EXPORT_ACTIN_TRACES = True

# Smoothing sigma (frames) for the actin ratio plot; set 0 to disable
ACTIN_PLOT_SMOOTH_SIGMA = 1.5

# -----------------------------------------------------------------------------
# --- 6. ADVANCED ---
# -----------------------------------------------------------------------------

N_WORKERS    = 3 #os.cpu_count() - 1 if os.cpu_count() > 1 else 1
FALLBACK_FPS = 1.0 # second(s)