# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS CONFIGURATION ---
======================================================

This file contains all the user-configurable settings for the 
GUV intensity analysis pipeline.

**Instructions for New Users:**
1.  Set the DATA_FOLDER to the directory containing your TIFF files.
2.  Set the EXPERIMENT_BASE_NAME to the unique name of your experiment.
3.  Set the CHANNEL_PREFIX strings to match your C1 (ROI) and C3 (Dye) files.
4.  Adjust 'JUMP_THRESHOLD_PERCENT' if the script triggers too early (increase it) 
    or too late (decrease it).

"""

import numpy as np
import os

# -------------------------------------------------------------------
# --- 1. FILE & EXPERIMENT IDENTIFICATION ---
# -------------------------------------------------------------------

# Set the path to the folder containing your .tif files
# Example: "C:/Users/YourName/Documents/GUV_Data/2023_05_10/"
DATA_FOLDER = r"C:\Data\Emma\Actin"

# Set the base name of the experiment
# The script will look for files matching:
# [PREFIX][BASE_NAME][SUFFIX]
# Example: "C1-MyExperiment_t*.tif"
EXPERIMENT_BASE_NAME = "Well2_390V"

# Prefixes for your different channels
# C1 is assumed to be the ROI/Membrane channel (for finding GUVs)
ROI_CHANNEL_PREFIX = "C1-"
# C3 is assumed to be the Dye/Intensity channel (for measurement)
DYE_CHANNEL_PREFIX = "C3-"

# File suffixes (wildcards allow for numbered sequences like _t001.tif)
TIF_SUFFIX = "_t*.tif"
CSV_SUFFIX = "_detected_vesicles.csv"


# -------------------------------------------------------------------
# --- 2. ANALYSIS & MASKING PARAMETERS ---
# -------------------------------------------------------------------

# --- Membrane Detection Mode ---
# 'LABELED':   Membrane is brighter than background (Fluorescence).
#              Uses Peak Detection (looking for bright ring).
# 'UNLABELED': Membrane is darker than background (Brightfield/Phase).
#              Uses Valley Detection (looking for dark ring).
MEMBRANE_DETECTION_MODE = 'UNLABELED'

# --- Membrane Detection Search Area ---
# Search window for the membrane peak, as a factor of the CSV radius.
# 0.3 means we look for the membrane within +/- 30% of the radius 
# found in the original CSV file.
MEMBRANE_SEARCH_FACTOR = 0.3

# --- Membrane Width Definition ---
# Defines the membrane region mask as: Detected Peak +/- this value (in pixels).
# Example: If 3, the membrane mask is 7 pixels wide total.
MEMBRANE_FIXED_HALF_WIDTH = 3

# --- Peak Finding Parameters (for scipy.signal.find_peaks) ---
# Minimum pixel distance between peaks to avoid detecting noise as multiple membranes.
PEAK_FIND_MIN_DISTANCE = 5
# Required peak prominence (contrast) as a fraction of max profile intensity.
PEAK_FIND_MIN_PROMINENCE = 0.05 

# --- Background Mask Geometry ---
# Buffer space (in pixels) between the GUV's outer edge and the
# *start* of the background ring. Used to ensure no membrane signal bleeds into bg.
BG_BUFFER_PIXELS = 2

# Width (in pixels) of the annular ring used for background subtraction.
BG_RING_WIDTH_PIXELS = 10

# --- Jump Detection (CRITICAL FOR KINETICS) ---

# The threshold for detecting the dye entry (The "Trigger" point).
# The algorithm calculates the Dynamic Range (Max - Min) of the trace.
# It defines the jump as the moment the intensity crosses X% of that range.
# 0.20 (20%) is robust for step-functions. 
# - Lower (e.g., 0.10) makes it more sensitive but risks triggering on noise.
# - Higher (e.g., 0.50) makes it safer but might detect the jump late.
JUMP_THRESHOLD_PERCENT = 0.20

# Minimum number of frames used for baseline calculation (before the jump)
# Used to calculate the initial intensity (I0) for normalization.
MIN_BASELINE_FRAMES = 3


# -------------------------------------------------------------------
# --- 3. FITTING & MODELING PARAMETERS ---
# -------------------------------------------------------------------

# --- Model Selection ---
# '4-PARAM': Exponential rise + linear drift (I = I_off + A(1-e^-t/tau) + Dt)
# '5-PARAM': Double exponential rise (Two timescales)
MODEL_TO_USE = '4-PARAM'

# --- Data Slicing ---
# Percentage of the data (from 0.0 to 1.0) to use for fitting.
# 0.9 means we ignore the last 10% of the video (useful if photobleaching occurs late).
FIT_DATA_PERCENTAGE = 0.9

# --- Initial Guesses for 5-PARAM Model ---
# Format: (Af, A1, tau1, A2, tau2)
FIT_INITIAL_GUESS_5PARAM = (
    1.0,  # Af (Final Amplitude)
    0.5,  # A1 (Amplitude of fast component)
    10.0, # tau1 (Time constant fast)
    0.5,  # A2 (Amplitude of slow component)
    100.0 # tau2 (Time constant slow)
)

# --- Initial Guesses for 4-PARAM Model ---
# Format: (I_offset, A, tau, D)
FIT_INITIAL_GUESS_4PARAM = (
    0.0,   # I_offset (Y-intercept)
    1.0,   # A (Amplitude)
    50.0,  # tau (Time constant)
    0.001  # D (Linear drift slope)
)

# -------------------------------------------------------------------
# --- 4. OUTPUT & VISUALIZATION PARAMETERS ---
# -------------------------------------------------------------------

# --- Image Export ---
OUTPUT_IMAGE_FOLDER = os.path.join(DATA_FOLDER, "output_images")

# Time points (in seconds, relative to the pulse) to export snapshot frames for.
EXPORT_TIME_POINTS_S = [0, 50, 100, 200, 300]

# --- Debugging ---
# If True, exports plots for GUVs with failed or suspicious jump detection
# (e.g., Jump detected at frame 0 or not found).
EXPORT_DEBUG_PLOTS = True

# --- Scale Bar ---
# Used for the exported snapshot frames.
MICRONS_PER_PIXEL = 0.108  
SCALE_BAR_LENGTH_MICRONS = 10

# --- Mask Visualization ---
# If True, saves an image showing the Inner, Membrane, and Background masks
# for the first GUV and any failed GUVs.
EXPORT_MASK_VISUALIZATION = True
MASK_VIZ_OVERLAY_ALPHA = 0.8
VIZ_MEMBRANE_THICKNESS_PIXELS = None


# -------------------------------------------------------------------
# --- 5. FALLBACKS & ADVANCED ---
# -------------------------------------------------------------------

# Number of parallel processes to use for GUV analysis.
# Defaults to (CPU Cores - 1). Set to 1 to force serial processing (easier debugging).
N_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

# Fallback FPS if time metadata cannot be read from the TIFF file headers.
FALLBACK_FPS = 1.0