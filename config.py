# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS CONFIGURATION ---
======================================================

This file contains all the user-configurable settings for the 
GUV intensity analysis pipeline.

**Instructions:**
1.  Set the DATA_FOLDER to the directory containing your TIFF files.
2.  Set the EXPERIMENT_BASE_NAME to the unique name of your experiment.
3.  Set the CHANNEL_PREFIX strings to match your C1 (ROI) and C3 (Dye) files.
4.  Adjust parameters in 'ANALYSIS & MASKING' as needed.
5.  Adjust parameters in 'FITTING & MODELING' as needed.
6.  Adjust parameters in 'OUTPUT & VISUALIZATION' as needed.

"""

import numpy as np
import os

# -------------------------------------------------------------------
# --- 1. FILE & EXPERIMENT IDENTIFICATION ---
# -------------------------------------------------------------------

# Set the path to the folder containing your .tif files
# Example: "C:/Users/YourName/Documents/GUV_Data/2023_05_10/"
DATA_FOLDER = r"C:\Data\Emma\NoMembrane"

# Set the base name of the experiment
# The script will look for files matching:
# [PREFIX][BASE_NAME][SUFFIX]
# Example: "C1-MyExperiment_t*.tif"
EXPERIMENT_BASE_NAME = "Well3_330V"

# Prefixes for your different channels
# C1 is assumed to be the ROI/Membrane channel (for finding GUVs)
ROI_CHANNEL_PREFIX = "C1-"
# C3 is assumed to be the Dye/Intensity channel (for measurement)
DYE_CHANNEL_PREFIX = "C2-"

# File suffixes
TIF_SUFFIX = "_t*.tif"
CSV_SUFFIX = "_detected_vesicles.csv"


# -------------------------------------------------------------------
# --- 2. ANALYSIS & MASKING PARAMETERS ---
# -------------------------------------------------------------------

# --- Membrane Detection Mode ---
# 'LABELED':   Membrane is brighter than background (Fluorescence -> Peak detection).
# 'UNLABELED': Membrane is darker than background (Brightfield/Phase -> Valley detection).
MEMBRANE_DETECTION_MODE = 'UNLABELED'

# --- Membrane Detection ---
# Search window for the membrane peak, as a factor of the CSV radius.
# 0.3 = search in a window of +/- 30% of the radius estimate.
MEMBRANE_SEARCH_FACTOR = 0.3

# --- Membrane Width ---
# Defines the membrane region as: Detected Peak +/- this value.
MEMBRANE_FIXED_HALF_WIDTH = 3

# --- Peak Finding Parameters (for scipy.signal.find_peaks) ---
# Minimum pixel distance between peaks for detection
PEAK_FIND_MIN_DISTANCE = 5
# Required peak prominence (as a fraction of max profile intensity)
PEAK_FIND_MIN_PROMINENCE = 0.05 


# --- Background Mask Geometry ---
# Buffer space (in pixels) between the GUV's outer edge and the
# *start* of the background ring.
BG_BUFFER_PIXELS = 2

# Width (in pixels) of the annular ring used for background subtraction.
BG_RING_WIDTH_PIXELS = 10

# --- Jump Detection ---
# Sensitivity for detecting the fluorescence jump.
JUMP_SENSITIVITY = 3.0 # Standard deviations from the mean difference

# --- Baseline Calculation ---
# Minimum number of frames to use for baseline calculation
MIN_BASELINE_FRAMES = 3


# -------------------------------------------------------------------
# --- 3. FITTING & MODELING PARAMETERS ---
# -------------------------------------------------------------------

# --- Model Selection ---
# '4-PARAM': Exponential rise + linear drift
# '5-PARAM': Double exponential rise
MODEL_TO_USE = '4-PARAM'

# --- Data Slicing ---
# Percentage of the data (from 0.0 to 1.0) to use for fitting.
FIT_DATA_PERCENTAGE = 0.9

# --- Initial Guesses for 5-PARAM Model ---
FIT_INITIAL_GUESS_5PARAM = (
    1.0,  # Af
    0.5,  # A1
    10.0, # tau1
    0.5,  # A2
    100.0 # tau2
)

# --- Initial Guesses for 4-PARAM Model ---
FIT_INITIAL_GUESS_4PARAM = (
    0.0,   # I_offset
    1.0,   # A
    50.0,  # tau
    0.001  # D
)

# -------------------------------------------------------------------
# --- 4. OUTPUT & VISUALIZATION PARAMETERS ---
# -------------------------------------------------------------------

# --- Image Export ---
OUTPUT_IMAGE_FOLDER = os.path.join(DATA_FOLDER, "output_images")

# Time points (in seconds, relative to the pulse) to export frames for.
EXPORT_TIME_POINTS_S = [0, 50, 100, 200, 300]

# --- Scale Bar ---
MICRONS_PER_PIXEL = 0.108  
SCALE_BAR_LENGTH_MICRONS = 10

# --- Mask Visualization ---
EXPORT_MASK_VISUALIZATION = True
MASK_VIZ_OVERLAY_ALPHA = 0.8
VIZ_MEMBRANE_THICKNESS_PIXELS = None


# -------------------------------------------------------------------
# --- 5. FALLBACKS & ADVANCED ---
# -------------------------------------------------------------------

# Number of parallel processes to use for GUV analysis.
# Using os.cpu_count() - 1 is a safe default.
# Set to 1 to disable parallel processing (useful for debugging).
N_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

# Fallback FPS if time metadata cannot be read from the TIFF file.
FALLBACK_FPS = 1.0