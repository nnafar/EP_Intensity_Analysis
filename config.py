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
DATA_FOLDER = r"C:\Data\Emma"

# Set the base name of the experiment
# The script will look for files matching:
# [PREFIX][BASE_NAME][SUFFIX]
# Example: "C1-MyExperiment_t*.tif"
EXPERIMENT_BASE_NAME = "A1Well4_exp360V"

# Prefixes for your different channels
# C1 is assumed to be the ROI/Membrane channel (for finding GUVs)
ROI_CHANNEL_PREFIX = "C1-"
# C3 is assumed to be the Dye/Intensity channel (for measurement)
DYE_CHANNEL_PREFIX = "C3-"

# File suffixes
TIF_SUFFIX = "_t*.tif"
CSV_SUFFIX = "_detected_vesicles.csv"


# -------------------------------------------------------------------
# --- 2. ANALYSIS & MASKING PARAMETERS ---
# -------------------------------------------------------------------

# --- Membrane Detection ---
# Search window for the membrane peak, as a factor of the CSV radius.
# 0.3 = search in a window of +/- 30% of the radius estimate.
# (e.g., 60px estimate -> search from 42px to 78px)
# Make this TIGHTER (e.g., 0.2) if artifacts are close to the membrane.
# Make this WIDER (e.g., 0.5) if the CSV estimate is poor.
MEMBRANE_SEARCH_FACTOR = 0.3

# --- NEW: Membrane Width ---
# Defines the membrane region as: Detected Peak +/- this value.
# The FWHM detection (peak_widths) was too broad for narrow peaks.
# A value of 4 creates an 8-pixel wide membrane region.
MEMBRANE_FIXED_HALF_WIDTH = 3


# --- Background Mask Geometry ---
# Buffer space (in pixels) between the GUV's outer edge and the
# *start* of the background ring.
BG_BUFFER_PIXELS = 5

# Width (in pixels) of the annular ring used for background subtraction.
BG_RING_WIDTH_PIXELS = 10

# --- Jump Detection ---
# Sensitivity for detecting the fluorescence jump.
# Higher values = less sensitive (needs a bigger jump).
# Lower values = more sensitive (may pick up noise).
JUMP_SENSITIVITY = 3.0 # Standard deviations from the mean difference


# -------------------------------------------------------------------
# --- 3. FITTING & MODELING PARAMETERS ---
# -------------------------------------------------------------------

# --- Model Selection ---
# Which model to use for fitting the average curve.
# '4-PARAM': Exponential rise + linear drift (good for simple uptake)
# '5-PARAM': Double exponential rise (good for complex resealing)
MODEL_TO_USE = '4-PARAM'

# --- Data Slicing ---
# Percentage of the data (from 0.0 to 1.0) to use for fitting.
# 1.0 = use all data
# 0.5 = use first 50% of data
FIT_DATA_PERCENTAGE = 0.99

# --- Initial Guesses for 5-PARAM Model ---
# Af: Final intensity
# A1: Amplitude of fast component
# tau1: Time constant of fast component (seconds)
# A2: Amplitude of slow component
# tau2: Time constant of slow component (seconds)
FIT_INITIAL_GUESS = (
    1.0,  # Af
    0.5,  # A1
    10.0, # tau1
    0.5,  # A2
    100.0 # tau2
)

# --- Initial Guesses for 4-PARAM Model ---
# I_offset: Baseline intensity (should be ~0)
# A: Amplitude of rise
# tau: Time constant (seconds)
# D: Linear drift term
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
# Folder to save output images and plots (relative to DATA_FOLDER or absolute)
OUTPUT_IMAGE_FOLDER = os.path.join(DATA_FOLDER, "output_images")

# Time points (in seconds, relative to the pulse) to export frames for.
EXPORT_TIME_POINTS_S = [0, 50, 100, 200, 300]

# --- Scale Bar ---
# Microns per pixel. Set to 0 to disable the scale bar.
MICRONS_PER_PIXEL = 0.108 # 0.1625 

# Length of the scale bar to draw (in microns).
SCALE_BAR_LENGTH_MICRONS = 10

# --- Mask Visualization ---
# Set to True to export a visualization of the masks for the first GUV.
# (Will also export for any GUV that has detection warnings)
EXPORT_MASK_VISUALIZATION = True

# Opacity of the colored mask overlay (0.0 = transparent, 1.0 = opaque)
MASK_VIZ_OVERLAY_ALPHA = 0.6

# Set a fixed thickness (in pixels) for the RED membrane visualization ring.
# If set to None, the ring will fill the entire detected membrane region.
# A fixed value (e.g., 5) can look cleaner.
VIZ_MEMBRANE_THICKNESS_PIXELS = 11


# -------------------------------------------------------------------
# --- 5. FALLBACKS & ADVANCED ---
# -------------------------------------------------------------------

# Fallback FPS if time metadata cannot be read from the TIFF file.
FALLBACK_FPS = 1.0