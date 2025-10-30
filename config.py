# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS CONFIGURATION ---
======================================================
This is the ONLY file you need to edit.
Set the folder, base name, and file suffixes.
"""

# --- 1. FILE PATHS ---
# Set the folder containing your experiment data
DATA_FOLDER = 'C:\\Data\\Emma'

# Set the common base name for your files (e.g., 'A1Well4_exp360V')
EXPERIMENT_BASE_NAME = 'A1Well4_exp360V'


# --- 2. CHANNEL & FILE SUFFIXES ---
# The script will build paths like:
# {DATA_FOLDER}/{PREFIX}{BASE_NAME}{SUFFIX}

# --- ROI Channel (Membrane) ---
# This channel is used to find the vesicle and draw the mask
ROI_CHANNEL_PREFIX = 'C1-'
# CSV is assumed to be named after the ROI channel.
CSV_SUFFIX = '_detected_vesicles.csv' 

# --- Dye Channel (for Measurement) ---
# This channel is used to measure the intensity uptake
DYE_CHANNEL_PREFIX = 'C3-'
# TIFs are assumed to be named after the dye channel, using a wildcard for the time point
TIF_SUFFIX = '_t*.tif'


# --- 3. ANALYSIS PARAMETERS ---
MODEL_TO_USE = '4-PARAM'  # Options: '4-PARAM' or '5-PARAM'
JUMP_SENSITIVITY = 3
FIT_DATA_PERCENTAGE = 0.99
MEMBRANE_THICKNESS_PIXELS = 15 

# Parameters for the Background Ring Mask
BG_BUFFER_PIXELS = 5     # Space between GUV outer edge and start of background ring
BG_RING_WIDTH_PIXELS = 20 # Radial width of the background ring

# Parameters for exporting a mask visualization image
EXPORT_MASK_VISUALIZATION = True # Set to True to save a debug image of the masks
MASK_VIZ_OVERLAY_ALPHA = 0.6     # Opacity of the base C1 image (0.0 to 1.0)


FIT_INITIAL_GUESS = [1.5, 0.5, 10.0, 0.5, 100.0] # [Af, A1, tau1, A2, tau2] 
# UPDATED FOR NORMALIZED DATA: [I_offset, A, tau, D]
FIT_INITIAL_GUESS_4PARAM = [0.0, 1.0, 10.0, 0.001] 

# --- 4. IMAGE EXPORT PARAMETERS ---
# List of time points (in seconds) to export as image frames
EXPORT_TIME_POINTS_S = [0, 50, 100, 200, 300]
# CORRECTED: Updated to the correct scaling factor
MICRONS_PER_PIXEL = 0.065  # e.g., 0.065 µm/pixel
SCALE_BAR_LENGTH_MICRONS = 10 # Adjusted for higher resolution
OUTPUT_IMAGE_FOLDER = 'C:\\Data\\Emma\\output_images'

# --- 5. FALLBACK PARAMETERS ---
# Used if metadata extraction fails
FALLBACK_FPS = 1.0