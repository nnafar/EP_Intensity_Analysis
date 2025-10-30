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
MODEL_TO_USE = '5-PARAM'  # Options: '4-PARAM' or '5-PARAM'
JUMP_SENSITIVITY = 3
FIT_DATA_PERCENTAGE = 0.5
FIT_INITIAL_GUESS = [1.0, 0.5, 10.0, 0.5, 100.0] # [Af, A1, tau1, A2, tau2]
FIT_INITIAL_GUESS_4PARAM = [1000.0, 1000.0, 10.0, 0.05] # [I_offset, A, tau, D]

# --- 4. IMAGE EXPORT PARAMETERS ---
# List of time points (in seconds) to export as image frames
EXPORT_TIME_POINTS_S = [0, 50, 100, 200, 300]
MICRONS_PER_PIXEL = 0.65  # e.g., 0.65 µm/pixel
SCALE_BAR_LENGTH_MICRONS = 20
OUTPUT_IMAGE_FOLDER = 'C:\\Data\\Emma\\output_images'

# --- 5. FALLBACK PARAMETERS ---
# Used if metadata extraction fails
FALLBACK_FPS = 1.0