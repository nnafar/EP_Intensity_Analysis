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
DATA_FOLDER = 'C:/Users/matya/Documents/Thesis code/Intensity Project/Data/W3_blank'

# Set the common base name for your files (e.g., 'W3')
EXPERIMENT_BASE_NAME = 'W3'


# --- 2. CHANNEL & FILE SUFFIXES ---
# The script will build paths like:
# {DATA_FOLDER}/{PREFIX}{BASE_NAME}{SUFFIX}

# --- ROI Channel (Membrane) ---
# This channel is used to find the vesicle and draw the mask
ROI_CHANNEL_PREFIX = 'C1-'
# CSV is assumed to be named after the ROI channel
CSV_SUFFIX = '-detected_vesicles.csv' 

# --- Dye Channel (for Measurement) ---
# This channel is used to measure the intensity uptake
DYE_CHANNEL_PREFIX = 'C3-'
# TIFs are assumed to be named after the dye channel
TIF_SUFFIX = '_*.tif'


# --- 3. ANALYSIS PARAMETERS ---
JUMP_SENSITIVITY = 3
FIT_DATA_PERCENTAGE = 0.5
FIT_INITIAL_GUESS = [1000, 10]
FALLBACK_FPS = 19.15


# --- 4. FRAME EXPORT SETTINGS ---
NUM_FRAMES_TO_EXPORT = 4
MICRONS_PER_PIXEL = 0.16  # !!! <--- EDIT THIS VALUE
SCALE_BAR_LENGTH_MICRONS = 10
OUTPUT_IMAGE_FOLDER = "analysis_output_frames"