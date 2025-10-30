# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
This file contains all the helper functions for the GUV analysis.
You should not need to edit this file.
All user-configurable settings are in 'config.py'.
"""

# --- Core Packages ---
import numpy as np
import cv2
import pandas as pd
import os

# --- Packages for Time Extraction ---
import re
from datetime import datetime
import tifffile
from PIL import Image
from PIL.ExifTags import TAGS
from typing import List, Dict, Any, Optional

# -------------------------------------------------------------------
# --- 1. TIME EXTRACTION FUNCTIONS ---
# -------------------------------------------------------------------

def extract_timestamps_from_metadata(file_paths: List[str]) -> tuple[Optional[np.ndarray], Optional[float]]:
    """
    Attempts to extract timestamps and frame interval from the ImageJ metadata
    in the first TIFF file.
    
    Returns: (time_array, frame_interval) or (None, None)
    """
    if not file_paths:
        return None, None
        
    try:
        # Read the first file's metadata
        with tifffile.TiffFile(file_paths[0]) as tif:
            # Check for ImageDescription which contains ImageJ metadata
            description_tag = tif.pages[0].tags.get('ImageDescription')
            if description_tag is None:
                return None, None
                
            description = description_tag.value
            
            # Use regex to find the frame interval (finterval=X)
            match = re.search(r'finterval=([0-9.]+)', description)
            if match:
                frame_interval = float(match.group(1))
                num_frames = len(file_paths)
                
                # Create the time array
                time_array = np.arange(num_frames) * frame_interval
                return time_array, frame_interval
                
    except Exception:
        # If tifffile fails or metadata is missing, return None to trigger fallback
        pass

    return None, None

def create_manual_timestamps(num_frames: int, fallback_fps: float = 1.0) -> tuple[np.ndarray, float]:
    """Creates a simple time array based on a fallback FPS. Returns (time_array, frame_interval)."""
    frame_interval = 1.0 / fallback_fps
    return np.arange(num_frames) * frame_interval, frame_interval


# -------------------------------------------------------------------
# --- 2. KINETIC MODEL FUNCTIONS ---
# -------------------------------------------------------------------

def dyn_model_4param(t: np.ndarray, I_offset: float, A: float, tau: float, D: float) -> np.ndarray:
    """
    4-parameter Exponential Rise with Linear Drift: I(t) = I_offset + A * (1 - np.exp(-t/tau)) + D * t
    - I_offset: Baseline intensity
    - A: Amplitude of the rise
    - tau: Time constant
    - D: Linear drift term
    """
    return I_offset + A * (1 - np.exp(-t / tau)) + D * t

def dyn_model(t: np.ndarray, Af: float, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
    """
    5-parameter Double Exponential Rise model for Dye Uptake (Resealing):
    I(t) = Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)
    - Af: Final normalized intensity (steady state)
    - A1: Amplitude of the fast component
    - tau1: Fast time constant
    - A2: Amplitude of the slow component
    - tau2: Slow time constant
    """
    return Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)

# -------------------------------------------------------------------
# --- 3. IMAGE PROCESSING & MASKING FUNCTIONS ---
# -------------------------------------------------------------------

def tifflist_to_numpy(file_paths: list) -> np.ndarray:
    """Loads a list of TIFF files into a 3D numpy array (stack)."""
    if not file_paths:
        return np.array([])
    # Load first image to get dimensions and data type
    first_frame = cv2.imread(file_paths[0], cv2.IMREAD_ANYDEPTH)
    if first_frame is None:
        raise FileNotFoundError(f"Could not load first frame: {file_paths[0]}")
    
    # Pre-allocate 3D array
    stack = np.empty((len(file_paths), first_frame.shape[0], first_frame.shape[1]), dtype=first_frame.dtype)
    stack[0] = first_frame
    
    # Load remaining images
    for i in range(1, len(file_paths)):
        frame = cv2.imread(file_paths[i], cv2.IMREAD_ANYDEPTH)
        if frame is not None:
            stack[i] = frame
        else:
            # Handle missing or corrupted files by filling with zeros
            stack[i] = np.zeros_like(first_frame)
            
    return stack

def create_circular_mask(img_shape: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    """Creates a circular mask for ROI selection."""
    h, w = img_shape.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def get_intensity_trace(im_stack: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Calculates the average intensity within the mask for every frame."""
    trace = np.zeros(im_stack.shape[0])
    num_pixels = np.sum(mask)
    
    for i in range(im_stack.shape[0]):
        # Apply the mask to the frame
        masked_frame = im_stack[i][mask]
        # Calculate the average intensity
        trace[i] = np.sum(masked_frame) / num_pixels if num_pixels > 0 else 0
        
    return trace

# -------------------------------------------------------------------
# --- 4. DATA PROCESSING & UTILITIES ---
# -------------------------------------------------------------------

def detect_intensity_jump(trace: np.ndarray, sensitivity: float = 3.0, baseline_frames: int = 5) -> int:
    """
    Detects a sudden jump in intensity (e.g., electroporation event) using the
    standard deviation of the difference trace.
    
    NOTE: The standard deviation is calculated using the difference trace *after*
    the first 'baseline_frames' to ensure a stable baseline noise estimate.
    Jump detection is most reliable for events occurring after this skipped region.

    Returns the frame index of the jump, or -1 if none is found.
    """
    if len(trace) < 10:
        return -1
        
    # Calculate the difference between consecutive frames
    diff_trace = np.diff(trace)
    
    # Calculate the standard deviation and mean of the difference trace (excluding the initial frames)
    # Use the configurable 'baseline_frames'
    std_diff = np.std(diff_trace[baseline_frames:]) 
    mean_diff = np.mean(diff_trace[baseline_frames:])
    
    # Threshold for jump detection (e.g., 3 * std_dev above the mean)
    threshold = mean_diff + sensitivity * std_diff
    
    # Find the first point where the difference exceeds the threshold
    jump_indices = np.where(diff_trace > threshold)[0]
    
    if jump_indices.size > 0:
        # Return the index of the *frame* after the difference occurred
        return jump_indices[0] + 1
    else:
        return -1
        
def find_closest_frame(time_array: np.ndarray, target_time_abs: float) -> tuple[int, float]:
    """Finds the frame index closest to the absolute target time."""
    idx = np.abs(time_array - target_time_abs).argmin()
    return idx, time_array[idx]

def style_image(frame: np.ndarray, time_label: str, microns_per_pixel: float, scale_bar_microns: int) -> np.ndarray:
    """
    Applies the red colormap and adds annotations. (Content omitted for brevity but assumed unchanged)
    """
    # 1. Normalize to 8-bit (0-255) for display
    img_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 2. Apply red colormap
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    img_color[:, :, 0] = 0  # Zero out Blue channel
    img_color[:, :, 1] = 0  # Zero out Green channel
    img_color[:, :, 2] = img_8bit # Set Red channel to intensity
    
    # 3. Add time stamp
    cv2.putText(
        img_color,
        time_label,
        (20, 40),  # Position (from top-left)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,  # Font scale
        (255, 255, 255),  # Color (white)
        2,  # Thickness
        cv2.LINE_AA
    )
    
    # 4. Add scale bar (if configured)
    if scale_bar_microns > 0 and microns_per_pixel > 0:
        bar_length_pixels = int(scale_bar_microns / microns_per_pixel)
        h, w, _ = img_color.shape
        
        # Position the scale bar: 30 pixels from bottom/right edge
        p1 = (w - 30 - bar_length_pixels, h - 30)
        p2 = (w - 30, h - 30)
        
        cv2.line(img_color, p1, p2, (255, 255, 255), 5) # Draw white line
        
        # Add label
        cv2.putText(
            img_color,
            f"{scale_bar_microns} µm",
            (p1[0], h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
    return img_color