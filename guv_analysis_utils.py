# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
This enhanced version includes membrane detection from radial intensity profiles,
adapted from the skeleton.py approach.
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
from typing import List, Dict, Any, Optional, Tuple

# --- Packages for Membrane Detection ---
from scipy.signal import find_peaks, peak_widths
from scipy import ndimage as nd

# -------------------------------------------------------------------
# --- 1. TIME EXTRACTION FUNCTIONS (unchanged) ---
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
# --- 2. KINETIC MODEL FUNCTIONS (unchanged) ---
# -------------------------------------------------------------------

def dyn_model_4param(t: np.ndarray, I_offset: float, A: float, tau: float, D: float) -> np.ndarray:
    """
    4-parameter Exponential Rise with Linear Drift: I(t) = I_offset + A * (1 - np.exp(-t/tau)) + D * t
    - I_offset: Baseline intensity (should be ~0 for normalized I_uptake)
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

def create_circular_mask(img_shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    """Creates a circular mask for ROI selection."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    # Center is (xc, yc) -> (X, Y)
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def create_annular_mask(img_shape: tuple[int, int], center: tuple[int, int], inner_radius: int, outer_radius: int) -> np.ndarray:
    """Creates an annular (ring) mask."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Inner and outer checks
    mask_outer = dist_from_center <= outer_radius
    mask_inner = dist_from_center >= inner_radius
    
    # Annular mask is the intersection of these two
    annular_mask = np.logical_and(mask_outer, mask_inner)
    return annular_mask

# -------------------------------------------------------------------
# --- 4. NEW: RADIAL PROFILE & MEMBRANE DETECTION ---
# -------------------------------------------------------------------

def calculate_linear_profiles(image: np.ndarray, center: Tuple[int, int], 
                              num_angles: int = 360, length_excess: float = 1.5, 
                              dr: int = 1, radius_estimate: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate linear intensity profiles radiating from the center.
    Adapted from skeleton.py linear_profiles function.
    
    Parameters:
        image: 2D image array
        center: (xc, yc) center coordinates
        num_angles: number of angular profiles to calculate
        length_excess: how far beyond radius to sample (multiplier)
        dr: step size in pixels along radius
        radius_estimate: estimated radius for determining profile length
        
    Returns:
        intensity_profiles: Array of shape (num_radial_points, num_angles)
        along_radius: Array of radial distances from center
        theta: Array of angles
    """
    xc, yc = center
    
    # Calculate the length of the linear profiles
    length = int(np.round(radius_estimate * length_excess))
    
    # Generate angles
    theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    
    # Generate radial distances
    along_radius = np.arange(0, length, dr)
    num_radial_points = len(along_radius)
    
    # Initialize intensity profiles array
    intensity_profiles = np.zeros((num_radial_points, num_angles))
    
    # For each angle
    for i, angle in enumerate(theta):
        # Calculate coordinates along this radial line
        x_coords = xc + along_radius * np.cos(angle)
        y_coords = yc + along_radius * np.sin(angle)
        
        # Ensure coordinates are within image bounds
        valid_mask = (x_coords >= 0) & (x_coords < image.shape[1]) & \
                     (y_coords >= 0) & (y_coords < image.shape[0])
        
        # Sample intensities using interpolation
        for j in range(num_radial_points):
            if valid_mask[j]:
                x_int, y_int = int(x_coords[j]), int(y_coords[j])
                intensity_profiles[j, i] = image[y_int, x_int]
            else:
                intensity_profiles[j, i] = 0  # Outside image bounds
                
    return intensity_profiles, along_radius, theta

def calculate_radial_profile(intensity_profiles: np.ndarray) -> np.ndarray:
    """
    Calculate the average radial profile by averaging over all angles.
    
    Parameters:
        intensity_profiles: Array of shape (num_radial_points, num_angles)
        
    Returns:
        radial_profile: 1D array of averaged intensities
    """
    return np.mean(intensity_profiles, axis=1)

def detect_membrane_from_profile(radial_profile: np.ndarray, radius_estimate: int) -> Tuple[int, int, List[str], bool]:
    """
    Detect membrane inner and outer borders from radial intensity profile using peak detection.
    Adapted from skeleton.py membrane_detection function.
    
    Parameters:
        radial_profile: 1D array of radial intensity profile
        radius_estimate: estimated radius from CSV (for guidance)
        
    Returns:
        index_border_in: Index of inner membrane border
        index_border_out: Index of outer membrane border  
        comments: List of quality/warning comments
        failed: True if detection failed completely
    """
    comments = []
    failed = False
    
    # Find peaks with adjusted parameters for better detection
    peaks, properties = find_peaks(
        radial_profile, 
        height=np.max(radial_profile) * 0.1,  # At least 10% of max
        distance=5, 
        prominence=np.max(radial_profile) * 0.05  # At least 5% prominence
    )
    
    if not np.any(peaks):
        index_border_in = 0
        index_border_out = 0 
        comments = ["no_membrane_peak_detected"]
        failed = True
        return index_border_in, index_border_out, comments, failed
    
    # If only one peak, use it
    if len(peaks) == 1:
        chosen_peak = 0
    else:
        # Multiple peaks - choose the best one based on several criteria
        peak_scores = []
        
        for i, peak_pos in enumerate(peaks):
            score = 0
            
            # Criterion 1: Prefer peaks closer to the expected radius (higher weight)
            distance_to_expected = abs(peak_pos - radius_estimate)
            distance_score = 1.0 / (1.0 + distance_to_expected / max(1, radius_estimate))
            score += distance_score * 3.0  # High weight for position
            
            # Criterion 2: Prefer higher peaks
            height_score = radial_profile[peak_pos] / np.max(radial_profile)
            score += height_score * 2.0  # Medium weight for height
            
            # Criterion 3: Prefer peaks with good prominence
            prominence_score = properties['prominences'][i] / np.max(properties['prominences'])
            score += prominence_score * 1.0  # Lower weight for prominence
            
            # Criterion 4: Penalize peaks that are too early (likely noise)
            if peak_pos < radius_estimate * 0.3:
                score *= 0.5
            
            # Criterion 5: Penalize peaks that are too late (likely artifacts)
            if peak_pos > radius_estimate * 2.0:
                score *= 0.3
                
            peak_scores.append(score)
        
        # Choose the peak with the highest score
        chosen_peak = np.argmax(peak_scores)
        
        # Add comment if we had to choose among many peaks
        if len(peaks) > 2:
            comments.append("multiple_peaks")
    
    # Calculate width at half maximum for the chosen peak
    try:
        width_half_max = peak_widths(radial_profile, [peaks[chosen_peak]], rel_height=0.5)
        index_border_in = int(np.round(width_half_max[2][0]))
        index_border_out = int(np.round(width_half_max[3][0]))
    except:
        # Fallback if width calculation fails
        index_border_in = max(0, peaks[chosen_peak] - 5)
        index_border_out = min(len(radial_profile) - 1, peaks[chosen_peak] + 5)
        comments.append("width_calc_failed")
    
    # Quality checks
    if index_border_out - index_border_in > radius_estimate/4:
        comments.append("wide_peak")
    
    if np.mean(radial_profile[:index_border_in]) >= 0.3 * radial_profile[peaks[chosen_peak]]:
        comments.append("signal_inside_vesicle")
    
    if len(radial_profile) > index_border_out and np.mean(radial_profile[index_border_out:]) >= 0.4 * radial_profile[peaks[chosen_peak]]:
        comments.append("signal_outside_vesicle")
    
    return index_border_in, index_border_out, comments, failed

def create_guv_masks_with_detection(roi_guide_frame: np.ndarray, center: tuple[int, int], 
                                   radius_estimate: int, bg_buffer: int, bg_width: int,
                                   num_angles: int = 360, length_excess: float = 1.5) -> \
                                   Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Creates GUV masks using membrane detection from radial intensity profiles.
    This is the ENHANCED version that actually detects where the membrane is.
    
    Parameters:
        roi_guide_frame: The membrane channel image
        center: (xc, yc) center coordinates
        radius_estimate: Initial radius estimate from CSV
        bg_buffer: Buffer space outside GUV before background ring starts
        bg_width: Width of background ring
        num_angles: Number of radial profiles for detection
        length_excess: How far to extend profiles beyond estimate
        
    Returns:
        inner_guv_mask: Mask for inner GUV area (excludes membrane)
        membrane_mask: Mask for membrane region (visualization only)
        background_ring_mask: Mask for background measurement
        detection_info: Dictionary with detection results and quality metrics
    """
    img_shape = roi_guide_frame.shape
    
    # Step 1: Calculate linear intensity profiles
    intensity_profiles, along_radius, theta = calculate_linear_profiles(
        roi_guide_frame, center, num_angles, length_excess, dr=1, 
        radius_estimate=radius_estimate
    )
    
    # Step 2: Calculate average radial profile
    radial_profile = calculate_radial_profile(intensity_profiles)
    
    # Step 3: Detect membrane from radial profile
    idx_inner, idx_outer, comments, failed = detect_membrane_from_profile(
        radial_profile, radius_estimate
    )
    
    # Store detection info
    detection_info = {
        'detected_inner_radius': along_radius[idx_inner] if not failed else 0,
        'detected_outer_radius': along_radius[idx_outer] if not failed else radius_estimate,
        'peak_position': along_radius[idx_outer] if not failed else radius_estimate,
        'detection_failed': failed,
        'comments': comments,
        'radial_profile': radial_profile,
        'along_radius': along_radius
    }
    
    # Step 4: Create masks based on detected membrane
    if failed:
        # Fallback to estimate-based masks
        print(f"  WARNING: Membrane detection failed. Using radius estimate: {radius_estimate}px")
        inner_radius = max(1, int(radius_estimate * 0.8))  # Conservative inner radius
        outer_radius = radius_estimate
    else:
        # Use detected radii
        inner_radius = max(1, int(detection_info['detected_inner_radius']))
        outer_radius = int(detection_info['detected_outer_radius'])
        
        print(f"  Detected membrane: inner={inner_radius}px, outer={outer_radius}px (estimate was {radius_estimate}px)")
    
    # 1. Inner GUV Mask (for dye uptake measurement)
    inner_guv_mask = create_circular_mask(img_shape, center, radius=inner_radius)
    
    # 2. Membrane Mask (for visualization)
    membrane_mask = create_annular_mask(img_shape, center, 
                                       inner_radius=inner_radius + 1, 
                                       outer_radius=outer_radius)
    
    # 3. Background Ring Mask
    bg_inner_r = outer_radius + bg_buffer
    bg_outer_r = outer_radius + bg_buffer + bg_width
    background_ring_mask = create_annular_mask(img_shape, center, 
                                              inner_radius=bg_inner_r, 
                                              outer_radius=bg_outer_r)
    
    return inner_guv_mask, membrane_mask, background_ring_mask, detection_info


# -------------------------------------------------------------------
# --- 5. LEGACY FUNCTION (for backward compatibility) ---
# -------------------------------------------------------------------
    
def create_guv_masks(roi_guide_frame: np.ndarray, center: tuple[int, int], guv_radius: int, 
                    membrane_thickness: int, bg_buffer: int, bg_width: int) -> \
                    tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY FUNCTION: Creates masks using fixed geometry (no detection).
    Kept for backward compatibility. Use create_guv_masks_with_detection instead.
    """
    img_shape = roi_guide_frame.shape
    
    R = int(np.round(guv_radius))
    
    # 1. Inner GUV Mask
    inner_guv_radius = max(1, R - membrane_thickness) 
    inner_guv_mask = create_circular_mask(img_shape, center, radius=inner_guv_radius)
    
    # 2. Membrane Mask
    membrane_mask = create_annular_mask(img_shape, center, 
                                       inner_radius=inner_guv_radius + 1, 
                                       outer_radius=R)

    # 3. Background Ring Mask
    bg_inner_r = R + bg_buffer
    bg_outer_r = R + bg_buffer + bg_width
    background_ring_mask = create_annular_mask(img_shape, center, 
                                              inner_radius=bg_inner_r, 
                                              outer_radius=bg_outer_r)
    
    return inner_guv_mask, membrane_mask, background_ring_mask

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
# --- 6. DATA PROCESSING & UTILITIES (unchanged) ---
# -------------------------------------------------------------------

def detect_intensity_jump(trace: np.ndarray, sensitivity: float = 3.0) -> int:
    """
    Detects a sudden jump in intensity (e.g., electroporation event) using the
    standard deviation of the difference trace.
    Returns the frame index of the jump, or -1 if none is found.
    """
    if len(trace) < 10:
        return -1
        
    # Calculate the difference between consecutive frames
    diff_trace = np.diff(trace)
    
    # Calculate the standard deviation and mean of the difference trace
    std_diff = np.std(diff_trace[5:]) 
    mean_diff = np.mean(diff_trace[5:])
    
    # Threshold for jump detection
    threshold = mean_diff + sensitivity * std_diff
    
    # Find the first point where the difference exceeds the threshold
    jump_indices = np.where(diff_trace > threshold)[0]
    
    if jump_indices.size > 0:
        return jump_indices[0] + 1
    else:
        return -1
        
def find_closest_frame(time_array: np.ndarray, target_time_abs: float) -> tuple[int, float]:
    """Finds the frame index closest to the absolute target time."""
    idx = np.abs(time_array - target_time_abs).argmin()
    return idx, time_array[idx]
    
def create_mask_visualization(base_image: np.ndarray, inner_mask: np.ndarray, 
                             membrane_mask: np.ndarray, background_mask: np.ndarray, 
                             alpha: float = 0.6) -> np.ndarray:
    """
    Overlays the inner, membrane, and background masks in color on top of the
    base C1 image for visualization.
    """
    # 1. Normalize base image to 8-bit and convert to BGR
    img_8bit = cv2.normalize(base_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    viz_image = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    # 2. Create colored overlay
    overlay = np.zeros_like(viz_image)
    overlay[inner_mask] = (128, 0, 128)      # Purple (BGR)
    overlay[membrane_mask] = (0, 0, 255)     # Red (BGR)
    overlay[background_mask] = (255, 0, 0)   # Blue (BGR)
    
    # 3. Blend images
    beta = 1.0 - alpha
    final_viz = cv2.addWeighted(viz_image, alpha, overlay, beta, 0)
    
    return final_viz

def style_image(frame: np.ndarray, time_label: str, microns_per_pixel: float, scale_bar_microns: int) -> np.ndarray:
    """
    Applies the red colormap and adds annotations.
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