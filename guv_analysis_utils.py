# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
This enhanced version includes membrane detection from radial intensity profiles.
It incorporates gradient-based and hybrid methods for more robust detection.
"""

# --- Core Packages ---
import numpy as np
import cv2
import pandas as pd
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- Packages for Time Extraction ---
import tifffile
from PIL import Image
from PIL.ExifTags import TAGS

# --- Packages for Analysis & Detection ---
from scipy.signal import find_peaks, peak_widths
from scipy import ndimage as nd
from scipy.ndimage import gaussian_filter1d, map_coordinates

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

def dyn_model_5param(t: np.ndarray, Af: float, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
    """
    5-parameter Double Exponential Rise model for Dye Uptake (Resealing):
    I(t) = Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)
    - Af: Final normalized intensity (steady state)
    - A1: Amplitude of the fast component
    - tau1: Fast time constant
    - A2: Amplitude of the slow component
    - tau2: Fast time constant
    """
    return Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)

# -------------------------------------------------------------------
# --- 3. IMAGE PROCESSING & MASKING FUNCTIONS (Core Utilities) ---
# -------------------------------------------------------------------

def _convert_to_8bit_gray(image: np.ndarray) -> np.ndarray:
    """
    Converts a 16-bit or 3-channel image to 8-bit grayscale for visualization.
    """
    if image.ndim == 3:
        image_gray = image[:,:,0]
    else:
        image_gray = image
    
    if image_gray.dtype != np.uint8:
         # Normalize from whatever bit-depth (e.g., 16-bit) to 8-bit
        return cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image_gray

def tifflist_to_numpy(file_paths: list) -> np.ndarray:
    """Loads a list of TIFF files into a 3D numpy array (stack)."""
    if not file_paths:
        return np.array([])
    # Load first image to get dimensions and data type
    # Ensure grayscale loading
    first_frame = cv2.imread(file_paths[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise FileNotFoundError(f"Could not load first frame: {file_paths[0]}")
    
    # Pre-allocate 3D array
    stack = np.empty((len(file_paths), first_frame.shape[0], first_frame.shape[1]), dtype=first_frame.dtype)
    stack[0] = first_frame
    
    # Load remaining images
    for i in range(1, len(file_paths)):
        # Ensure grayscale loading
        frame = cv2.imread(file_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            stack[i] = frame
        else:
            # Warning for missing/corrupt files
            print(f"  - Warning: Could not load file: {file_paths[i]}. Frame {i} will be zeros.")
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

def refine_guv_center(image: np.ndarray, center_guess: Tuple[int, int], radius_estimate: int, 
                      search_box_factor: float = 1.0) -> Tuple[int, int]:
    """
    Refines the GUV center coordinates by finding the "center of mass" (centroid)
    of the *membrane ring* using edge detection.
    """
    xc, yc = center_guess
    
    # Define a search box around the guess
    box_half_width = int(max(15, radius_estimate * search_box_factor))
    x_start = max(0, xc - box_half_width)
    x_end = min(image.shape[1], xc + box_half_width)
    y_start = max(0, yc - box_half_width)
    y_end = min(image.shape[0], yc + box_half_width)
    
    if x_start >= x_end or y_start >= y_end:
        return center_guess
        
    roi = image[y_start:y_end, x_start:x_end]
    
    # Handle cases where cv2.imread loads a grayscale TIFF as 3-channel
    if roi.ndim == 3:
        # Assume all channels are identical and take the first one
        roi = roi[:, :, 0]
    
    # --- Use Laplacian edge detection to find the ring ---
    # This is more robust than simple thresholding, which gets pulled
    # by the bright interior.

    # 1. Blur to reduce noise before edge detection
    if roi.dtype == np.uint16:
        # Need to work with a float type for Laplacian, then scale
        roi_float = roi.astype(np.float32)
        roi_blurred = cv2.GaussianBlur(roi_float, (3, 3), 0)
        # Using 16-bit signed output to capture positive/negative slopes
        
        laplacian = cv2.Laplacian(roi_blurred, cv2.CV_32F, ksize=3)

    else:
        # Assume 8-bit or similar
        roi_blurred = cv2.GaussianBlur(roi.astype(np.float32), (3, 3), 0)
        
        laplacian = cv2.Laplacian(roi_blurred, cv2.CV_32F, ksize=3)

    # 2. Convert back to absolute 8-bit scale
    # This image now highlights *only* the edges (the ring)
    edge_image_8u = cv2.convertScaleAbs(laplacian)

    # 3. Normalize for thresholding (Otsu works best on 8-bit)
    if np.max(edge_image_8u) > 0:
         roi_8bit = cv2.normalize(edge_image_8u, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
         roi_8bit = edge_image_8u # all zeros

    # 4. Threshold the *edge image* to get a mask of the ring
    try:
        thresh_val, thresh_mask = cv2.threshold(
            roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except cv2.error:
         # This can happen if roi_8bit is all one value (e.g., all black)
         thresh_mask = np.zeros_like(roi_8bit) # Fallback to an empty mask
    
    # 5. Find the center of mass of the *ring mask*
    M = cv2.moments(thresh_mask)
    if M["m00"] == 0:
        # Fallback: if no moments, find max brightness pixel in the *edge image*
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(roi_8bit)
        cX, cY = max_loc
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
    xc_new = x_start + cX
    yc_new = y_start + cY
    
    shift = np.sqrt((xc_new - xc)**2 + (yc_new - yc)**2)
    
    # Keep the shift check, it's still good practice
    if shift > radius_estimate * 0.7:
        print(f"  - Warning: Center refinement shifted by {shift:.1f}px. Reverting to original center.")
        return center_guess
        
    return (xc_new, yc_new)

# -------------------------------------------------------------------
# --- 4. RADIAL PROFILE & MEMBRANE DETECTION ---
# -------------------------------------------------------------------

def linear_profiles(image: np.ndarray, 
                    center: Tuple[int, int], 
                    num_angles: int, 
                    length_excess: float, 
                    radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate linear intensity profiles radiating from the center using
    scipy.ndimage.map_coordinates for sub-pixel interpolation.
    
    (Adapted from skeleton.py)
    """
    xc, yc = center
    
    # --- For 3-channel image ---
    if image.ndim == 3:
        image = image[:,:,0]
    
    length = int(radius * length_excess)
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    # 1D array of radii
    along_radius = np.arange(0, length, 1)
    
    # 2D arrays of x and y coordinates
    # (num_angles, length)
    x_coords = xc + along_radius[np.newaxis, :] * np.cos(theta[:, np.newaxis])
    y_coords = yc + along_radius[np.newaxis, :] * np.sin(theta[:, np.newaxis])

    # map_coordinates requires coordinates as a (2, N) array
    # We want a (num_angles, length) output, so we flatten and stack
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)

    # Perform interpolation
    # 'order=1' is linear interpolation
    profiles_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
    
    # Reshape back to (num_angles, length) and transpose
    # to (length, num_angles) to match old function's output shape
    profiles = profiles_flat.reshape(num_angles, length).T
    
    return profiles, along_radius, theta

def average_radial_profile(profiles: np.ndarray) -> np.ndarray:
    """
    Calculate the average radial profile using the median over all angles.
    (Adapted from skeleton.py)
    """
    # Use median to be robust against outliers (e.g., lipid clusters)
    return np.median(profiles, axis=1)

def membrane_search(profile: np.ndarray, 
                    radii: np.ndarray, 
                    expected_radius: int, 
                    search_factor: float,
                    membrane_half_width: int,
                    peak_min_dist: int,      
                    peak_min_prom: float     
                    ) -> Tuple[int, int, int, List[str], bool]:
    """
    Detects membrane location by finding the highest peak in a search window
    defined by expected_radius +/- (expected_radius * search_factor).
    
    (Adapted from skeleton.py + previous fixes)
    
    Returns:
        peak_pos: Radius (in pixels) of the detected peak.
        inner_b: Radius (in pixels) of the inner membrane border.
        outer_b: Radius (in pixels) of the outer membrane border.
        comments: List of quality/warning comments.
        failed: True if detection failed.
    """
    comments = []
    failed = False
    
    # --- Smooth the profile to be robust to noise/clusters ---
    # sigma=1 applies a light smoothing to average out high-frequency noise
    profile_smooth = gaussian_filter1d(profile, sigma=1)
    
    # --- 1. Define search window around hint ---
    data_to_search = profile_smooth # Use the smoothed profile for detection
    
    # Find array indices corresponding to the radius window
    min_radius = max(0, expected_radius * (1.0 - search_factor))
    max_radius = expected_radius * (1.0 + search_factor)
    
    min_idx = np.searchsorted(radii, min_radius, side='left')
    max_idx = np.searchsorted(radii, max_radius, side='right')
    
    if min_idx < max_idx: # Ensure window is valid
        # Create a *mask* to nullify data outside this window
        search_mask = np.zeros_like(profile_smooth)
        search_mask[min_idx:max_idx] = 1.0
        data_to_search = profile_smooth * search_mask
    else:
        comments.append("invalid_search_window")
        # data_to_search is already profile_smooth
    
    # --- 2. Find peaks in the masked, smoothed data ---
    try:
        # Use parameterized peak finding
        max_prof_val = np.max(profile)  # Base prominence on original max
        peaks, props = find_peaks(
            data_to_search,             # Uses the smoothed data
            height=max_prof_val * 0.1,  # Height relative to *original* max
            distance=peak_min_dist,     
            prominence=max_prof_val * peak_min_prom 
        )

    except Exception as e:
        comments.append(f"find_peaks_failed: {e}")
        peaks = np.array([]) # Ensure peaks is an empty array

    if not np.any(peaks):
        comments.append("no_peak_in_window")
        failed = True
        # Fallback to using the estimate
        peak_idx = np.searchsorted(radii, expected_radius) # Find index closest to hint
        peak_idx = min(peak_idx, len(radii) - 1)           # Clamp to bounds
    else:
        # Find the *highest* peak within the allowed window
        heights = props['peak_heights']
        chosen_peak_idx_in_peaks_array = np.argmax(heights)
        peak_idx = peaks[chosen_peak_idx_in_peaks_array]

    # --- 3. Calculate width ---
    # Use the fixed half-width from config
    try:
        inner_idx = max(0, peak_idx - membrane_half_width)
        outer_idx = min(len(profile) - 1, peak_idx + membrane_half_width)
        
        if inner_idx >= outer_idx:
            raise ValueError("Inner border >= outer border (width is zero or negative)")
            
    except Exception as e:
        comments.append(f"width_calc_failed: {e}")
        # Fallback: use a default width (e.g., 5 pixels) around the peak
        inner_idx = max(0, peak_idx - 5)
        outer_idx = min(len(profile) - 1, peak_idx + 5)
        
    # --- 4. Convert indices back to pixel radii ---
    # Ensure indices are within the bounds of the radii array
    peak_idx = min(peak_idx, len(radii) - 1)
    inner_idx = min(inner_idx, len(radii) - 1)
    outer_idx = min(outer_idx, len(radii) - 1)
    
    peak_pos = int(radii[peak_idx])
    inner_b = int(radii[inner_idx])
    outer_b = int(radii[outer_idx])
    
    # Final sanity check
    if inner_b == 0 and outer_b == 0:
        comments.append("zero_radius_detected")
        failed = True
        inner_b = max(1, int(expected_radius * 0.8))
        outer_b = expected_radius
        peak_pos = expected_radius

    return peak_pos, inner_b, outer_b, comments, failed


def create_guv_masks_with_detection(roi_guide_frame: np.ndarray, 
                                   center: Tuple[int, int],
                                   radius_estimate: int, 
                                   bg_buffer: int, 
                                   bg_width: int,
                                   search_factor: float,
                                   membrane_half_width: int,
                                   peak_min_dist: int,      
                                   peak_min_prom: float,     
                                   num_angles: int = 360, 
                                   length_excess: float = 1.5,
                                   viz_thickness: Optional[int] = None) -> \
                                   Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Creates GUV masks using the detection logic from skeleton.py, constrained
    by a search window based on the radius_estimate.
    
    (This function now wraps the new skeleton.py-based methods)
    """
    img_shape = roi_guide_frame.shape
    
    # --- 1. Call functions ---
    
    # Step 1: Calculate linear intensity profiles
    profiles, along_radius, theta = linear_profiles(
        roi_guide_frame, 
        center, 
        num_angles=num_angles, 
        length_excess=length_excess,
        radius=radius_estimate 
    )
    
    # Step 2: Calculate average radial profile
    radial_profile = average_radial_profile(profiles)
    
    # Step 3: Detect membrane
    peak_pos, inner_b, outer_b, comments, failed = membrane_search(
        radial_profile, 
        along_radius,
        expected_radius=radius_estimate,
        search_factor=search_factor,
        membrane_half_width=membrane_half_width,
        peak_min_dist=peak_min_dist,          
        peak_min_prom=peak_min_prom           
    )

    # --- 2. Create Masks ---
    
    # 1. Inner GUV Mask (for dye uptake measurement)
    inner_guv_mask = create_circular_mask(img_shape, center, radius=inner_b)
    
    # 2. Membrane Mask (for visualization)
    if viz_thickness is not None and viz_thickness > 0:
        # Use a fixed thickness for visualization, anchored to the *inner* radius
        viz_mem_inner_r = inner_b + 1
        viz_mem_outer_r = inner_b + viz_thickness
        membrane_mask = create_annular_mask(img_shape, center, 
                                           inner_radius=viz_mem_inner_r, 
                                           outer_radius=viz_mem_outer_r)
    else:
        # Original behavior: fill the whole detected membrane area
        membrane_mask = create_annular_mask(img_shape, center, 
                                           inner_radius=inner_b + 1, 
                                           outer_radius=outer_b)
    
    # 3. Background Ring Mask
    bg_inner_r = outer_b + bg_buffer
    bg_outer_r = outer_b + bg_buffer + bg_width
    background_ring_mask = create_annular_mask(img_shape, center, 
                                              inner_radius=bg_inner_r, 
                                              outer_radius=bg_outer_r)

    # --- 3. Populate detection_info dictionary for run_analysis.py ---
    detection_info = {
        'detected_inner_radius': inner_b,
        'detected_outer_radius': outer_b,
        'bg_inner_radius': bg_inner_r, 
        'bg_outer_radius': bg_outer_r, 
        'peak_position': peak_pos,
        'detection_failed': failed,
        'comments': comments,
        'radial_profile': radial_profile, # Save the *original* noisy profile for plotting
        'along_radius': along_radius,
        'method': 'skeleton_search',
        'search_factor_used': search_factor
    }

    if not failed:
         print(f"  Detected membrane: inner={inner_b}px, outer={outer_b}px (estimate was {radius_estimate}px)")
    else:
         print(f"  WARNING: Membrane detection failed. Review comments: {comments}")

    
    return inner_guv_mask, membrane_mask, background_ring_mask, detection_info


# -------------------------------------------------------------------
# --- 5. LEGACY & TRACE FUNCTIONS ---
# -------------------------------------------------------------------
    
def create_guv_masks(roi_guide_frame: np.ndarray, center: tuple[int, int], guv_radius: int, 
                    membrane_thickness: int, bg_buffer: int, bg_width: int) -> \
                    tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    LEGACY FUNCTION: Creates masks using fixed geometry (no detection).
    """
    img_shape = roi_guide_frame.shape
    R = int(np.round(guv_radius))
    inner_guv_radius = max(1, R - membrane_thickness) 
    inner_guv_mask = create_circular_mask(img_shape, center, radius=inner_guv_radius)
    membrane_mask = create_annular_mask(img_shape, center, 
                                       inner_radius=inner_guv_radius + 1, 
                                       outer_radius=R)
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
    
    if num_pixels == 0:
        return trace
        
    for i in range(im_stack.shape[0]):
        masked_frame = im_stack[i][mask]
        trace[i] = np.sum(masked_frame) / num_pixels
        
    return trace

# -------------------------------------------------------------------
# --- 6. DATA PROCESSING & UTILITIES ---
# -------------------------------------------------------------------

def detect_intensity_jump(trace: np.ndarray, sensitivity: float = 3.0) -> int:
    """
    Detects a sudden jump in intensity (e.g., electroporation event).
    """
    if len(trace) < 10:
        return -1
        
    diff_trace = np.diff(trace)
    std_diff = np.std(diff_trace[5:]) 
    mean_diff = np.mean(diff_trace[5:])
    
    if std_diff == 0:
        return -1
        
    threshold = mean_diff + sensitivity * std_diff
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
    Overlays the inner, membrane, and background masks in color.
    """
    img_8bit = _convert_to_8bit_gray(base_image)
    
    viz_image = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(viz_image)
    overlay[inner_mask] = (201, 87, 188)      # Magenta (BGR)
    overlay[membrane_mask] = (82, 0, 249)     # Bright Red (BGR)
    overlay[background_mask] = (114, 48, 19)  # Purple (BGR)
    beta = 1.0 - alpha
    final_viz = cv2.addWeighted(viz_image, alpha, overlay, beta, 0)
    return final_viz

def style_image(frame: np.ndarray, time_label: str, microns_per_pixel: float, scale_bar_microns: int) -> np.ndarray:
    """
    Applies the red colormap and adds annotations.
    """
    img_8bit = _convert_to_8bit_gray(frame)

    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    img_color[:, :, 0] = 0
    img_color[:, :, 1] = 0
    img_color[:, :, 2] = img_8bit
    
    cv2.putText(
        img_color,
        time_label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    if scale_bar_microns > 0 and microns_per_pixel > 0:
        bar_length_pixels = int(scale_bar_microns / microns_per_pixel)
        h, w, _ = img_color.shape
        p1 = (w - 30 - bar_length_pixels, h - 30)
        p2 = (w - 30, h - 30)
        cv2.line(img_color, p1, p2, (255, 255, 255), 5)
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