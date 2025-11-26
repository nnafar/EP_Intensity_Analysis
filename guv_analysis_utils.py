# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
Helper functions for:
1. Timestamp extraction
2. Image processing (masking, center refinement)
3. Signal processing (Jump detection, Kinetic models)
4. Visualization
"""

# --- Core Packages ---
import numpy as np
import cv2
import pandas as pd
import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- Packages for Time Extraction ---
import tifffile
from PIL import Image
from PIL.ExifTags import TAGS

# --- Packages for Analysis & Detection ---
from scipy.signal import find_peaks
from scipy import ndimage as nd
from scipy.ndimage import gaussian_filter1d, map_coordinates
import matplotlib.pyplot as plt

# --- Local Imports ---
import config as cfg


# -------------------------------------------------------------------
# --- 0. LOGGING SETUP ---
# -------------------------------------------------------------------

def setup_logging(output_folder: str, experiment_name: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger that writes to both file and console."""
    log_filename = os.path.join(
        output_folder, 
        f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)-8s - %(funcName)-20s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logger

# -------------------------------------------------------------------
# --- 1. TIME EXTRACTION FUNCTIONS ---
# -------------------------------------------------------------------

def extract_timestamps_from_metadata(file_paths: List[str]) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Attempts to read 'finterval' from ImageJ TIFF tags."""
    if not file_paths: return None, None
    try:
        with tifffile.TiffFile(file_paths[0]) as tif:
            description_tag = tif.pages[0].tags.get('ImageDescription')
            if description_tag is None: return None, None
            description = description_tag.value
            # Regex to look for "finterval=0.5" etc.
            match = re.search(r'finterval=([0-9.]+)', description)
            if match:
                frame_interval = float(match.group(1))
                num_frames = len(file_paths)
                time_array = np.arange(num_frames) * frame_interval
                return time_array, frame_interval
    except Exception: pass
    return None, None

def create_manual_timestamps(num_frames: int, fallback_fps: float = 1.0) -> tuple[np.ndarray, float]:
    """Generates timestamps based on a fixed FPS if metadata fails."""
    frame_interval = 1.0 / fallback_fps
    return np.arange(num_frames) * frame_interval, frame_interval


# -------------------------------------------------------------------
# --- 2. KINETIC MODEL FUNCTIONS ---
# -------------------------------------------------------------------

def dyn_model_4param(t: np.ndarray, I_offset: float, A: float, tau: float, D: float) -> np.ndarray:
    """Standard uptake model: Exponential rise + Linear Drift."""
    return I_offset + A * (1 - np.exp(-t / tau)) + D * t

def dyn_model_5param(t: np.ndarray, Af: float, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
    """Double exponential model: Fast and Slow uptake components."""
    return Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)

# -------------------------------------------------------------------
# --- 3. IMAGE PROCESSING & MASKING FUNCTIONS ---
# -------------------------------------------------------------------

def _convert_to_8bit_gray(image: np.ndarray) -> np.ndarray:
    """Helper to ensure image is 8-bit for OpenCV drawing functions."""
    if image.ndim == 3: image_gray = image[:,:,0]
    else: image_gray = image
    if image_gray.dtype != np.uint8:
        return cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image_gray

def create_circular_mask(img_shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    """Creates a boolean mask for a filled circle."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return dist_from_center <= radius

def create_annular_mask(img_shape: tuple[int, int], center: tuple[int, int], inner_radius: int, outer_radius: int) -> np.ndarray:
    """Creates a boolean mask for a ring (annulus)."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return np.logical_and(dist_from_center <= outer_radius, dist_from_center >= inner_radius)

def refine_guv_center(image: np.ndarray, center_guess: Tuple[int, int], radius_estimate: int, 
                      search_box_factor: float = 1.0) -> Tuple[int, int]:
    """
    Refines the GUV center coordinates by running Edge Detection + Otsu Thresholding
    within a local ROI around the original CSV coordinate.
    """
    xc, yc = center_guess
    box_half_width = int(max(15, radius_estimate * search_box_factor))
    x_start = max(0, xc - box_half_width)
    x_end = min(image.shape[1], xc + box_half_width)
    y_start = max(0, yc - box_half_width)
    y_end = min(image.shape[0], yc + box_half_width)
    
    if x_start >= x_end or y_start >= y_end: return center_guess
    roi = image[y_start:y_end, x_start:x_end]
    if roi.ndim == 3: roi = roi[:, :, 0]
    
    # Smooth and detect edges
    roi_blurred = cv2.GaussianBlur(roi.astype(np.float32), (3, 3), 0)
    laplacian = cv2.Laplacian(roi_blurred, cv2.CV_32F, ksize=3)
    edge_image_8u = cv2.convertScaleAbs(laplacian)

    try:
        thresh_val, thresh_mask = cv2.threshold(edge_image_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except cv2.error:
         thresh_mask = np.zeros_like(edge_image_8u)
    
    # Calculate centroid of the edges
    M = cv2.moments(thresh_mask)
    if M["m00"] == 0:
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(edge_image_8u)
        cX, cY = max_loc
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
    xc_new, yc_new = x_start + cX, y_start + cY
    
    # Sanity check: If shift is too large (> 70% of radius), reject it.
    shift = np.sqrt((xc_new - xc)**2 + (yc_new - yc)**2)
    if shift > radius_estimate * 0.7: return center_guess
    return (xc_new, yc_new)

# -------------------------------------------------------------------
# --- 4. RADIAL PROFILE & MEMBRANE DETECTION ---
# -------------------------------------------------------------------

def linear_profiles(image: np.ndarray, center: Tuple[int, int], num_angles: int, 
                    length_excess: float, radius: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts intensity profiles radiating from the center at multiple angles."""
    xc, yc = center
    if image.ndim == 3: image = image[:,:,0]
    length = int(radius * length_excess)
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    along_radius = np.arange(0, length, 1)
    
    x_coords = xc + along_radius[np.newaxis, :] * np.cos(theta[:, np.newaxis])
    y_coords = yc + along_radius[np.newaxis, :] * np.sin(theta[:, np.newaxis])
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)
    profiles_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
    return profiles_flat.reshape(num_angles, length).T, along_radius, theta

def average_radial_profile(profiles: np.ndarray) -> np.ndarray:
    """Computes the median profile to remove artifacts (other GUVs, dirt)."""
    return np.median(profiles, axis=1)

def membrane_search(profile: np.ndarray, radii: np.ndarray, expected_radius: int, 
                    search_factor: float, membrane_half_width: int, peak_min_dist: int,      
                    peak_min_prom: float, detect_bright: bool = True) -> Tuple[int, int, int, List[str], bool]:
    """
    Finds the membrane location in the radial profile using Peak (or Valley) detection.
    """
    comments = []
    failed = False
    profile_smooth = gaussian_filter1d(profile, sigma=1)
    
    # If searching for dark membrane (unlabeled), invert the profile so we can use find_peaks
    search_profile = profile_smooth if detect_bright else np.max(profile_smooth) - profile_smooth

    # Define search window based on config
    min_radius = max(0, expected_radius * (1.0 - search_factor))
    max_radius = expected_radius * (1.0 + search_factor)
    min_idx = np.searchsorted(radii, min_radius, side='left')
    max_idx = np.searchsorted(radii, max_radius, side='right')
    
    if min_idx < max_idx:
        search_mask = np.zeros_like(search_profile)
        search_mask[min_idx:max_idx] = 1.0
        data_to_search = search_profile * search_mask
    else:
        comments.append("invalid_search_window")
        data_to_search = search_profile # Fallback

    try:
        max_search_val = np.max(search_profile)
        if max_search_val == 0: max_search_val = 1.0
        peaks, props = find_peaks(data_to_search, height=max_search_val * 0.1, distance=peak_min_dist, prominence=max_search_val * peak_min_prom)
    except Exception as e:
        comments.append(f"find_peaks_failed: {e}")
        peaks = np.array([])

    if not np.any(peaks):
        comments.append("no_peak_in_window")
        failed = True
        # Fallback: use the expected radius from CSV
        peak_idx = min(np.searchsorted(radii, expected_radius), len(radii) - 1)
    else:
        # Select the highest peak in the window
        peak_idx = peaks[np.argmax(props['peak_heights'])]

    inner_idx = max(0, peak_idx - membrane_half_width)
    outer_idx = min(len(profile) - 1, peak_idx + membrane_half_width)
    
    # Map indices back to physical radius units
    peak_pos = int(radii[min(peak_idx, len(radii) - 1)])
    inner_b = int(radii[min(inner_idx, len(radii) - 1)])
    outer_b = int(radii[min(outer_idx, len(radii) - 1)])
    
    if inner_b == 0 and outer_b == 0:
        comments.append("zero_radius_detected")
        failed = True
        inner_b = max(1, int(expected_radius * 0.8))
        outer_b = expected_radius
        peak_pos = expected_radius

    return peak_pos, inner_b, outer_b, comments, failed

def create_guv_masks_with_detection(roi_guide_frame: np.ndarray, center: Tuple[int, int], radius_estimate: int, 
                                   bg_buffer: int, bg_width: int, search_factor: float, membrane_half_width: int,
                                   peak_min_dist: int, peak_min_prom: float, detect_bright: bool = True,
                                   num_angles: int = 360, length_excess: float = 1.5, viz_thickness: Optional[int] = None):
    """
    Orchestrates the creation of Inner, Membrane, and Background masks based on radial profile analysis.
    """
    
    profiles, along_radius, theta = linear_profiles(roi_guide_frame, center, num_angles, length_excess, radius_estimate)
    radial_profile = average_radial_profile(profiles)
    
    peak_pos, inner_b, outer_b, comments, failed = membrane_search(
        radial_profile, along_radius, radius_estimate, search_factor,
        membrane_half_width, peak_min_dist, peak_min_prom, detect_bright
    )

    inner_guv_mask = create_circular_mask(roi_guide_frame.shape, center, radius=inner_b)
    
    if viz_thickness and viz_thickness > 0:
        membrane_mask = create_annular_mask(roi_guide_frame.shape, center, inner_b + 1, inner_b + viz_thickness)
    else:
        membrane_mask = create_annular_mask(roi_guide_frame.shape, center, inner_b + 1, outer_b)
    
    bg_inner = outer_b + bg_buffer
    bg_outer = bg_inner + bg_width
    background_ring_mask = create_annular_mask(roi_guide_frame.shape, center, bg_inner, bg_outer)

    detection_info = {
        'detected_inner_radius': inner_b, 'detected_outer_radius': outer_b,
        'bg_inner_radius': bg_inner, 'bg_outer_radius': bg_outer, 
        'peak_position': peak_pos, 'detection_failed': failed, 'comments': comments,
        'radial_profile': radial_profile, 'along_radius': along_radius
    }
    return inner_guv_mask, membrane_mask, background_ring_mask, detection_info


# -------------------------------------------------------------------
# --- 5. PARALLEL WORKER FUNCTION ---
# -------------------------------------------------------------------

def process_single_guv(guv_id: str, center_orig: tuple, radius_csv_raw: int, 
                       is_first_guv: bool, 
                       roi_frame: np.ndarray, dye_files: list) -> tuple:
    """
    Runs the full analysis pipeline for a single GUV.
    This function is designed to be pickle-able for multiprocessing.
    """
    guv_radius_estimate = int(np.round(radius_csv_raw / 2.0))
    (xc_refined, yc_refined) = refine_guv_center(roi_frame, center_orig, guv_radius_estimate, 0.8)
    
    # Determine detection mode from config
    detect_bright = (getattr(cfg, 'MEMBRANE_DETECTION_MODE', 'LABELED').upper() != 'UNLABELED')
    
    # Generate masks
    inner_mask, membrane_mask, background_mask, detection_info = create_guv_masks_with_detection(
        roi_frame, (xc_refined, yc_refined), guv_radius_estimate,
        cfg.BG_BUFFER_PIXELS, cfg.BG_RING_WIDTH_PIXELS, cfg.MEMBRANE_SEARCH_FACTOR,
        cfg.MEMBRANE_FIXED_HALF_WIDTH, cfg.PEAK_FIND_MIN_DISTANCE, cfg.PEAK_FIND_MIN_PROMINENCE,
        detect_bright=detect_bright, viz_thickness=getattr(cfg, 'VIZ_MEMBRANE_THICKNESS_PIXELS', None)
    )
    
    quality_entry = {
        'guv_id': guv_id, 'estimated_radius': guv_radius_estimate,
        'detected_inner': detection_info['detected_inner_radius'], 'failed': detection_info['detection_failed'],
        'comments': ', '.join(detection_info['comments']) if detection_info['comments'] else 'OK'
    }

    # --- EXPORT MASK VISUALIZATION ---
    if (cfg.EXPORT_MASK_VISUALIZATION and (quality_entry['failed'] or is_first_guv)):
        viz_image = create_mask_visualization(roi_frame, inner_mask, membrane_mask, background_mask, cfg.MASK_VIZ_OVERLAY_ALPHA)
        try:
            cv2.imwrite(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_mask_viz_GUV_{guv_id}.png"), viz_image)
        except Exception: pass

    # --- GET TRACES ---
    # Calculate mean intensity over time for GUV interior and Background
    intensity_trace = get_intensity_trace_lazy(dye_files, inner_mask)
    background_trace = get_intensity_trace_lazy(dye_files, background_mask)
    
    if np.all(intensity_trace == 0): return None, quality_entry
    
    # --- DETECT JUMP (UPDATED: Using Robust Threshold Method) ---
    try:
        jump_frame = detect_intensity_jump_robust(
            intensity_trace, 
            threshold_percent=getattr(cfg, 'JUMP_THRESHOLD_PERCENT', 0.20)
        )
        
        # --- DEBUG PLOTTING FOR JUMPS ---
        # If no jump is found (0), we flag it for debugging
        is_suspicious = (jump_frame == 0)
        
        if is_suspicious and getattr(cfg, 'EXPORT_DEBUG_PLOTS', False):
            try:
                # 'Agg' backend prevents thread issues during plot generation
                plt.switch_backend('Agg') 
                fig, ax = plt.subplots(figsize=(6,4))
                
                # Plot both raw and smoothed to help user diagnose why it failed
                smoothed = gaussian_filter1d(intensity_trace, sigma=2)
                ax.plot(intensity_trace, 'b-', alpha=0.3, label='Raw')
                ax.plot(smoothed, 'k-', alpha=0.8, label='Smoothed')
                
                if jump_frame > 0:
                    ax.axvline(jump_frame, color='r', linestyle='--', label=f'Detected: {jump_frame}')
                else:
                    ax.text(0.5, 0.5, "No Jump Found", transform=ax.transAxes, color='red')
                    
                ax.set_title(f"Check Jump GUV {guv_id}")
                ax.legend()
                fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"DEBUG_JUMP_{guv_id}.png"))
                plt.close(fig)
                
                if jump_frame == 0: quality_entry['comments'] += "; JUMP_NOT_FOUND"
                
            except Exception: pass
            
    except Exception:
        jump_frame = 0 
        
    result = { "intensity_trace": intensity_trace, "background_trace": background_trace, "jump_frame": jump_frame }
    return result, quality_entry

# -------------------------------------------------------------------
# --- 6. UTILITIES (TRACING, ETC) ---
# -------------------------------------------------------------------
    
def get_intensity_trace_lazy(file_paths: List[str], mask: np.ndarray) -> np.ndarray:
    """
    Calculates the mean intensity inside 'mask' for every file in 'file_paths'.
    Loads files one by one to save memory.
    """
    trace = np.zeros(len(file_paths))
    num_pixels = np.sum(mask)
    if num_pixels == 0: return trace
    for i, filepath in enumerate(file_paths):
        frame = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is None: trace[i] = 0.0
        else: trace[i] = np.sum(frame[mask]) / num_pixels
    return trace

def detect_intensity_jump_robust(trace: np.ndarray, threshold_percent: float = 0.20) -> int:
    """
    Robustly detects the start of the intensity jump using a Threshold Crossing method.
    
    Why this is better than Gradient/Derivative methods:
    Gradient methods look for the steepest slope. However, noise in the baseline 
    (before dye entry) often has steep slopes on a micro-scale, causing false positives 
    at Frame 0.
    
    This method assumes the dye uptake is a "Step Function" (Low -> High).
    It waits until the signal definitively crosses a percentage of the total height.
    
    Args:
        trace: 1D array of intensity values.
        threshold_percent: The fraction of the dynamic range (Max - Min) the signal 
                           must cross to be considered a jump (default 0.20 or 20%).
                           
    Returns:
        int: The frame index where the jump *starts* (the "toe" of the curve).
    """
    if len(trace) < 5:
        return 0
        
    # 1. Smooth the data to remove single-pixel noise spikes
    # Sigma=2 is gentle enough to keep the shape but remove jitter.
    smoothed = gaussian_filter1d(trace, sigma=2)
    
    # 2. Calculate Dynamic Range (Max - Min)
    min_val = np.min(smoothed)
    max_val = np.max(smoothed)
    dynamic_range = max_val - min_val
    
    # Safety Check: Is there a real jump? 
    # If the total change is < 20 counts (on 12/16-bit img), it's likely just a flat line.
    if dynamic_range < 20: 
        return 0
        
    # 3. Define Threshold Level
    # e.g. Baseline + 20% of the total height
    threshold = min_val + (threshold_percent * dynamic_range)
    
    # 4. Find the first frame that crosses this threshold
    # We search specifically in the first 95% of the trace to avoid end-artifacts.
    search_window = int(len(smoothed) * 0.95)
    crossings = np.where(smoothed[:search_window] > threshold)[0]
    
    if len(crossings) == 0:
        return 0 # No crossing found
        
    first_crossing = crossings[0]
    
    # 5. Refine: Find the "Toe" of the curve
    # The 'first_crossing' is where we hit 20% height (mid-rise). 
    # We want the exact moment it STARTED rising from the baseline.
    # Strategy: Look backwards from the crossing point to find the local minimum.
    
    search_back = min(first_crossing, 20) # Look back max 20 frames
    local_segment = smoothed[first_crossing - search_back : first_crossing + 1]
    
    # The index of the minimum value within this local segment
    local_min_relative_idx = np.argmin(local_segment)
    
    # Convert relative index back to absolute frame number
    refined_jump_frame = first_crossing - search_back + local_min_relative_idx
    
    return max(0, refined_jump_frame)

def find_closest_frame(time_array: np.ndarray, target_time_abs: float) -> tuple[int, float]:
    """Finds the index in time_array closest to target_time_abs."""
    idx = np.abs(time_array - target_time_abs).argmin()
    return idx, time_array[idx]
    
def create_mask_visualization(base_image: np.ndarray, inner_mask: np.ndarray, 
                             membrane_mask: np.ndarray, background_mask: np.ndarray, 
                             alpha: float = 0.6) -> np.ndarray:
    """Overlays mask colors onto the base image for quality control."""
    img_8bit = _convert_to_8bit_gray(base_image)
    viz_image = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(viz_image)
    
    # Define Colors (BGR format)
    overlay[inner_mask] = (201, 87, 188)      # Magenta (Inner)
    overlay[membrane_mask] = (82, 0, 249)     # Red (Membrane)
    overlay[background_mask] = (114, 48, 19)  # Dark Blue/Purple (Background)
    
    return cv2.addWeighted(viz_image, alpha, overlay, 1.0 - alpha, 0)

def style_image(frame: np.ndarray, time_label: str, microns_per_pixel: float, scale_bar_microns: int) -> np.ndarray:
    """Adds timestamp text and a scale bar to an image frame."""
    img_8bit = _convert_to_8bit_gray(frame)
    img_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    
    # Apply Red False Color (Set Blue and Green channels to 0)
    img_color[:, :, 0] = 0
    img_color[:, :, 1] = 0
    img_color[:, :, 2] = img_8bit
    
    # Add Time Label
    cv2.putText(img_color, time_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Add Scale Bar
    if scale_bar_microns > 0 and microns_per_pixel > 0:
        bar_len = int(scale_bar_microns / microns_per_pixel)
        h, w = img_color.shape[:2]
        p1, p2 = (w - 30 - bar_len, h - 30), (w - 30, h - 30)
        cv2.line(img_color, p1, p2, (255, 255, 255), 5)
        cv2.putText(img_color, f"{scale_bar_microns} um", (p1[0], h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img_color