# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
This enhanced version includes membrane detection from radial intensity profiles.
It incorporates gradient-based and hybrid methods for more robust detection.
Supports 'LABELED' (fluorescence) and 'UNLABELED' (brightfield) modes.
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
from scipy.signal import find_peaks, peak_widths
from scipy import ndimage as nd
from scipy.ndimage import gaussian_filter1d, map_coordinates
import matplotlib.pyplot as plt

# --- Local Imports ---
import config as cfg
import guv_analysis_utils as utils # Imports itself for internal calls


# -------------------------------------------------------------------
# --- 0. LOGGING SETUP ---
# -------------------------------------------------------------------

def setup_logging(output_folder: str, experiment_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configures logging to file and console.
    """
    log_filename = os.path.join(
        output_folder, 
        f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    
    # Ensure handlers are not added multiple times if reloaded
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)-8s - %(funcName)-20s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logger

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
        with tifffile.TiffFile(file_paths[0]) as tif:
            description_tag = tif.pages[0].tags.get('ImageDescription')
            if description_tag is None:
                return None, None
                
            description = description_tag.value
            
            match = re.search(r'finterval=([0-9.]+)', description)
            if match:
                frame_interval = float(match.group(1))
                num_frames = len(file_paths)
                
                time_array = np.arange(num_frames) * frame_interval
                return time_array, frame_interval
                
    except Exception:
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
    """
    return I_offset + A * (1 - np.exp(-t / tau)) + D * t

def dyn_model_5param(t: np.ndarray, Af: float, A1: float, tau1: float, A2: float, tau2: float) -> np.ndarray:
    """
    5-parameter Double Exponential Rise model for Dye Uptake (Resealing):
    I(t) = Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)
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
        return cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return image_gray

def tifflist_to_numpy(file_paths: List[str]) -> np.ndarray:
    """Loads a list of TIFF files into a 3D numpy array (stack)."""
    if not file_paths:
        return np.array([])
    
    first_frame = cv2.imread(file_paths[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        raise FileNotFoundError(f"Could not load first frame: {file_paths[0]}")
    
    stack = np.empty((len(file_paths), first_frame.shape[0], first_frame.shape[1]), dtype=first_frame.dtype)
    stack[0] = first_frame
    
    for i in range(1, len(file_paths)):
        frame = cv2.imread(file_paths[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            stack[i] = frame
        else:
            stack[i] = np.zeros_like(first_frame)
            
    return stack

def create_circular_mask(img_shape: tuple[int, int], center: tuple[int, int], radius: int) -> np.ndarray:
    """Creates a circular mask for ROI selection."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def create_annular_mask(img_shape: tuple[int, int], center: tuple[int, int], inner_radius: int, outer_radius: int) -> np.ndarray:
    """Creates an annular (ring) mask."""
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask_outer = dist_from_center <= outer_radius
    mask_inner = dist_from_center >= inner_radius
    
    annular_mask = np.logical_and(mask_outer, mask_inner)
    return annular_mask

def refine_guv_center(image: np.ndarray, center_guess: Tuple[int, int], radius_estimate: int, 
                      search_box_factor: float = 1.0) -> Tuple[int, int]:
    """
    Refines the GUV center coordinates by finding the "center of mass" (centroid)
    of the *membrane ring* using edge detection.
    """
    xc, yc = center_guess
    
    box_half_width = int(max(15, radius_estimate * search_box_factor))
    x_start = max(0, xc - box_half_width)
    x_end = min(image.shape[1], xc + box_half_width)
    y_start = max(0, yc - box_half_width)
    y_end = min(image.shape[0], yc + box_half_width)
    
    if x_start >= x_end or y_start >= y_end:
        return center_guess
        
    roi = image[y_start:y_end, x_start:x_end]
    
    if roi.ndim == 3:
        roi = roi[:, :, 0]
    
    if roi.dtype == np.uint16:
        roi_float = roi.astype(np.float32)
        roi_blurred = cv2.GaussianBlur(roi_float, (3, 3), 0)
        laplacian = cv2.Laplacian(roi_blurred, cv2.CV_32F, ksize=3)
    else:
        roi_blurred = cv2.GaussianBlur(roi.astype(np.float32), (3, 3), 0)
        laplacian = cv2.Laplacian(roi_blurred, cv2.CV_32F, ksize=3)

    edge_image_8u = cv2.convertScaleAbs(laplacian)

    if np.max(edge_image_8u) > 0:
         roi_8bit = cv2.normalize(edge_image_8u, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
         roi_8bit = edge_image_8u

    try:
        thresh_val, thresh_mask = cv2.threshold(
            roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    except cv2.error:
         thresh_mask = np.zeros_like(roi_8bit)
    
    M = cv2.moments(thresh_mask)
    if M["m00"] == 0:
        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(roi_8bit)
        cX, cY = max_loc
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    
    xc_new = x_start + cX
    yc_new = y_start + cY
    
    shift = np.sqrt((xc_new - xc)**2 + (yc_new - yc)**2)
    
    if shift > radius_estimate * 0.7:
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
    """
    xc, yc = center
    
    if image.ndim == 3:
        image = image[:,:,0]
    
    length = int(radius * length_excess)
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    
    along_radius = np.arange(0, length, 1)
    
    x_coords = xc + along_radius[np.newaxis, :] * np.cos(theta[:, np.newaxis])
    y_coords = yc + along_radius[np.newaxis, :] * np.sin(theta[:, np.newaxis])

    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)

    profiles_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0.0)
    
    profiles = profiles_flat.reshape(num_angles, length).T
    
    return profiles, along_radius, theta

def average_radial_profile(profiles: np.ndarray) -> np.ndarray:
    """
    Calculate the average radial profile using the median over all angles.
    """
    return np.median(profiles, axis=1)

def membrane_search(
    profile: np.ndarray, 
    radii: np.ndarray, 
    expected_radius: int, 
    search_factor: float,
    membrane_half_width: int,
    peak_min_dist: int,      
    peak_min_prom: float,
    detect_bright: bool = True
) -> Tuple[int, int, int, List[str], bool]:
    """
    Detects membrane location by finding the highest peak (or deepest valley).

    The algorithm searches for the highest intensity peak within a window
    centered on expected_radius. If detect_bright is False, it searches for
    a valley by inverting the profile.
    """
    comments = []
    failed = False
    
    # --- Smooth the profile to be robust to noise/clusters ---
    profile_smooth = gaussian_filter1d(profile, sigma=1)
    
    # --- HANDLE DETECTION MODE (Peak vs Valley) ---
    if detect_bright:
        # Standard mode: look for high intensity peaks
        search_profile = profile_smooth
    else:
        # Valley mode: Invert profile so valleys become peaks
        # We subtract profile from max to keep values positive for 'find_peaks'
        search_profile = np.max(profile_smooth) - profile_smooth

    # --- 1. Define search window around hint ---
    data_to_search = search_profile
    
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
    
    # --- 2. Find peaks in the masked, smoothed data ---
    try:
        max_search_val = np.max(search_profile)
        if max_search_val == 0:
            max_search_val = 1.0
            
        peaks, props = find_peaks(
            data_to_search,
            height=max_search_val * 0.1,
            distance=peak_min_dist,     
            prominence=max_search_val * peak_min_prom 
        )

    except Exception as e:
        comments.append(f"find_peaks_failed: {e}")
        peaks = np.array([])

    if not np.any(peaks):
        comments.append("no_peak_in_window")
        failed = True
        peak_idx = np.searchsorted(radii, expected_radius)
        peak_idx = min(peak_idx, len(radii) - 1)
    else:
        heights = props['peak_heights']
        chosen_peak_idx_in_peaks_array = np.argmax(heights)
        peak_idx = peaks[chosen_peak_idx_in_peaks_array]

    # --- 3. Calculate width ---
    try:
        inner_idx = max(0, peak_idx - membrane_half_width)
        outer_idx = min(len(profile) - 1, peak_idx + membrane_half_width)
        
        if inner_idx >= outer_idx:
            raise ValueError("Inner border >= outer border (width is zero or negative)")
            
    except Exception as e:
        comments.append(f"width_calc_failed: {e}")
        inner_idx = max(0, peak_idx - 5)
        outer_idx = min(len(profile) - 1, peak_idx + 5)
        
    # --- 4. Convert indices back to pixel radii ---
    peak_idx = min(peak_idx, len(radii) - 1)
    inner_idx = min(inner_idx, len(radii) - 1)
    outer_idx = min(outer_idx, len(radii) - 1)
    
    peak_pos = int(radii[peak_idx])
    inner_b = int(radii[inner_idx])
    outer_b = int(radii[outer_idx])
    
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
                                   detect_bright: bool = True,
                                   num_angles: int = 360, 
                                   length_excess: float = 1.5,
                                   viz_thickness: Optional[int] = None) -> \
                                   Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Creates GUV masks using the detection logic.
    """
    img_shape = roi_guide_frame.shape
    
    profiles, along_radius, theta = linear_profiles(
        roi_guide_frame, 
        center, 
        num_angles=num_angles, 
        length_excess=length_excess,
        radius=radius_estimate 
    )
    
    radial_profile = average_radial_profile(profiles)
    
    peak_pos, inner_b, outer_b, comments, failed = membrane_search(
        radial_profile, 
        along_radius,
        expected_radius=radius_estimate,
        search_factor=search_factor,
        membrane_half_width=membrane_half_width,
        peak_min_dist=peak_min_dist,          
        peak_min_prom=peak_min_prom,
        detect_bright=detect_bright
    )

    inner_guv_mask = create_circular_mask(img_shape, center, radius=inner_b)
    
    if viz_thickness is not None and viz_thickness > 0:
        viz_mem_inner_r = inner_b + 1
        viz_mem_outer_r = inner_b + viz_thickness
        membrane_mask = create_annular_mask(img_shape, center, 
                                           inner_radius=viz_mem_inner_r, 
                                           outer_radius=viz_mem_outer_r)
    else:
        membrane_mask = create_annular_mask(img_shape, center, 
                                           inner_radius=inner_b + 1, 
                                           outer_radius=outer_b)
    
    bg_inner_r = outer_b + bg_buffer
    bg_outer_r = outer_b + bg_buffer + bg_width
    background_ring_mask = create_annular_mask(img_shape, center, 
                                              inner_radius=bg_inner_r, 
                                              outer_radius=bg_outer_r)

    detection_info = {
        'detected_inner_radius': inner_b,
        'detected_outer_radius': outer_b,
        'bg_inner_radius': bg_inner_r, 
        'bg_outer_radius': bg_outer_r, 
        'peak_position': peak_pos,
        'detection_failed': failed,
        'comments': comments,
        'radial_profile': radial_profile,
        'along_radius': along_radius,
        'method': 'skeleton_search',
        'search_factor_used': search_factor
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
    """
    
    # --- Assume CSV 'size' is DIAMETER, divide by 2 for radius ---
    guv_radius_estimate = int(np.round(radius_csv_raw / 2.0))

    # --- GUV CENTER COORDINATES REFINEMENT ---
    (xc_refined, yc_refined) = utils.refine_guv_center(
        roi_frame, 
        center_orig, 
        guv_radius_estimate,
        search_box_factor=0.8
    )
    
    # --- 5a. Define Masks using Membrane Detection ---
    viz_thickness = getattr(cfg, 'VIZ_MEMBRANE_THICKNESS_PIXELS', None)
    
    # --- HANDLE DETECTION MODE ---
    # If user sets 'UNLABELED', we look for dark membranes (detect_bright=False)
    # Default is 'LABELED' (detect_bright=True)
    mode = getattr(cfg, 'MEMBRANE_DETECTION_MODE', 'LABELED').upper()
    detect_bright = (mode != 'UNLABELED')
    
    inner_mask, membrane_mask, background_mask, detection_info = utils.create_guv_masks_with_detection(
        roi_frame, 
        (xc_refined, yc_refined),
        radius_estimate=guv_radius_estimate,
        bg_buffer=cfg.BG_BUFFER_PIXELS,
        bg_width=cfg.BG_RING_WIDTH_PIXELS,
        search_factor=cfg.MEMBRANE_SEARCH_FACTOR,
        membrane_half_width=cfg.MEMBRANE_FIXED_HALF_WIDTH,
        peak_min_dist=cfg.PEAK_FIND_MIN_DISTANCE,
        peak_min_prom=cfg.PEAK_FIND_MIN_PROMINENCE,
        detect_bright=detect_bright,
        num_angles=360,
        length_excess=1.5,
        viz_thickness=viz_thickness
    )
    
    # Log detection quality
    quality_entry = {
        'guv_id': guv_id,
        'estimated_radius': guv_radius_estimate,
        'detected_inner': detection_info['detected_inner_radius'],
        'detected_outer': detection_info['detected_outer_radius'],
        'failed': detection_info['detection_failed'],
        'comments': ', '.join(detection_info['comments']) if detection_info['comments'] else 'OK'
    }

    # --- Export mask visualization ---
    should_export_viz = (cfg.EXPORT_MASK_VISUALIZATION and quality_entry['failed']) or \
                       (cfg.EXPORT_MASK_VISUALIZATION and 'OK' not in quality_entry['comments'])
    
    if cfg.EXPORT_MASK_VISUALIZATION and is_first_guv:
         should_export_viz = True

    if should_export_viz:
        viz_image = utils.create_mask_visualization(
            roi_frame, inner_mask, membrane_mask, background_mask,
            alpha=cfg.MASK_VIZ_OVERLAY_ALPHA
        )
        viz_name = f"{cfg.EXPERIMENT_BASE_NAME}_mask_viz_GUV_{guv_id}.png"
        viz_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, viz_name)
        
        try:
            cv2.imwrite(viz_path, viz_image)
        except Exception:
            pass # Cannot log from a worker process
            
        # --- SAVE RADIAL PLOT ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(detection_info['along_radius'], detection_info['radial_profile'], 'b-', linewidth=2)
        ax.axvspan(0, detection_info['detected_inner_radius'], color='purple', alpha=0.2, label='Uptake (Inner) Region')
        ax.axvspan(detection_info['detected_inner_radius'], detection_info['detected_outer_radius'], color='red', alpha=0.2, label='Membrane Region')
        bg_r_start = detection_info['bg_inner_radius']
        bg_r_end = detection_info['bg_outer_radius']
        ax.axvspan(bg_r_start, bg_r_end, color='blue', alpha=0.2, label=f'Background Region')
        ax.axvline(x=guv_radius_estimate, color='orange', linestyle=':', label=f"CSV estimate ({guv_radius_estimate}px)")
        ax.axvline(x=detection_info['peak_position'], color='black', linestyle=':', label=f"Detected Position ({detection_info['peak_position']:.1f}px)")
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'Radial Intensity Profile & Regions - GUV {guv_id} (Center: {xc_refined}, {yc_refined})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        profile_name = f"{cfg.EXPERIMENT_BASE_NAME}_radial_profile_GUV_{guv_id}.png"
        profile_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, profile_name)
        fig.savefig(profile_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- 5b. Get Intensity Traces ---
    intensity_trace = utils.get_intensity_trace_lazy(dye_files, inner_mask)
    background_trace = utils.get_intensity_trace_lazy(dye_files, background_mask)
    
    if np.all(intensity_trace == 0):
        # Mask was empty or invalid
        return None, quality_entry
    
    # --- 5c. Detect Jump ---
    try:
        jump_frame = utils.detect_intensity_jump(intensity_trace, sensitivity=cfg.JUMP_SENSITIVITY)
        if jump_frame == -1:
            jump_frame = 0 
        
    except Exception:
        jump_frame = 0 
        
    result = {
        "intensity_trace": intensity_trace,
        "background_trace": background_trace,
        "jump_frame": jump_frame
    }
    # Return a tuple of (data, quality_log)
    return result, quality_entry

# -------------------------------------------------------------------
# --- 6. LEGACY & TRACE FUNCTIONS ---
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

def get_intensity_trace_lazy(file_paths: List[str], mask: np.ndarray) -> np.ndarray:
    """
    Calculates the average intensity within the mask for every frame
    by loading one frame at a time ("lazy loading").
    """
    trace = np.zeros(len(file_paths))
    num_pixels = np.sum(mask)
    
    if num_pixels == 0:
        return trace
        
    for i, filepath in enumerate(file_paths):
        frame = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is None:
            trace[i] = 0.0
            continue
            
        masked_frame = frame[mask]
        trace[i] = np.sum(masked_frame) / num_pixels
        
    return trace

# -------------------------------------------------------------------
# --- 7. DATA PROCESSING & UTILITIES ---
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