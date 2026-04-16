# -*- coding: utf-8 -*-
"""
======================================================
--- GUV ANALYSIS UTILITIES ---
======================================================
Helper functions for:
1. Timestamp extraction
2. Image processing (masking)
3. GUV tracking via ring-score grid search  ← replaces template matching
4. Signal processing (jump detection, kinetic models)
5. Visualization
"""

import numpy as np
import cv2
import pandas as pd
import os
import re
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import tifffile
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, map_coordinates
import matplotlib.pyplot as plt

import config as cfg


# -------------------------------------------------------------------
# --- 0. LOGGING ---
# -------------------------------------------------------------------

def setup_logging(output_folder: str, experiment_name: str,
                  level: int = logging.INFO) -> logging.Logger:
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
# --- 1. TIMESTAMP EXTRACTION ---
# -------------------------------------------------------------------

def extract_timestamps_from_metadata(
        file_paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    
    if not file_paths:
        return None, None
        
    num_frames = len(file_paths)
    schedule = getattr(cfg, 'FRAME_INTERVAL_SCHEDULE', None)
    
    if schedule is not None and len(schedule) > 0:
        dt_array = np.zeros(num_frames)
        for (start_idx, end_idx, interval) in schedule:
            actual_end = num_frames if end_idx is None else min(end_idx, num_frames)
            if start_idx < num_frames:
                dt_array[start_idx:actual_end] = interval
                
        # The first frame sits at t=0
        dt_array[0] = 0.0
        t_array = np.cumsum(dt_array)
        return t_array, float(np.median(dt_array[1:]))

    try:
        with tifffile.TiffFile(file_paths[0]) as tif:
            tag = tif.pages[0].tags.get('ImageDescription')
            if tag is None:
                return None, None
            match = re.search(r'finterval=([0-9.]+)', tag.value)
            if match:
                dt = float(match.group(1))
                return np.arange(num_frames) * dt, dt
    except Exception:
        pass
    
    return None, None


def create_manual_timestamps(num_frames: int,
                              fallback_fps: float = 1.0
                              ) -> Tuple[np.ndarray, float]:
    dt = 1.0 / fallback_fps
    return np.arange(num_frames) * dt, dt


# -------------------------------------------------------------------
# --- 2. KINETIC MODELS ---
# -------------------------------------------------------------------

# Dye Influx
def influx_1exp(t, I0, Iinf, tau, D=0):
    return I0 + (Iinf - I0) * (1 - np.exp(-t / tau)) + D * t

def influx_2exp(t, I0, a1, tau1, a2, tau2, D=0):
    return I0 + a1 * (1 - np.exp(-t / tau1)) + a2 * (1 - np.exp(-t / tau2)) + D * t

# Dye Efflux
def efflux_1exp(t, I0, Iinf, tau, D=0):
    return Iinf + (I0 - Iinf) * np.exp(-t / tau) + D * t

def efflux_2exp(t, Iinf, a1, tau1, a2, tau2, D=0):
    return Iinf + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + D * t

# -------------------------------------------------------------------
# --- 3. PULSE-FRAME DETECTION ---
# -------------------------------------------------------------------

def detect_pulse_frame_efflux(
        intensity_curves: list,
        background_curves: list,
        smooth_sigma: float = 2.0,
        output_folder: str  = None,
        experiment_name: str = "",
        time_array: Optional[np.ndarray] = None,
) -> int:
    """
    Auto-detects the pulse frame for DYE EFFLUX experiments.

    Strategy
    --------
    1. Compute net signal (intensity − background) for every GUV.
    2. Average across all GUVs (NaN-safe) to get a single mean trace.
    3. Smooth with a Gaussian kernel to suppress single-frame noise.
    4. Differentiate with respect to *time* (dI/dt) rather than frame
       index so that variable acquisition rates (e.g. 1 s → 0.1 s → 5 s)
       do not bias the detection toward long-interval transitions.
       Falls back to dI/dframe when no time_array is supplied.
    5. The pulse frame is the index of the most negative dI/dt step.

    A diagnostic PNG is written to *output_folder* (if provided) so the
    detection can be visually verified.

    Parameters
    ----------
    intensity_curves  : list of 1-D float arrays, one per GUV
    background_curves : list of 1-D float arrays, one per GUV
    smooth_sigma      : Gaussian σ in frames for pre-smoothing
    output_folder     : directory for the diagnostic plot (optional)
    experiment_name   : used in the plot filename
    time_array        : 1-D array of frame timestamps (seconds).
                        When provided the derivative is dI/dt; otherwise
                        dI/dframe is used (legacy behaviour).

    Returns
    -------
    int  – detected pulse frame index (0-based)
    """
    if not intensity_curves:
        return 0

    # Net signal per GUV, stacked into a 2-D array (n_guvs × n_frames)
    net_traces = np.vstack([
        np.asarray(d, float) - np.asarray(b, float)
        for d, b in zip(intensity_curves, background_curves)
    ])

    mean_trace = np.nanmean(net_traces, axis=0)
    smoothed   = gaussian_filter1d(mean_trace, sigma=smooth_sigma)

    # ── dI/dt: normalise by the actual time step between frames ───────────
    raw_diff = np.diff(smoothed)          # dI/dframe  (length = n_frames − 1)
    if time_array is not None and len(time_array) >= len(smoothed):
        dt = np.diff(time_array[:len(smoothed)])
        dt = np.where(dt > 0, dt, 1.0)   # guard against zero-length intervals
        diff = raw_diff / dt              # dI/dt
        diff_label = 'dI/dt  (AU/s, smoothed)'
        diff_ylabel = 'Δ intensity / second'
    else:
        diff = raw_diff
        diff_label = 'dI/dframe  (smoothed)'
        diff_ylabel = 'Δ intensity / frame'

    pulse_frame = int(np.argmin(diff))    # steepest negative step in dI/dt

    # ── Diagnostic plot ────────────────────────────────────────────────────
    if output_folder:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        n_frames = len(mean_trace)
        frames   = np.arange(n_frames)

        axes[0].plot(frames, mean_trace, color='steelblue', lw=1.0,
                     alpha=0.6, label='Mean net signal (raw)')
        axes[0].plot(frames, smoothed, color='navy', lw=1.5,
                     label=f'Smoothed (σ={smooth_sigma} fr)')
        axes[0].axvline(pulse_frame, color='crimson', ls='--', lw=1.5,
                        label=f'Detected pulse  (frame {pulse_frame})')
        axes[0].set_ylabel('Mean net fluorescence (AU)')
        axes[0].legend(fontsize=8)
        axes[0].set_title('Pulse-frame detection — efflux')

        axes[1].plot(np.arange(len(diff)), diff, color='darkorange', lw=1.0,
                     label=diff_label)
        axes[1].axvline(pulse_frame, color='crimson', ls='--', lw=1.5)
        axes[1].axhline(0, color='gray', lw=0.5, ls=':')
        axes[1].set_xlabel('Frame index')
        axes[1].set_ylabel(diff_ylabel)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        out_path = os.path.join(
            output_folder,
            f"{experiment_name}_pulse_frame_detection.png"
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    return pulse_frame


# -------------------------------------------------------------------
# --- 4. MASK HELPERS ---
# -------------------------------------------------------------------

def _convert_to_8bit_gray(image: np.ndarray) -> np.ndarray:
    """ Boosts contrast by clipping outliers before normalizing to 8-bit. """
    p_low  = getattr(cfg, 'CONTRAST_P_LOW', 1.0)
    p_high = getattr(cfg, 'CONTRAST_P_HIGH', 99.0)
    
    img = image[:, :, 0] if image.ndim == 3 else image
    if img.dtype != np.uint8:
        low, high = np.percentile(img, (p_low, p_high))
        img_clipped = np.clip(img, low, high)
        return cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img
def generate_vectorized_masks(shape, center, axes, angle, mem_hw, bg_buf, bg_w) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates inner, membrane, and background boolean masks simultaneously."""
    h, w = shape[:2]
    cx, cy = center
    a, b = axes

    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    
    # Rotate coordinates to align with ellipse angle
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    x_rot = cos_t * (x - cx) + sin_t * (y - cy)
    y_rot = -sin_t * (x - cx) + cos_t * (y - cy)

    # Compute normalized elliptical distance squared
    # A value of 1.0 lies exactly on the ellipse contour
    dist_sq = (x_rot**2) / (a**2) + (y_rot**2) / (b**2)
    dist = np.sqrt(dist_sq)

    # Calculate fractional thresholds based on semi-major axis
    mem_inner_frac = max(0.01, (a - mem_hw) / a)
    mem_outer_frac = (a + mem_hw) / a
    bg_inner_frac  = (a + mem_hw + bg_buf) / a
    bg_outer_frac  = (a + mem_hw + bg_buf + bg_w) / a

    inner_mask = dist < mem_inner_frac
    mem_mask   = (dist >= mem_inner_frac) & (dist <= mem_outer_frac)
    bg_mask    = (dist >= bg_inner_frac)  & (dist <= bg_outer_frac)

    return inner_mask, mem_mask, bg_mask


# -------------------------------------------------------------------
# --- 4. RADIAL PROFILE ---
# -------------------------------------------------------------------

_POLAR_CACHE = {}

def get_polar_offsets(r_max: int, n_angles: int) -> Tuple[np.ndarray, np.ndarray]:
    """Caches trigonometric arrays for radial extraction."""
    key = (r_max, n_angles)
    if key not in _POLAR_CACHE:
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        radii  = np.arange(r_max)
        dx = radii[:, None] * np.cos(thetas)[None, :]
        dy = radii[:, None] * np.sin(thetas)[None, :]
        _POLAR_CACHE[key] = (dx, dy)
    return _POLAR_CACHE[key]

def radial_profile_2d_local(local_crop: np.ndarray, local_center: Tuple[float, float],
                            r_max: int, n_angles: int = 72) -> np.ndarray:
    """Sub-pixel sampling operating only on a small bounding box crop."""
    cx, cy = local_center
    dx, dy = get_polar_offsets(r_max, n_angles)

    xs = cx + dx
    ys = cy + dy

    coords = np.vstack((ys.flatten(), xs.flatten()))
    return map_coordinates(local_crop, coords, order=1, mode='nearest').reshape(r_max, n_angles)

def radial_profile_local(local_crop: np.ndarray, local_center: Tuple[float, float],
                         r_max: int, n_angles: int = 72) -> np.ndarray:
    return np.median(radial_profile_2d_local(local_crop, local_center, r_max, n_angles), axis=1).astype(float)


# -------------------------------------------------------------------
# --- 5. RING-SCORE GRID SEARCH  (replaces NCC template matching)
# -------------------------------------------------------------------

def _ring_score_from_profile(profile: np.ndarray, expected_r: float,
                              search_factor: float, detect_bright: bool) -> float:
    """
    Compute a scalar quality score for a ring at *expected_r* in *profile*.

    For dark rings (UNLABELED):
        score = mean of surrounding region − minimum inside search window
        → large positive value when a deep valley exists at expected_r

    For bright rings (LABELED):
        score = maximum inside search window − mean of surrounding region
        → large positive value when a sharp peak exists at expected_r
    """
    profile = gaussian_filter1d(profile, sigma=2)
    lo = max(2, int(expected_r * (1.0 - search_factor)))
    hi = min(len(profile) - 2, int(expected_r * (1.0 + search_factor)))
    if lo >= hi:
        return 0.0

    window = profile[lo:hi]

    # Surroundings = thin bands just inside and outside the search window
    band = max(5, (hi - lo) // 2)
    inner_band = profile[max(0, lo - band):lo]
    outer_band = profile[hi:min(len(profile), hi + band)]
    both = np.concatenate([inner_band, outer_band])
    surround = both.mean() if len(both) > 0 else profile.mean()

    if detect_bright:
        return float(window.max() - surround)
    else:
        return float(surround - window.min())


def _score_at_center(frame: np.ndarray, cx: int, cy: int,
                     expected_r: int, search_factor: float,
                     detect_bright: bool, n_angles: int = 72) -> float:
    """
    Fast ring-score at a single candidate centre.
    Uses angle sampling (72) for speed during the grid search.
    Returns 0.0 if the centre is too close to the image boundary.
    """
    r_max = int(expected_r * (1.0 + search_factor) + 5)
    h, w  = frame.shape[:2]
    
    x_min = max(0, int(cx - r_max - 2))
    x_max = min(w, int(cx + r_max + 2))
    y_min = max(0, int(cy - r_max - 2))
    y_max = min(h, int(cy + r_max + 2))
    
    
    if cx - r_max < 0 or cx + r_max >= w or cy - r_max < 0 or cy + r_max >= h:
        return 0.0
    
    local_frame = frame[y_min:y_max, x_min:x_max]
    local_cx = cx - x_min
    local_cy = cy - y_min
    
    prof = radial_profile_local(local_frame, (local_cx, local_cy), r_max, n_angles)
    return _ring_score_from_profile(prof, expected_r, search_factor, detect_bright)


def find_best_guv_center(
        frame: np.ndarray,
        prev_center: Tuple[int, int],
        prev_radius: int,
        search_window_factor: float,
        search_factor: float,
        detect_bright: bool,
        grid_step_factor: float = 0.15,
) -> Tuple[Tuple[int, int], float]:
    """
    Translational search for the GUV centre using ring-quality scoring.

    Strategy
    --------
    Evaluate the ring-score at every grid point inside a circle of radius
    `prev_radius × search_window_factor` around `prev_center`.  Return the
    highest-scoring candidate.
    """
    cx, cy  = prev_center
    r       = prev_radius
    h, w    = frame.shape[:2]

    search_r = max(8, int(r * search_window_factor))
    step     = max(2, int(r * grid_step_factor))
    r_max    = int(r * (1.0 + search_factor) + 5)
    
    # 1. Define bounding box for the entire search space
    bound = search_r + step + r_max + 2
    x_min = max(0, int(cx - bound))
    x_max = min(w, int(cx + bound))
    y_min = max(0, int(cy - bound))
    y_max = min(h, int(cy + bound))
    
    local_frame = frame[y_min:y_max, x_min:x_max]

    best_score  = -1.0
    best_center = prev_center
    
    # 2. Grid search over the local crop
    for dx in range(-search_r, search_r + step, step):
        for dy in range(-search_r, search_r + step, step):
            if dx * dx + dy * dy > search_r * search_r:
                continue
            ncx, ncy = cx + dx, cy + dy
            # Calculate coordinates relative to the local crop
            local_cx = ncx - x_min
            local_cy = ncy - y_min
            
            if local_cx - r_max < 0 or local_cx + r_max >= local_frame.shape[1] or \
               local_cy - r_max < 0 or local_cy + r_max >= local_frame.shape[0]:
                continue
                
            prof = radial_profile_local(local_frame, (local_cx, local_cy), r_max, 72)
            score = _ring_score_from_profile(prof, r, search_factor, detect_bright)
            
            if score > best_score:
                best_score  = score
                best_center = (ncx, ncy)

    return best_center, best_score


def detect_membrane_ellipse(frame: np.ndarray, center: Tuple[float, float],
                            expected_axes: Tuple[float, float],
                            search_factor: float, detect_bright: bool
                            ) -> Tuple[Tuple[float, float], Tuple[float, float], float, bool]:
    
    expected_r = sum(expected_axes) / 2.0
    r_max = int(expected_r * (1.0 + search_factor) + 10)
    h, w = frame.shape[:2]
    cx, cy = center

    if cx - r_max < 0 or cx + r_max >= w or cy - r_max < 0 or cy + r_max >= h:
        return center, expected_axes, 0.0, False
    
    x_min = max(0, int(cx - r_max - 2))
    x_max = min(w, int(cx + r_max + 2))
    y_min = max(0, int(cy - r_max - 2))
    y_max = min(h, int(cy + r_max + 2))

    local_frame = frame[y_min:y_max, x_min:x_max]
    local_cx, local_cy = cx - x_min, cy - y_min
    
    sampled_2d = radial_profile_2d_local(local_frame, (local_cx, local_cy), r_max, n_angles=72)
    smooth_2d = gaussian_filter1d(sampled_2d, sigma=1.0, axis=0)

    lo = max(2, int(expected_r * (1.0 - search_factor)))
    hi = min(sampled_2d.shape[0] - 2, int(expected_r * (1.0 + search_factor)))
    if lo >= hi:
        return center, expected_axes, 0.0, False

    search_sig = smooth_2d if detect_bright else (smooth_2d.max(axis=0) - smooth_2d)
    window = search_sig[lo:hi, :]

    peak_indices = np.argmax(window, axis=0) + lo

    thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    pts_x = cx + peak_indices * np.cos(thetas)
    pts_y = cy + peak_indices * np.sin(thetas)
    points = np.column_stack((pts_x, pts_y)).astype(np.float32)

    try:
        ellipse = cv2.fitEllipse(points)
        ecx, ecy = ellipse[0]
        ea, eb = ellipse[1][0] / 2.0, ellipse[1][1] / 2.0 
        e_angle = ellipse[2]

        dist = np.hypot(ecx - cx, ecy - cy)
        if dist > expected_r * search_factor:
            return center, expected_axes, 0.0, False

        return (ecx, ecy), (ea, eb), e_angle, True

    except Exception:
        return center, expected_axes, 0.0, False


# -------------------------------------------------------------------
# --- 6. FRAME-BY-FRAME TRACKER ---
# -------------------------------------------------------------------

def track_guv_across_frames(
        roi_stack: np.ndarray,
        n_frames: int,
        initial_center: Tuple[int, int],
        initial_radius: int,
        search_window_factor: float,
        grid_step_factor: float,
        search_factor: float,
        membrane_half_width: int,
        detect_bright: bool,
        bg_buffer: int,
        bg_width: int,
        max_radius_change_factor: float = 0.20,
        rupture_score_threshold: float  = 4.0,
        rupture_consecutive_fails: int  = 3,
) -> Dict[str, Any]:

    inner_masks: List[Optional[np.ndarray]] = [None] * n_frames
    mem_masks:   List[Optional[np.ndarray]] = [None] * n_frames
    bg_masks:    List[Optional[np.ndarray]] = [None] * n_frames
    centers:     List[Optional[Tuple]]      = [None] * n_frames
    radii:       List[Optional[int]]        = [None] * n_frames
    ellipses:    List[Optional[Dict]]       = [None] * n_frames 
    ring_scores: List[float]                = [0.0]  * n_frames

    cx, cy = float(initial_center[0]), float(initial_center[1])
    axes   = (float(initial_radius), float(initial_radius))
    angle  = 0.0
    vx, vy = 0.0, 0.0

    consecutive_low = 0
    ruptured_at     = None
    n_valid         = 0

    for i in range(n_frames):
        frame = roi_stack[i]
        if frame is None:
            consecutive_low += 1
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                break
            continue

        pred_cx = cx + vx
        pred_cy = cy + vy
        avg_r = (axes[0] + axes[1]) / 2.0

        (new_cx, new_cy), score = find_best_guv_center(
            frame, (pred_cx, pred_cy), avg_r,
            search_window_factor, search_factor,
            detect_bright, grid_step_factor
        )
        ring_scores[i] = score

        if score < rupture_score_threshold:
            consecutive_low += 1
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                break
        else:
            consecutive_low = 0

        det_center, det_axes, det_angle, r_ok = detect_membrane_ellipse(
            frame, (new_cx, new_cy), axes, search_factor, detect_bright
        )

        # Clamp axes change
        max_da = axes[0] * max_radius_change_factor
        max_db = axes[1] * max_radius_change_factor
        new_a  = max(3.0, np.clip(det_axes[0], axes[0] - max_da, axes[0] + max_da))
        new_b  = max(3.0, np.clip(det_axes[1], axes[1] - max_db, axes[1] + max_db))
        new_axes = (new_a, new_b)
        new_avg_r = int((new_a + new_b) / 2.0)

        img_shape = frame.shape[:2]
        
        inner_masks[i], mem_masks[i], bg_masks[i] = generate_vectorized_masks(
            img_shape, det_center, new_axes, det_angle,
            membrane_half_width, bg_buffer, bg_width
        )
        
        centers[i]  = det_center
        radii[i]    = new_avg_r
        ellipses[i] = {'center': det_center, 'axes': new_axes, 'angle': det_angle}
        n_valid    += 1

        vx = 0.5 * vx + 0.5 * (det_center[0] - cx)
        vy = 0.5 * vy + 0.5 * (det_center[1] - cy)
        cx, cy = det_center
        axes, angle = new_axes, det_angle

    return {
        'inner_masks':    inner_masks,
        'mem_masks':      mem_masks,
        'bg_masks':       bg_masks,
        'centers':        centers,
        'radii':          radii,
        'ellipses':       ellipses,
        'ring_scores':    ring_scores,
        'ruptured_at':    ruptured_at,
        'n_valid_frames': n_valid,
    }


# -------------------------------------------------------------------
# --- 7. INTENSITY EXTRACTION ---
# -------------------------------------------------------------------

def get_intensity_trace_tracked(dye_stack: np.ndarray,
                                n_frames: int,
                                per_frame_masks: List[Optional[np.ndarray]],
                                method: str = 'mean'
                                ) -> np.ndarray:
    """
    Extract intensity inside *per_frame_masks[i]* from *dye_files[i]*.
    Returns NaN for frames whose mask is None (post-rupture / untracked).
    """
    trace = np.full(n_frames, np.nan)
    for i, mask in enumerate(per_frame_masks):
        if mask is None:
            continue
        
        pixels = dye_stack[i][mask]
        if len(pixels) == 0:
            continue
            
        if method == 'median':
            trace[i] = float(np.median(pixels))
        else:
            trace[i] = float(np.mean(pixels))
        
    return trace

def find_closest_frame(time_array: np.ndarray, target: float) -> Tuple[int, float]:
    """Find the index and actual time value closest to the target time."""
    idx = int(np.abs(time_array - target).argmin())
    return idx, float(time_array[idx])


# -------------------------------------------------------------------
# --- 8. PARALLEL WORKER ---
# -------------------------------------------------------------------

def process_single_guv(guv_id: str,
                        initial_center: Tuple[int, int],
                        initial_radius: int,
                        is_first_guv: bool,
                        roi_mmap_info: Tuple,
                        dye_mmap_info: Tuple,
                        n_frames: int,
                        actin_mmap_info: Tuple = (None, None, None)) -> Tuple:
    """
    Full analysis pipeline for a single GUV using frame-by-frame tracking.
    Modified to strip heavy masks and return only coordinate data to save RAM.
    """
    detect_bright = (getattr(cfg, 'MEMBRANE_DETECTION_MODE', 'LABELED').upper() != 'UNLABELED')

    roi_path, roi_shape, roi_dtype = roi_mmap_info
    roi_stack = np.memmap(roi_path, dtype=roi_dtype, mode='r', shape=roi_shape)
    
    dye_stack = None
    if dye_mmap_info[0] is not None:
        dye_path, dye_shape, dye_dtype = dye_mmap_info
        dye_stack = np.memmap(dye_path, dtype=dye_dtype, mode='r', shape=dye_shape)

    actin_stack = None
    if actin_mmap_info[0] is not None:
        actin_path, actin_shape, actin_dtype = actin_mmap_info
        actin_stack = np.memmap(actin_path, dtype=actin_dtype, mode='r', shape=actin_shape)

    # --- 1. Track GUV ---
    tracking = track_guv_across_frames(
        roi_stack=roi_stack, n_frames=n_frames, 
        initial_center=initial_center, initial_radius=initial_radius,
        search_window_factor=getattr(cfg, 'TRACKING_SEARCH_WINDOW_FACTOR', 0.7),
        grid_step_factor=getattr(cfg, 'TRACKING_GRID_STEP_FACTOR', 0.15),
        search_factor=cfg.MEMBRANE_SEARCH_FACTOR,
        membrane_half_width=cfg.MEMBRANE_FIXED_HALF_WIDTH,
        detect_bright=detect_bright,
        bg_buffer=cfg.BG_BUFFER_PIXELS,
        bg_width=cfg.BG_RING_WIDTH_PIXELS,
        max_radius_change_factor=getattr(cfg, 'TRACKING_MAX_RADIUS_CHANGE_FACTOR', 0.20),
        rupture_score_threshold=getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0),
        rupture_consecutive_fails=getattr(cfg, 'RUPTURE_DETECTION_CONSECUTIVE_FAILS', 3),
    )

    # --- 2. Extract Intensity (Calculation happens here while masks are still in RAM) ---
    intensity_trace  = get_intensity_trace_tracked(dye_stack, n_frames, tracking['inner_masks'], method='mean')
    background_trace = get_intensity_trace_tracked(dye_stack, n_frames, tracking['bg_masks'], method='median')

    # --- 2b. Extract Actin Cortex Traces (C2 channel) ---
    actin_data = None
    if actin_stack is not None and getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False):
        img_shape = roi_stack[0].shape[:2]
        actin_data = extract_actin_traces(
            actin_stack   = actin_stack,
            n_frames      = n_frames,
            mem_masks     = tracking['mem_masks'],
            inner_masks   = tracking['inner_masks'],
            cortex_hw     = getattr(cfg, 'ACTIN_CORTEX_HALF_WIDTH', 4),
            centers       = tracking['centers'],
            ellipses      = tracking['ellipses'],
            img_shape     = img_shape,
            bg_buffer     = cfg.BG_BUFFER_PIXELS,
            bg_width      = cfg.BG_RING_WIDTH_PIXELS,
        )

    # --- 3. Diagnostics and Export ---
    ruptured_at = tracking['ruptured_at']
    n_valid     = tracking['n_valid_frames']
    
    quality_entry = {
        'guv_id': guv_id, 'initial_radius': initial_radius, 'failed': ruptured_at is not None,
        'ruptured_at_frame': ruptured_at, 'n_valid_frames': n_valid,
        'mean_ring_score': float(np.mean([s for s in tracking['ring_scores'] if s > 0])) if n_valid > 0 else 0.0,
        'comments': f"RUPTURED_AT_F{ruptured_at}" if ruptured_at is not None else 'OK',
    }

    if getattr(cfg, 'EXPORT_TRACK_VISUALIZATION', True):
        # Workers run before the pulse frame is auto-detected, so we pass
        # pulse_frame=None here.  main() will redraw these plots with the
        # confirmed PULSE_FRAME once detection is complete.
        export_track_visualization(
            cfg.FOLDER_TRACKING, cfg.FOLDER_SCORES, cfg.EXPERIMENT_BASE_NAME, guv_id,
            roi_stack, tracking['centers'], tracking['ellipses'],
            tracking['ring_scores'], ruptured_at, pulse_frame=None
        )

    # --- 4. LIGHTWEIGHT RETURN (Discard heavy masks to prevent MemoryError) ---
    # We strip 'inner_masks', 'mem_masks', and 'bg_masks' here
    light_tracking = {
        'centers':     tracking['centers'],
        'radii':       tracking['radii'],
        'ellipses':    tracking['ellipses'],
        'ring_scores': tracking['ring_scores']
    }

    result = {
        'intensity_trace':  intensity_trace,
        'background_trace': background_trace,
        'tracking':         light_tracking,
        'actin_data':       actin_data,   # None if ANALYZE_ACTIN_CHANNEL is False
    }
    return result, quality_entry

# -------------------------------------------------------------------
# --- 9. VISUALIZATION HELPERS ---
# -------------------------------------------------------------------

def create_mask_visualization(base_image, inner_mask, membrane_mask,
                               background_mask, alpha: float = 0.6):
    img8  = _convert_to_8bit_gray(base_image)
    viz   = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    ov    = np.zeros_like(viz)
    if inner_mask is not None:      ov[inner_mask]      = (201,  87, 188)
    if membrane_mask is not None:   ov[membrane_mask]   = ( 82,   0, 249)
    if background_mask is not None: ov[background_mask] = (114,  48,  19)
    return cv2.addWeighted(viz, alpha, ov, 1.0 - alpha, 0)


def export_track_visualization(track_folder, score_folder, experiment_name, guv_id,
                                roi_stack, centers, ellipses, ring_scores,
                                ruptured_at, pulse_frame=None):
    """
    Saves a diagnostic PNG showing the GUV trajectory and ring scores.
    Left panel: last valid ROI frame with centre path overlaid.
    Right panel: ring score time series with rupture threshold.
    """
    valid_pairs = [(i, c) for i, c in enumerate(centers) if c is not None]
    if not valid_pairs:
        return
    
    last_i, _ = valid_pairs[-1]
    frame = roi_stack[last_i]
    if frame is None:
        return

    # ── Left: trajectory on ROI frame ─────────────────────────────────────
    img8   = _convert_to_8bit_gray(frame)
    canvas = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    pts = np.array([c for _, c in valid_pairs], dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(canvas, [pts], False, (0, 230, 0), 1, cv2.LINE_AA)

    # Start (green) and end (yellow=survived, red=ruptured)
    _, c0 = valid_pairs[0]
    el0 = ellipses[valid_pairs[0][0]]
    if el0:
        cv2.ellipse(canvas, (int(el0['center'][0]), int(el0['center'][1])),
                    (int(el0['axes'][0]), int(el0['axes'][1])), el0['angle'], 0, 360, (0, 230, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, (int(c0[0]), int(c0[1])), 3, (0, 230, 0), -1)

    _, cf = valid_pairs[-1]
    elf = ellipses[valid_pairs[-1][0]]
    ec  = (0, 0, 220) if ruptured_at is not None else (0, 220, 220)
    if elf:
        cv2.ellipse(canvas, (int(elf['center'][0]), int(elf['center'][1])),
                    (int(elf['axes'][0]), int(elf['axes'][1])), elf['angle'], 0, 360, ec, 2, cv2.LINE_AA)
    cv2.circle(canvas, (int(cf[0]), int(cf[1])), 3, ec, -1)

    label = f"GUV {guv_id}  {'RUPTURED @F'+str(ruptured_at) if ruptured_at is not None else 'SURVIVED'}"
    cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ec, 1, cv2.LINE_AA)

    # ── Right: ring score plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ring_scores, 'b-', lw=1, alpha=0.8, label='Ring score')
   
    # Add Pulse Line
    if pulse_frame is not None:
        ax.axvline(pulse_frame, color='gray', ls='--', lw=1.5, label='Pulse', zorder=0)
   
    ax.axhline(getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0), color='orange', ls='--', lw=1, label='Rupture threshold')
    
    if ruptured_at is not None:
        ax.axvline(ruptured_at, color='red', ls=':', lw=1, label=f'Rupture F{ruptured_at}')
    ax.set_xlabel("Frame"); ax.set_ylabel("Ring quality score")
    ax.set_title(f"GUV {guv_id}")
    ax.legend(fontsize=7); fig.tight_layout()

    score_path = os.path.join(score_folder, f"_score_plot_GUV_{guv_id}.png")
    fig.savefig(score_path, dpi=120); plt.close(fig)

    # Combine side-by-side (resize score plot to match canvas height)
    score_img = cv2.imread(score_path)
    if score_img is not None:
        th = canvas.shape[0]
        tw = int(score_img.shape[1] * th / score_img.shape[0])
        score_img = cv2.resize(score_img, (tw, th))
        combined  = np.hstack([canvas, score_img])
        cv2.imwrite(os.path.join(track_folder, f"{experiment_name}_track_GUV_{guv_id}.png"), combined)
        # Score plot is kept in score_folder — not deleted
    else:
        cv2.imwrite(os.path.join(track_folder, f"{experiment_name}_track_GUV_{guv_id}.png"), canvas)


def style_image(frame, time_label, microns_per_pixel, scale_bar_microns):
    img8  = _convert_to_8bit_gray(frame)
    out   = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    out[:, :, 0] = 0; out[:, :, 1] = 0; out[:, :, 2] = img8
    cv2.putText(out, time_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    if scale_bar_microns > 0 and microns_per_pixel > 0:
        bar = int(scale_bar_microns / microns_per_pixel)
        h, w = out.shape[:2]
        cv2.line(out, (w-30-bar, h-30), (w-30, h-30), (255,255,255), 5)
        cv2.putText(out, f"{scale_bar_microns} um", (w-30-bar, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return out


def export_track_video(output_folder: str, experiment_name: str, guv_id: str,
                       roi_stack: np.ndarray, centers: List[Optional[Tuple]],
                       ellipses: List[Optional[Dict]], fps: float = 10.0):
    """
    Generates an AVI video of the tracking process.
    Draws the live contour (yellow) and the historical trajectory (green).
    """
    valid_pairs = [(i, c) for i, c in enumerate(centers) if c is not None]
    if not valid_pairs:
        return

    # Extract dimensions from the first valid frame
    first_i = valid_pairs[0][0]
    frame = roi_stack[first_i]
    if frame is None:
        return

    h, w = frame.shape[:2]
    vid_path = os.path.join(output_folder, f"{experiment_name}_track_GUV_{guv_id}.avi")

    # MJPG codec is highly compatible for AVI containers
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    valid_pts = []

    for i, (center, el) in enumerate(zip(centers, ellipses)):
        img = roi_stack[i]
        if img is None: continue

        # Use the improved contrast function here
        img8 = _convert_to_8bit_gray(img)
        canvas = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        if center is not None and el is not None:
            valid_pts.append(center)
            cv2.ellipse(canvas, (int(el['center'][0]), int(el['center'][1])),
                        (int(el['axes'][0]), int(el['axes'][1])), el['angle'], 0, 360, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw the historical trajectory path (Green)
        if len(valid_pts) > 1:
            pts_array = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_array], False, (0, 230, 0), 1, cv2.LINE_AA)

        # Add a frame counter overlay
        cv2.putText(canvas, f"Frame: {i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(canvas)

    out.release()

def export_full_stack_videos(track_folder: str, mask_folder: str, experiment_name: str,
                             roi_stack: np.ndarray, raw_results: list, 
                             circles: list[dict], fps: float = 10.0):
    """
    Generates consolidated videos by recalculating masks on-the-fly.
    This prevents the MemoryError seen during data transfer.
    """
    h, w = roi_stack[0].shape[:2]
    n_frames = roi_stack.shape[0]
    
    track_path = os.path.join(track_folder, f"{experiment_name}_ALL_TRACKING.avi")
    mask_path = os.path.join(mask_folder, f"{experiment_name}_ALL_MASKS.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out_track = cv2.VideoWriter(track_path, fourcc, fps, (w, h))
    out_mask = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))

    # Constants needed for on-the-fly masks (pulling from config)
    mem_hw = cfg.MEMBRANE_FIXED_HALF_WIDTH
    bg_buf = cfg.BG_BUFFER_PIXELS
    bg_w   = cfg.BG_RING_WIDTH_PIXELS

    for i in tqdm(range(n_frames), desc="Rendering consolidated videos"):
        frame = roi_stack[i]
        if frame is None: continue

        img8 = _convert_to_8bit_gray(frame)
        canvas_track = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        canvas_mask = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(canvas_mask)
        
        for idx, res in enumerate(raw_results):
            if res is None: continue
            tr = res.get('tracking', {})
            gid = circles[idx]['id']
            
            el = tr['ellipses'][i]
            if el is None: continue

            # --- Draw Tracking Video ---
            cv2.ellipse(canvas_track, (int(el['center'][0]), int(el['center'][1])),
                        (int(el['axes'][0]), int(el['axes'][1])), el['angle'], 
                        0, 360, (0, 255, 255), 2, cv2.LINE_AA)

            # --- Generate Masks on-the-fly to save memory ---
            inner_m, mem_m, bg_m = generate_vectorized_masks(
                (h, w), el['center'], el['axes'], el['angle'],
                mem_hw, bg_buf, bg_w
            )
            
            if inner_m is not None: overlay[inner_m] = (201, 87, 188) # Purple
            if mem_m is not None:   overlay[mem_m]   = (82, 0, 249)   # Pink
            if bg_m is not None:    overlay[bg_m]    = (114, 48, 19)  # Dark Blue

        # Finalize Mask Video Frame
        alpha = getattr(cfg, 'MASK_VIZ_OVERLAY_ALPHA', 0.6)
        cv2.addWeighted(canvas_mask, alpha, overlay, 1.0 - alpha, 0, canvas_mask)
        
        # Draw IDs on top of both
        for idx, res in enumerate(raw_results):
            if res is None: continue
            center = res.get('tracking', {}).get('centers', [None]*n_frames)[i]
            if center is not None:
                txt_pos = (int(center[0]), int(center[1]))
                gid = circles[idx]['id']
                cv2.putText(canvas_track, f"ID:{gid}", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(canvas_mask, f"ID:{gid}", txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_track.write(canvas_track)
        out_mask.write(canvas_mask)

    out_track.release()
    out_mask.release()
    
# -------------------------------------------------------------------
# --- 10. ACTIN CORTEX ANALYSIS ---
# -------------------------------------------------------------------

def extract_actin_traces(
        actin_stack: np.ndarray,
        n_frames: int,
        mem_masks: list,
        inner_masks: list,
        cortex_hw: int,
        centers: list,
        ellipses: list,
        img_shape: tuple,
        bg_buffer: int,
        bg_width: int,
) -> dict:
    """
    Extracts per-frame actin intensity from the C2 channel for one GUV.

    Two regions are sampled:
      - Cortex  : a thin ring at the membrane position
                  (re-generated with *cortex_hw* so it can differ from the
                  dye-channel membrane-half-width).
      - Lumen   : the full interior of the GUV.

    Returns
    -------
    dict with keys:
        'cortex_trace'      – mean intensity in the cortex ring  (n_frames,)
        'lumen_trace'       – mean intensity in the lumen        (n_frames,)
        'bg_trace'          – median background (reused from dye pipeline)
        'enrichment_ratio'  – cortex_net / lumen_net per frame
    """
    cortex_tr = np.full(n_frames, np.nan)
    lumen_tr  = np.full(n_frames, np.nan)
    bg_tr     = np.full(n_frames, np.nan)

    for i in range(n_frames):
        el = ellipses[i]
        if el is None or actin_stack is None:
            continue

        # Re-generate cortex mask with actin-specific half-width
        inner_m, cortex_m, bg_m = generate_vectorized_masks(
            img_shape,
            el['center'],
            el['axes'],
            el['angle'],
            cortex_hw,
            bg_buffer,
            bg_width,
        )

        frame = actin_stack[i]
        if frame is None:
            continue

        if cortex_m is not None and cortex_m.any():
            cortex_tr[i] = float(np.mean(frame[cortex_m]))
        if inner_m is not None and inner_m.any():
            lumen_tr[i] = float(np.mean(frame[inner_m]))
        if bg_m is not None and bg_m.any():
            bg_tr[i] = float(np.median(frame[bg_m]))

    # Net signals (background-subtracted)
    with np.errstate(divide='ignore', invalid='ignore'):
        net_cortex = cortex_tr - bg_tr
        net_lumen  = lumen_tr  - bg_tr
        enrichment = np.where(net_lumen > 0, net_cortex / net_lumen, np.nan)

    return {
        'cortex_trace':     cortex_tr,
        'lumen_trace':      lumen_tr,
        'bg_trace':         bg_tr,
        'enrichment_ratio': enrichment,
    }


def normalize_actin_curves(
        actin_data_list: list,
        pulse_frame: int,
        time_array: np.ndarray,
) -> dict:
    """
    Normalise actin traces to their pre-pulse baseline (first 5 frames)
    and align to t=0 at the pulse frame.

    Parameters
    ----------
    actin_data_list : list of dicts from extract_actin_traces(), one per GUV
    pulse_frame     : frame index of the electroporation pulse
    time_array      : absolute timestamps (s)

    Returns
    -------
    dict with aligned arrays and GUV-averaged traces
    """
    cortex_norm_list    = []
    lumen_norm_list     = []
    enrichment_al_list  = []
    pre_enrichment_list = []   # scalar pre-pulse baseline per GUV

    # Use up to 5 frames strictly before the pulse for the baseline.
    # If pulse_frame == 0 there is no pre-pulse window; fall back to
    # the first post-pulse frame so the ratio still makes dimensional sense.
    safe_pre_end   = pulse_frame if pulse_frame > 0 else 1
    safe_pre_start = max(0, safe_pre_end - 5)

    for ad in actin_data_list:
        cortex = np.asarray(ad['cortex_trace'], float)
        lumen  = np.asarray(ad['lumen_trace'],  float)
        bg     = np.asarray(ad['bg_trace'],     float)
        enrich = np.asarray(ad['enrichment_ratio'], float)

        # Baseline net values
        c0 = float(np.nanmean((cortex - bg)[safe_pre_start:safe_pre_end]))
        l0 = float(np.nanmean((lumen  - bg)[safe_pre_start:safe_pre_end]))

        with np.errstate(divide='ignore', invalid='ignore'):
            c_norm = (cortex - bg) / c0 if c0 != 0 else np.full_like(cortex, np.nan)
            l_norm = (lumen  - bg) / l0 if l0 != 0 else np.full_like(lumen,  np.nan)

        # Pre-pulse mean enrichment ratio (the actual "before" value for the bar chart)
        pre_enrich = float(np.nanmean(enrich[safe_pre_start:safe_pre_end]))
        pre_enrichment_list.append(pre_enrich)

        cortex_norm_list.append(c_norm[pulse_frame:])
        lumen_norm_list .append(l_norm[pulse_frame:])
        enrichment_al_list.append(enrich[pulse_frame:])

    t_aligned = time_array[pulse_frame:] - time_array[pulse_frame]

    cortex_arr = np.array(cortex_norm_list)
    lumen_arr  = np.array(lumen_norm_list)
    enrich_arr = np.array(enrichment_al_list)

    return {
        't_aligned':          t_aligned,
        'cortex_norm':        cortex_arr,
        'lumen_norm':         lumen_arr,
        'enrichment_aligned': enrich_arr,
        'avg_cortex':         np.nanmean(cortex_arr,  axis=0),
        'avg_lumen':          np.nanmean(lumen_arr,   axis=0),
        'avg_enrichment':     np.nanmean(enrich_arr,  axis=0),
        'pre_enrichment':     np.array(pre_enrichment_list),  # shape (n_guvs,)
    }


def plot_actin_analysis(
        aligned_actin: dict,
        valid_guv_ids: list,
        output_folder: str,
        experiment_name: str,
        smooth_sigma: float = 1.5,
):
    """
    Three-panel actin summary figure:
      Panel 1 – Normalised cortex intensity (per GUV + mean)
      Panel 2 – Normalised lumen intensity  (per GUV + mean)
      Panel 3 – Cortex enrichment ratio     (per GUV + mean)

    A vertical dashed line marks the pulse (t = 0).
    Optionally smoothed with a Gaussian for display.
    """
    from scipy.ndimage import gaussian_filter1d

    t             = aligned_actin['t_aligned']
    cortex_arr    = aligned_actin['cortex_norm']
    lumen_arr     = aligned_actin['lumen_norm']
    enrich_arr    = aligned_actin['enrichment_aligned']
    avg_cortex    = aligned_actin['avg_cortex']
    avg_lumen     = aligned_actin['avg_lumen']
    avg_enrich    = aligned_actin['avg_enrichment']

    def _smooth(x):
        if smooth_sigma > 0:
            finite = np.isfinite(x)
            if finite.sum() < 3:
                return x
            s = gaussian_filter1d(np.where(finite, x, 0), smooth_sigma)
            n = gaussian_filter1d(finite.astype(float), smooth_sigma)
            return np.where(n > 0, s / n, np.nan)
        return x

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = plt.cm.tab10.colors

    for i, (gid, c_row, l_row, e_row) in enumerate(
            zip(valid_guv_ids, cortex_arr, lumen_arr, enrich_arr)):
        col = palette[i % len(palette)]
        axes[0].plot(t, _smooth(c_row), color=col, alpha=0.4, lw=1,   label=f'GUV {gid}')
        axes[1].plot(t, _smooth(l_row), color=col, alpha=0.4, lw=1,   label=f'GUV {gid}')
        axes[2].plot(t, _smooth(e_row), color=col, alpha=0.4, lw=1,   label=f'GUV {gid}')

    axes[0].plot(t, _smooth(avg_cortex), 'k-', lw=2.5, label='Mean')
    axes[1].plot(t, _smooth(avg_lumen),  'k-', lw=2.5, label='Mean')
    axes[2].plot(t, _smooth(avg_enrich), 'k-', lw=2.5, label='Mean')

    for ax in axes:
        ax.axvline(0, color='crimson', ls='--', lw=1.5, label='Pulse', zorder=3)
        ax.axhline(1, color='gray',    ls=':',  lw=0.8)
        ax.set_xlabel('Time relative to pulse (s)')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_title('Cortex actin (C2, membrane mask)\nNormalised to pre-pulse')
    axes[0].set_ylabel('Normalised intensity (a.u.)')

    axes[1].set_title('Lumenal actin (C2, inner mask)\nNormalised to pre-pulse')
    axes[1].set_ylabel('Normalised intensity (a.u.)')

    axes[2].set_title('Cortex enrichment ratio\n(cortex_net / lumen_net)')
    axes[2].set_ylabel('Enrichment ratio')
    axes[2].axhline(1, color='tomato', ls='--', lw=1, alpha=0.6,
                    label='No enrichment (=1)')

    plt.suptitle(f'{experiment_name}  –  Actin cortex analysis', fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_folder, f'{experiment_name}_actin_cortex_analysis.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def export_actin_csv(
        aligned_actin: dict,
        valid_guv_ids: list,
        output_folder: str,
        experiment_name: str,
):
    """Save per-GUV aligned actin traces to a single tidy CSV."""
    t = aligned_actin['t_aligned']
    rows = [{'time_s': tv} for tv in t]
    df = pd.DataFrame(rows)

    for gid, c_row, l_row, e_row in zip(
            valid_guv_ids,
            aligned_actin['cortex_norm'],
            aligned_actin['lumen_norm'],
            aligned_actin['enrichment_aligned']):
        df[f'GUV_{gid}_cortex']     = c_row
        df[f'GUV_{gid}_lumen']      = l_row
        df[f'GUV_{gid}_enrichment'] = e_row

    df['avg_cortex']     = aligned_actin['avg_cortex']
    df['avg_lumen']      = aligned_actin['avg_lumen']
    df['avg_enrichment'] = aligned_actin['avg_enrichment']

    csv_path = os.path.join(output_folder,
                            f'{experiment_name}_actin_cortex_traces.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f', na_rep='NaN')
    return csv_path


def plot_actin_pre_post(
        aligned_actin: dict,
        valid_guv_ids: list,
        time_array_aligned: np.ndarray,
        output_folder: str,
        experiment_name: str,
        post_window_s: float = 60.0,
):
    """
    Paired bar chart comparing the cortex enrichment ratio BEFORE vs AFTER
    the electroporation pulse, per GUV.

    The "pre" value is the actual pre-pulse baseline stored in aligned_actin
    (computed in normalize_actin_curves from the frames just before the pulse).

    post_window_s : average the first N seconds of post-pulse data as "after".
    """
    t      = time_array_aligned           # starts at 0 (pulse = t=0)
    enrich = aligned_actin['enrichment_aligned']
    pre_vals = aligned_actin['pre_enrichment']   # shape (n_guvs,) — actual pre-pulse

    post_mask  = (t >= 0) & (t <= post_window_s)
    if not post_mask.any():
        post_mask = np.ones(len(t), bool)

    post_means = np.nanmean(enrich[:, post_mask], axis=1)

    x     = np.arange(len(valid_guv_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(valid_guv_ids) * 1.2), 5))
    ax.bar(x - width/2, pre_vals,   width, label='Pre-pulse baseline',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, post_means, width,
           label=f'Post-pulse  (first {post_window_s:.0f} s)',
           color='tomato', alpha=0.8)
    ax.axhline(1, color='gray', ls=':', lw=0.8, label='No enrichment (=1)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'GUV {g}' for g in valid_guv_ids], rotation=45, ha='right')
    ax.set_ylabel('Cortex enrichment ratio  (cortex_net / lumen_net)')
    ax.set_title(f'{experiment_name}\nCortex enrichment: pre vs post electroporation')
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_actin_pre_post_bar.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_actin_spatial_snapshot(
        actin_stack: np.ndarray,
        ellipses_per_guv: list,   # list of (guv_id, frame_idx, ellipse_dict)
        output_folder: str,
        experiment_name: str,
        microns_per_pixel: float = 1.0,
):
    """
    Montage of per-GUV radial intensity profiles from the actin channel
    at the pre-pulse frame, showing membrane peak vs lumenal actin.

    ellipses_per_guv : [(guv_id, frame_idx, el_dict), ...]
    """
    n = len(ellipses_per_guv)
    if n == 0 or actin_stack is None:
        return

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)

    for k, (gid, fi, el) in enumerate(ellipses_per_guv):
        ax = axes[k // ncols][k % ncols]
        frame = actin_stack[fi]
        if frame is None or el is None:
            ax.set_visible(False)
            continue

        cx, cy = el['center']
        avg_r  = int(sum(el['axes']) / 2.0)
        r_max  = int(avg_r * 1.6)

        h, w = frame.shape[:2]
        x_min = max(0, int(cx - r_max - 2))
        x_max = min(w, int(cx + r_max + 2))
        y_min = max(0, int(cy - r_max - 2))
        y_max = min(h, int(cy + r_max + 2))

        # Skip if we cannot fit a full radial profile inside the image
        if (cx - r_max < 0 or cx + r_max >= w or
                cy - r_max < 0 or cy + r_max >= h):
            ax.set_visible(False)
            continue

        local   = frame[y_min:y_max, x_min:x_max]
        profile = radial_profile_local(local, (cx - x_min, cy - y_min), r_max)

        r_um = np.arange(r_max) * microns_per_pixel
        mem_um = avg_r * microns_per_pixel

        ax.plot(r_um, profile, color='steelblue', lw=1.4)
        ax.axvline(mem_um, color='crimson', ls='--', lw=1.2, label='Membrane')
        ax.axvspan(0, mem_um, alpha=0.06, color='gold',  label='Lumen')
        ax.axvspan(mem_um, r_um[-1], alpha=0.06, color='tomato', label='Extracellular')
        ax.set_title(f'GUV {gid}  (frame {fi})', fontsize=9)
        ax.set_xlabel(r'Radius ($\mu$m)', fontsize=8)
        ax.set_ylabel('Intensity (AU)', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty panels
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.suptitle(f'{experiment_name} – Actin radial profiles (pre-pulse)',
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_actin_radial_profiles.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_dye_actin_overlay(
        aligned_dye: dict,
        aligned_actin: dict,
        valid_guv_ids: list,
        output_folder: str,
        experiment_name: str,
        smooth_sigma: float = 1.5,
):
    """
    Per-GUV dual-axis figure correlating dye efflux with actin cortex dynamics.

    Left y-axis  (blue)   – Normalised dye intensity (fractional retention)
    Right y-axis (orange) – Cortex enrichment ratio (cortex_net / lumen_net)

    One subplot per GUV, up to 4 per row.  A summary panel (last subplot)
    shows population means of both signals.

    Parameters
    ----------
    aligned_dye   : dict returned by normalize_and_align_curves()
    aligned_actin : dict returned by normalize_actin_curves()
    valid_guv_ids : list of GUV id strings in the same order as both arrays
    """
    from scipy.ndimage import gaussian_filter1d

    t_dye    = aligned_dye['t_aligned']
    t_act    = aligned_actin['t_aligned']

    dye_arr  = aligned_dye['all_curves_aligned']     # (n, T_dye)
    act_arr  = aligned_actin['enrichment_aligned']   # (n, T_act)

    avg_dye  = aligned_dye['average_curve']
    avg_act  = aligned_actin['avg_enrichment']

    def _sm(x):
        if smooth_sigma <= 0:
            return x
        finite = np.isfinite(x)
        if finite.sum() < 3:
            return x
        s = gaussian_filter1d(np.where(finite, x, 0.0), smooth_sigma)
        n = gaussian_filter1d(finite.astype(float),      smooth_sigma)
        return np.where(n > 0, s / n, np.nan)

    n_guvs = len(valid_guv_ids)
    # +1 panel for the population summary
    n_panels = n_guvs + 1
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    DYE_COLOR = '#1f77b4'    # blue
    ACT_COLOR = '#ff7f0e'    # orange

    def _draw_panel(ax, t_d, dye_row, t_a, act_row, title):
        ax2 = ax.twinx()

        ax .plot(t_d, _sm(dye_row), color=DYE_COLOR, lw=1.6, label='Dye (ret.)')
        ax2.plot(t_a, _sm(act_row), color=ACT_COLOR, lw=1.6, label='Cortex enrich.')

        ax .axvline(0, color='crimson', ls='--', lw=1.2, zorder=3)
        ax .axhline(1, color=DYE_COLOR, ls=':', lw=0.6, alpha=0.5)
        ax2.axhline(1, color=ACT_COLOR, ls=':', lw=0.6, alpha=0.5)

        ax .set_xlabel('Time relative to pulse (s)', fontsize=8)
        ax .set_ylabel('Dye retention (a.u.)',        fontsize=8, color=DYE_COLOR)
        ax2.set_ylabel('Cortex enrichment ratio',     fontsize=8, color=ACT_COLOR)
        ax .tick_params(axis='y', labelcolor=DYE_COLOR, labelsize=7)
        ax2.tick_params(axis='y', labelcolor=ACT_COLOR, labelsize=7)
        ax .set_title(title, fontsize=9)
        ax .grid(True, alpha=0.2)

        # Combined legend
        lines  = ax .get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=7, loc='upper right')

    # Per-GUV panels
    for i, gid in enumerate(valid_guv_ids):
        row_idx = i // ncols
        col_idx = i  % ncols
        ax = axes[row_idx][col_idx]

        dye_row = dye_arr[i] if i < len(dye_arr) else np.full_like(t_dye, np.nan)
        act_row = act_arr[i] if i < len(act_arr) else np.full_like(t_act, np.nan)

        _draw_panel(ax, t_dye, dye_row, t_act, act_row, f'GUV {gid}')

    # Population summary panel (last)
    summary_idx  = n_guvs
    sr, sc       = summary_idx // ncols, summary_idx % ncols
    ax_sum       = axes[sr][sc]
    _draw_panel(ax_sum, t_dye, avg_dye, t_act, avg_act,
                'Population mean ± shade')

    # Add shaded ±1 SD to the summary panel
    ax_sum2 = ax_sum.get_shared_x_axes()   # twin already created inside _draw_panel
    # Recreate twin for shading (ax_sum.twinx() would add a third axis; shade on existing)
    if len(dye_arr) > 1:
        dye_sd  = np.nanstd(dye_arr,  axis=0)
        act_sd  = np.nanstd(act_arr,  axis=0)
        sm_dye  = _sm(avg_dye)
        sm_act  = _sm(avg_act)
        ax_sum .fill_between(t_dye,
                             _sm(avg_dye - dye_sd), _sm(avg_dye + dye_sd),
                             color=DYE_COLOR, alpha=0.15)
        # The twin axis was created inside _draw_panel; we can't reach it directly.
        # Plot the actin SD on top of ax_sum with a secondary colour for clarity.
        ax_sum.fill_between(t_act,
                            _sm(avg_act - act_sd), _sm(avg_act + act_sd),
                            color=ACT_COLOR, alpha=0.12, transform=ax_sum.transData)

    # Hide unused panels
    for k in range(n_panels, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.suptitle(f'{experiment_name}  –  Dye efflux vs Actin cortex dynamics',
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_actin_overlay.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def redraw_score_plots(
        guv_results: dict,
        circles: list,
        roi_mmap_info: tuple,
        pulse_frame: int,
        track_folder: str,
        score_folder: str,
        experiment_name: str,
):
    """
    Re-exports every per-GUV tracking PNG with the confirmed pulse-frame
    marker.  Called from main() after PULSE_FRAME is finalised so all score
    plots are consistent, regardless of what the filename tag said.
    """
    roi_path, roi_shape, roi_dtype = roi_mmap_info
    roi_stack = np.memmap(roi_path, dtype=roi_dtype, mode='r', shape=roi_shape)

    for i, (result, qe) in enumerate(guv_results['raw_results']):
        if result is None:
            continue
        guv_id    = qe['guv_id']
        tr        = result['tracking']
        rup       = qe.get('ruptured_at_frame')
        export_track_visualization(
            track_folder, score_folder, experiment_name, guv_id,
            roi_stack,
            tr['centers'], tr['ellipses'], tr['ring_scores'],
            rup,
            pulse_frame=pulse_frame,   # ← confirmed value
        )

    del roi_stack


def plot_tracking_metrics(df: pd.DataFrame, time_array: np.ndarray, 
                          output_folder: str, experiment_name: str, 
                          um_per_px: float = 1.0, pulse_time=None):
    """
    Generates a 3-panel figure plotting Size (Radius), Deformation (Eccentricity),
    and Mean Square Displacement (MSD) for each tracked GUV.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    guv_ids = df['guv_id'].unique()
    
    for gid in guv_ids:
        guv_data = df[df['guv_id'] == gid].sort_values('frame')
        frames = guv_data['frame'].values
        
        valid_t_idx = [f for f in frames if f < len(time_array)]
        if not valid_t_idx:
            continue
        
        t_vals = time_array[valid_t_idx]
        
        # 1. Size Plot
        r_um = guv_data['radius'].values[:len(valid_t_idx)] * um_per_px
        axes[0].plot(t_vals, r_um, label=f'GUV {gid}', alpha=0.8)
        
        # 2. Deformation Plot
        if 'eccentricity' in guv_data.columns:
            ecc = guv_data['eccentricity'].values[:len(valid_t_idx)]
            axes[1].plot(t_vals, ecc, label=f'GUV {gid}', alpha=0.8)
        
        # 3. MSD Plot
        x = guv_data['x'].values * um_per_px
        y = guv_data['y'].values * um_per_px
        
        max_lag = len(x) // 4
        if max_lag > 1:
            msd = []
            for lag in range(1, max_lag + 1):
                dx = x[lag:] - x[:-lag]
                dy = y[lag:] - y[:-lag]
                msd.append(np.nanmean(dx**2 + dy**2))
            
            axes[2].plot(range(1, max_lag + 1), msd, label=f'GUV {gid}', alpha=0.8)

    axes[0].set_title('GUV Size over Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(r'Radius ($\mu$m)')
    
    axes[1].set_title('Deformation (Eccentricity)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Eccentricity (0=Circle)')
    axes[1].set_ylim(-0.05, 1.05)
    
    axes[2].set_title('Mean Square Displacement')
    axes[2].set_xlabel('Frame Lag (frames)')
    axes[2].set_ylabel(r'MSD ($\mu$m$^2$)')
    
    # Add Pulse lines to Size and Deformation plots
    if pulse_time is not None:
        for i in [0, 1]:
            axes[i].axvline(pulse_time, color='k', ls='--', alpha=0.5, label='Pulse')
            axes[i].legend(fontsize=8)
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{experiment_name}_tracking_metrics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)