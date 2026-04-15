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
# --- 3. MASK HELPERS ---
# -------------------------------------------------------------------

def _convert_to_8bit_gray(image: np.ndarray) -> np.ndarray:
    img = image[:, :, 0] if image.ndim == 3 else image
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
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

# guv_analysis_utils.py

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


# -------------------------------------------------------------------
# --- 8. PARALLEL WORKER ---
# -------------------------------------------------------------------

def process_single_guv(guv_id: str,
                        initial_center: Tuple[int, int],
                        initial_radius: int,
                        is_first_guv: bool,
                        roi_mmap_info: Tuple,
                        dye_mmap_info: Tuple,
                        n_frames: int) -> Tuple:
    """
    Full analysis pipeline for a single GUV using frame-by-frame tracking.
    """
    detect_bright = (getattr(cfg, 'MEMBRANE_DETECTION_MODE', 'LABELED').upper() != 'UNLABELED')

    # Reconstruct the memmaps in read-only mode for this specific worker process
    roi_path, roi_shape, roi_dtype = roi_mmap_info
    roi_stack = np.memmap(roi_path, dtype=roi_dtype, mode='r', shape=roi_shape)
    
    if dye_mmap_info[0] is not None:
        dye_path, dye_shape, dye_dtype = dye_mmap_info
        dye_stack = np.memmap(dye_path, dtype=dye_dtype, mode='r', shape=dye_shape)
    else:
        dye_stack = None

    # ── Track GUV across all frames ────────────────────────────────────────
    tracking = track_guv_across_frames(
        roi_stack        = roi_stack,
        n_frames         = n_frames,
        initial_center   = initial_center,
        initial_radius   = initial_radius,
        search_window_factor    = getattr(cfg, 'TRACKING_SEARCH_WINDOW_FACTOR', 0.7),
        grid_step_factor        = getattr(cfg, 'TRACKING_GRID_STEP_FACTOR', 0.15),
        search_factor           = cfg.MEMBRANE_SEARCH_FACTOR,
        membrane_half_width     = cfg.MEMBRANE_FIXED_HALF_WIDTH,
        detect_bright           = detect_bright,
        bg_buffer               = cfg.BG_BUFFER_PIXELS,
        bg_width                = cfg.BG_RING_WIDTH_PIXELS,
        max_radius_change_factor   = getattr(cfg, 'TRACKING_MAX_RADIUS_CHANGE_FACTOR', 0.20),
        rupture_score_threshold    = getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0),
        rupture_consecutive_fails  = getattr(cfg, 'RUPTURE_DETECTION_CONSECUTIVE_FAILS', 3),
    )

    ruptured_at = tracking['ruptured_at']
    n_valid     = tracking['n_valid_frames']

    quality_entry = {
        'guv_id':            guv_id,
        'initial_radius':    initial_radius,
        'failed':            ruptured_at is not None,
        'ruptured_at_frame': ruptured_at,
        'n_valid_frames':    n_valid,
        'mean_ring_score':   float(np.mean([s for s in tracking['ring_scores'] if s > 0])) if n_valid > 0 else 0.0,
        'comments':          f"RUPTURED_AT_F{ruptured_at}" if ruptured_at is not None else 'OK',
    }

    # ── Export first-frame mask visualisation ──────────────────────────────
    if cfg.EXPORT_MASK_VISUALIZATION and (quality_entry['failed'] or is_first_guv):
        if tracking['inner_masks'][0] is not None:
            roi_f0 = roi_stack[0].copy()
            if roi_f0 is not None:
                mem_m = tracking['mem_masks'][0]
                if mem_m is None:
                    c0 = tracking['centers'][0] if tracking['centers'][0] is not None else initial_center
                    el0 = tracking['ellipses'][0]
                    ax0 = el0['axes'] if el0 is not None else (initial_radius, initial_radius)
                    ang0 = el0['angle'] if el0 is not None else 0.0
                    
                    _, mem_m, _ = generate_vectorized_masks(
                        roi_f0.shape, c0, ax0, ang0,
                        cfg.MEMBRANE_FIXED_HALF_WIDTH,
                        cfg.BG_BUFFER_PIXELS,
                        cfg.BG_RING_WIDTH_PIXELS
                    )
                viz = create_mask_visualization(roi_f0, tracking['inner_masks'][0],
                                                mem_m, tracking['bg_masks'][0],
                                                cfg.MASK_VIZ_OVERLAY_ALPHA)
                try:
                    cv2.imwrite(os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                               f"{cfg.EXPERIMENT_BASE_NAME}_mask_viz_GUV_{guv_id}.png"), viz)
                except Exception:
                    pass

    # ── Export tracking trajectory ─────────────────────────────────────────
    if getattr(cfg, 'EXPORT_TRACK_VISUALIZATION', True):
        try:
            export_track_visualization(
                cfg.OUTPUT_IMAGE_FOLDER, cfg.EXPERIMENT_BASE_NAME, guv_id,
                roi_stack, tracking['centers'], tracking['ellipses'],
                tracking['ring_scores'], ruptured_at
            )
        except Exception:
            pass
        
    if getattr(cfg, 'EXPORT_TRACK_VIDEO', False):
        try:
            export_track_video(
                cfg.OUTPUT_IMAGE_FOLDER, cfg.EXPERIMENT_BASE_NAME, guv_id,
                roi_stack, tracking['centers'], tracking['ellipses'],
                fps=getattr(cfg, 'VIDEO_EXPORT_FPS', 10.0)
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to export video for GUV {guv_id}: {e}")
    
    if getattr(cfg, 'TRACKING_ONLY_MODE', False):
        result = {
            'intensity_trace':  None,
            'background_trace': None,
            'jump_frame':       None,
            'tracking':         tracking,
        }
        return result, quality_entry

    # ── Extract intensity traces ───────────────────────────────────────────
    intensity_trace  = get_intensity_trace_tracked(dye_stack, n_frames, tracking['inner_masks'], method='mean')
    background_trace = get_intensity_trace_tracked(dye_stack, n_frames, tracking['bg_masks'], method='median')
    trace_for_jump = np.where(np.isnan(intensity_trace), 0.0, intensity_trace)

    if np.all(trace_for_jump == 0) or np.all(np.isnan(intensity_trace)):
        return None, quality_entry

    # ── Detect dye-entry jump ──────────────────────────────────────────────
    try:
        jump_frame = detect_intensity_jump_robust(
            trace_for_jump,
            threshold_percent=getattr(cfg, 'JUMP_THRESHOLD_PERCENT', 0.20)
        )
    except Exception:
        jump_frame = 0

    result = {
        'intensity_trace':  intensity_trace,
        'background_trace': background_trace,
        'jump_frame':       jump_frame,
        'tracking':         tracking,
    }
    return result, quality_entry


# -------------------------------------------------------------------
# --- 9. JUMP DETECTION ---
# -------------------------------------------------------------------

def detect_intensity_jump_robust(trace: np.ndarray,
                                  threshold_percent: float = 0.20) -> int:
    """
    Detects the 'toe' of the dye-entry step function.

    Uses threshold-crossing rather than gradient peak so that baseline
    noise does not trigger a false positive at frame 0.
    """
    if len(trace) < 5:
        return 0

    smoothed = gaussian_filter1d(trace, sigma=2)
    mn, mx   = smoothed.min(), smoothed.max()
    if (mx - mn) < 20:
        return 0

    threshold    = mn + threshold_percent * (mx - mn)
    search_end   = int(len(smoothed) * 0.95)
    crossings    = np.where(smoothed[:search_end] > threshold)[0]

    if len(crossings) == 0:
        return 0
    
    first     = crossings[0]
    look_back = min(first, 20)
    segment   = smoothed[first - look_back: first + 1]
    return max(0, first - look_back + int(np.argmin(segment)))

def find_closest_frame(time_array: np.ndarray, target: float) -> Tuple[int, float]:
    idx = int(np.abs(time_array - target).argmin())
    return idx, float(time_array[idx])


# -------------------------------------------------------------------
# --- 10. VISUALIZATION HELPERS ---
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


def export_track_visualization(output_folder, experiment_name, guv_id,
                                roi_stack, centers, ellipses, ring_scores,
                                ruptured_at):
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
    ax.axhline(getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0), color='orange', ls='--', lw=1, label='Rupture threshold')
    if ruptured_at is not None:
        ax.axvline(ruptured_at, color='red', ls=':', lw=1, label=f'Rupture F{ruptured_at}')
    ax.set_xlabel("Frame"); ax.set_ylabel("Ring quality score")
    ax.set_title(f"GUV {guv_id}")
    ax.legend(fontsize=7); fig.tight_layout()

    score_path = os.path.join(output_folder, f"_score_plot_GUV_{guv_id}.png")
    fig.savefig(score_path, dpi=120); plt.close(fig)

    # Combine side-by-side (resize score plot to match canvas height)
    score_img = cv2.imread(score_path)
    if score_img is not None:
        th = canvas.shape[0]
        tw = int(score_img.shape[1] * th / score_img.shape[0])
        score_img = cv2.resize(score_img, (tw, th))
        combined  = np.hstack([canvas, score_img])
        cv2.imwrite(os.path.join(output_folder, f"{experiment_name}_track_GUV_{guv_id}.png"), combined)
        os.remove(score_path)
    else:
        cv2.imwrite(os.path.join(output_folder, f"{experiment_name}_track_GUV_{guv_id}.png"), canvas)


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
        if img is None:
            continue

        # Apply percentile clipping for visibility
        p_low, p_high = np.percentile(img, (1, 99.5))
        clipped = np.clip(img, p_low, p_high)
        img8 = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).astype(np.uint8)
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

def export_full_stack_videos(output_folder: str, experiment_name: str, 
                             roi_stack: np.ndarray, raw_results: list, 
                             circles: list[dict], fps: float = 10.0):
    """
    Generates two consolidated videos for all GUVs:
    1. Tracking: Contours and IDs overlaid on ROI frames.
    2. Masks: Inner, membrane, and background masks overlaid on ROI frames.
    """
    h, w = roi_stack[0].shape[:2]
    n_frames = roi_stack.shape[0]
    
    track_path = os.path.join(output_folder, f"{experiment_name}_ALL_TRACKING.avi")
    mask_path = os.path.join(output_folder, f"{experiment_name}_ALL_MASKS.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    out_track = cv2.VideoWriter(track_path, fourcc, fps, (w, h))
    out_mask = cv2.VideoWriter(mask_path, fourcc, fps, (w, h))

    # Wrap frame loop in tqdm for a progress bar in the console
    for i in range(n_frames):
        frame = roi_stack[i]
        if frame is None: continue

        img8 = _convert_to_8bit_gray(frame)
        canvas_track = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        
        # Prepare Mask Frame
        canvas_mask = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(canvas_mask)
        
        for idx, res in enumerate(raw_results):
            if res is None: continue
            tr = res.get('tracking', {})
            gid = circles[idx]['id']
            
            # --- Draw Tracking ---
            center = tr['centers'][i]
            el = tr['ellipses'][i]
            if center is not None and el is not None:
                cv2.ellipse(canvas_track, (int(el['center'][0]), int(el['center'][1])),
                            (int(el['axes'][0]), int(el['axes'][1])), el['angle'], 
                            0, 360, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas_track, f"ID:{gid}", (int(center[0]), int(center[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # --- Draw Masks ---
            inner = tr['inner_masks'][i]
            mem = tr['mem_masks'][i]
            bg = tr['bg_masks'][i]
            
            if inner is not None: overlay[inner] = (201, 87, 188) # Purple
            if mem is not None:   overlay[mem]   = (82, 0, 249)   # Pink
            if bg is not None:    overlay[bg]    = (114, 48, 19)  # Dark Blue
        
        # Apply mask overlay
        alpha = getattr(cfg, 'MASK_VIZ_OVERLAY_ALPHA', 0.6)
        cv2.addWeighted(canvas_mask, alpha, overlay, 1.0 - alpha, 0, canvas_mask)
        
        out_track.write(canvas_track)
        out_mask.write(canvas_mask)

    out_track.release()
    out_mask.release()
    
def plot_tracking_metrics(df: pd.DataFrame, time_array: np.ndarray, 
                          output_folder: str, experiment_name: str, 
                          um_per_px: float = 1.0):
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
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    plt.tight_layout()
    out_path = os.path.join(output_folder, f"{experiment_name}_tracking_metrics.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)