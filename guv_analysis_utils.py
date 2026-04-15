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
    """Reads 'finterval' from the ImageJ TIFF ImageDescription tag."""
    if not file_paths:
        return None, None
    try:
        with tifffile.TiffFile(file_paths[0]) as tif:
            tag = tif.pages[0].tags.get('ImageDescription')
            if tag is None:
                return None, None
            match = re.search(r'finterval=([0-9.]+)', tag.value)
            if match:
                dt = float(match.group(1))
                return np.arange(len(file_paths)) * dt, dt
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

def dyn_model_4param(t, I_offset, A, tau, D):
    return I_offset + A * (1 - np.exp(-t / tau)) + D * t

def dyn_model_5param(t, Af, A1, tau1, A2, tau2):
    return Af - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)


# -------------------------------------------------------------------
# --- 3. MASK HELPERS ---
# -------------------------------------------------------------------

def _convert_to_8bit_gray(image: np.ndarray) -> np.ndarray:
    img = image[:, :, 0] if image.ndim == 3 else image
    if img.dtype != np.uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

def create_circular_mask(shape, center, radius) -> np.ndarray:
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    return (X - center[0])**2 + (Y - center[1])**2 <= radius**2

def create_annular_mask(shape, center, r_inner, r_outer) -> np.ndarray:
    h, w = shape[:2]
    Y, X = np.ogrid[:h, :w]
    d2 = (X - center[0])**2 + (Y - center[1])**2
    return (d2 >= r_inner**2) & (d2 <= r_outer**2)


# -------------------------------------------------------------------
# --- 4. RADIAL PROFILE ---
# -------------------------------------------------------------------

def radial_profile(frame: np.ndarray, center: Tuple[int, int],
                   r_max: int, n_angles: int = 72) -> np.ndarray:
    """
    Median radial profile from centre out to r_max.

    Using the median across angles makes the profile robust to other
    GUVs, dirt, or anything that only affects a subset of directions.

    Parameters
    ----------
    frame    : greyscale image (any dtype)
    center   : (x, y) in pixel coordinates
    r_max    : profile length in pixels
    n_angles : number of radial lines to sample

    Returns
    -------
    1-D array of length r_max
    """
    cx, cy   = center
    img      = frame[:, :, 0] if frame.ndim == 3 else frame
    thetas   = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii    = np.arange(r_max)

    # Calculate precise floating-point coordinates
    xs = cx + radii[:, None] * np.cos(thetas)[None, :]
    ys = cy + radii[:, None] * np.sin(thetas)[None, :]

    # Map coordinates for bilinear sub-pixel interpolation
    coords = np.vstack((ys.flatten(), xs.flatten()))
    sampled = map_coordinates(img, coords, order=1, mode='nearest').reshape(r_max, n_angles)

    return np.median(sampled, axis=1).astype(float)


# -------------------------------------------------------------------
# --- 5. RING-SCORE GRID SEARCH  (replaces NCC template matching)
# -------------------------------------------------------------------

def _ring_score_from_profile(profile: np.ndarray, expected_r: int,
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
                     detect_bright: bool, n_angles: int = 24) -> float:
    """
    Fast ring-score at a single candidate centre.
    Uses sparse angle sampling (24) for speed during the grid search.
    Returns 0.0 if the centre is too close to the image boundary.
    """
    r_max = int(expected_r * (1.0 + search_factor) + 5)
    h, w  = frame.shape[:2]
    if cx - r_max < 0 or cx + r_max >= w or cy - r_max < 0 or cy + r_max >= h:
        return 0.0
    prof = radial_profile(frame, (cx, cy), r_max, n_angles)
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

    This approach is invariant to:
    • Translation  – handled by searching a spatial grid
    • Size changes – the radius estimate is updated separately each frame
    • Appearance   – we only measure the ring contrast, not pixel values

    Parameters
    ----------
    frame               : current ROI frame (any greyscale dtype)
    prev_center         : (x, y) centre estimate from the previous frame
    prev_radius         : radius estimate from the previous frame (pixels)
    search_window_factor: search circle radius = prev_radius × this factor
    search_factor       : ± fraction of prev_radius to accept as membrane
    detect_bright       : True → look for bright ring; False → dark ring
    grid_step_factor    : grid spacing = prev_radius × this factor

    Returns
    -------
    (best_center, best_score) — best_center is (x, y) in original coords
    """
    cx, cy  = prev_center
    r       = prev_radius
    h, w    = frame.shape[:2]

    search_r = max(8, int(r * search_window_factor))
    step     = max(2, int(r * grid_step_factor))

    best_score  = -1.0
    best_center = prev_center

    for dx in range(-search_r, search_r + step, step):
        for dy in range(-search_r, search_r + step, step):
            if dx * dx + dy * dy > search_r * search_r:
                continue
            ncx, ncy = cx + dx, cy + dy
            if ncx < 0 or ncx >= w or ncy < 0 or ncy >= h:
                continue
            score = _score_at_center(frame, ncx, ncy, r,
                                     search_factor, detect_bright,
                                     n_angles=72)
            if score > best_score:
                best_score  = score
                best_center = (ncx, ncy)

    return best_center, best_score


def detect_membrane_radius(frame: np.ndarray, center: Tuple[int, int],
                           expected_r: int, search_factor: float,
                           peak_min_dist: int, peak_min_prom: float,
                           detect_bright: bool) -> Tuple[int, bool]:
    """
    Find the membrane radius by locating the peak/valley in the full
    radial profile (360 angles for accuracy).

    Returns (detected_radius, success_flag).
    On failure, returns (expected_r, False).
    """
    r_max  = int(expected_r * (1.0 + search_factor) + 10)
    h, w   = frame.shape[:2]
    cx, cy = center
    if cx - r_max < 0 or cx + r_max >= w or cy - r_max < 0 or cy + r_max >= h:
        return expected_r, False

    prof   = radial_profile(frame, center, r_max, n_angles=72)
    smooth = gaussian_filter1d(prof, sigma=1) # sigma was 2 before

    lo = max(2, int(expected_r * (1.0 - search_factor)))
    hi = min(len(smooth) - 2, int(expected_r * (1.0 + search_factor)))
    if lo >= hi:
        return expected_r, False

    # For dark ring: invert so the valley becomes a peak
    search_sig = smooth if detect_bright else (smooth.max() - smooth)

    # Restrict find_peaks to the search window
    masked = np.full_like(search_sig, -np.inf)
    masked[lo:hi] = search_sig[lo:hi]

    prom_thresh = (masked[lo:hi].max() - masked[lo:hi].min()) * peak_min_prom
    try:
        peaks, props = find_peaks(
            masked,
            distance=peak_min_dist,
            prominence=max(1.0, prom_thresh)
        )
    except Exception:
        return expected_r, False

    if len(peaks) == 0:
        return expected_r, False

    # Pick the most prominent peak inside the window
    in_window = peaks[(peaks >= lo) & (peaks < hi)]
    if len(in_window) == 0:
        return expected_r, False

    proms = props['prominences'][(peaks >= lo) & (peaks < hi)]
    best  = in_window[np.argmax(proms)]
    return int(best), True


# -------------------------------------------------------------------
# --- 6. FRAME-BY-FRAME TRACKER ---
# -------------------------------------------------------------------

def track_guv_across_frames(
        roi_files: List[str],
        initial_center: Tuple[int, int],
        initial_radius: int,
        search_window_factor: float,
        grid_step_factor: float,
        search_factor: float,
        membrane_half_width: int,
        peak_min_dist: int,
        peak_min_prom: float,
        detect_bright: bool,
        bg_buffer: int,
        bg_width: int,
        max_radius_change_factor: float = 0.20,
        rupture_score_threshold: float  = 4.0,
        rupture_consecutive_fails: int  = 3,
) -> Dict[str, Any]:
    """
    Track a single GUV across all frames in *roi_files*.

    Per-frame pipeline
    ------------------
    1. Ring-score grid search  →  best centre candidate.
    2. Membrane radius detection via radial profile peak/valley finding.
    3. Radius-change clamp  →  rejects single-frame artefacts.
    4. Build inner (lumen), membrane, and background masks.
    5. Rupture check: if best ring-score < rupture_score_threshold for
       *rupture_consecutive_fails* frames in a row → declare ruptured.

    Why no template matching?
    -------------------------
    Template matching (NCC) requires appearance stability between frames.
    GUVs undergoing electroporation change their size, internal content,
    and contrast.  The ring-score approach only asks "is there a circular
    ring here?" — invariant to what's inside the GUV or how it is lit.

    Returns
    -------
    dict with keys:
        inner_masks, mem_masks, bg_masks   – per-frame boolean arrays (None after rupture)
        centers, radii, ring_scores        – per-frame tracking state
        ruptured_at                        – frame index or None
        n_valid_frames                     – count of successfully tracked frames
    """
    n = len(roi_files)
    inner_masks: List[Optional[np.ndarray]] = [None] * n
    mem_masks:   List[Optional[np.ndarray]] = [None] * n
    bg_masks:    List[Optional[np.ndarray]] = [None] * n
    centers:     List[Optional[Tuple]]      = [None] * n
    radii:       List[Optional[int]]        = [None] * n
    ring_scores: List[float]                = [0.0]  * n

    cx, cy = initial_center
    r      = initial_radius
    
    vx, vy = 0.0, 0.0

    consecutive_low = 0
    ruptured_at     = None
    n_valid         = 0

    consecutive_low = 0
    ruptured_at     = None
    n_valid         = 0

    for i, fp in enumerate(roi_files):
        frame = cv2.imread(fp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is None:
            consecutive_low += 1
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                break
            continue

        # ── 1. Grid search for new centre (with velocity prediction) ──────
        pred_cx = int(cx + vx)
        pred_cy = int(cy + vy)

        (new_cx, new_cy), score = find_best_guv_center(
            frame, (pred_cx, pred_cy), r,
            search_window_factor, search_factor,
            detect_bright, grid_step_factor
        )
        ring_scores[i] = score

        # ── 2. Rupture check ───────────────────────────────────────────────
        if score < rupture_score_threshold:
            consecutive_low += 1
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                break
            # Use fallback: keep previous state for this frame but record it
        else:
            consecutive_low = 0

        # ── 3. Radius detection ────────────────────────────────────────────
        detected_r, r_ok = detect_membrane_radius(
            frame, (new_cx, new_cy), r,
            search_factor, peak_min_dist, peak_min_prom, detect_bright
        )

        # Clamp radius change
        max_dr  = r * max_radius_change_factor
        new_r   = int(np.clip(detected_r, r - max_dr, r + max_dr))
        new_r   = max(new_r, 3)

        # ── 4. Build masks ─────────────────────────────────────────────────
        img_shape  = frame.shape[:2]
        inner_r    = max(1, new_r - membrane_half_width)
        outer_r    = new_r + membrane_half_width
        bg_inner_r = outer_r + bg_buffer
        bg_outer_r = bg_inner_r + bg_width

        inner_masks[i] = create_circular_mask(img_shape, (new_cx, new_cy), inner_r)
        mem_masks[i]   = create_annular_mask (img_shape, (new_cx, new_cy), inner_r + 1, outer_r)
        bg_masks[i]    = create_annular_mask (img_shape, (new_cx, new_cy), bg_inner_r, bg_outer_r)
        centers[i]     = (new_cx, new_cy)
        radii[i]       = new_r
        n_valid       += 1

        # ── 5. Update state and velocity for next frame ────────────────────
        # Calculate the delta from the previous actual center, damped by 50% 
        # to prevent single-frame glitches from throwing the tracker off course.
        vx = 0.5 * vx + 0.5 * (new_cx - cx)
        vy = 0.5 * vy + 0.5 * (new_cy - cy)

        cx, cy = new_cx, new_cy
        r      = new_r

    return {
        'inner_masks':    inner_masks,
        'mem_masks':      mem_masks,
        'bg_masks':       bg_masks,
        'centers':        centers,
        'radii':          radii,
        'ring_scores':    ring_scores,
        'ruptured_at':    ruptured_at,
        'n_valid_frames': n_valid,
    }


# -------------------------------------------------------------------
# --- 7. INTENSITY EXTRACTION ---
# -------------------------------------------------------------------

def get_intensity_trace_tracked(dye_files: List[str],
                                 per_frame_masks: List[Optional[np.ndarray]]
                                 ) -> np.ndarray:
    """
    Extract mean intensity inside *per_frame_masks[i]* from *dye_files[i]*.
    Returns NaN for frames whose mask is None (post-rupture / untracked).
    """
    trace = np.full(len(dye_files), np.nan)
    for i, (fp, mask) in enumerate(zip(dye_files, per_frame_masks)):
        if mask is None:
            continue
        n_px = int(np.sum(mask))
        if n_px == 0:
            continue
        frame = cv2.imread(fp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            trace[i] = float(np.sum(frame[mask])) / n_px
    return trace


# -------------------------------------------------------------------
# --- 8. PARALLEL WORKER ---
# -------------------------------------------------------------------

def process_single_guv(guv_id: str,
                        initial_center: Tuple[int, int],
                        initial_radius: int,
                        is_first_guv: bool,
                        roi_files: List[str],
                        dye_files: List[str]) -> Tuple:
    """
    Full analysis pipeline for a single GUV using frame-by-frame tracking.

    Parameters
    ----------
    guv_id         : string identifier (from circle selector)
    initial_center : (x, y) in original image coords (from circle selector)
    initial_radius : radius in pixels (from circle selector)
    is_first_guv   : if True, force mask-visualisation export
    roi_files      : all ROI (guide) frames — used for tracking
    dye_files      : all dye (measurement) frames
    """
    detect_bright = (getattr(cfg, 'MEMBRANE_DETECTION_MODE', 'LABELED').upper() != 'UNLABELED')

    # ── Track GUV across all frames ────────────────────────────────────────
    tracking = track_guv_across_frames(
        roi_files        = roi_files,
        initial_center   = initial_center,
        initial_radius   = initial_radius,
        search_window_factor    = getattr(cfg, 'TRACKING_SEARCH_WINDOW_FACTOR', 0.7),
        grid_step_factor        = getattr(cfg, 'TRACKING_GRID_STEP_FACTOR', 0.15),
        search_factor           = cfg.MEMBRANE_SEARCH_FACTOR,
        membrane_half_width     = cfg.MEMBRANE_FIXED_HALF_WIDTH,
        peak_min_dist           = cfg.PEAK_FIND_MIN_DISTANCE,
        peak_min_prom           = cfg.PEAK_FIND_MIN_PROMINENCE,
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
            roi_f0 = cv2.imread(roi_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if roi_f0 is not None:
                mem_m = tracking['mem_masks'][0]
                if mem_m is None:
                    c0 = tracking['centers'][0] if tracking['centers'][0] is not None else initial_center
                    r0 = tracking['radii'][0] if tracking['radii'][0] is not None else initial_radius
                    mem_m = create_annular_mask(
                        roi_f0.shape, c0,
                        max(1, r0 - cfg.MEMBRANE_FIXED_HALF_WIDTH) + 1,
                        r0 + cfg.MEMBRANE_FIXED_HALF_WIDTH
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
                roi_files, tracking['centers'], tracking['radii'],
                tracking['ring_scores'], ruptured_at
            )
        except Exception:
            pass
        
    if getattr(cfg, 'EXPORT_TRACK_VIDEO', False):
        try:
            export_track_video(
                cfg.OUTPUT_IMAGE_FOLDER, cfg.EXPERIMENT_BASE_NAME, guv_id,
                roi_files, tracking['centers'], tracking['radii'],
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
    intensity_trace  = get_intensity_trace_tracked(dye_files, tracking['inner_masks'])
    background_trace = get_intensity_trace_tracked(dye_files, tracking['bg_masks'])

    trace_for_jump = np.where(np.isnan(intensity_trace), 0.0, intensity_trace)

    if np.all(trace_for_jump == 0) or np.all(np.isnan(intensity_trace)):
        return None, quality_entry

    # ── Detect dye-entry jump ──────────────────────────────────────────────
    try:
        jump_frame = detect_intensity_jump_robust(
            trace_for_jump,
            threshold_percent=getattr(cfg, 'JUMP_THRESHOLD_PERCENT', 0.20)
        )

        if jump_frame == 0 and getattr(cfg, 'EXPORT_DEBUG_PLOTS', False):
            try:
                plt.switch_backend('Agg')
                fig, ax = plt.subplots(figsize=(8, 4))
                ax2 = ax.twinx()
                ax.plot(intensity_trace, 'b-', alpha=0.4, label='Intensity (NaN=ruptured)')
                ax.plot(gaussian_filter1d(trace_for_jump, sigma=2), 'k-', lw=1.5,
                        label='Smoothed')
                ax2.plot(tracking['ring_scores'], 'g--', alpha=0.6, lw=1, label='Ring score')
                ax2.axhline(getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0),
                            color='orange', ls=':', lw=1, label='Rupture threshold')
                if ruptured_at is not None:
                    ax.axvline(ruptured_at, color='orange', ls='--',
                               label=f'Rupture F{ruptured_at}')
                ax.set_title(f"GUV {guv_id} — No Jump Found")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Intensity")
                ax2.set_ylabel("Ring score")
                lines1, labs1 = ax.get_legend_handles_labels()
                lines2, labs2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=7)
                fig.tight_layout()
                fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                            f"DEBUG_JUMP_GUV_{guv_id}.png"), dpi=150)
                plt.close(fig)
                quality_entry['comments'] += "; JUMP_NOT_FOUND"
            except Exception:
                pass

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
                                roi_files, centers, radii, ring_scores,
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
    frame = cv2.imread(roi_files[last_i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
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
    r0    = radii[valid_pairs[0][0]] or 5
    cv2.circle(canvas, c0, r0, (0, 230, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, c0, 3, (0, 230, 0), -1)

    _, cf = valid_pairs[-1]
    rf    = radii[valid_pairs[-1][0]] or 5
    ec    = (0, 0, 220) if ruptured_at is not None else (0, 220, 220)
    cv2.circle(canvas, cf, rf, ec, 2, cv2.LINE_AA)
    cv2.circle(canvas, cf, 3, ec, -1)

    label = f"GUV {guv_id}  {'RUPTURED @F'+str(ruptured_at) if ruptured_at is not None else 'SURVIVED'}"
    cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, ec, 1, cv2.LINE_AA)

    # ── Right: ring score plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ring_scores, 'b-', lw=1, alpha=0.8, label='Ring score')
    ax.axhline(getattr(cfg, 'RUPTURE_SCORE_THRESHOLD', 4.0),
               color='orange', ls='--', lw=1, label='Rupture threshold')
    if ruptured_at is not None:
        ax.axvline(ruptured_at, color='red', ls=':', lw=1, label=f'Rupture F{ruptured_at}')
    ax.set_xlabel("Frame"); ax.set_ylabel("Ring quality score")
    ax.set_title(f"GUV {guv_id}")
    ax.legend(fontsize=7); fig.tight_layout()

    score_path = os.path.join(output_folder,
                              f"_score_plot_GUV_{guv_id}.png")
    fig.savefig(score_path, dpi=120); plt.close(fig)

    # Combine side-by-side (resize score plot to match canvas height)
    score_img = cv2.imread(score_path)
    if score_img is not None:
        th = canvas.shape[0]
        tw = int(score_img.shape[1] * th / score_img.shape[0])
        score_img = cv2.resize(score_img, (tw, th))
        combined  = np.hstack([canvas, score_img])
        cv2.imwrite(os.path.join(output_folder,
                    f"{experiment_name}_track_GUV_{guv_id}.png"), combined)
        os.remove(score_path)
    else:
        cv2.imwrite(os.path.join(output_folder,
                    f"{experiment_name}_track_GUV_{guv_id}.png"), canvas)


def style_image(frame, time_label, microns_per_pixel, scale_bar_microns):
    img8  = _convert_to_8bit_gray(frame)
    out   = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    out[:, :, 0] = 0; out[:, :, 1] = 0; out[:, :, 2] = img8
    cv2.putText(out, time_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)
    if scale_bar_microns > 0 and microns_per_pixel > 0:
        bar = int(scale_bar_microns / microns_per_pixel)
        h, w = out.shape[:2]
        cv2.line(out, (w-30-bar, h-30), (w-30, h-30), (255,255,255), 5)
        cv2.putText(out, f"{scale_bar_microns} um", (w-30-bar, h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return 


def export_track_video(output_folder: str, experiment_name: str, guv_id: str,
                       roi_files: List[str], centers: List[Optional[Tuple]],
                       radii: List[Optional[int]], fps: float = 10.0):
    """
    Generates an AVI video of the tracking process.
    Draws the live contour (yellow) and the historical trajectory (green).
    """
    valid_pairs = [(i, c) for i, c in enumerate(centers) if c is not None]
    if not valid_pairs:
        return

    # Extract dimensions from the first valid frame
    first_i = valid_pairs[0][0]
    frame = cv2.imread(roi_files[first_i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return

    h, w = frame.shape[:2]
    vid_path = os.path.join(output_folder, f"{experiment_name}_track_GUV_{guv_id}.avi")

    # MJPG codec is highly compatible for AVI containers
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    valid_pts = []

    for i, (fp, center, radius) in enumerate(zip(roi_files, centers, radii)):
        img = cv2.imread(fp, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Apply percentile clipping for visibility
        p_low, p_high = np.percentile(img, (1, 99.5))
        clipped = np.clip(img, p_low, p_high)
        img8 = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).astype(np.uint8)
        canvas = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

        if center is not None and radius is not None:
            valid_pts.append(center)
            # Draw the live membrane contour (Yellow)
            cv2.circle(canvas, center, radius, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw the historical trajectory path (Green)
        if len(valid_pts) > 1:
            pts_array = np.array(valid_pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_array], False, (0, 230, 0), 1, cv2.LINE_AA)

        # Add a frame counter overlay
        cv2.putText(canvas, f"Frame: {i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(canvas)

    out.release()