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

def _apply_fallback_schedule(num_frames: int) -> Tuple[np.ndarray, float]:
    """Fallback routine: applies config schedule, or defaults to constant FPS."""
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

    return create_manual_timestamps(num_frames, getattr(cfg, 'FALLBACK_FPS', 1.0))


def extract_timestamps_from_metadata(file_paths: List[str]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not file_paths:
        return None, None
        
    num_frames = len(file_paths)
    
    # Attempt 1: Embedded TIFF metadata
    try:
        with tifffile.TiffFile(file_paths[0]) as tif:
            tag = tif.pages[0].tags.get('ImageDescription')
            if tag is not None:
                match = re.search(r'finterval=([0-9.]+)', tag.value)
                if match:
                    dt = float(match.group(1))
                    return np.arange(num_frames) * dt, dt
    except Exception:
        pass
    
    # Attempt 2 & 3: Custom Schedule -> Constant FPS
    return _apply_fallback_schedule(num_frames)


def extract_timestamps_nd2(nd2_file) -> Tuple[np.ndarray, float]:
    num_frames = nd2_file.sizes.get('T', 1)
    
    # Attempt 1: Embedded ND2 hardware metadata
    try:
        times = [
            frame.channels[0].time.relativeTimeMs / 1000.0 
            for frame in nd2_file.frame_metadata()
        ]
        if len(times) > 0:
            t_array = np.array(times)
            dt = float(np.median(np.diff(t_array)))
            return t_array, dt
    except Exception:
        pass
        
    # Attempt 2 & 3: Custom Schedule -> Constant FPS
    return _apply_fallback_schedule(num_frames)

def create_memmap_from_nd2_channel(nd2_file, ch_idx: int, mmap_path: str, desc: str) -> tuple:
    """Reads a specific channel from an ND2 file into a contiguous binary memmap using out-of-core streaming."""
    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    # Load file lazily to prevent RAM saturation
    lazy_data = nd2_file.to_dask()
    
    # Handle standard (Time, Channel, Y, X) shape
    if lazy_data.ndim == 4:
        ch_data = lazy_data[:, ch_idx, :, :]
    elif lazy_data.ndim == 3:
        ch_data = lazy_data
    else:
        raise ValueError(f"Unexpected ND2 array shape: {lazy_data.shape}")

    shape = ch_data.shape
    dtype = ch_data.dtype
    mmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)

    for i in tqdm(range(shape[0]), desc=desc, unit="frame"):
        # .compute() evaluates and loads only the current frame into system RAM
        mmap_arr[i] = ch_data[i].compute()
        
    mmap_arr.flush()

    return mmap_path, shape, str(dtype)


def create_manual_timestamps(num_frames: int, fallback_fps: float = 1.0) -> Tuple[np.ndarray, float]:
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

def _get_fast_window(time_array: np.ndarray) -> Tuple[int, int]:
    """
    Identify the frame range of the shortest-interval acquisition segment
    from FRAME_INTERVAL_SCHEDULE (or by inspecting the time_array directly).

    Returns (fast_start, fast_end) as inclusive frame indices of the fast
    segment.  The search window for pulse detection is (fast_start - 1,
    fast_end) so that the transition INTO the fast segment is included.

    Falls back to (0, n-1) if no schedule is defined.
    """
    n = len(time_array)
    schedule = getattr(cfg, 'FRAME_INTERVAL_SCHEDULE', None)

    if schedule is not None and len(schedule) > 0:
        min_interval = min(interval for (_, _, interval) in schedule)
        for (start, end, interval) in schedule:
            if interval == min_interval:
                lo = int(start)
                hi = (int(end) - 1) if end is not None else (n - 1)
                return max(0, lo), min(n - 1, hi)

    # Fallback: find the run of shortest dt steps in the actual time array
    if n > 2:
        dt = np.diff(time_array)
        dt = np.where(dt > 0, dt, np.inf)
        min_dt = dt.min()
        fast_mask = dt <= min_dt * 1.5
        fast_idx  = np.where(fast_mask)[0]
        if len(fast_idx) > 0:
            return int(fast_idx[0]), int(fast_idx[-1] + 1)

    return 0, n - 1


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
    1.  Stack net signals (intensity − background) for every GUV into a
        2-D array (n_guvs × n_frames).
    2.  Restrict the search to the fast-acquisition window only
        (shortest-dt segment from FRAME_INTERVAL_SCHEDULE, including the
        one frame immediately before the window so the transition step
        F(fast_start−1) → F(fast_start) is captured in the derivative).
        The pulse is always delivered within this window; including slow-
        interval frames in a dI/dt derivative causes artefactually large
        values at the interval-change boundary.
    3.  Within the fast window all frames share the same dt, so dI/dframe
        is equivalent to dI/dt for ranking purposes.  Compute per-GUV
        dI/dframe for each consecutive pair of frames in the window.
    4.  Take the **median** across GUVs at each step.  The median is
        intrinsically robust to single-GUV dropouts (mask loss, transient
        tracking failure): a minority of GUVs with artefactual dips cannot
        shift the median.  No Gaussian smoothing is needed within the
        window — averaging across GUVs already suppresses noise.
    5.  The pulse frame is the frame that *arrives* at the steepest median
        drop, i.e.  fast_start − 1 + argmin(median_diff) + 1.

    The diagnostic plot shows the full mean trace (smoothed for display)
    with the search window highlighted, and a bar chart of the median
    dI/dframe within that window.

    Parameters
    ----------
    intensity_curves  : list of 1-D float arrays, one per GUV
    background_curves : list of 1-D float arrays, one per GUV
    smooth_sigma      : Gaussian σ in seconds for the display-only
                        smoothed trace.  Not used in detection logic.
    output_folder     : directory for the diagnostic plot (optional)
    experiment_name   : used in the plot filename
    time_array        : 1-D array of frame timestamps (seconds).
                        Required for window identification; falls back to
                        full-trace mean dI/dframe if not supplied.

    Returns
    -------
    int  – detected pulse frame index (0-based)
    """
    if not intensity_curves:
        return 0

    n_frames = len(intensity_curves[0])

    # ── 1. Per-GUV net signal ──────────────────────────────────────────────
    net_arr = np.vstack([
        np.asarray(d, float) - np.asarray(b, float)
        for d, b in zip(intensity_curves, background_curves)
    ])   # shape: (n_guvs, n_frames)

    mean_trace = np.nanmean(net_arr, axis=0)

    # ── 2. Fast-window bounds ──────────────────────────────────────────────
    if time_array is not None and len(time_array) >= n_frames:
        t = time_array[:n_frames]
        fast_start, fast_end = _get_fast_window(t)
    else:
        fast_start, fast_end = 0, n_frames - 1
        t = np.arange(n_frames, dtype=float)

    # Include the frame immediately before the fast segment so the
    # transition step (pre-pulse → first fast frame) is captured.
    search_lo = max(0, fast_start - 1)
    search_hi = fast_end   # inclusive

    # ── 3 & 4. Median dI/dframe across GUVs within the window ─────────────
    window_arr   = net_arr[:, search_lo : search_hi + 1]   # (n_guvs, window)
    per_guv_diff = np.diff(window_arr, axis=1)              # (n_guvs, window-1)
    median_diff  = np.nanmedian(per_guv_diff, axis=0)       # (window-1,)

    # ── 5. Pulse frame = frame that ARRIVES at the steepest drop ──────────
    argmin_local = int(np.argmin(median_diff))
    pulse_frame  = search_lo + argmin_local + 1   # +1: diff[i]=frame[i+1]-frame[i]

    # ── Diagnostic plot ────────────────────────────────────────────────────
    if output_folder:
        # Smooth mean trace on a uniform time grid for display only
        dt_min     = max(float(np.diff(t).min()), 1e-6)
        dt_uniform = min(0.01, dt_min / 2.0)
        t_unif     = np.arange(t[0], t[-1] + dt_uniform, dt_uniform)
        sig_unif   = np.interp(t_unif, t, mean_trace)
        sigma_uni  = smooth_sigma / dt_uniform
        sm_unif    = gaussian_filter1d(sig_unif, sigma=sigma_uni)
        smoothed   = np.interp(t, t_unif, sm_unif)

        frames = np.arange(n_frames)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        axes[0].plot(frames, mean_trace, color='steelblue', lw=1.0,
                     alpha=0.6, label='Mean net signal (raw)')
        axes[0].plot(frames, smoothed, color='navy', lw=1.5,
                     label=f'Smoothed (σ={smooth_sigma} s, display only)')
        axes[0].axvspan(search_lo, search_hi, alpha=0.10, color='gold',
                        label=f'Search window  [{search_lo}–{search_hi}]')
        axes[0].axvline(pulse_frame, color='crimson', ls='--', lw=1.5,
                        label=f'Detected pulse  (frame {pulse_frame})')
        axes[0].set_ylabel('Mean net fluorescence (AU)')
        axes[0].legend(fontsize=8)
        axes[0].set_title('Pulse-frame detection — efflux')

        win_frames = np.arange(search_lo, search_hi)
        axes[1].bar(win_frames, median_diff, color='darkorange', alpha=0.7,
                    label='Median dI/dframe across GUVs (within window)')
        axes[1].axvline(pulse_frame - 0.5, color='crimson', ls='--', lw=1.5)
        axes[1].axhline(0, color='gray', lw=0.5, ls=':')
        axes[1].set_xlabel('Frame index')
        axes[1].set_ylabel('Median Δ intensity / frame')
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
    """Generates inner, membrane, and background boolean masks simultaneously.

    ``mem_hw`` may be either:
      * a scalar  – symmetric half-width (original behaviour), or
      * a 2-tuple ``(inner_hw, outer_hw)`` – asymmetric ring, extending
        *inner_hw* pixels inward from the membrane contour and *outer_hw*
        pixels outward.  Background placement is based on ``outer_hw``.
    """
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

    # Resolve symmetric vs. asymmetric half-width
    if isinstance(mem_hw, (tuple, list)):
        mem_inner_hw, mem_outer_hw = mem_hw
    else:
        mem_inner_hw = mem_outer_hw = mem_hw

    # Calculate fractional thresholds based on semi-major axis
    mem_inner_frac = max(0.01, (a - mem_inner_hw) / a)
    mem_outer_frac = (a + mem_outer_hw) / a
    bg_inner_frac  = (a + mem_outer_hw + bg_buf) / a
    bg_outer_frac  = (a + mem_outer_hw + bg_buf + bg_w) / a

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


def _compute_peak_fwhm(
        profile: np.ndarray,
        expected_r: float,
        search_factor: float,
        min_prominence_fraction: float = 0.10,
        lo_override: Optional[int] = None,
        hi_override: Optional[int] = None,
) -> Optional[Tuple[int, float, int, int]]:
    """
    Locate the membrane peak and return its FWHM using a shared baseline
    so that an elevated lumen signal does not artificially narrow the
    measured width.

    Baseline strategy
    -----------------
    GUV actin profiles have a bright lumenal plateau that rises toward
    the membrane peak from the inside.  If the left (lumenal) minimum is
    used as the left baseline, ``left_half_max`` is set too close to the
    peak and the left crossing is found just 1–2 px from the apex,
    severely under-estimating FWHM.

    Instead we use a **single shared baseline = right_base** (the
    extracellular / outer side), which is the true background level the
    peak sits above.  The lumenal plateau is cortex-adjacent signal, not
    background, and should not be treated as the reference floor on the
    inward side.  Both left and right crossings are therefore found at the
    same threshold:  peak_val − (peak_val − right_base) * 0.5.

    lo_override / hi_override
        When provided (both must be given together), these replace the
        expected_r ± search_factor window with the actual inner/outer
        boundary indices detected from the membrane (ROI) channel profile.
        This anchors the FWHM search to the physically meaningful region of
        the membrane ring rather than a generic radial fraction.
    """
    n  = len(profile)
    if lo_override is not None and hi_override is not None:
        lo = max(2, lo_override)
        hi = min(n - 2, hi_override)
    else:
        lo = max(2, int(expected_r * (1.0 - search_factor)))
        hi = min(n - 2, int(expected_r * (1.0 + search_factor)))
    if lo >= hi:
        return None

    window     = profile[lo:hi]
    peak_local = int(np.argmax(window))
    peak_idx   = peak_local + lo
    peak_val   = float(profile[peak_idx])

    # Ensure peak is brighter than lumen interior reference
    inner_end  = max(1, int(expected_r * 0.70))
    lumen_mean = float(np.mean(profile[:inner_end]))
    if peak_val <= lumen_mean:
        return None

    # ── Baseline: use the extracellular (outer) side only ─────────────────
    # The lumenal plateau is signal, not background.  Using the outer
    # minimum as the shared baseline means both crossings are found at the
    # same level, producing a physically correct peak width even when the
    # inner signal is elevated (e.g. lumenal actin release post-pulse).
    band       = max(5, (hi - lo) // 2)
    right_zone = profile[peak_idx : min(n, hi + band)]
    right_base = float(right_zone.min()) if len(right_zone) > 0 else float(profile.min())

    # Prominence validation: peak must rise meaningfully above the shared baseline
    amplitude     = peak_val - right_base
    profile_range = float(profile.max() - profile.min())

    if profile_range > 0 and (amplitude / profile_range) < min_prominence_fraction:
        return None

    # Shared half-maximum threshold (same for both sides)
    half_max = right_base + amplitude * 0.5

    # Find left crossing — walk inward from the peak; search all the way to
    # the start of the profile so the crossing is not missed when the lumenal
    # plateau sits above half_max (in that case left_idx stays at lo,
    # correctly capturing the full inward extent of the peak).
    left_idx = lo
    for j in range(peak_idx - 1, -1, -1):
        if profile[j] <= half_max:
            left_idx = j + 1
            break

    # Find right crossing — walk outward from the peak
    right_idx = hi
    for j in range(peak_idx + 1, min(n, hi + band)):
        if profile[j] <= half_max:
            right_idx = j - 1
            break

    fwhm_px = float(max(1, right_idx - left_idx + 1))
    return peak_idx, fwhm_px, left_idx, right_idx


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
            actin_stack       = actin_stack,
            n_frames          = n_frames,
            mem_masks         = tracking['mem_masks'],
            inner_masks       = tracking['inner_masks'],
            cortex_hw         = getattr(cfg, 'ACTIN_CORTEX_HALF_WIDTH', 4),
            centers           = tracking['centers'],
            ellipses          = tracking['ellipses'],
            img_shape         = img_shape,
            bg_buffer         = cfg.BG_BUFFER_PIXELS,
            bg_width          = cfg.BG_RING_WIDTH_PIXELS,
            search_factor     = cfg.MEMBRANE_SEARCH_FACTOR,
            microns_per_pixel = getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
            roi_stack         = roi_stack,
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
        export_track_visualization(
            cfg.FOLDER_TRACKING, cfg.EXPERIMENT_BASE_NAME, guv_id,
            roi_stack, tracking['centers'], tracking['ellipses'],
            ruptured_at
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


def export_track_visualization(track_folder, experiment_name, guv_id,
                                roi_stack, centers, ellipses, ruptured_at):
    """Saves a diagnostic PNG showing the GUV trajectory."""
    valid_pairs = [(i, c) for i, c in enumerate(centers) if c is not None]
    if not valid_pairs:
        return
    
    last_i, _ = valid_pairs[-1]
    frame = roi_stack[last_i]
    if frame is None:
        return

    img8   = _convert_to_8bit_gray(frame)
    canvas = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    pts = np.array([c for _, c in valid_pairs], dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(canvas, [pts], False, (0, 230, 0), 1, cv2.LINE_AA)

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

    cv2.imwrite(os.path.join(track_folder, f"{experiment_name}_track_GUV_{guv_id}.png"), canvas)


def normalize_tracking_data(df: pd.DataFrame, pulse_frame: int) -> pd.DataFrame:
    """Calculates normalized radius and circularity relative to pre-pulse averages."""
    df['norm_radius'] = np.nan
    df['norm_circularity'] = np.nan
    
    for gid in df['guv_id'].unique():
        mask = df['guv_id'] == gid
        guv_df = df[mask]
        
        pre_mask = guv_df['frame'] < pulse_frame
        if not pre_mask.any():
            pre_mask = guv_df['frame'] == guv_df['frame'].min()
            
        mean_r = guv_df.loc[pre_mask, 'radius'].mean()
        mean_c = guv_df.loc[pre_mask, 'circularity'].mean()
        
        if mean_r > 0:
            df.loc[mask, 'norm_radius'] = df.loc[mask, 'radius'] / mean_r
        if mean_c > 0:
            df.loc[mask, 'norm_circularity'] = df.loc[mask, 'circularity'] / mean_c
            
    return df


def export_tracking_summary_csv(df: pd.DataFrame, time_array: np.ndarray, output_folder: str, experiment_name: str):
    """Exports a pivoted CSV summarizing normalized tracking metrics for all GUVs."""
    frames = sorted(df['frame'].unique())
    t_vals = [time_array[f] if f < len(time_array) else np.nan for f in frames]
    
    summary_df = pd.DataFrame({'frame': frames, 'time_s': t_vals})
    
    for gid in df['guv_id'].unique():
        guv_df = df[df['guv_id'] == gid].set_index('frame')
        summary_df[f'GUV_{gid}_norm_radius'] = summary_df['frame'].map(guv_df['norm_radius'])
        summary_df[f'GUV_{gid}_norm_circularity'] = summary_df['frame'].map(guv_df['norm_circularity'])
        
    csv_path = os.path.join(output_folder, f"{experiment_name}_tracking_summary.csv")
    summary_df.to_csv(csv_path, index=False, float_format='%.6f', na_rep='NaN')
    return csv_path


def plot_guv_shape_metrics(df: pd.DataFrame, time_array: np.ndarray, 
                           output_folder: str, experiment_name: str, pulse_time: float):
    """Generates a 1x2 panel plot for each GUV showing normalized circularity and size starting from 0."""
    guv_ids = df['guv_id'].unique()
    
    for gid in guv_ids:
        guv_data = df[df['guv_id'] == gid].sort_values('frame')
        frames = guv_data['frame'].values
        
        valid_t_idx = [f for f in frames if f < len(time_array)]
        if not valid_t_idx:
            continue
        
        t_vals = time_array[valid_t_idx]
        norm_c = guv_data['norm_circularity'].values[:len(valid_t_idx)]
        norm_r = guv_data['norm_radius'].values[:len(valid_t_idx)]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Panel 1: Circularity
        axes[0].plot(t_vals, norm_c, 'b-', lw=1.5, label='Norm. Circularity')
        axes[0].axhline(1.0, color='gray', ls=':', lw=1)
        axes[0].set_title(f'GUV {gid} Circularity Change')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Normalized Circularity')
        
        # Dynamic upper bound calculation to avoid cutting off data if a GUV expands
        max_c = np.nanmax(norm_c) if len(norm_c) > 0 else 1.0
        axes[0].set_ylim(0, max(1.1, max_c * 1.05))
        
        # Panel 2: Size
        axes[1].plot(t_vals, norm_r, 'r-', lw=1.5, label='Norm. Size (Radius)')
        axes[1].axhline(1.0, color='gray', ls=':', lw=1)
        axes[1].set_title(f'GUV {gid} Size Change')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized Radius')
        
        max_r = np.nanmax(norm_r) if len(norm_r) > 0 else 1.0
        axes[1].set_ylim(0, max(1.1, max_r * 1.05))
        
        # Shared formatting
        for ax in axes:
            ax.axvline(pulse_time, color='k', ls='--', alpha=0.5, label='Pulse')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
        plt.tight_layout()
        out_path = os.path.join(output_folder, f"{experiment_name}_shape_GUV_{gid}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

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
        search_factor: float = 0.35,
        microns_per_pixel: float = 1.0,
        roi_stack: np.ndarray = None,
) -> dict:
    """
    Extracts per-frame actin intensity from the C2 channel for one GUV.

    The cortex mask is defined **frame-by-frame** from the FWHM of the
    membrane peak in the actin radial intensity profile.  This means the
    mask automatically adapts to changes in peak width over time (e.g. as
    the cortex thickens or dissolves) rather than using a fixed ring.

    If no clear peak is found in a given frame the fallback half-width
    ``cortex_hw`` is used symmetrically instead.

    Signals extracted
    -----------------
    Cortex        : mean intensity within the FWHM-defined ring (mask-based)
    Peak intensity: max of the smoothed radial profile within the FWHM window
                    — directly measures actin density at the cortex apex
    Lumen         : mean intensity inside the GUV (inner mask from tracking)
    BG            : median background annulus
    FWHM          : full-width at half-maximum of the membrane peak, in both px
                    and µm (µm requires *microns_per_pixel* to be set correctly)

    Returns
    -------
    dict with keys:
        'cortex_trace'      – mean intensity in the FWHM cortex ring (n_frames,)
        'peak_intensity_trace' – max of radial profile within FWHM   (n_frames,)
        'lumen_trace'       – mean intensity in the GUV lumen         (n_frames,)
        'bg_trace'          – median background                       (n_frames,)
        'fwhm_px_trace'     – FWHM of the membrane peak in pixels     (n_frames,)
        'fwhm_um_trace'     – FWHM in µm                              (n_frames,)
    """
    cortex_tr       = np.full(n_frames, np.nan)
    peak_tr         = np.full(n_frames, np.nan)
    lumen_tr        = np.full(n_frames, np.nan)
    bg_tr           = np.full(n_frames, np.nan)
    fwhm_px_tr      = np.full(n_frames, np.nan)
    fwhm_um_tr      = np.full(n_frames, np.nan)
    peak_detected_tr = np.zeros(n_frames, dtype=np.int8)  # 1 = peak found, 0 = no cortex

    n_angles = getattr(cfg, 'ACTIN_N_ANGLES', 72)
    gini_tr           = np.full(n_frames, np.nan)
    # angular_profile_store[i] is a 1-D array of length n_angles (or None when
    # no peak was detected). Stored as an object array so the CSV exporter and
    # normaliser can handle it cleanly without pre-allocating a fixed n_angles.
    angular_profile_store: List[Optional[np.ndarray]] = [None] * n_frames

    h, w = img_shape[:2]

    for i in range(n_frames):
        el = ellipses[i]
        if el is None or actin_stack is None:
            continue

        frame = actin_stack[i]
        if frame is None:
            continue

        cx, cy = el['center']
        avg_r  = float((el['axes'][0] + el['axes'][1]) / 2.0)
        r_max  = int(avg_r * (1.0 + search_factor) + 10)

        # --- Lumen and background: always computed, independent of peak ---
        inner_m_base, _, bg_m_base = generate_vectorized_masks(
            img_shape, el['center'], el['axes'], el['angle'],
            cortex_hw, bg_buffer, bg_width,
        )
        if inner_m_base is not None and inner_m_base.any():
            lumen_tr[i] = float(np.mean(frame[inner_m_base]))
        if bg_m_base is not None and bg_m_base.any():
            bg_tr[i] = float(np.median(frame[bg_m_base]))

        # --- Cortex and peak: only filled when a distinct peak is found ---
        can_profile = (cx - r_max >= 0 and cx + r_max < w and
                       cy - r_max >= 0 and cy + r_max < h)
        if not can_profile:
            # cortex_tr[i], peak_tr[i], fwhm traces remain NaN
            continue

        x_min = max(0, int(cx - r_max - 2))
        x_max = min(w, int(cx + r_max + 2))
        y_min = max(0, int(cy - r_max - 2))
        y_max = min(h, int(cy + r_max + 2))
        local_crop = frame[y_min:y_max, x_min:x_max]
        profile    = radial_profile_local(
            local_crop, (cx - x_min, cy - y_min), r_max
        )
        profile_sm = gaussian_filter1d(profile, sigma=1.5)

        # --- Membrane boundary constraints from the ROI channel ---
        # Run _compute_peak_fwhm on the membrane (ROI) channel profile for
        # this same frame and position.  The resulting left_idx / right_idx
        # define the physical inner and outer edge of the membrane ring.
        # These are passed as hard lo/hi overrides into the actin FWHM call,
        # mirroring the skeleton.py approach where membrane_detection()
        # provides index_border_in / index_border_out to constrain the actin
        # analysis.  Falls back to None (generic window) if unavailable.
        lo_mem, hi_mem = None, None
        if roi_stack is not None:
            roi_frame = roi_stack[i]
            if roi_frame is not None:
                roi_crop    = roi_frame[y_min:y_max, x_min:x_max]
                roi_profile = radial_profile_local(
                    roi_crop, (cx - x_min, cy - y_min), r_max
                )
                roi_sm = gaussian_filter1d(roi_profile, sigma=1.5)
                mem_result = _compute_peak_fwhm(
                    roi_sm, avg_r, search_factor,
                    min_prominence_fraction=0.05,  # relaxed: membrane is bright
                )
                if mem_result is not None:
                    _, _, raw_lo, raw_hi = mem_result
                    # Pad the membrane border outward so the actin search window
                    # is slightly generous — guards against a hairline-underestimated
                    # membrane edge clipping the cortex peak search.
                    # Matches the ±3 px padding in skeleton.py's membrane_detection().
                    pad = getattr(cfg, 'ACTIN_MEMBRANE_BORDER_PADDING', 3)
                    lo_mem = max(0,           raw_lo - pad)
                    hi_mem = min(r_max - 1,   raw_hi + pad)

        fwhm_result = _compute_peak_fwhm(
            profile_sm, avg_r, search_factor,
            min_prominence_fraction=getattr(cfg, 'ACTIN_PEAK_MIN_PROMINENCE', 0.10),
            lo_override=lo_mem,
            hi_override=hi_mem,
        )
        if fwhm_result is None:
            # No distinct peak → cortex/peak/FWHM stay NaN for this frame.
            # This is the correct representation: if there is no cortex there
            # is nothing to measure; a fallback mask value would be misleading.
            peak_detected_tr[i] = 0
            continue

        peak_idx, fwhm_px, left_idx, right_idx = fwhm_result
        inner_hw = max(1, int(np.ceil(avg_r - left_idx)))
        outer_hw = max(1, int(np.ceil(right_idx - avg_r)))
        fwhm_px_tr[i]       = fwhm_px
        fwhm_um_tr[i]       = fwhm_px * microns_per_pixel
        peak_tr[i]          = float(profile_sm[peak_idx])
        peak_detected_tr[i] = 1

        # Cortex mean: mask-based, width derived from the FWHM
        _, cortex_m, _ = generate_vectorized_masks(
            img_shape, el['center'], el['axes'], el['angle'],
            (inner_hw, outer_hw), bg_buffer, bg_width,
        )
        if cortex_m is not None and cortex_m.any():
            cortex_tr[i] = float(np.mean(frame[cortex_m]))

        # --- Angular profile and Gini index ---
        # Sample the full 2-D polar array at this frame and take np.max()
        # within the membrane-border window [left_idx : right_idx+1] for
        # each angle.  This gives the brightest cortex value per direction
        # (matching skeleton.py's angular_profile() convention), which is
        # then used to compute the Gini coefficient as a measure of how
        # unevenly the actin cortex is distributed around the circumference.
        polar_2d = radial_profile_2d_local(
            local_crop, (cx - x_min, cy - y_min), r_max, n_angles
        )  # shape: (r_max, n_angles)

        # Apply the same Gaussian smoothing radially before collapsing
        polar_sm = gaussian_filter1d(polar_2d.astype(float), sigma=1.5, axis=0)

        # Clamp border indices to valid range
        ang_lo = max(0, left_idx)
        ang_hi = min(r_max, right_idx + 1)
        if ang_hi > ang_lo:
            ang_profile = np.max(polar_sm[ang_lo:ang_hi, :], axis=0)  # (n_angles,)
        else:
            # Fallback: single radial slice at the peak
            ang_profile = polar_sm[min(peak_idx, r_max - 1), :]

        angular_profile_store[i] = ang_profile

        # Gini coefficient over the angular profile (background-corrected)
        # A value near 0 = uniformly distributed cortex; near 1 = highly patchy.
        bg_val_i = bg_tr[i] if np.isfinite(bg_tr[i]) else 0.0
        ang_corrected = np.clip(ang_profile - bg_val_i, 0.0, None)
        total_ang = ang_corrected.sum()
        if total_ang > 0:
            n_ang = len(ang_corrected)
            sorted_ang = np.sort(ang_corrected)
            idx_arr    = np.arange(1, n_ang + 1)
            gini_tr[i] = float(
                (2.0 * np.sum(idx_arr * sorted_ang) - (n_ang + 1) * total_ang)
                / (n_ang * total_ang)
            )

    return {
        'cortex_trace':         cortex_tr,
        'peak_intensity_trace': peak_tr,
        'lumen_trace':          lumen_tr,
        'bg_trace':             bg_tr,
        'fwhm_px_trace':        fwhm_px_tr,
        'fwhm_um_trace':        fwhm_um_tr,
        'peak_detected_trace':  peak_detected_tr,  # 1 = peak found, 0 = no cortex/edge issue
        'gini_trace':           gini_tr,            # Gini coefficient of angular actin distribution
        'angular_profiles':     angular_profile_store,  # list of (n_angles,) arrays, None where no peak
    }


def normalize_actin_curves(
        actin_data_list: list,
        pulse_frame: int,
        time_array: np.ndarray,
) -> dict:
    """
    Normalise actin traces to their pre-pulse baseline and align to t = 0
    at the pulse frame.

    Cortex and lumen intensities are normalised to their respective pre-pulse
    means (fractional change).  FWHM is aligned in time but kept in µm since
    the absolute width is the biologically meaningful quantity — it directly
    reflects how the cortex thickness changes after electroporation.

    Parameters
    ----------
    actin_data_list : list of dicts from extract_actin_traces(), one per GUV
    pulse_frame     : frame index of the electroporation pulse
    time_array      : absolute timestamps (s)

    Returns
    -------
    dict with aligned arrays and GUV-averaged traces
    """
    cortex_norm_list = []
    peak_norm_list   = []
    lumen_norm_list  = []
    fwhm_al_list     = []
    peak_det_list    = []
    gini_al_list     = []

    safe_pre_end   = pulse_frame if pulse_frame > 0 else 1
    safe_pre_start = max(0, safe_pre_end - 5)

    for ad in actin_data_list:
        cortex    = np.asarray(ad['cortex_trace'],         float)
        peak      = np.asarray(ad['peak_intensity_trace'], float)
        lumen     = np.asarray(ad['lumen_trace'],          float)
        bg        = np.asarray(ad['bg_trace'],             float)
        fwhm      = np.asarray(ad['fwhm_um_trace'],        float)
        peak_det  = np.asarray(ad['peak_detected_trace'],  np.int8)
        gini      = np.asarray(ad.get('gini_trace', np.full(len(cortex), np.nan)), float)

        c0 = float(np.nanmean((cortex - bg)[safe_pre_start:safe_pre_end]))
        p0 = float(np.nanmean((peak   - bg)[safe_pre_start:safe_pre_end]))
        l0 = float(np.nanmean((lumen  - bg)[safe_pre_start:safe_pre_end]))

        with np.errstate(divide='ignore', invalid='ignore'):
            c_norm = (cortex - bg) / c0 if (c0 != 0 and np.isfinite(c0)) else np.full_like(cortex, np.nan)
            p_norm = (peak   - bg) / p0 if (p0 != 0 and np.isfinite(p0)) else np.full_like(peak,   np.nan)
            l_norm = (lumen  - bg) / l0 if (l0 != 0 and np.isfinite(l0)) else np.full_like(lumen,  np.nan)

        cortex_norm_list.append(c_norm[pulse_frame:])
        peak_norm_list  .append(p_norm[pulse_frame:])
        lumen_norm_list .append(l_norm[pulse_frame:])
        fwhm_al_list    .append(fwhm[pulse_frame:])
        peak_det_list   .append(peak_det[pulse_frame:])
        gini_al_list    .append(gini[pulse_frame:])   # absolute (0–1); not normalised

    t_aligned  = time_array[pulse_frame:] - time_array[pulse_frame]
    cortex_arr   = np.array(cortex_norm_list)
    peak_arr     = np.array(peak_norm_list)
    lumen_arr    = np.array(lumen_norm_list)
    fwhm_arr     = np.array(fwhm_al_list)
    peak_det_arr = np.array(peak_det_list, dtype=np.int8)
    gini_arr     = np.array(gini_al_list)

    return {
        't_aligned':       t_aligned,
        'cortex_norm':     cortex_arr,
        'peak_norm':       peak_arr,
        'lumen_norm':      lumen_arr,
        'fwhm_aligned':    fwhm_arr,                         # µm, absolute
        'gini_aligned':    gini_arr,                         # Gini (0–1), absolute
        'peak_detected':   peak_det_arr,                     # (n_guvs, n_frames) 0/1
        'avg_cortex':      np.nanmean(cortex_arr, axis=0),
        'avg_peak':        np.nanmean(peak_arr,   axis=0),
        'avg_lumen':       np.nanmean(lumen_arr,  axis=0),
        'avg_fwhm':        np.nanmean(fwhm_arr,   axis=0),
        'avg_gini':        np.nanmean(gini_arr,   axis=0),
        # Fraction of GUVs with a detected cortex peak per frame (0–1)
        'peak_detect_frac': peak_det_arr.mean(axis=0),
    }


def plot_actin_analysis(
        aligned_actin: dict,
        valid_guv_ids: list,
        output_folder: str,
        experiment_name: str,
        smooth_sigma: float = 1.5,
):
    """
    2x2 actin summary figure:
      Top-left     - Normalised cortex peak intensity  (radial profile max)
      Top-right    - Normalised lumenal actin intensity
      Bottom-left  - Cortex FWHM over time             (um, absolute)
      Bottom-right - Gini index                        (angular heterogeneity, absolute)

    All panels share the same time axis aligned to the pulse (t = 0).
    """
    from scipy.ndimage import gaussian_filter1d

    t         = aligned_actin['t_aligned']
    peak_arr  = aligned_actin['peak_norm']
    lumen_arr = aligned_actin['lumen_norm']
    fwhm_arr  = aligned_actin['fwhm_aligned']
    gini_arr  = aligned_actin['gini_aligned']
    avg_peak  = aligned_actin['avg_peak']
    avg_lumen = aligned_actin['avg_lumen']
    avg_fwhm  = aligned_actin['avg_fwhm']
    avg_gini  = aligned_actin['avg_gini']

    def _smooth(x):
        if smooth_sigma > 0:
            finite = np.isfinite(x)
            if finite.sum() < 3:
                return x
            s = gaussian_filter1d(np.where(finite, x, 0), smooth_sigma)
            n = gaussian_filter1d(finite.astype(float), smooth_sigma)
            return np.where(n > 0, s / n, np.nan)
        return x

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_cpeak = axes[0, 0]
    ax_lumen = axes[0, 1]
    ax_fwhm  = axes[1, 0]
    ax_gini  = axes[1, 1]

    palette = plt.cm.tab10.colors

    for i, (gid, p_row, l_row, f_row, g_row) in enumerate(
            zip(valid_guv_ids, peak_arr, lumen_arr, fwhm_arr, gini_arr)):
        col = palette[i % len(palette)]
        ax_cpeak.plot(t, _smooth(p_row), color=col, alpha=0.4, lw=1, label=f'GUV {gid}')
        ax_lumen.plot(t, _smooth(l_row), color=col, alpha=0.4, lw=1, label=f'GUV {gid}')
        ax_fwhm .plot(t, _smooth(f_row), color=col, alpha=0.4, lw=1, label=f'GUV {gid}')
        ax_gini .plot(t, _smooth(g_row), color=col, alpha=0.4, lw=1, label=f'GUV {gid}')

    ax_cpeak.plot(t, _smooth(avg_peak),  'k-', lw=2.5, label='Mean')
    ax_lumen.plot(t, _smooth(avg_lumen), 'k-', lw=2.5, label='Mean')
    ax_fwhm .plot(t, _smooth(avg_fwhm),  'k-', lw=2.5, label='Mean')
    ax_gini .plot(t, _smooth(avg_gini),  'k-', lw=2.5, label='Mean')

    # Tighten the FWHM y-axis so variation is readable.
    all_fwhm_vals = fwhm_arr[np.isfinite(fwhm_arr)]
    if len(all_fwhm_vals) > 0:
        ax_fwhm.set_ylim(max(0.0, all_fwhm_vals.min() * 0.85),
                         all_fwhm_vals.max() * 1.10)

    # Gini: bounded 0-1 coefficient; anchor at 0 and give headroom above max.
    all_gini_vals = gini_arr[np.isfinite(gini_arr)]
    if len(all_gini_vals) > 0:
        ax_gini.set_ylim(0.0, min(1.0, all_gini_vals.max() * 1.15))

    for ax in (ax_cpeak, ax_lumen, ax_fwhm, ax_gini):
        ax.axvline(0, color='crimson', ls='--', lw=1.5, label='Pulse', zorder=3)
        ax.set_xlabel('Time relative to pulse (s)')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    ax_cpeak.axhline(1, color='gray', ls=':', lw=0.8)
    ax_lumen.axhline(1, color='gray', ls=':', lw=0.8)

    ax_cpeak.set_title('Cortex peak intensity\n(radial profile max, normalised to pre-pulse)')
    ax_cpeak.set_ylabel('Normalised intensity (a.u.)')

    ax_lumen.set_title('Lumenal actin intensity\n(inner mask, normalised to pre-pulse)')
    ax_lumen.set_ylabel('Normalised intensity (a.u.)')

    ax_fwhm.set_title('Cortex thickness (FWHM)\n(absolute, \u00b5m)')
    ax_fwhm.set_ylabel('FWHM (\u00b5m)')

    ax_gini.set_title('Cortex Gini index\n(angular heterogeneity, 0 = uniform, 1 = patchy)')
    ax_gini.set_ylabel('Gini coefficient (a.u.)')

    plt.suptitle(f'{experiment_name}  -  Actin cortex analysis', fontsize=12)
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
    df = pd.DataFrame({'time_s': t})

    for gid, c_row, p_row, l_row, f_row, g_row, d_row in zip(
            valid_guv_ids,
            aligned_actin['cortex_norm'],
            aligned_actin['peak_norm'],
            aligned_actin['lumen_norm'],
            aligned_actin['fwhm_aligned'],
            aligned_actin['gini_aligned'],
            aligned_actin['peak_detected']):
        df[f'GUV_{gid}_cortex_mean']    = c_row
        df[f'GUV_{gid}_cortex_peak']    = p_row
        df[f'GUV_{gid}_lumen']          = l_row
        df[f'GUV_{gid}_fwhm_um']        = f_row
        df[f'GUV_{gid}_gini']           = g_row
        df[f'GUV_{gid}_peak_detected']  = d_row

    df['avg_cortex_mean']  = aligned_actin['avg_cortex']
    df['avg_cortex_peak']  = aligned_actin['avg_peak']
    df['avg_lumen']        = aligned_actin['avg_lumen']
    df['avg_fwhm_um']      = aligned_actin['avg_fwhm']
    df['avg_gini']         = aligned_actin['avg_gini']
    df['peak_detect_frac'] = aligned_actin['peak_detect_frac']

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
    Paired bar chart comparing the normalised cortex peak intensity BEFORE
    vs AFTER the electroporation pulse, per GUV.

    Pre-pulse value is always 1.0 by construction (normalised to baseline).
    The post-pulse value is the mean of the first *post_window_s* seconds
    after the pulse.  Values < 1 indicate cortex loss; values > 1 indicate
    cortex enrichment.

    post_window_s : window (s) after the pulse used to compute the post value.
    """
    t        = time_array_aligned
    cortex   = aligned_actin['cortex_norm']   # mean within FWHM mask
    peak     = aligned_actin['peak_norm']     # radial profile max

    post_mask = (t >= 0) & (t <= post_window_s)
    if not post_mask.any():
        post_mask = np.ones(len(t), bool)

    pre_vals        = np.ones(len(valid_guv_ids))          # always 1 by normalisation
    post_mean_means = np.nanmean(cortex[:, post_mask], axis=1)
    post_peak_means = np.nanmean(peak[:, post_mask],   axis=1)

    x     = np.arange(len(valid_guv_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(valid_guv_ids) * 1.4), 5))
    ax.bar(x - width,     pre_vals,        width, label='Pre-pulse (=1)',
           color='steelblue', alpha=0.8)
    ax.bar(x,             post_mean_means, width,
           label=f'Post — cortex mean  (first {post_window_s:.0f} s)',
           color='tomato', alpha=0.8)
    ax.bar(x + width,     post_peak_means, width,
           label=f'Post — cortex peak  (first {post_window_s:.0f} s)',
           color='darkorange', alpha=0.8)
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'GUV {g}' for g in valid_guv_ids], rotation=45, ha='right')
    ax.set_ylabel('Normalised cortex intensity  (a.u.)')
    ax.set_title(f'{experiment_name}\nCortex intensity: pre vs post electroporation')
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
        profile_sm = gaussian_filter1d(profile, sigma=1.5)

        r_um   = np.arange(r_max) * microns_per_pixel
        mem_um = avg_r * microns_per_pixel

        ax.plot(r_um, profile,    color='steelblue', lw=1.4, alpha=0.5, label='Raw')
        ax.plot(r_um, profile_sm, color='steelblue', lw=1.8, label='Smoothed')

        # FWHM annotation
        fwhm_result = _compute_peak_fwhm(profile_sm, avg_r,
                                         search_factor=0.35,
                                         min_prominence_fraction=getattr(
                                             cfg, 'ACTIN_PEAK_MIN_PROMINENCE', 0.10))
        if fwhm_result is not None:
            _, fwhm_px, left_idx, right_idx = fwhm_result
            left_um  = left_idx  * microns_per_pixel
            right_um = right_idx * microns_per_pixel
            fwhm_um  = fwhm_px   * microns_per_pixel
            half_max_val = (profile_sm[left_idx] + profile_sm[right_idx]) / 2.0
            ax.hlines(half_max_val, left_um, right_um,
                      color='crimson', lw=1.5, ls='-',
                      label=f'FWHM = {fwhm_um:.2f} µm')
            ax.axvspan(left_um, right_um, alpha=0.12, color='crimson')

        ax.axvline(mem_um, color='dimgray', ls='--', lw=1.2, label='Membrane')
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

def plot_actin_radial_evolution(
        actin_stack: np.ndarray,
        ellipses_per_guv: list,
        output_folder: str,
        experiment_name: str,
        microns_per_pixel: float = 1.0,
):
    """
    Generates a multigrid plot showing radial intensity profile evolution
    over time for each GUV, using a colormap from light gray to black.
    X-axis is in µm when microns_per_pixel is supplied.
    """
    n = len(ellipses_per_guv)
    if n == 0 or actin_stack is None:
        return

    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)

    for k, (gid, el_list) in enumerate(ellipses_per_guv):
        ax = axes[k // ncols][k % ncols]

        # Filter to valid frames for this GUV
        valid_frames = [(i, el) for i, el in enumerate(el_list) if el is not None]

        # Normalize color map: light gray (start) to black (end)
        colors = plt.cm.Greys(np.linspace(0.2, 1.0, len(valid_frames)))

        for idx, (fi, el) in enumerate(valid_frames):
            frame = actin_stack[fi]
            if frame is None: continue

            cx, cy = el['center']
            avg_r = int(sum(el['axes']) / 2.0)
            r_max = int(avg_r * 1.6)

            h_f, w_f = frame.shape[:2]
            if (cx - r_max < 0 or cx + r_max >= w_f or
                    cy - r_max < 0 or cy + r_max >= h_f):
                continue

            x_min = max(0, int(cx - r_max - 2))
            x_max = min(w_f, int(cx + r_max + 2))
            y_min = max(0, int(cy - r_max - 2))
            y_max = min(h_f, int(cy + r_max + 2))
            local = frame[y_min:y_max, x_min:x_max]
            profile = radial_profile_local(
                local, (cx - x_min, cy - y_min), r_max
            )
            profile_sm = gaussian_filter1d(profile, sigma=1.5)
            r_um = np.arange(r_max) * microns_per_pixel

            ax.plot(r_um, profile_sm, color=colors[idx], lw=1.0, alpha=0.7)

        ax.set_title(f'GUV {gid} Profile Evolution')
        ax.set_xlabel(r'Radius ($\mu$m)')
        ax.set_ylabel('Intensity (AU)')
        ax.grid(True, alpha=0.2)

    # Hide unused panels
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, f'{experiment_name}_actin_evolution.png'), dpi=300)
    plt.close(fig)


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
    Right y-axis (orange) – Normalised cortex peak intensity (FWHM-masked)

    One subplot per GUV, up to 4 per row.  A summary panel (last subplot)
    shows population means of both signals with ±1 SD shading.

    Parameters
    ----------
    aligned_dye   : dict returned by normalize_and_align_curves()
    aligned_actin : dict returned by normalize_actin_curves()
    valid_guv_ids : list of GUV id strings in the same order as both arrays
    """
    from scipy.ndimage import gaussian_filter1d

    t_dye   = aligned_dye['t_aligned']
    t_act   = aligned_actin['t_aligned']

    dye_arr = aligned_dye['all_curves_aligned']  # (n, T_dye)
    act_arr = aligned_actin['cortex_norm']        # (n, T_act) — normalised cortex

    avg_dye = aligned_dye['average_curve']
    avg_act = aligned_actin['avg_cortex']

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
        ax2.plot(t_a, _sm(act_row), color=ACT_COLOR, lw=1.6, label='Cortex (norm.)')

        ax .axvline(0, color='crimson', ls='--', lw=1.2, zorder=3)
        ax .axhline(1, color=DYE_COLOR, ls=':', lw=0.6, alpha=0.5)
        ax2.axhline(1, color=ACT_COLOR, ls=':', lw=0.6, alpha=0.5)

        ax .set_xlabel('Time relative to pulse (s)', fontsize=8)
        ax .set_ylabel('Dye retention (a.u.)',          fontsize=8, color=DYE_COLOR)
        ax2.set_ylabel('Cortex intensity (norm., a.u.)', fontsize=8, color=ACT_COLOR)
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

    plt.suptitle(f'{experiment_name}  –  Dye efflux vs Actin cortex intensity',
                 fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_actin_overlay.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

# -------------------------------------------------------------------
# --- 11. DYE CHANNEL SUMMARY PLOTS ---
# -------------------------------------------------------------------

def plot_dye_fits_grid(
        fit_data: list,
        output_folder: str,
        experiment_name: str,
        model_name: str,
) -> str:
    """
    Multigrid plot of per-GUV kinetic fits — one subplot per GUV plus a
    population-mean panel.

    Parameters
    ----------
    fit_data : list of dicts, one per GUV, each containing:
        'guv_id'  – str
        't_data'  – 1-D array of finite time points used in the fit
        'y_data'  – corresponding normalised intensity values
        'y_fit'   – model evaluated at t_data with fitted parameters
        'r_um'    – GUV radius in µm
        'param_str' – short label string, e.g. "τ = 12.3 s"
    output_folder : str
    experiment_name : str
    model_name : str  – used for the figure title
    """
    n = len(fit_data)
    if n == 0:
        return ""

    n_panels = n + 1                       # +1 for population mean
    ncols    = min(4, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))
    palette  = plt.cm.tab10.colors

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for k, d in enumerate(fit_data):
        ax  = axes[k // ncols][k % ncols]
        col = palette[k % len(palette)]

        ax.plot(d['t_data'], d['y_data'], '.', color=col,
                alpha=0.45, ms=3, label='Data')
        ax.plot(d['t_data'], d['y_fit'],  '-', color=col,
                lw=1.8, label='Fit')
        ax.axvline(0, color='gray', ls='--', lw=1.2, label='Pulse', zorder=0)

        ax.set_title(
            rf"GUV {d['guv_id']}  (R={d['r_um']:.1f} µm)"
            f"\n{d['param_str']}",
            fontsize=8
        )
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Norm. intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6)

    # ── Population mean panel ─────────────────────────────────────────────
    # Each GUV's t_data covers only its finite points and may have a
    # different length.  Build a common dense grid spanning the union of
    # all time ranges, then interpolate each fit onto that grid before
    # computing the mean and SD.
    mean_ax = axes[n // ncols][n % ncols]

    t_min = min(float(d['t_data'][0])  for d in fit_data)
    t_max = max(float(d['t_data'][-1]) for d in fit_data)
    n_pts = max(200, max(len(d['t_data']) for d in fit_data))
    all_t = np.linspace(t_min, t_max, n_pts)

    y_stack = []
    for k, d in enumerate(fit_data):
        col = palette[k % len(palette)]
        mean_ax.plot(d['t_data'], d['y_data'], '.', color=col,
                     alpha=0.20, ms=2)
        mean_ax.plot(d['t_data'], d['y_fit'],  '-', color=col,
                     lw=0.8, alpha=0.4)
        # Interpolate this GUV's fit onto the shared grid; NaN outside range
        y_interp = np.interp(all_t, d['t_data'], d['y_fit'],
                             left=np.nan, right=np.nan)
        y_stack.append(y_interp)

    if len(y_stack) > 0:
        y_arr  = np.array(y_stack)          # now always (n_guvs, n_pts)
        y_mean = np.nanmean(y_arr, axis=0)
        y_sd   = np.nanstd(y_arr,  axis=0)
        mean_ax.plot(all_t, y_mean, 'k-', lw=2.5, label='Mean fit')
        mean_ax.fill_between(all_t, y_mean - y_sd, y_mean + y_sd,
                             color='black', alpha=0.12, label='±1 SD')

    mean_ax.axvline(0, color='gray', ls='--', lw=1.2, label='Pulse', zorder=0)
    mean_ax.set_title('Population mean (fits)', fontsize=8)
    mean_ax.set_xlabel('Time (s)', fontsize=8)
    mean_ax.set_ylabel('Norm. intensity', fontsize=8)
    mean_ax.tick_params(labelsize=7)
    mean_ax.grid(True, alpha=0.25)
    mean_ax.legend(fontsize=6)

    # Hide unused panels
    for k in range(n_panels, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.suptitle(
        f'{experiment_name}  –  Individual kinetic fits  ({model_name})',
        fontsize=11
    )
    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_fits_grid.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_dye_summary(
        aligned: dict,
        output_folder: str,
        experiment_name: str,
        smooth_sigma: float = 0.0,
) -> str:
    """
    Single-panel summary of all normalised dye traces + population mean
    (± 1 SD shading).  Mirrors the leftmost panel of the actin cortex
    analysis figure.

    Parameters
    ----------
    aligned : dict returned by normalize_and_align_curves()
        Keys used: 't_aligned', 'all_curves_aligned', 'average_curve',
                   'valid_guv_ids'
    output_folder : str
    experiment_name : str
    smooth_sigma : float  Gaussian σ (frames) for display smoothing; 0 = off
    """
    t       = aligned['t_aligned']
    arr     = aligned['all_curves_aligned']   
    avg     = aligned['average_curve']
    ids     = aligned['valid_guv_ids']
    palette = plt.cm.tab10.colors

    def _smooth(x):
        if smooth_sigma > 0:
            finite = np.isfinite(x)
            if finite.sum() < 3:
                return x
            s = gaussian_filter1d(np.where(finite, x, 0.0), smooth_sigma)
            n = gaussian_filter1d(finite.astype(float), smooth_sigma)
            return np.where(n > 0, s / n, np.nan)
        return x

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (gid, row) in enumerate(zip(ids, arr)):
        ax.plot(t, _smooth(row), color=palette[i % len(palette)],
                alpha=0.40, lw=1.0, label=f'GUV {gid}')

    # Mean ± SD
    sd  = np.nanstd(arr, axis=0)
    sm  = _smooth(avg)
    sd_sm_lo = _smooth(avg - sd)
    sd_sm_hi = _smooth(avg + sd)

    ax.fill_between(t, sd_sm_lo, sd_sm_hi, color='black', alpha=0.12,
                    label='±1 SD')
    ax.plot(t, sm, 'k-', lw=2.5, label='Mean')

    ax.axvline(0, color='crimson', ls='--', lw=1.5, label='Pulse', zorder=3)
    ax.axhline(1, color='gray',    ls=':',  lw=0.8)

    ax.set_xlabel('Time relative to pulse (s)', fontsize=11)
    ax.set_ylabel('Normalised dye intensity (a.u.)', fontsize=11)
    ax.set_title(
        f'{experiment_name}\nDye channel – all GUVs + population mean',
        fontsize=11
    )
    ax.grid(True, alpha=0.25)

    # Legend: all GUVs + stat lines, arranged in 3 columns to keep it compact
    handles, labels = ax.get_legend_handles_labels()
    ncols_leg = 3 if len(handles) > 10 else 2
    ax.legend(handles, labels, fontsize=8, ncol=ncols_leg,
              loc='upper right' if avg[-1] < avg[0] else 'lower right')

    plt.tight_layout()
    out_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_summary.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path