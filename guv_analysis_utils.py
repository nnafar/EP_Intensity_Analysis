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
# --- 2b. MODEL SELECTION (AIC/AICc/BIC) ---
# -------------------------------------------------------------------

def compute_information_criteria(rss: float, n: int, k: int) -> Tuple[float, float, float]:
    """
    AIC, AICc, and BIC for a least-squares fit, using the standard Gaussian-
    likelihood form for i.i.d. residuals:

        AIC  = n * ln(RSS / n) + 2 * k_eff
        AICc = AIC + 2 * k_eff * (k_eff + 1) / (n - k_eff - 1)
        BIC  = n * ln(RSS / n) + k_eff * ln(n)

    where `k_eff = k + 1` includes the residual variance as an estimated
    parameter (the conventional convention for comparing least-squares fits
    with these criteria, e.g. as used by R's AIC() on an nls() object).

    Lower values indicate a better trade-off between fit quality and model
    complexity. AICc adds a small-sample correction over AIC and should be
    preferred whenever n is not much larger than k_eff (as is typical here:
    tens of frames vs. 4-6 fitted parameters) — it converges to AIC as
    n → ∞. BIC penalises additional parameters more heavily than AICc for
    the sample sizes typical of a single GUV trace, so it will tend to
    favour the simpler (1EXP) model more often; treat agreement between
    AICc and BIC as the stronger signal.

    Parameters
    ----------
    rss : residual sum of squares of the fit
    n   : number of data points fitted
    k   : number of *mean-function* parameters (i.e. len(names) from the
          model spec) — do NOT include the variance itself here.

    Returns
    -------
    (aic, aicc, bic) — NaN for any quantity that is undefined (rss <= 0,
    or, for AICc, when n - k_eff - 1 <= 0, i.e. too few points to support
    the correction).
    """
    if not np.isfinite(rss) or rss <= 0 or n <= 0:
        return np.nan, np.nan, np.nan

    k_eff = k + 1
    aic = float(n * np.log(rss / n) + 2 * k_eff)
    bic = float(n * np.log(rss / n) + k_eff * np.log(n))

    denom = n - k_eff - 1
    aicc = float(aic + (2 * k_eff * (k_eff + 1)) / denom) if denom > 0 else np.nan

    return aic, aicc, bic


# -------------------------------------------------------------------
# --- 3. PULSE-FRAME DETECTION ---
# -------------------------------------------------------------------

def _get_fast_window(time_array: np.ndarray) -> Tuple[int, int]:
    """
    Returns (fast_start, fast_end) — inclusive frame indices of the
    shortest-interval acquisition segment from FRAME_INTERVAL_SCHEDULE,
    or by inspecting the time_array dt distribution.
    The pulse-detection search window is (fast_start - 1, fast_end) so the
    transition INTO the fast segment is included.
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
    if n > 2:
        dt = np.diff(time_array)
        dt = np.where(dt > 0, dt, np.inf)
        min_dt    = dt.min()
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
    1.  Build a (n_guvs × n_frames) net-signal array.
    2.  Restrict the search to the fast-acquisition window only
        (shortest-dt segment from FRAME_INTERVAL_SCHEDULE), including the
        one frame immediately before the window so the pre→fast transition
        is captured.
    3.  Within the fast window all frames share the same dt, so dI/dframe
        is equivalent to dI/dt for ranking.  Compute per-GUV dI/dframe.
    4.  Take the **median** across GUVs at each step — robust to minority
        GUVs with mask dropouts or transient tracking failures, with no
        smoothing required.
    5.  Pulse frame = frame that *arrives* at the steepest median drop
        (search_lo + argmin + 1).

    smooth_sigma is used only for the display trace in the diagnostic plot.
    """
    if not intensity_curves:
        return 0

    n_frames = len(intensity_curves[0])
    net_arr  = np.vstack([
        np.asarray(d, float) - np.asarray(b, float)
        for d, b in zip(intensity_curves, background_curves)
    ])
    mean_trace = np.nanmean(net_arr, axis=0)

    if time_array is not None and len(time_array) >= n_frames:
        t = time_array[:n_frames]
        fast_start, fast_end = _get_fast_window(t)
    else:
        fast_start, fast_end = 0, n_frames - 1
        t = np.arange(n_frames, dtype=float)

    search_lo = max(0, fast_start - 1)
    
    # Restrict the upper bound if a max frame limit is set
    max_search = getattr(cfg, 'PULSE_SEARCH_MAX_FRAMES', None)
    if max_search is not None:
        search_hi = min(fast_end, search_lo + max_search)
    else:
        search_hi = fast_end

    window_arr   = net_arr[:, search_lo : search_hi + 1]
    per_guv_diff = np.diff(window_arr, axis=1)
    
    # Extract corresponding time window and compute actual dt
    window_t = t[search_lo : search_hi + 1]
    dt = np.diff(window_t)
    dt = np.where(dt <= 0, 1e-6, dt) # Prevent division by zero
    
    # Compute the true rate of change (dI/dt)
    per_guv_rate = per_guv_diff / dt[np.newaxis, :]
    median_rate  = np.nanmedian(per_guv_rate, axis=0)

    argmin_local = int(np.argmin(median_rate))
    pulse_frame  = search_lo + argmin_local + 1

    if output_folder:
        dt_min     = max(float(np.diff(t).min()), 1e-6)
        dt_uniform = min(0.01, dt_min / 2.0)
        t_unif     = np.arange(t[0], t[-1] + dt_uniform, dt_uniform)
        sig_unif   = np.interp(t_unif, t, mean_trace)
        sm_unif    = gaussian_filter1d(sig_unif, sigma=smooth_sigma / dt_uniform)
        smoothed   = np.interp(t, t_unif, sm_unif)

        frames = np.arange(n_frames)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(frames, mean_trace, color='steelblue', lw=1.0, alpha=0.6,
                     label='Mean net signal (raw)')
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
        axes[1].bar(win_frames, median_rate, color='darkorange', alpha=0.7,
                    label='Median dI/dt across GUVs (within window)')
        axes[1].axvline(pulse_frame - 0.5, color='crimson', ls='--', lw=1.5)
        axes[1].axhline(0, color='gray', lw=0.5, ls=':')
        axes[1].set_xlabel('Frame index')
        axes[1].set_ylabel('Median dI/dt (AU/s)')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder,
                    f"{experiment_name}_pulse_frame_detection.png"), dpi=150)
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
    """Generates inner, membrane, and background boolean masks using a memory-optimized local bounding box."""
    h, w = shape[:2]
    cx, cy = center
    a, b = axes

    # Resolve symmetric vs. asymmetric half-width
    if isinstance(mem_hw, (tuple, list)):
        mem_inner_hw, mem_outer_hw = mem_hw
    else:
        mem_inner_hw = mem_outer_hw = mem_hw

    # 1. Define a localized bounding box to prevent massive RAM allocation
    max_r = max(a, b) + mem_outer_hw + bg_buf + bg_w + 5
    x_min = max(0, int(cx - max_r))
    x_max = min(w, int(cx + max_r))
    y_min = max(0, int(cy - max_r))
    y_max = min(h, int(cy + max_r))

    # Check if the GUV is entirely out of bounds
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool)

    # Create coordinate grid ONLY for the local bounding box
    y, x = np.ogrid[y_min:y_max, x_min:x_max]
    
    # Rotate coordinates to align with ellipse angle
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    x_rot = cos_t * (x - cx) + sin_t * (y - cy)
    y_rot = -sin_t * (x - cx) + cos_t * (y - cy)

    # Compute normalized elliptical distance squared
    dist_sq = (x_rot**2) / (a**2) + (y_rot**2) / (b**2)
    dist = np.sqrt(dist_sq)

    # Calculate fractional thresholds based on semi-major axis
    mem_inner_frac = max(0.01, (a - mem_inner_hw) / a)
    mem_outer_frac = (a + mem_outer_hw) / a
    bg_inner_frac  = (a + mem_outer_hw + bg_buf) / a
    bg_outer_frac  = (a + mem_outer_hw + bg_buf + bg_w) / a

    # Generate masks in local space
    local_inner = dist < mem_inner_frac
    local_mem   = (dist >= mem_inner_frac) & (dist <= mem_outer_frac)
    local_bg    = (dist >= bg_inner_frac)  & (dist <= bg_outer_frac)

    # 2. Embed the local bool masks into the full-frame bool arrays (lightweight)
    inner_mask = np.zeros((h, w), dtype=bool)
    mem_mask   = np.zeros((h, w), dtype=bool)
    bg_mask    = np.zeros((h, w), dtype=bool)

    inner_mask[y_min:y_max, x_min:x_max] = local_inner
    mem_mask[y_min:y_max, x_min:x_max]   = local_mem
    bg_mask[y_min:y_max, x_min:x_max]    = local_bg

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
    Locate the membrane peak and return its FWHM using a shared extracellular
    baseline so that an elevated lumen plateau does not artificially narrow
    the measured width.

    Both the left and right crossings use the same threshold:
        half_max = right_base + (peak_val - right_base) * 0.5
    where right_base is the minimum of the profile on the outer (extracellular)
    side of the peak.  The lumenal plateau is cortex-adjacent signal, not
    background, and must not be used as the inward baseline.

    lo_override / hi_override
        When provided, replace the generic search window with the actual
        membrane inner/outer boundary from the ROI channel.
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

    # Establish left (lumenal) baseline using median to reject noise
    inner_end  = max(1, int(expected_r * 0.70))
    left_base  = float(np.median(profile[:inner_end]))
    
    if peak_val <= left_base:
        return None

    # Establish right (extracellular) baseline
    band       = max(5, (hi - lo) // 2)
    right_zone = profile[peak_idx : min(n, hi + band)]
    right_base = float(right_zone.min()) if len(right_zone) > 0 else float(profile.min())

    # Prominence check against the deeper right-side baseline
    amplitude     = peak_val - right_base
    profile_range = float(profile.max() - profile.min())
    if profile_range > 0 and (amplitude / profile_range) < min_prominence_fraction:
        return None

    # Calculate independent half-max thresholds for asymmetric peaks
    left_half_max  = left_base + (peak_val - left_base) * 0.5
    right_half_max = right_base + (peak_val - right_base) * 0.5

    # Left crossing using left_half_max
    left_idx = lo
    for j in range(peak_idx - 1, -1, -1):
        if profile[j] <= left_half_max:
            left_idx = j + 1
            break

    # Right crossing using right_half_max
    right_idx = hi
    for j in range(peak_idx + 1, min(n, hi + band)):
        if profile[j] <= right_half_max:
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
) -> Tuple[Tuple[int, int], float, bool]:
    """
    Translational search for the GUV centre using ring-quality scoring.

    Strategy
    --------
    Evaluate the ring-score at every grid point inside a circle of radius
    `prev_radius × search_window_factor` around `prev_center`.  Return the
    highest-scoring candidate.

    Returns
    -------
    (best_center, best_score, in_bounds)
        in_bounds is False only when EVERY candidate in the search grid was
        too close to the frame edge to fit a full radial profile — i.e. the
        vesicle has drifted to (or past) the field-of-view boundary. This is
        the signal used upstream to distinguish "lost because it left the
        frame" from "lost because the ring genuinely disappeared" (rupture).
        When True, at least one candidate was scored normally, even if the
        best score is still low for other reasons (e.g. rupture).
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
    in_bounds   = False
    
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

            in_bounds = True
            prof = radial_profile_local(local_frame, (local_cx, local_cy), r_max, 72)
            score = _ring_score_from_profile(prof, r, search_factor, detect_bright)
            
            if score > best_score:
                best_score  = score
                best_center = (ncx, ncy)

    return best_center, best_score, in_bounds


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
    exit_reason     = None   # 'RUPTURED' | 'OUT_OF_FRAME' | None
    n_valid         = 0

    # Flags for the CURRENT consecutive-low-score streak: True where that
    # frame's failure was because the search grid fell outside the frame
    # (find_best_guv_center found no evaluable candidate), False where a
    # candidate WAS evaluated but scored poorly (consistent with rupture).
    # Reset whenever a good-quality frame breaks the streak.
    streak_boundary_flags: List[bool] = []

    for i in range(n_frames):
        frame = roi_stack[i]
        if frame is None:
            consecutive_low += 1
            streak_boundary_flags.append(False)  # missing frame, not a boundary issue
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                exit_reason = 'RUPTURED'
                break
            continue

        pred_cx = cx + vx
        pred_cy = cy + vy
        avg_r = (axes[0] + axes[1]) / 2.0

        (new_cx, new_cy), score, in_bounds = find_best_guv_center(
            frame, (pred_cx, pred_cy), avg_r,
            search_window_factor, search_factor,
            detect_bright, grid_step_factor
        )
        ring_scores[i] = score

        if score < rupture_score_threshold:
            consecutive_low += 1
            streak_boundary_flags.append(not in_bounds)
            if consecutive_low >= rupture_consecutive_fails:
                ruptured_at = i - consecutive_low + 1
                # Majority vote over the streak: if most of the failing
                # frames had no evaluable ring candidate at all (vesicle at
                # or past the FOV edge), this is a tracking limitation, not
                # a biological rupture.
                n_boundary = sum(streak_boundary_flags)
                exit_reason = ('OUT_OF_FRAME'
                                if n_boundary > len(streak_boundary_flags) / 2.0
                                else 'RUPTURED')
                break
        else:
            consecutive_low = 0
            streak_boundary_flags = []

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
        'exit_reason':    exit_reason,   # 'RUPTURED' | 'OUT_OF_FRAME' | None
        'n_valid_frames': n_valid,
    }


# -------------------------------------------------------------------
# --- 7. INTENSITY EXTRACTION ---
# -------------------------------------------------------------------

def get_intensity_trace_tracked(dye_stack: np.ndarray,
                                n_frames: int,
                                per_frame_masks: List[Optional[np.ndarray]],
                                method: str = 'mean',
                                percentile: float = 50.0
                                ) -> np.ndarray:
    """
    Extract intensity inside *per_frame_masks[i]* from *dye_files[i]*.
    Returns NaN for frames whose mask is None (post-rupture / untracked).

    method : 'mean' | 'median' | 'percentile'
        'percentile' takes `percentile` of the masked pixels; at 50.0 it is
        identical to 'median'. Used for the background annulus so the
        background estimator is set in one place (cfg.BG_PERCENTILE) rather
        than hard-coded.
    """
    trace = np.full(n_frames, np.nan)
    for i, mask in enumerate(per_frame_masks):
        if mask is None:
            continue
        
        pixels = dye_stack[i][mask]
        if len(pixels) == 0:
            continue
            
        if method == 'percentile':
            trace[i] = float(np.percentile(pixels, percentile))
        elif method == 'median':
            trace[i] = float(np.median(pixels))
        else:
            trace[i] = float(np.mean(pixels))
        
    return trace

def find_closest_frame(time_array: np.ndarray, target: float) -> Tuple[int, float]:
    """Find the index and actual time value closest to the target time."""
    idx = int(np.abs(time_array - target).argmin())
    return idx, float(time_array[idx])


def recompute_background_traces(
        dye_stack: np.ndarray,
        n_frames: int,
        all_ellipses: List[List[Optional[Dict]]],
        target_idx: int,
        membrane_half_width,
        bg_buffer: int,
        bg_width: int,
        img_shape: tuple,
        exclusion_padding: int = 2,
        min_bg_pixels: int = 20,
        neighbor_hold_frames: int = 5,
        sigma_clip: float = 3.0,
        sigma_clip_candidates: Optional[List[float]] = None,
        percentile: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Re-extracts the background median AND std for one GUV, frame by frame,
    excluding pixels that belong to a neighboring GUV.

    Two independent lines of defense are used, because "neighboring GUV" can
    mean a vesicle we know the position of or one we don't:

    1. Position-based exclusion (tracked neighbors)
       ------------------------------------------
       For every OTHER GUV that has a valid tracked ellipse *this frame*,
       its inner + membrane footprint (padded by `exclusion_padding`) is
       removed from the annulus, as before. If a neighbor's ellipse is
       momentarily missing (transient tracking loss, not yet declared
       ruptured), its *last known* position is held and still excluded for
       up to `neighbor_hold_frames` frames — a vesicle doesn't vanish just
       because one frame's ring-score search failed. Beyond that grace
       period the hold is dropped (either it ruptured and dispersed, or it
       drifted out of frame, in which case continuing to exclude a stale
       position would just shrink the usable background for no benefit).

    2. Statistical outlier rejection (untracked or unselected neighbors)
       -------------------------------------------------------------
       Any vesicle the user never circled has no entry in `all_ellipses` at
       all, so step 1 cannot know about it. After position-based exclusion,
       a MAD-based sigma-clip is applied directly to the remaining annulus
       pixel *values*: pixels more than `sigma_clip` scaled-MADs from the
       median are dropped before computing statistics. This catches a
       bright membrane rim or dark lumen from an untracked/unselected
       neighbor sitting in the ring without needing to know where it is.

    Choice of estimator
    -------------------
    The value finally reported per frame is `percentile` of the surviving
    annulus pixels (cfg.BG_PERCENTILE; 50.0 reproduces the median that this
    function used previously). The MAD sigma-clip in step 2 deliberately
    keeps using the TRUE MEDIAN as its centre regardless of `percentile` —
    centring a symmetric ±sigma clip on, say, the 25th percentile would
    make the rejection lopsided, throwing away the upper half of a
    perfectly clean annulus. So `percentile` changes what is reported, not
    which pixels are considered contaminated.

    Diagnostics for tuning `sigma_clip`
    ------------------------------------
    While the raw (position-excluded, pre-statistical-clip) pixels for a
    frame are still in memory, the median/MAD are recorded, and the
    exclusion fraction that *would* result is computed for each value in
    `sigma_clip_candidates` — not just the active `sigma_clip`. This makes
    it possible to inspect, after the fact, how aggressive different
    thresholds would have been on real data (e.g. "does sigma=2.5 already
    exclude pixels even on frames with no neighbor overlap?") without
    re-running the pipeline for every candidate value.

    Parameters
    ----------
    all_ellipses : list of length n_guvs, each a list of length n_frames
        of {'center','axes','angle'} dicts (or None) — the tracked GUVs'
        light tracking['ellipses'] output. Vesicles never circled by the
        user simply have no entry here and are only caught by step 2.
    target_idx : index into all_ellipses for the GUV being processed.
    neighbor_hold_frames : how many consecutive frames to keep excluding a
        tracked neighbor's last known footprint after its ellipse goes
        missing, before giving up on that neighbor for this frame.
    sigma_clip : MAD-multiplier threshold for the statistical outlier
        rejection step actually applied. Set to 0 or None to disable step 2
        (diagnostics are still computed).
    sigma_clip_candidates : list of MAD-multiplier values to evaluate for
        diagnostic purposes only (does not affect the returned traces).
        Defaults to [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0].

    Returns
    -------
    (bg_value_trace, bg_std_trace, n_excluded_pixels_trace, diagnostics)
        bg_value_trace : (n_frames,) float array of the `percentile` of the
            surviving pixels (median when percentile=50). NaN where no
            ellipse/frame was available. Reflects the ACTIVE `sigma_clip`.
        bg_std_trace : (n_frames,) per-pixel std of the surviving pixels.
        n_excluded_pixels_trace : (n_frames,) int array — total pixels
            removed by either mechanism at the active threshold.
        diagnostics : dict of (n_frames,) arrays —
            'n_pixels_total'        : ring pixels remaining after position exclusion
            'n_excluded_position'   : pixels removed by neighbor-position exclusion
            'n_excluded_stats'      : pixels removed by the active sigma-clip
            'bg_median_raw'         : median before statistical clipping
                (also the centre used for the MAD clip)
            'bg_percentile_raw'     : the ACTIVE `percentile` before
                statistical clipping — compare against 'bg_median_raw' to
                see how much the estimator choice actually moves I_bg,t
            'bg_mean_raw'           : mean before statistical clipping
            'bg_std_raw'            : std before statistical clipping
            'bg_mad_raw'            : median absolute deviation (unscaled)
            'bg_scaled_mad_raw'     : MAD * 1.4826 (normal-consistent scale)
            'frac_excluded_sigma_{s}' : one array per candidate in
                sigma_clip_candidates — fraction of raw pixels that would be
                excluded at that threshold.
    """
    if sigma_clip_candidates is None:
        sigma_clip_candidates = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    n_guvs = len(all_ellipses)
    bg_value  = np.full(n_frames, np.nan)   # the reported estimator (percentile)
    bg_std    = np.full(n_frames, np.nan)
    n_excl    = np.zeros(n_frames, dtype=int)

    n_pixels_total      = np.zeros(n_frames, dtype=int)
    n_excluded_position = np.zeros(n_frames, dtype=int)
    n_excluded_stats    = np.zeros(n_frames, dtype=int)
    bg_median_raw       = np.full(n_frames, np.nan)
    bg_percentile_raw   = np.full(n_frames, np.nan)
    bg_mean_raw          = np.full(n_frames, np.nan)
    bg_std_raw            = np.full(n_frames, np.nan)
    bg_mad_raw            = np.full(n_frames, np.nan)
    bg_scaled_mad_raw     = np.full(n_frames, np.nan)
    frac_excluded_by_sigma = {
        s: np.full(n_frames, np.nan) for s in sigma_clip_candidates
    }

    diagnostics = {
        'n_pixels_total':      n_pixels_total,
        'n_excluded_position': n_excluded_position,
        'n_excluded_stats':    n_excluded_stats,
        'bg_median_raw':       bg_median_raw,
        'bg_percentile_raw':   bg_percentile_raw,
        'bg_mean_raw':         bg_mean_raw,
        'bg_std_raw':          bg_std_raw,
        'bg_mad_raw':          bg_mad_raw,
        'bg_scaled_mad_raw':   bg_scaled_mad_raw,
    }
    for s in sigma_clip_candidates:
        diagnostics[f'frac_excluded_sigma_{s}'] = frac_excluded_by_sigma[s]

    if dye_stack is None:
        return bg_value, bg_std, n_excl, diagnostics

    target_ellipses = all_ellipses[target_idx]

    # Per-neighbor "last known position" cache for the hold-frames mechanism.
    last_known: Dict[int, Dict] = {}
    missing_streak: Dict[int, int] = {}

    for i in range(n_frames):
        el = target_ellipses[i] if i < len(target_ellipses) else None
        if el is None:
            continue

        frame = dye_stack[i]
        if frame is None:
            continue

        _, _, bg_mask = generate_vectorized_masks(
            img_shape, el['center'], el['axes'], el['angle'],
            membrane_half_width, bg_buffer, bg_width,
        )
        if bg_mask is None or not bg_mask.any():
            continue

        # --- Step 1: position-based exclusion of tracked neighbors ---
        occupied = np.zeros(img_shape[:2], dtype=bool)
        pad_hw = membrane_half_width + exclusion_padding
        for j in range(n_guvs):
            if j == target_idx:
                continue
            other_list = all_ellipses[j]
            other_el = other_list[i] if i < len(other_list) else None

            if other_el is not None:
                last_known[j] = other_el
                missing_streak[j] = 0
            else:
                missing_streak[j] = missing_streak.get(j, 0) + 1
                if j in last_known and missing_streak[j] <= neighbor_hold_frames:
                    other_el = last_known[j]   # hold last known position
                else:
                    other_el = None            # gap too long — give up on it

            if other_el is None:
                continue

            other_inner, other_mem, _ = generate_vectorized_masks(
                img_shape, other_el['center'], other_el['axes'], other_el['angle'],
                pad_hw, bg_buffer, bg_width,
            )
            if other_inner is not None:
                occupied |= other_inner
            if other_mem is not None:
                occupied |= other_mem

        effective_bg = bg_mask & ~occupied
        excluded_by_position = int(bg_mask.sum() - effective_bg.sum())

        if effective_bg.sum() < min_bg_pixels:
            # Not enough clean pixels remain — fall back to the full
            # annulus rather than computing stats on a near-empty sample.
            effective_bg = bg_mask
            excluded_by_position = 0

        pixels = frame[effective_bg]
        if len(pixels) == 0:
            continue

        n_pixels_total[i]      = len(pixels)
        n_excluded_position[i] = excluded_by_position
        bg_median_raw[i]     = float(np.median(pixels))
        bg_percentile_raw[i] = float(np.percentile(pixels, percentile))
        bg_mean_raw[i]       = float(np.mean(pixels))
        bg_std_raw[i]    = float(np.std(pixels))

        med = bg_median_raw[i]
        mad = float(np.median(np.abs(pixels - med)))
        scaled_mad = mad * 1.4826   # normal-consistent scale estimate
        bg_mad_raw[i]        = mad
        bg_scaled_mad_raw[i] = scaled_mad

        # --- Diagnostics: exclusion fraction at each candidate threshold ---
        if scaled_mad > 0:
            dev = np.abs(pixels - med)
            for s in sigma_clip_candidates:
                frac_excluded_by_sigma[s][i] = float(np.mean(dev > s * scaled_mad))
        else:
            for s in sigma_clip_candidates:
                frac_excluded_by_sigma[s][i] = 0.0

        # --- Step 2: statistical outlier rejection at the ACTIVE threshold
        # (catches untracked/unselected neighbors that step 1 has no
        # position for) ---
        excluded_by_stats = 0
        if sigma_clip and scaled_mad > 0 and len(pixels) >= max(min_bg_pixels, 10):
            keep = np.abs(pixels - med) <= sigma_clip * scaled_mad
            if keep.sum() >= min_bg_pixels:
                excluded_by_stats = int(len(pixels) - keep.sum())
                pixels = pixels[keep]

        n_excluded_stats[i] = excluded_by_stats
        n_excl[i] = excluded_by_position + excluded_by_stats
        bg_value[i] = float(np.percentile(pixels, percentile))
        bg_std[i]   = float(np.std(pixels))

    return bg_value, bg_std, n_excl, diagnostics


def export_background_diagnostics_csv(
        diagnostics_per_guv: Dict[str, Dict[str, np.ndarray]],
        time_array: np.ndarray,
        output_folder: str,
        experiment_name: str,
        sigma_clip_active: float,
        logger: Optional[logging.Logger] = None,
) -> str:
    """
    Writes a tidy long-format CSV (one row per GUV per frame) of the raw
    background statistics and candidate sigma-clip exclusion fractions
    produced by `recompute_background_traces`, so `BG_SIGMA_CLIP` can be
    tuned from real data instead of guessed.

    How to use it to pick BG_SIGMA_CLIP
    ------------------------------------
    - `frac_excluded_sigma_X` columns show what fraction of ring pixels
      would be dropped at threshold X, per frame. On a "clean" frame (no
      neighbor nearby) this should be small and roughly constant across
      candidates — if a low threshold (e.g. 2.0) is already excluding a
      meaningful fraction on clean frames, it's too aggressive and will
      eat into legitimate background pixels.
    - On frames where `n_excluded_position` is 0 but you independently know
      (e.g. by eye, from the ROI images) that an untracked neighbor is
      present, look at which threshold first produces a large
      `frac_excluded_sigma_X` jump — that's roughly the smallest threshold
      that still catches it.
    - `bg_std_raw` vs `bg_scaled_mad_raw` compares the ordinary std (pulled
      up by contamination) against the outlier-robust MAD-based scale —
      a large gap between them is itself a sign of contamination.

    Also logs a one-line summary (mean raw background level, mean
    scaled-MAD, and the average exclusion fraction at the currently
    configured `sigma_clip_active`) if a logger is provided.
    """
    rows = []
    for gid, diag in diagnostics_per_guv.items():
        n_frames = len(diag['n_pixels_total'])
        for i in range(n_frames):
            t = float(time_array[i]) if i < len(time_array) else np.nan
            row = {
                'guv_id': gid,
                'frame': i,
                'time_s': t,
                'n_pixels_total':      diag['n_pixels_total'][i],
                'n_excluded_position': diag['n_excluded_position'][i],
                'n_excluded_stats':    diag['n_excluded_stats'][i],
                'bg_median_raw':       diag['bg_median_raw'][i],
                'bg_percentile_raw':   diag['bg_percentile_raw'][i],
                'bg_mean_raw':         diag['bg_mean_raw'][i],
                'bg_std_raw':          diag['bg_std_raw'][i],
                'bg_mad_raw':          diag['bg_mad_raw'][i],
                'bg_scaled_mad_raw':   diag['bg_scaled_mad_raw'][i],
            }
            for key in diag:
                if key.startswith('frac_excluded_sigma_'):
                    row[key] = diag[key][i]
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        output_folder, f'{experiment_name}_background_diagnostics.csv'
    )
    df.to_csv(csv_path, index=False, float_format='%.6f', na_rep='NaN')

    if logger is not None and not df.empty:
        mean_raw   = np.nanmean(df['bg_median_raw'])
        mean_pct   = np.nanmean(df['bg_percentile_raw'])
        mean_mad   = np.nanmean(df['bg_scaled_mad_raw'])
        bg_pct     = getattr(cfg, 'BG_PERCENTILE', 50.0)
        active_col = f'frac_excluded_sigma_{sigma_clip_active}'
        if active_col in df.columns:
            mean_excl_active = np.nanmean(df[active_col])
            logger.info(
                f"Background diagnostics: mean raw median = {mean_raw:.2f}, "
                f"mean scaled-MAD = {mean_mad:.2f}, average fraction excluded "
                f"at sigma_clip={sigma_clip_active} = {mean_excl_active:.3%}."
            )
        else:
            logger.info(
                f"Background diagnostics: mean raw median = {mean_raw:.2f}, "
                f"mean scaled-MAD = {mean_mad:.2f}."
            )

        # The active estimator is reported alongside the median, and their
        # difference is the downward bias that BG_PERCENTILE introduces. It
        # is the number to check before trusting a non-50 setting, so it is
        # logged rather than left buried in the diagnostics CSV.
        if abs(bg_pct - 50.0) > 1e-6:
            offset = mean_raw - mean_pct
            rel = (offset / mean_mad) if np.isfinite(mean_mad) and mean_mad > 0 else np.nan
            logger.info(
                f"  Active estimator = {bg_pct:g}th percentile "
                f"(mean = {mean_pct:.2f}), i.e. {offset:.2f} AU below the "
                f"median" + (f" ({rel:.2f} x scaled-MAD)." if np.isfinite(rel) else ".")
            )
            logger.info(
                "  This offset shifts fully-permeabilised GUVs below zero in "
                "the normalised curves; the pre-pulse end stays pinned at 1."
            )

    return csv_path


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
    background_trace = get_intensity_trace_tracked(
        dye_stack, n_frames, tracking['bg_masks'], method='percentile',
        percentile=getattr(cfg, 'BG_PERCENTILE', 50.0)
    )

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
    exit_reason = tracking.get('exit_reason')
    n_valid     = tracking['n_valid_frames']
    
    quality_entry = {
        'guv_id': guv_id, 'initial_radius': initial_radius, 'failed': ruptured_at is not None,
        'ruptured_at_frame': ruptured_at, 'exit_reason': exit_reason, 'n_valid_frames': n_valid,
        'mean_ring_score': float(np.mean([s for s in tracking['ring_scores'] if s > 0])) if n_valid > 0 else 0.0,
        'comments': f"{exit_reason}_AT_F{ruptured_at}" if ruptured_at is not None else 'OK',
    }

    if getattr(cfg, 'EXPORT_TRACK_VISUALIZATION', True):
        export_track_visualization(
            cfg.FOLDER_TRACKING, cfg.EXPERIMENT_BASE_NAME, guv_id,
            roi_stack, tracking['centers'], tracking['ellipses'],
            ruptured_at, exit_reason
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
                                roi_stack, centers, ellipses, ruptured_at,
                                exit_reason=None):
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
    # Red = ruptured (biological event); blue = lost to FOV edge (tracking
    # limitation, not a rupture); cyan = survived to the end of the movie.
    if ruptured_at is None:
        ec = (0, 220, 220)
    elif exit_reason == 'OUT_OF_FRAME':
        ec = (216, 125, 59)   # BGR blue, matches the #3B7DD8 used in plots
    else:
        ec = (0, 0, 220)
    if elf:
        cv2.ellipse(canvas, (int(elf['center'][0]), int(elf['center'][1])),
                    (int(elf['axes'][0]), int(elf['axes'][1])), elf['angle'], 0, 360, ec, 2, cv2.LINE_AA)
    cv2.circle(canvas, (int(cf[0]), int(cf[1])), 3, ec, -1)

    if ruptured_at is None:
        status_label = 'SURVIVED'
    elif exit_reason == 'OUT_OF_FRAME':
        status_label = f'OUT OF FRAME @F{ruptured_at}'
    else:
        status_label = f'RUPTURED @F{ruptured_at}'
    label = f"GUV {guv_id}  {status_label}"
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


def classify_guv_fates(
        df_track: pd.DataFrame,
        ruptured_guv_ids: set,
        out_of_frame_guv_ids: set,
        shrinkage_fraction_threshold: float = 0.15,
        terminal_n_frames: int = 3,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Refines the tracker's binary RUPTURED/OUT_OF_FRAME exit classification
    into four mutually exclusive fates, adding a SHRUNK category:

      'OUT_OF_FRAME' - unchanged: tracking was lost because the vesicle
                        drifted past the field-of-view edge. A tracking
                        limitation, not a biological event.
      'SHRUNK'       - the vesicle's radius declined by at least
                        shrinkage_fraction_threshold from its own pre-pulse
                        baseline (norm_radius, from normalize_tracking_data),
                        regardless of how tracking ended. This deliberately
                        overrides a raw 'RUPTURED' tag: a vesicle that
                        deflates gradually and only THEN drops below the
                        ring-detector's minimum resolvable size will also
                        trigger the tracker's rupture-score exit condition,
                        but that's a detection-limit artifact of shrinkage,
                        not a membrane burst — lumping the two together
                        would misrepresent slow deflation as catastrophic
                        rupture.
      'RUPTURED'     - ring score collapsed abruptly WITHOUT a preceding,
                        sustained radius decline — i.e. the vesicle was
                        still close to its pre-pulse size right up to the
                        point tracking was lost. This is what's left of the
                        tracker's raw 'RUPTURED' tag after shrinkage-driven
                        exits are pulled out.
      'SURVIVED'     - tracked through to the end of the movie with no
                        meaningful (< threshold) radius loss.

    terminal_n_frames : how many of a GUV's LAST valid tracked frames are
        averaged (in norm_radius) to get its terminal size — smooths
        single-frame ring-detection noise right at the endpoint rather than
        keying the whole classification off one potentially noisy frame.

    Requires df_track already have 'norm_radius' (radius / pre-pulse mean
    radius per GUV) — i.e. normalize_tracking_data() must be called first.

    Returns
    -------
    (fate_map, df_fate)
        fate_map : dict of guv_id (str) -> fate label
        df_fate  : one row per GUV with guv_id, fate, frac_radius_loss
                   (positive = shrinkage, NaN if no valid radius data or
                   OUT_OF_FRAME), terminal_norm_radius, exit_reason_raw
                   (the tracker's original, pre-refinement tag)
    """
    fate_map: Dict[str, str] = {}
    records = []

    for gid_raw, g in df_track.groupby('guv_id'):
        gid = str(gid_raw)
        exit_reason_raw = (
            'RUPTURED' if gid in ruptured_guv_ids else
            'OUT_OF_FRAME' if gid in out_of_frame_guv_ids else
            None
        )

        if exit_reason_raw == 'OUT_OF_FRAME':
            fate_map[gid] = 'OUT_OF_FRAME'
            records.append({
                'guv_id': gid, 'fate': 'OUT_OF_FRAME',
                'frac_radius_loss': np.nan, 'terminal_norm_radius': np.nan,
                'exit_reason_raw': exit_reason_raw,
            })
            continue

        norm_r = g.sort_values('frame')['norm_radius'].dropna().values
        if len(norm_r) == 0:
            fate = exit_reason_raw or 'SURVIVED'
            fate_map[gid] = fate
            records.append({
                'guv_id': gid, 'fate': fate,
                'frac_radius_loss': np.nan, 'terminal_norm_radius': np.nan,
                'exit_reason_raw': exit_reason_raw,
            })
            continue

        terminal_norm_r = float(np.mean(norm_r[-terminal_n_frames:]))
        frac_loss = 1.0 - terminal_norm_r   # positive = net shrinkage

        if frac_loss >= shrinkage_fraction_threshold:
            fate = 'SHRUNK'
        elif exit_reason_raw == 'RUPTURED':
            fate = 'RUPTURED'
        else:
            fate = 'SURVIVED'

        fate_map[gid] = fate
        records.append({
            'guv_id': gid, 'fate': fate,
            'frac_radius_loss': frac_loss, 'terminal_norm_radius': terminal_norm_r,
            'exit_reason_raw': exit_reason_raw,
        })

    df_fate = pd.DataFrame(records)
    return fate_map, df_fate


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
    BG            : background annulus at cfg.BG_PERCENTILE (median at 50)
    FWHM          : full-width at half-maximum of the membrane peak, in both px
                    and µm (µm requires *microns_per_pixel* to be set correctly)

    Returns
    -------
    dict with keys:
        'cortex_trace'      – mean intensity in the FWHM cortex ring (n_frames,)
        'peak_intensity_trace' – max of radial profile within FWHM   (n_frames,)
        'lumen_trace'       – mean intensity in the GUV lumen         (n_frames,)
        'bg_trace'          – background annulus percentile            (n_frames,)
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
            # Same estimator as the dye background (cfg.BG_PERCENTILE), so
            # the two channels' backgrounds are defined consistently.
            bg_tr[i] = float(np.percentile(frame[bg_m_base],
                                           getattr(cfg, 'BG_PERCENTILE', 50.0)))

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


def _guv_status_style(gid, ruptured_guv_ids: Optional[set] = None,
                       out_of_frame_guv_ids: Optional[set] = None,
                       shrunk_guv_ids: Optional[set] = None) -> Tuple[str, str]:
    """
    Returns (color, label_suffix) for one GUV based on how its tracking
    ended. Rupture, shrinkage, and out-of-frame are kept visually and
    categorically distinct everywhere a plot colors GUVs by status:
    rupture is an abrupt biological event, shrinkage is a gradual one
    (and should not be counted as rupture), and drifting out of the
    tracked field of view is a tracking limitation that should never be
    counted or read as either.
    """
    gid = str(gid)
    if gid in (ruptured_guv_ids or set()):
        return '#CC2222', ' (ruptured)'
    if gid in (shrunk_guv_ids or set()):
        return '#E08214', ' (shrunk)'
    if gid in (out_of_frame_guv_ids or set()):
        return '#3B7DD8', ' (lost, out of frame)'
    return '#888888', ''


def plot_actin_analysis(
        aligned_actin: dict,
        valid_guv_ids: list,
        output_folder: str,
        experiment_name: str,
        smooth_sigma: float = 1.5,
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
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

    C_SURV = '#888888'
    C_RUPT = '#CC2222'

    for gid, p_row, l_row, f_row, g_row in zip(
            valid_guv_ids, peak_arr, lumen_arr, fwhm_arr, gini_arr):
        col, suffix = _guv_status_style(gid, ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids)
        lw    = 1.2 if suffix else 0.9
        label = f'GUV {gid}{suffix}'
        ax_cpeak.plot(t, _smooth(p_row), color=col, alpha=0.75, lw=lw, label=label)
        ax_lumen.plot(t, _smooth(l_row), color=col, alpha=0.75, lw=lw, label=label)
        ax_fwhm .plot(t, _smooth(f_row), color=col, alpha=0.75, lw=lw, label=label)
        ax_gini .plot(t, _smooth(g_row), color=col, alpha=0.75, lw=lw, label=label)

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
    
    # Pre-allocate dictionary to prevent pandas DataFrame fragmentation
    data_dict = {'time_s': t}

    for gid, c_row, p_row, l_row, f_row, g_row, d_row in zip(
            valid_guv_ids,
            aligned_actin['cortex_norm'],
            aligned_actin['peak_norm'],
            aligned_actin['lumen_norm'],
            aligned_actin['fwhm_aligned'],
            aligned_actin['gini_aligned'],
            aligned_actin['peak_detected']):
        
        data_dict[f'GUV_{gid}_cortex_mean']   = c_row
        data_dict[f'GUV_{gid}_cortex_peak']   = p_row
        data_dict[f'GUV_{gid}_lumen']         = l_row
        data_dict[f'GUV_{gid}_fwhm_um']       = f_row
        data_dict[f'GUV_{gid}_gini']          = g_row
        data_dict[f'GUV_{gid}_peak_detected'] = d_row

    data_dict['avg_cortex_mean']  = aligned_actin['avg_cortex']
    data_dict['avg_cortex_peak']  = aligned_actin['avg_peak']
    data_dict['avg_lumen']        = aligned_actin['avg_lumen']
    data_dict['avg_fwhm_um']      = aligned_actin['avg_fwhm']
    data_dict['avg_gini']         = aligned_actin['avg_gini']
    data_dict['peak_detect_frac'] = aligned_actin['peak_detect_frac']

    # Initialize the DataFrame in a single operation
    df = pd.DataFrame(data_dict)

    csv_path = os.path.join(output_folder,
                            f'{experiment_name}_actin_cortex_traces.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f', na_rep='NaN')
    return csv_path


# -------------------------------------------------------------------
# --- ACTIN ANGULAR KYMOGRAPH (directionality of cortex breakdown) ---
# -------------------------------------------------------------------

def build_actin_angular_kymograph(
        ad: dict,
        pulse_frame: int,
        time_array: np.ndarray,
        n_angles: int = 72,
) -> Optional[dict]:
    """
    Stacks one GUV's per-frame angular actin profile (the 'angular_profiles'
    entry from extract_actin_traces — one (n_angles,) array per frame,
    sampled at fixed angles in ABSOLUTE IMAGE-FRAME coordinates around the
    GUV centre, None where no cortex peak was detected that frame) into a
    (n_frames_aligned, n_angles) kymograph: angle vs. time, aligned to
    t = 0 at the pulse frame.

    Because sampling is in image-frame coordinates (not rotated with the
    fitted ellipse), angle 0 in the output is the same physical direction
    for every frame and every GUV in one experiment — which is what makes
    it meaningful to draw electrode-pole reference lines on the result.

    Frames with no detected peak become an all-NaN row rather than being
    dropped, so gaps show up as gaps in the kymograph instead of silently
    compressing the time axis.

    Two intensity representations are returned:
      kymo_raw  : background-subtracted intensity per angle (AU). Only
                  comparable within one GUV — absolute brightness differs
                  GUV-to-GUV with labeling efficiency, so don't pool this
                  across GUVs.
      kymo_norm : each angle's own pre-pulse mean used as that angle's
                  baseline (fold-change, 1 = unchanged). This IS comparable
                  across GUVs, since it factors out per-GUV/per-angle
                  brightness differences and isolates the relative loss
                  pattern — this is what the population-average function
                  pools.

    Returns None if the GUV never had a detected peak in any frame.
    """
    profiles = ad.get('angular_profiles')
    if profiles is None:
        return None

    bg = np.asarray(ad['bg_trace'], float)
    n_frames = len(profiles)
    kymo = np.full((n_frames, n_angles), np.nan)
    for i, prof in enumerate(profiles):
        if prof is None:
            continue
        bg_i = bg[i] if (i < len(bg) and np.isfinite(bg[i])) else 0.0
        kymo[i, :] = np.asarray(prof, float) - bg_i

    if not np.isfinite(kymo).any():
        return None

    safe_pre_end   = pulse_frame if pulse_frame > 0 else 1
    safe_pre_start = max(0, safe_pre_end - 5)
    baseline_per_angle = np.nanmean(kymo[safe_pre_start:safe_pre_end, :], axis=0)

    # Angles whose pre-pulse baseline is non-positive or missing (peak never
    # detected there before the pulse) can't be sensibly turned into a fold
    # change — dividing by ~0 background noise would produce a spurious huge
    # or negative ratio. Those angle columns are left as NaN in kymo_norm
    # rather than showing a fabricated value.
    safe_baseline = np.where(
        np.isfinite(baseline_per_angle) & (baseline_per_angle > 0),
        baseline_per_angle, np.nan
    )

    kymo_aligned = kymo[pulse_frame:, :]
    t_aligned    = time_array[pulse_frame:] - time_array[pulse_frame]

    with np.errstate(divide='ignore', invalid='ignore'):
        kymo_norm = kymo_aligned / safe_baseline[None, :]

    return {
        't_aligned':          t_aligned,
        'kymo_raw':           kymo_aligned,          # (n_frames_aligned, n_angles) AU
        'kymo_norm':          kymo_norm,              # fold-change vs. pre-pulse, per angle
        'baseline_per_angle': baseline_per_angle,
        'angles_deg':         np.degrees(np.linspace(0, 2 * np.pi, n_angles, endpoint=False)),
    }


def _rotate_for_display(data_2d: np.ndarray, angles_deg: np.ndarray,
                         offset_deg: float) -> np.ndarray:
    """
    Rolls a (n_frames, n_angles) array along its angle axis so that, when
    plotted against the UNCHANGED angles_deg tick values (0, step, 2*step,
    ...), row k now shows the data that was physically sampled at
    code_angle = (k*step - offset_deg) mod 360.

    In other words this implements display_angle = (code_angle + offset_deg)
    mod 360 by moving data rather than relabeling ticks, which keeps the
    axis monotonic (no wrap-around jump) and requires no change to how
    angles_deg/masks are computed elsewhere (they stay in code-frame).

    offset_deg must be an integer multiple of the angular step (360 /
    n_angles) so the roll lands exactly on a sampled angle; a fractional
    shift is rounded to the nearest valid step and a warning is not raised
    here — the caller (plotting functions) is expected to pass a config
    value that's already a clean multiple (e.g. 90 deg with 72 angles).
    """
    n_angles = len(angles_deg)
    step = 360.0 / n_angles
    shift = int(round(offset_deg / step)) % n_angles
    return np.roll(data_2d, shift=shift, axis=1)


def plot_actin_angular_kymograph(
        kymo_data: dict,
        guv_id: str,
        output_folder: str,
        experiment_name: str,
        electrode_angle_deg: float = 0.0,
        angle_display_offset_deg: float = 90.0,
        vmax_percentile: float = 99.0,
) -> str:
    """
    Two-panel angular kymograph for one GUV.
      Left  - raw (background-subtracted) cortex intensity, angle vs. time
      Right - normalised to this GUV's own pre-pulse angular baseline
              (fold-change; 1 = unchanged, <1 = lost, >1 = gained)

    The angle axis is shown in DISPLAY convention:
        display_angle = (code_angle + angle_display_offset_deg) mod 360
    Default 90 deg matches the lab diagram: 0 deg at the top of the vesicle
    (an equator reference point, not a pole), increasing clockwise so that
    90 deg = cathode-facing pole and 270 deg = anode-facing pole. This is
    purely a rendering choice — electrode_angle_deg (and everything the
    poles/equator are computed from) stays in code-frame.

    Solid white lines mark the electrode-facing poles; dotted white lines
    mark the equator. A uniform (all-over) breakdown shows a horizontal
    band of loss spanning all angles; a directional (polar) breakdown shows
    loss concentrated at/near the solid lines while the equator stays
    bright.
    """
    t      = kymo_data['t_aligned']
    angles = kymo_data['angles_deg']
    raw    = _rotate_for_display(kymo_data['kymo_raw'],  angles, angle_display_offset_deg)
    norm   = _rotate_for_display(kymo_data['kymo_norm'], angles, angle_display_offset_deg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    finite_raw = raw[np.isfinite(raw)]
    vmax_raw = float(np.percentile(finite_raw, vmax_percentile)) if finite_raw.size else 1.0
    im1 = ax1.pcolormesh(t, angles, raw.T, shading='nearest',
                         cmap='inferno', vmin=0, vmax=max(vmax_raw, 1e-9))
    fig.colorbar(im1, ax=ax1, label='Cortex intensity (bg-subtracted, AU)')
    ax1.set_title('Raw')

    finite_dev = np.abs(norm - 1.0)
    finite_dev = finite_dev[np.isfinite(finite_dev)]
    vlim = float(np.percentile(finite_dev, vmax_percentile)) if finite_dev.size else 1.0
    vlim = max(vlim, 1e-9)
    im2 = ax2.pcolormesh(t, angles, norm.T, shading='nearest',
                         cmap='RdBu_r', vmin=1 - vlim, vmax=1 + vlim)
    fig.colorbar(im2, ax=ax2, label='Fold change vs. pre-pulse (per angle)')
    ax2.set_title('Normalised to pre-pulse baseline (per angle)')

    pole1    = (electrode_angle_deg + angle_display_offset_deg) % 360
    pole2    = (electrode_angle_deg + 180 + angle_display_offset_deg) % 360
    equator1 = (electrode_angle_deg + 90 + angle_display_offset_deg) % 360
    equator2 = (electrode_angle_deg + 270 + angle_display_offset_deg) % 360

    for ax in (ax1, ax2):
        ax.axvline(0, color='cyan', ls='--', lw=1.2, alpha=0.8)
        for pole_ang in (pole1, pole2):
            ax.axhline(pole_ang, color='white', ls='-', lw=1.0, alpha=0.7)
        for eq_ang in (equator1, equator2):
            ax.axhline(eq_ang, color='white', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel('Time relative to pulse (s)')
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])

    ax1.set_ylabel(
        'Angle (deg, 0\u00b0 = top, clockwise)\n'
        'solid = electrode-facing poles, dotted = equator'
    )

    plt.suptitle(f'{experiment_name}  -  GUV {guv_id}  -  '
                f'Actin cortex angular kymograph', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(
        output_folder, f'{experiment_name}_GUV{guv_id}_actin_kymograph.png'
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_actin_angular_kymograph_group(
        kymo_data_list: list,
        guv_ids: list,
        output_folder: str,
        experiment_name: str,
        electrode_angle_deg: float = 0.0,
        angle_display_offset_deg: float = 90.0,
        vmax_percentile: float = 95.0,
) -> Optional[str]:
    """
    Population-average angular kymograph across all actin-positive GUVs in
    one experiment. Only the normalised (fold-change) representation is
    pooled — raw AU is not comparable GUV-to-GUV (labeling efficiency,
    cortex density, and detector settings differ), but each GUV's own
    fold-change relative to its own pre-pulse baseline is.

    Individual GUVs are truncated/padded (NaN) to a common time axis (the
    shortest post-pulse trace among them) before averaging, so early
    ruptures don't bias later time points toward whichever GUVs happen to
    still be tracked.

    Angle axis display convention matches plot_actin_angular_kymograph()
    (see angle_display_offset_deg there) — the underlying averaging happens
    in code-frame; the rotation is applied only for rendering.

    Returns None if no GUV produced a usable kymogram.
    """
    valid = [(gid, kd) for gid, kd in zip(guv_ids, kymo_data_list) if kd is not None]
    if not valid:
        return None

    min_len = min(len(kd['t_aligned']) for _, kd in valid)
    t_common = valid[0][1]['t_aligned'][:min_len]
    angles   = valid[0][1]['angles_deg']

    stack = np.stack([kd['kymo_norm'][:min_len, :] for _, kd in valid], axis=0)
    mean_norm = np.nanmean(stack, axis=0)   # (min_len, n_angles)
    mean_norm = _rotate_for_display(mean_norm, angles, angle_display_offset_deg)
    n_guvs = len(valid)

    fig, ax = plt.subplots(figsize=(8, 6))
    finite_dev = np.abs(mean_norm - 1.0)
    finite_dev = finite_dev[np.isfinite(finite_dev)]
    vlim = float(np.percentile(finite_dev, vmax_percentile)) if finite_dev.size else 1.0
    vlim = max(vlim, 1e-9)
    im = ax.pcolormesh(t_common, angles, mean_norm.T, shading='nearest',
                       cmap='RdBu_r', vmin=1 - vlim, vmax=1 + vlim)
    fig.colorbar(im, ax=ax, label='Mean fold change vs. pre-pulse (per angle)')

    pole1    = (electrode_angle_deg + angle_display_offset_deg) % 360
    pole2    = (electrode_angle_deg + 180 + angle_display_offset_deg) % 360
    equator1 = (electrode_angle_deg + 90 + angle_display_offset_deg) % 360
    equator2 = (electrode_angle_deg + 270 + angle_display_offset_deg) % 360

    ax.axvline(0, color='cyan', ls='--', lw=1.2, alpha=0.8)
    for pole_ang in (pole1, pole2):
        ax.axhline(pole_ang, color='white', ls='-', lw=1.0, alpha=0.7)
    for eq_ang in (equator1, equator2):
        ax.axhline(eq_ang, color='white', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time relative to pulse (s)')
    ax.set_ylabel(
        'Angle (deg, 0\u00b0 = top, clockwise)\n'
        'solid = electrode-facing poles, dotted = equator'
    )
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_title(
        f'{experiment_name}  -  Population-average actin angular kymograph\n'
        f'(n = {n_guvs} GUV(s) with a detected cortex)'
    )
    plt.tight_layout()
    out_path = os.path.join(
        output_folder, f'{experiment_name}_actin_kymograph_population_avg.png'
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def compute_pole_vs_equator_trace(
        kymo_data: dict,
        electrode_angle_deg: float = 0.0,
        angular_half_width_deg: float = 22.5,
) -> dict:
    """
    Collapses a 2-D angular kymogram (from build_actin_angular_kymograph) into
    two direct per-frame scalars, so "does the pole lose actin faster than
    the equator" can be read off a line plot instead of judged by eye on a
    heatmap.

    pole_mean(t)     : mean fold-change (kymo_norm) within
                        +/- angular_half_width_deg of EITHER electrode-facing
                        pole (electrode_angle_deg and +180 deg)
    equator_mean(t)   : same, but within +/- angular_half_width_deg of either
                        equator direction (electrode_angle_deg +/- 90 deg)
    polarization_index(t) : (equator_mean - pole_mean) / (equator_mean + pole_mean)
                        ~ 0 when pole and equator lose actin at the same rate
                          (uniform breakdown)
                        > 0 and growing when the pole loses faster than the
                          equator (directional breakdown at the poles)
                        < 0 if, unexpectedly, the equator loses faster —
                          worth a second look rather than assuming poles.
                        NaN for a frame where both windows are NaN or the
                        pole+equator sum is ~0 (nothing to compare).

    angular_half_width_deg : half-width of the angular window averaged around
        each pole/equator direction. 22.5 deg (default) means each of the 4
        windows spans 45 deg total, so all 4 windows together cover the full
        360 deg with no gaps and no overlap.
    """
    norm   = kymo_data['kymo_norm']
    angles = kymo_data['angles_deg']
    t      = kymo_data['t_aligned']

    def _circ_dist(a, b):
        return np.abs((a - b + 180) % 360 - 180)

    pole1, pole2 = electrode_angle_deg % 360, (electrode_angle_deg + 180) % 360
    eq1, eq2     = (electrode_angle_deg + 90) % 360, (electrode_angle_deg + 270) % 360

    pole_mask = (_circ_dist(angles, pole1) <= angular_half_width_deg) | \
                (_circ_dist(angles, pole2) <= angular_half_width_deg)
    eq_mask   = (_circ_dist(angles, eq1)   <= angular_half_width_deg) | \
                (_circ_dist(angles, eq2)   <= angular_half_width_deg)

    with np.errstate(invalid='ignore'):
        pole_mean = np.nanmean(norm[:, pole_mask], axis=1)
        eq_mean   = np.nanmean(norm[:, eq_mask],   axis=1)

    denom = eq_mean + pole_mean
    with np.errstate(divide='ignore', invalid='ignore'):
        pol_index = np.where(np.abs(denom) > 1e-9, (eq_mean - pole_mean) / denom, np.nan)

    return {
        't':                  t,
        'pole_mean':          pole_mean,
        'equator_mean':       eq_mean,
        'polarization_index': pol_index,
    }


def plot_pole_vs_equator(
        pe_data: dict,
        guv_id: str,
        output_folder: str,
        experiment_name: str,
) -> str:
    """
    Two-panel line plot from compute_pole_vs_equator_trace():
      Top    - pole_mean vs. equator_mean fold-change over time (the direct
               comparison: do the two lines separate, and which one drops?)
      Bottom - polarization_index over time (single number: 0 = uniform,
               rising positive = increasingly pole-concentrated loss)
    """
    t   = pe_data['t']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(t, pe_data['pole_mean'],    color='#CC3333', lw=2.0, label='Pole (electrode-facing)')
    ax1.plot(t, pe_data['equator_mean'], color='#3366CC', lw=2.0, label='Equator')
    ax1.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax1.axvline(0.0, color='cyan', ls='--', lw=1.2, alpha=0.8, label='Pulse (t = 0)')
    ax1.set_ylabel('Fold change vs. pre-pulse\n(mean within angular window)')
    ax1.set_title(f'GUV {guv_id} — pole vs. equator actin retention')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, pe_data['polarization_index'], color='black', lw=1.8)
    ax2.axhline(0.0, color='gray', ls=':', lw=0.8)
    ax2.axvline(0.0, color='cyan', ls='--', lw=1.2, alpha=0.8)
    ax2.set_ylabel('Polarization index\n(equator−pole)/(equator+pole)')
    ax2.set_xlabel('Time relative to pulse (s)')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(
        output_folder, f'{experiment_name}_GUV{guv_id}_pole_vs_equator.png'
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


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
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
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

    DYE_COLOR = '#1f77b4'    # blue  (kept for dual-axis lines)
    ACT_COLOR = '#ff7f0e'    # orange (kept for dual-axis lines)

    def _draw_panel(ax, t_d, dye_row, t_a, act_row, title, status_color=None):
        ax2 = ax.twinx()

        trace_col = status_color or DYE_COLOR
        ax .plot(t_d, _sm(dye_row), color=trace_col, lw=1.6, label='Dye (ret.)')
        ax2.plot(t_a, _sm(act_row), color=status_color or ACT_COLOR,
                 lw=1.6, label='Cortex (norm.)')

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

        col, suffix = _guv_status_style(gid, ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids)
        _draw_panel(ax, t_dye, dye_row, t_act, act_row, f'GUV {gid}{suffix}',
                    status_color=col if suffix else None)

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





def _crop_around(frame: np.ndarray, center: Tuple[float, float],
                 half_size: int) -> np.ndarray:
    """
    Fixed-size square crop centred on `center`, zero-padded where the box
    runs off the edge of the image. Returning a constant (2*half_size)^2
    array regardless of position is what keeps the pixel scale identical
    between the two timepoints, so a vesicle that deflates shrinks inside
    the frame instead of being re-zoomed to fill it.
    """
    h, w = frame.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    size = 2 * half_size
    out = np.zeros((size, size), dtype=frame.dtype)

    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out

    src = frame[sy0:sy1, sx0:sx1]
    if src.ndim == 3:
        src = src[:, :, 0]
    out[sy0 - y0: sy0 - y0 + src.shape[0],
        sx0 - x0: sx0 - x0 + src.shape[1]] = src
    return out


def _last_valid_center(centers: List[Optional[Tuple]], frame_idx: int
                       ) -> Tuple[Optional[Tuple], Optional[int]]:
    """
    Centre at `frame_idx`, falling back to the most recent earlier frame
    that has one. Returns (center, frame_actually_used).
    """
    if frame_idx is None or frame_idx < 0:
        frame_idx = len(centers) - 1
    frame_idx = min(int(frame_idx), len(centers) - 1)
    for i in range(frame_idx, -1, -1):
        if centers[i] is not None:
            return centers[i], i
    return None, None


def _fluor_cmap(color: str):
    """Black -> `color` linear colormap, the usual look for a fluorescence channel."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    return LinearSegmentedColormap.from_list('fluor', [(0, 0, 0), to_rgb(color)])


def export_guv_crops_prepost(
        prepost_df: pd.DataFrame,
        gid_to_tracking: dict,
        pulse_frame: int,
        time_full: np.ndarray,
        roi_mmap_info: Tuple,
        dye_mmap_info: Tuple,
        actin_mmap_info: Tuple,
        output_folder: str,
        experiment_name: str,
        crop_factor: float = 2.0,
        microns_per_pixel: float = 1.0,
        scale_bar_microns: float = 10.0,
        analyze_actin: bool = False,
        save_individual: bool = True,
        uniform_box: bool = True,
        channel_colors: Optional[dict] = None,
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
) -> str:
    """
    Cropped single-vesicle images accompanying the pre-vs-final endpoint
    figure: for every GUV, the membrane and dye channels (and the actin
    channel when it is enabled) at the last pre-pulse frame and at that
    GUV's own final frame.

    Two conventions here matter for the images to mean anything:

    1. DISPLAY SCALING IS SHARED BETWEEN THE TWO TIMEPOINTS. The intensity
       range is taken from the PRE-pulse crop of each channel and reused
       unchanged for the final crop. Per-image autoscaling (what
       _convert_to_8bit_gray does, correctly, for the tracking overlays)
       would renormalise a dim post-pulse vesicle back up to full range and
       erase the very dye loss the figure exists to show. Scaling is per
       GUV rather than per experiment so that dim and bright vesicles are
       each visible, which means brightness is comparable ACROSS the two
       timepoints of one GUV but NOT across GUVs — read the endpoint CSV,
       not the pixel brightness, for between-vesicle comparisons.

    2. CROP BOX SIZE IS FIXED FROM THE PRE-PULSE RADIUS. Both timepoints
       are cropped at crop_factor x the pre-pulse radius, so a vesicle that
       deflates visibly shrinks inside a constant frame instead of being
       re-zoomed to fill it. With uniform_box=True (the default) ONE box
       size, set by the largest pre-pulse radius in the experiment, is used
       for every GUV, so the montage is at a single pixel scale throughout
       and vesicles can be size-compared against each other by eye. Set it
       to False to size each box to its own GUV — tighter framing for small
       vesicles, but then imshow rescales each panel independently and
       relative sizes across rows become meaningless.

    The final frame is per-GUV, taken from `prepost_df['frame_final']`, so
    a vesicle that ruptured or drifted out of frame shows its own last
    observation. Its timestamp is printed on the panel.

    Returns the path of the montage PNG. Individual crops (one PNG per GUV
    per channel per timepoint, 8-bit, same shared scaling) are written to a
    `crops/` subfolder when save_individual is True.
    """
    if channel_colors is None:
        channel_colors = {'membrane': '#00FF66', 'dye': '#FF3355', 'actin': '#33CCFF'}

    # ── Assemble the channel list actually available ──────────────────────
    stacks = []
    if roi_mmap_info and roi_mmap_info[0] is not None:
        p, s, d = roi_mmap_info
        stacks.append(('membrane', np.memmap(p, dtype=d, mode='r', shape=s)))
    if dye_mmap_info and dye_mmap_info[0] is not None:
        p, s, d = dye_mmap_info
        stacks.append(('dye', np.memmap(p, dtype=d, mode='r', shape=s)))
    if analyze_actin and actin_mmap_info and actin_mmap_info[0] is not None:
        p, s, d = actin_mmap_info
        stacks.append(('actin', np.memmap(p, dtype=d, mode='r', shape=s)))

    if not stacks or len(prepost_df) == 0:
        return ""

    pre_frame = max(0, int(pulse_frame) - 1)
    time_full = np.asarray(time_full, dtype=float)

    crop_dir = os.path.join(output_folder, 'crops')
    if save_individual:
        os.makedirs(crop_dir, exist_ok=True)

    # One box size for the whole montage (see docstring point 2). Taken from
    # the largest pre-pulse radius so no vesicle is clipped.
    global_half = None
    if uniform_box:
        r_all = []
        for _, rr in prepost_df.iterrows():
            trk = gid_to_tracking.get(str(rr['guv_id']))
            if trk is None:
                continue
            _, fp = _last_valid_center(trk['centers'], pre_frame)
            if fp is not None and trk['radii'][fp]:
                r_all.append(float(trk['radii'][fp]))
        if r_all:
            global_half = max(8, int(round(crop_factor * max(r_all))))

    n_rows = len(prepost_df)
    n_cols = 2 * len(stacks)   # (pre, final) per channel

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.6 * n_rows),
                             squeeze=False)

    for r_i, (_, r) in enumerate(prepost_df.iterrows()):
        gid = str(r['guv_id'])
        tracking = gid_to_tracking.get(gid)
        status_col, suffix = _guv_status_style(
            gid, ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids
        )

        if tracking is None:
            for c_i in range(n_cols):
                axes[r_i][c_i].axis('off')
            axes[r_i][0].text(0.5, 0.5, f'GUV {gid}\nno tracking data',
                              ha='center', va='center', fontsize=8)
            continue

        centers = tracking['centers']
        radii   = tracking['radii']

        c_pre,  f_pre  = _last_valid_center(centers, pre_frame)
        f_final_req    = int(r['frame_final']) if r['frame_final'] is not None else -1
        c_fin,  f_fin  = _last_valid_center(centers, f_final_req)
        if c_pre is None or c_fin is None:
            for c_i in range(n_cols):
                axes[r_i][c_i].axis('off')
            continue

        # Box size from the PRE-pulse radius (see docstring point 2)
        r_pre = radii[f_pre] if (f_pre is not None and radii[f_pre]) else None
        if not r_pre:
            r_pre = next((x for x in radii if x), 20)
        half = global_half or max(8, int(round(crop_factor * float(r_pre))))

        for ch_i, (ch_name, stack) in enumerate(stacks):
            n_f = stack.shape[0]
            fi_pre = min(f_pre, n_f - 1)
            fi_fin = min(f_fin, n_f - 1)

            crop_pre = _crop_around(np.asarray(stack[fi_pre]), c_pre, half)
            crop_fin = _crop_around(np.asarray(stack[fi_fin]), c_fin, half)

            # Shared display range, taken from the PRE crop (docstring point 1)
            vmin, vmax = np.percentile(
                crop_pre[crop_pre > 0] if np.any(crop_pre > 0) else crop_pre,
                (getattr(cfg, 'CONTRAST_P_LOW', 0.5),
                 getattr(cfg, 'CONTRAST_P_HIGH', 98.0))
            )
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1.0

            cmap = _fluor_cmap(channel_colors.get(ch_name, '#FFFFFF'))

            for k, (crop, tag, fi) in enumerate(
                    [(crop_pre, 'pre', fi_pre), (crop_fin, 'final', fi_fin)]):
                ax = axes[r_i][2 * ch_i + k]
                ax.imshow(crop, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(status_col if suffix else '#333333')
                    sp.set_linewidth(1.8 if suffix else 0.8)

                t_rel = (float(time_full[fi] - time_full[pulse_frame])
                         if fi < len(time_full) and pulse_frame < len(time_full)
                         else np.nan)
                lbl = 'pre-pulse' if tag == 'pre' else 'final'
                ax.set_xlabel(f'{lbl}  (f{fi}, {t_rel:+.0f} s)', fontsize=6)

                if r_i == 0:
                    ax.set_title(f'{ch_name.capitalize()} – {lbl}', fontsize=8)
                if 2 * ch_i + k == 0:
                    ax.set_ylabel(f"GUV {gid}{suffix}", fontsize=7,
                                  color=status_col if suffix else 'black')

                # Scale bar on the first channel only, to avoid clutter
                if ch_i == 0 and microns_per_pixel > 0 and scale_bar_microns > 0:
                    bar_px = scale_bar_microns / microns_per_pixel
                    if bar_px < 0.9 * crop.shape[1]:
                        y = crop.shape[0] - 0.10 * crop.shape[0]
                        x0 = 0.06 * crop.shape[1]
                        ax.plot([x0, x0 + bar_px], [y, y], '-', color='white', lw=2.5)
                        ax.text(x0, y - 0.05 * crop.shape[0],
                                f'{scale_bar_microns:g} µm',
                                color='white', fontsize=5, va='bottom')

                if save_individual:
                    lo, hi = float(vmin), float(vmax)
                    scaled = np.clip((crop.astype(np.float32) - lo) / (hi - lo), 0, 1)
                    cv2.imwrite(
                        os.path.join(
                            crop_dir,
                            f'{experiment_name}_GUV{gid}_{ch_name}_{tag}_f{fi}.png'),
                        (scaled * 255).astype(np.uint8)
                    )

    plt.suptitle(
        f'{experiment_name}  –  Single-vesicle crops, pre-pulse vs. final frame\n'
        f'(display range shared between timepoints; crop box fixed at '
        f'{crop_factor:g}x pre-pulse radius)',
        fontsize=10
    )
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = os.path.join(output_folder,
                            f'{experiment_name}_guv_crops_prepost.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    for _, st in stacks:
        del st

    return out_path


def compute_prepulse_vs_final(
        curves_full: np.ndarray,
        time_full: np.ndarray,
        pulse_frame: int,
        guv_ids: List[str],
        n_pre: int = 5,
        n_final: int = 3,
) -> pd.DataFrame:
    """
    Per-GUV endpoint metric: normalised dye intensity BEFORE the pulse vs.
    at the END of the movie.

    Motivation
    ----------
    When the post-pulse acquisition is too coarse to resolve the efflux
    transient (i.e. the vesicle has already largely equilibrated by the
    first or second post-pulse frame), a single-exponential tau is not
    identifiable and the fitted time constant is dominated by the sampling
    interval rather than by membrane permeability. The pre-vs-final
    contrast is the quantity the data *can* support: it measures how much
    dye was lost in total, without any assumption about the shape of the
    trajectory in between.

    Definitions
    -----------
    I_pre   : mean of the normalised curve over the `n_pre` frames
              immediately preceding `pulse_frame` (same baseline window
              used by the normalisation, so I_pre is ~1 by construction;
              it is recomputed here rather than assumed, and its spread
              gives an honest pre-pulse noise estimate for the panel).
    I_final : mean of the last `n_final` FINITE frames of the curve. The
              last frame is taken per-GUV, not globally: a GUV whose
              tracking ends early (rupture, drift out of frame) has its
              own true last observation, and `t_final_s` records when that
              was so a shortened observation window is visible rather than
              silently averaged in with full-length traces.

    Returns
    -------
    DataFrame with one row per GUV:
        guv_id, I_pre, I_pre_sd, I_final, I_final_sd, delta_I,
        released_fraction, released_pct, frame_final, t_final_s,
        n_final_used
    where delta_I = I_final - I_pre (negative for efflux) and
    released_fraction = (I_pre - I_final) / I_pre.
    """
    curves_full = np.asarray(curves_full, dtype=float)
    time_full   = np.asarray(time_full,   dtype=float)

    pre_end   = pulse_frame if pulse_frame > 0 else 1
    pre_start = max(0, pre_end - int(n_pre))

    rows = []
    for gid, row in zip(guv_ids, curves_full):
        pre_vals = row[pre_start:pre_end]
        pre_vals = pre_vals[np.isfinite(pre_vals)]

        finite_idx = np.flatnonzero(np.isfinite(row))
        # Only frames at or after the pulse count as "final" — a GUV whose
        # entire post-pulse trace is NaN must not fall back onto its own
        # pre-pulse baseline and report zero release.
        finite_idx = finite_idx[finite_idx >= pulse_frame]

        if pre_vals.size == 0 or finite_idx.size == 0:
            # I_pre is still reported when only the endpoint is missing, so a
            # GUV with a clean baseline but no usable post-pulse frame is
            # distinguishable in the CSV from one that failed outright.
            rows.append({
                'guv_id': str(gid),
                'I_pre':    float(np.mean(pre_vals)) if pre_vals.size else np.nan,
                'I_pre_sd': float(np.std(pre_vals))  if pre_vals.size > 1 else np.nan,
                'I_final': np.nan, 'I_final_sd': np.nan, 'delta_I': np.nan,
                'released_fraction': np.nan, 'released_pct': np.nan,
                'frame_final': -1, 't_final_s': np.nan, 'n_final_used': 0,
            })
            continue

        tail_idx   = finite_idx[-int(max(1, n_final)):]
        final_vals = row[tail_idx]

        I_pre   = float(np.mean(pre_vals))
        I_final = float(np.mean(final_vals))
        f_last  = int(finite_idx[-1])

        t_last = (float(time_full[f_last] - time_full[pulse_frame])
                  if f_last < len(time_full) and pulse_frame < len(time_full)
                  else np.nan)

        released = (I_pre - I_final) / I_pre if abs(I_pre) > 1e-9 else np.nan

        rows.append({
            'guv_id':            str(gid),
            'I_pre':             I_pre,
            'I_pre_sd':          float(np.std(pre_vals))   if pre_vals.size   > 1 else 0.0,
            'I_final':           I_final,
            'I_final_sd':        float(np.std(final_vals)) if final_vals.size > 1 else 0.0,
            'delta_I':           I_final - I_pre,
            'released_fraction': released,
            'released_pct':      100.0 * released if np.isfinite(released) else np.nan,
            'frame_final':       f_last,
            't_final_s':         t_last,
            'n_final_used':      int(tail_idx.size),
        })

    return pd.DataFrame(rows)


def plot_dye_prepulse_vs_final(
        curves_full: np.ndarray,
        time_full: np.ndarray,
        pulse_frame: int,
        guv_ids: List[str],
        output_folder: str,
        experiment_name: str,
        n_pre: int = 5,
        n_final: int = 3,
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
) -> Tuple[str, str, pd.DataFrame]:
    """
    Multipanel per-GUV figure of the pre-pulse -> final-frame intensity
    change, plus a population summary panel.

    One panel per GUV shows the full normalised trace (pre-pulse frames at
    negative time, pulse at t = 0), the shaded pre-pulse baseline window
    with its mean level, and the endpoint level averaged over the last
    `n_final` finite frames. The vertical arrow between the two levels is
    the reported delta_I. The panel title carries the released fraction and
    the time of the last observation, so a GUV that stopped being tracked
    early is not mistaken for one measured over the full window.

    The final panel is a paired slope graph (pre -> final for every GUV)
    coloured by tracking fate, giving the population picture without
    invoking any kinetic model.

    Colour scheme matches the rest of the dye figures via
    _guv_status_style(): surviving grey, ruptured red, shrunk orange,
    out-of-frame blue.

    Returns
    -------
    (png_path, csv_path, dataframe)
    """
    df = compute_prepulse_vs_final(
        curves_full, time_full, pulse_frame, guv_ids,
        n_pre=n_pre, n_final=n_final
    )

    curves_full = np.asarray(curves_full, dtype=float)
    time_full   = np.asarray(time_full,   dtype=float)
    t_rel = time_full - time_full[pulse_frame]

    pre_end   = pulse_frame if pulse_frame > 0 else 1
    pre_start = max(0, pre_end - int(n_pre))

    C_TRACE   = '#888888'
    C_PRE     = '#2C7FB8'   # baseline level
    C_FIN     = '#D95F0E'   # endpoint level
    C_PRE_BAND = '#DCE9F2'  # shaded pre-pulse window

    n        = len(df)
    if n == 0:
        return "", "", df
    n_panels = n + 1
    ncols    = min(4, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for k, (_, r) in enumerate(df.iterrows()):
        ax = axes[k // ncols][k % ncols]
        row = curves_full[k]
        status_col, suffix = _guv_status_style(
            r['guv_id'], ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids
        )
        c_trace = status_col if suffix else C_TRACE

        # Shaded baseline window
        if pre_end > pre_start:
            ax.axvspan(t_rel[pre_start], t_rel[max(pre_start, pre_end - 1)],
                       color=C_PRE_BAND, alpha=0.9, zorder=0,
                       label='Pre-pulse window')

        ax.plot(t_rel, row, '-', color=c_trace, lw=1.0, alpha=0.55, zorder=1)
        ax.plot(t_rel, row, '.', color=c_trace, ms=4, alpha=0.85, zorder=2,
                label='Data' + suffix)

        if np.isfinite(r['I_pre']):
            ax.axhline(r['I_pre'], color=C_PRE, ls='--', lw=1.4, zorder=3,
                       label=f"Pre-pulse = {r['I_pre']:.2f}")
        if np.isfinite(r['I_final']):
            ax.axhline(r['I_final'], color=C_FIN, ls='--', lw=1.4, zorder=3,
                       label=f"Final = {r['I_final']:.2f}")

        # Endpoint marker(s) actually averaged
        f_last = int(r['frame_final'])
        if f_last >= 0:
            n_used = int(r['n_final_used'])
            finite_idx = np.flatnonzero(np.isfinite(row))
            finite_idx = finite_idx[finite_idx >= pulse_frame]
            tail_idx = finite_idx[-n_used:] if n_used > 0 else finite_idx[-1:]
            ax.plot(t_rel[tail_idx], row[tail_idx], 'o', mfc='none',
                    mec=C_FIN, mew=1.6, ms=8, zorder=4,
                    label=f'Final {n_used} frame(s)')

            # Delta arrow at the last time point
            if np.isfinite(r['I_pre']) and np.isfinite(r['I_final']):
                ax.annotate(
                    '', xy=(t_rel[f_last], r['I_final']),
                    xytext=(t_rel[f_last], r['I_pre']),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.6),
                    zorder=5
                )
                ax.text(t_rel[f_last], 0.5 * (r['I_pre'] + r['I_final']),
                        rf"  $\Delta$I = {r['delta_I']:+.2f}",
                        fontsize=7, va='center', ha='right', color='black')

        ax.axvline(0, color='crimson', ls='--', lw=1.0, zorder=1)

        title_color = status_col if suffix else 'black'
        rel_txt = (f"{r['released_pct']:.0f}% released"
                   if np.isfinite(r['released_pct']) else 'no valid endpoint')
        t_txt = (f"last obs. t = {r['t_final_s']:.0f} s"
                 if np.isfinite(r['t_final_s']) else 'last obs. n/a')
        ax.set_title(f"GUV {r['guv_id']}{suffix}\n{rel_txt}   |   {t_txt}",
                     fontsize=8, color=title_color)
        ax.set_xlabel('Time relative to pulse (s)', fontsize=8)
        ax.set_ylabel('Norm. intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.20)
        ax.legend(fontsize=6, loc='best')

    # ── Population summary panel: paired slope graph ──────────────────────
    sum_ax = axes[n // ncols][n % ncols]
    for _, r in df.iterrows():
        if not (np.isfinite(r['I_pre']) and np.isfinite(r['I_final'])):
            continue
        col, suffix = _guv_status_style(
            r['guv_id'], ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids
        )
        sum_ax.plot([0, 1], [r['I_pre'], r['I_final']], '-o',
                    color=col, lw=1.2, ms=5, alpha=0.75)
        sum_ax.annotate(str(r['guv_id']), xy=(1.02, r['I_final']),
                        fontsize=6, va='center', color=col)

    ok = df[np.isfinite(df['I_pre']) & np.isfinite(df['I_final'])]
    if len(ok) > 0:
        m_pre, m_fin = ok['I_pre'].mean(), ok['I_final'].mean()
        s_pre, s_fin = ok['I_pre'].std(),  ok['I_final'].std()
        sum_ax.errorbar([0, 1], [m_pre, m_fin], yerr=[s_pre, s_fin],
                        fmt='-s', color='black', lw=2.5, ms=8, capsize=4,
                        zorder=5, label='Mean ± SD')
        sum_ax.set_title(
            f'Pre-pulse → final (n = {len(ok)})\n'
            f"mean released = {ok['released_pct'].mean():.0f} ± "
            f"{ok['released_pct'].std():.0f} %",
            fontsize=8
        )
        sum_ax.legend(fontsize=6)
    else:
        sum_ax.set_title('Pre-pulse → final (no valid GUVs)', fontsize=8)

    sum_ax.set_xticks([0, 1])
    sum_ax.set_xticklabels(['Pre-pulse', 'Final frame'], fontsize=8)
    sum_ax.set_xlim(-0.35, 1.35)
    sum_ax.set_ylabel('Norm. intensity', fontsize=8)
    sum_ax.tick_params(labelsize=7)
    sum_ax.grid(True, axis='y', alpha=0.20)

    for k in range(n_panels, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    plt.suptitle(
        f'{experiment_name}  –  Pre-pulse vs. final-frame dye intensity '
        f'(model-free endpoint)',
        fontsize=11
    )
    plt.tight_layout()

    png_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_prepulse_vs_final.png')
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    csv_path = os.path.join(output_folder,
                            f'{experiment_name}_dye_prepulse_vs_final.csv')
    df.to_csv(csv_path, index=False)

    return png_path, csv_path, df


def plot_dye_fits_grid(
        fit_data: list,
        output_folder: str,
        experiment_name: str,
        model_name: str,
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
) -> str:
    """
    Multigrid plot of per-GUV kinetic fits — one subplot per GUV plus a
    population-mean panel.

    Colour scheme
    -------------
    Surviving GUVs   : medium grey (#888888) data dots, dark grey (#444444) fit line.
    Ruptured GUVs    : red (#CC2222) data dots and fit line — abrupt membrane
        burst while the vesicle was still close to its pre-pulse size.
    Shrunk GUVs      : orange (#E08214) data dots and fit line — gradual
        radius decline (deflation), not a burst.
    Out-of-frame GUVs: blue (#3B7DD8) data dots and fit line — tracking was lost
        because the vesicle drifted beyond the field of view, NOT a rupture.
    Population mean panel: grey individual fits, bold black mean, light grey ±1 SD band.

    Parameters
    ----------
    fit_data : list of dicts, one per GUV, each containing:
        'guv_id', 't_data', 'y_data', 'y_fit', 'r_um', 'param_str'
    ruptured_guv_ids     : set of guv_id strings that ruptured (coloured red)
    out_of_frame_guv_ids : set of guv_id strings lost to the FOV edge (coloured blue)
    """
    n = len(fit_data)
    if n == 0:
        return ""

    if ruptured_guv_ids is None:
        ruptured_guv_ids = set()
    if out_of_frame_guv_ids is None:
        out_of_frame_guv_ids = set()

    C_DATA_SURV = '#888888'   # mid-grey  — surviving data points
    C_FIT_SURV  = '#444444'   # dark grey — surviving fit line
    C_SD_BAND   = '#BBBBBB'   # light grey — ±1 SD shading (distinct from traces)

    n_panels = n + 1
    ncols    = min(4, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for k, d in enumerate(fit_data):
        ax = axes[k // ncols][k % ncols]
        status_col, suffix = _guv_status_style(
            d['guv_id'], ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids
        )
        c_data = status_col if suffix else C_DATA_SURV
        c_fit  = status_col if suffix else C_FIT_SURV

        ax.plot(d['t_data'], d['y_data'], '.', color=c_data,
                alpha=0.55, ms=3, label='Data' + suffix)
        ax.plot(d['t_data'], d['y_fit'],  '-', color=c_fit,
                lw=2.0, label='Fit')
        ax.axvline(0, color='dimgray', ls='--', lw=1.0, label='Pulse', zorder=0)

        title_color = status_col if suffix else 'black'
        ax.set_title(
            rf"GUV {d['guv_id']}  (R={d['r_um']:.1f} µm){suffix}"
            f"\n{d['param_str']}",
            fontsize=8, color=title_color
        )
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Norm. intensity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.20)
        ax.legend(fontsize=6)

    # ── Population mean panel ─────────────────────────────────────────────
    mean_ax = axes[n // ncols][n % ncols]

    t_min = min(float(d['t_data'][0])  for d in fit_data)
    t_max = max(float(d['t_data'][-1]) for d in fit_data)
    n_pts = max(200, max(len(d['t_data']) for d in fit_data))
    all_t = np.linspace(t_min, t_max, n_pts)

    y_stack = []
    for d in fit_data:
        status_col, suffix = _guv_status_style(
            d['guv_id'], ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids
        )
        c_fit = status_col if suffix else C_FIT_SURV
        mean_ax.plot(d['t_data'], d['y_data'], '.', color=c_fit,
                     alpha=0.18, ms=2)
        mean_ax.plot(d['t_data'], d['y_fit'],  '-', color=c_fit,
                     lw=0.8, alpha=0.45)
        y_interp = np.interp(all_t, d['t_data'], d['y_fit'],
                             left=np.nan, right=np.nan)
        y_stack.append(y_interp)

    if len(y_stack) > 0:
        y_arr  = np.array(y_stack)
        y_mean = np.nanmean(y_arr, axis=0)
        y_sd   = np.nanstd(y_arr,  axis=0)
        mean_ax.fill_between(all_t, y_mean - y_sd, y_mean + y_sd,
                             color=C_SD_BAND, alpha=0.55, label='±1 SD', zorder=1)
        mean_ax.plot(all_t, y_mean, 'k-', lw=2.5, label='Mean fit', zorder=2)

    mean_ax.axvline(0, color='dimgray', ls='--', lw=1.0, label='Pulse', zorder=0)
    mean_ax.set_title('Population mean (fits)', fontsize=8)
    mean_ax.set_xlabel('Time (s)', fontsize=8)
    mean_ax.set_ylabel('Norm. intensity', fontsize=8)
    mean_ax.tick_params(labelsize=7)
    mean_ax.grid(True, alpha=0.20)
    mean_ax.legend(fontsize=6)

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
        ruptured_guv_ids: Optional[set] = None,
        out_of_frame_guv_ids: Optional[set] = None,
        shrunk_guv_ids: Optional[set] = None,
) -> str:
    """
    Single-panel summary of all normalised dye traces + population mean.

    Colour scheme: surviving GUVs in medium grey (#888888), ruptured GUVs
    in red (#CC2222) — a real biological event — out-of-frame GUVs in blue
    (#3B7DD8) — tracking lost to the vesicle drifting beyond the field of
    view, NOT a rupture — bold black mean, light grey (#BBBBBB) ±1 SD band.
    """
    t       = aligned['t_aligned']
    arr     = aligned['all_curves_aligned']
    avg     = aligned['average_curve']
    ids     = aligned['valid_guv_ids']

    C_SD    = '#BBBBBB'

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

    for gid, row in zip(ids, arr):
        col, suffix = _guv_status_style(gid, ruptured_guv_ids, out_of_frame_guv_ids, shrunk_guv_ids)
        lw    = 1.2 if suffix else 0.9
        label = f'GUV {gid}{suffix}'
        ax.plot(t, _smooth(row), color=col, alpha=0.70, lw=lw, label=label)

    sd       = np.nanstd(arr, axis=0)
    sm       = _smooth(avg)
    ax.fill_between(t, _smooth(avg - sd), _smooth(avg + sd),
                    color=C_SD, alpha=0.55, label='±1 SD', zorder=1)
    ax.plot(t, sm, 'k-', lw=2.5, label='Mean', zorder=2)

    ax.axvline(0, color='crimson', ls='--', lw=1.5, label='Pulse', zorder=3)
    ax.axhline(1, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Time relative to pulse (s)', fontsize=11)
    ax.set_ylabel('Normalised dye intensity (a.u.)', fontsize=11)
    ax.set_title(
        f'{experiment_name}\nDye channel – all GUVs + population mean',
        fontsize=11
    )
    ax.grid(True, alpha=0.25)
    handles, labels_leg = ax.get_legend_handles_labels()
    ncols_leg = 3 if len(handles) > 10 else 2
    ax.legend(handles, labels_leg, fontsize=8, ncol=ncols_leg,
              loc='upper right' if avg[-1] < avg[0] else 'lower right')
    plt.tight_layout()
    out_path = os.path.join(output_folder, f'{experiment_name}_dye_summary.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path