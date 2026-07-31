# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS ---
======================================================
Main execution script.

Pipeline
--------
1. If CIRCLES_JSON_PATH doesn't exist → launch interactive_select.py
   so the user can draw starting circles on the first ROI frame.
2. Load Data:  ROI frames (for tracking), dye frames, timestamps,
               circle definitions (from JSON).
3. Process GUVs (parallel):
   - Track centre + radius frame-by-frame using ring-score grid search.
   - Detect rupture events.
   - Extract per-frame intensity traces.
4. Normalise & align curves (t = 0 at pulse frame extracted from filename).
5. Fit kinetic model to individual GUV trajectories.
6. Export plots, tracking data, per-GUV CSVs, and parameter boxplots grouped by size.
"""

import glob
import os
import json
import logging
import multiprocessing
import sys
import re
from functools import partial
from natsort import natsorted

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import scipy.optimize as sc
import pandas as pd
import tempfile
from tqdm import tqdm
import gc

import config as cfg
import guv_analysis_utils as utils


# -------------------------------------------------------------------
# --- 1. GUV CIRCLE DEFINITIONS
# -------------------------------------------------------------------

def get_guv_circles(logger: logging.Logger) -> list[dict]:
    """
    Returns the list of user-drawn GUV circles.
    If CIRCLES_JSON_PATH already exists, loads it.
    Otherwise, extracts the first frame (ND2 or TIFF) and launches the interactive selector.
    """
    path = cfg.CIRCLES_JSON_PATH

    if os.path.exists(path):
        with open(path) as f:
            circles = json.load(f)
        logger.info(f"Loaded {len(circles)} GUV circle(s) from: {path}")
        return circles

    logger.info("No circle definitions found — launching interactive selector.")

    input_format = getattr(cfg, 'INPUT_FORMAT', 'TIFF').upper()
    first_frame = None

    if input_format == 'ND2':
        import nd2
        nd2_path = os.path.join(cfg.DATA_FOLDER, getattr(cfg, 'ND2_FILE_NAME', f"{cfg.EXPERIMENT_BASE_NAME}.nd2"))
        if not os.path.exists(nd2_path):
            logger.error(f"ND2 file not found: {nd2_path}")
            return []
            
        with nd2.ND2File(nd2_path) as f:
            idx_mem = getattr(cfg, 'ND2_CHANNEL_IDX_MEMBRANE', 0)
            data = f.asarray()
            # Extract first frame, membrane channel. Handle 3D or 4D ND2 arrays.
            if data.ndim == 4:
                first_frame = data[0, idx_mem, :, :]
            elif data.ndim == 3:
                first_frame = data[0, :, :]
    else:
        import glob
        from natsort import natsorted
        roi_pattern = os.path.join(
            cfg.DATA_FOLDER,
            cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
        )
        roi_files = natsorted(glob.glob(roi_pattern))
        if not roi_files:
            logger.error(f"No ROI files found matching: {roi_pattern}")
            return []
        first_frame = cv2.imread(roi_files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

    if first_frame is None:
        logger.error("Could not load first ROI frame.")
        return []

    from interactive_select import run_selector
    circles = run_selector(first_frame, save_path=path)

    if not circles:
        logger.error("No circles were drawn — aborting.")
        return []

    logger.info(f"Saved {len(circles)} circle(s) → {path}")
    return circles


# -------------------------------------------------------------------
# --- 2. DATA LOADING ---
# -------------------------------------------------------------------

def create_memmap_stack(files: list, mmap_path: str, desc: str = "Processing") -> tuple:
    """Reads individual TIFFs and writes them to a contiguous binary memmap file."""
    if not files:
        return None, None, None

    # On Windows a file left open by a crashed previous run causes [Errno 22].
    # Explicitly remove the stale file before np.memmap tries to create it.
    if os.path.exists(mmap_path):
        try:
            os.remove(mmap_path)
        except OSError as e:
            raise OSError(
                f"Cannot remove stale temp file '{mmap_path}'. "
                f"Close any process that may still have it open, then retry.\n"
                f"Original error: {e}"
            ) from e

    first_frame = cv2.imread(files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    dtype = first_frame.dtype
    shape = (len(files), first_frame.shape[0], first_frame.shape[1])

    mmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)

    for i, f in enumerate(tqdm(files, desc=desc, unit="frame")):
        mmap_arr[i] = cv2.imread(f, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    mmap_arr.flush()

    return mmap_path, shape, str(dtype)


def load_input_data(circles: list[dict], logger: logging.Logger) -> dict | None:
    """
    Loads image frames and timestamps, supporting both TIFF sequences and ND2 stacks.
    """
    mmap_dir = r"C:\temp_guv_data" 
    os.makedirs(mmap_dir, exist_ok=True)
   
    roi_mmap_path   = os.path.join(mmap_dir, "roi_temp.dat")
    dye_mmap_path   = os.path.join(mmap_dir, "dye_temp.dat")
    actin_mmap_path = os.path.join(mmap_dir, "actin_temp.dat")

    input_format = getattr(cfg, 'INPUT_FORMAT', 'TIFF').upper()

    if input_format == 'ND2':
        import nd2
        nd2_path = os.path.join(cfg.DATA_FOLDER, getattr(cfg, 'ND2_FILE_NAME', f"{cfg.EXPERIMENT_BASE_NAME}.nd2"))
        
        if not os.path.exists(nd2_path):
            logger.error(f"ND2 file not found: {nd2_path}")
            return None

        logger.info(f"Loading ND2 file: {nd2_path}")
        with nd2.ND2File(nd2_path) as f:
            ch_names = [c.channel.name for c in f.metadata.channels]
            logger.info(f"ND2 Channels found: {ch_names}")

            idx_mem = getattr(cfg, 'ND2_CHANNEL_IDX_MEMBRANE', 0)
            idx_dye = getattr(cfg, 'ND2_CHANNEL_IDX_DYE', 1)
            idx_act = getattr(cfg, 'ND2_CHANNEL_IDX_ACTIN', 2)

            logger.info(f"Compiling ND2 channels into local memory-mapped stacks at {mmap_dir}...")
            roi_mmap_info = utils.create_memmap_from_nd2_channel(f, idx_mem, roi_mmap_path, "ROI Stack")
            
            if getattr(cfg, 'TRACKING_ONLY_MODE', False):
                dye_mmap_info = (None, None, None)
            else:
                dye_mmap_info = utils.create_memmap_from_nd2_channel(f, idx_dye, dye_mmap_path, "Dye Stack")

            if getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False):
                actin_mmap_info = utils.create_memmap_from_nd2_channel(f, idx_act, actin_mmap_path, "Actin Stack")
            else:
                actin_mmap_info = (None, None, None)

            time_array, frame_interval = utils.extract_timestamps_nd2(f)
            dye_files = [f"nd2_frame_{i}" for i in range(roi_mmap_info[1][0])]

    else:
        # Existing TIFF processing logic
        dye_files = natsorted(glob.glob(os.path.join(
            cfg.DATA_FOLDER, cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
        )))
        roi_files = natsorted(glob.glob(os.path.join(
            cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
        )))
        actin_files = natsorted(glob.glob(os.path.join(
            cfg.DATA_FOLDER, cfg.ACTIN_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
        ))) if getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False) else []

        if not dye_files or not roi_files:
            logger.error("Required TIFF files not found.")
            return None

        time_array, frame_interval = utils.extract_timestamps_from_metadata(dye_files)
        if time_array is None:
            time_array, frame_interval = utils.create_manual_timestamps(
                len(dye_files), getattr(cfg, 'FALLBACK_FPS', 1.0)
            )

        logger.info(f"Compiling TIFFs into local memory-mapped stacks at {mmap_dir}...")
        roi_mmap_info = create_memmap_stack(roi_files, roi_mmap_path, desc="ROI Stack")
        
        dye_mmap_info = (None, None, None) if getattr(cfg, 'TRACKING_ONLY_MODE', False) else \
                        create_memmap_stack(dye_files, dye_mmap_path, desc="Dye Stack")
        
        actin_mmap_info = create_memmap_stack(actin_files, actin_mmap_path, desc="Actin Stack") \
                          if actin_files and getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False) else (None, None, None)

    logger.info(f"Frame interval: {frame_interval:.3f} s  ({len(time_array)} frames)")
    
    if len(time_array) > 1:
        steps = np.zeros(len(time_array))
        steps[1:] = np.round(np.diff(time_array), 3)
        steps[0]  = steps[1]
        
        changes = np.where(steps[:-1] != steps[1:])[0] + 1
        splits  = [0] + changes.tolist() + [len(time_array)]
        
        for i in range(len(splits) - 1):
            start = splits[i]
            end   = splits[i+1] - 1
            count = end - start + 1
            
            interval = steps[start]
            t_start  = np.round(time_array[start], 3)
            t_end    = np.round(time_array[end], 3)
            
            logger.info(
                f"Frames {start}–{end} ({count} frames): {interval} s intervals. "
                f"The timestamps progress from {t_start} to {t_end}."
            )
    

    return {
        'dye_files':        dye_files,
        'roi_mmap_info':    roi_mmap_info,
        'dye_mmap_info':    dye_mmap_info,
        'actin_mmap_info':  actin_mmap_info,
        'roi_mmap_path':    roi_mmap_path,
        'dye_mmap_path':    dye_mmap_path,
        'actin_mmap_path':  actin_mmap_path,
        'time_array':       time_array,
        'frame_interval':   frame_interval,
        'circles':          circles,
    }

# -------------------------------------------------------------------
# --- 3. PARALLEL GUV PROCESSING ---
# -------------------------------------------------------------------

def process_all_guvs(circles: list[dict],
                     roi_mmap_info: tuple,
                     dye_mmap_info: tuple,
                     logger: logging.Logger,
                     actin_mmap_info: tuple = (None, None, None)) -> dict:
    """
    Runs process_single_guv for each circle using multiprocessing.
    """
    # Extract the total number of frames from the shape tuple
    n_frames = roi_mmap_info[1][0]

    process_func = partial(
        utils.process_single_guv,
        roi_mmap_info=roi_mmap_info,
        dye_mmap_info=dye_mmap_info,
        n_frames=n_frames,
        actin_mmap_info=actin_mmap_info,
    )

    tasks = [
        (str(c['id']),           # guv_id
         (c['x'], c['y']),       # initial_center
         c['r'],                 # initial_radius
         (i == 0))               # is_first_guv
        for i, c in enumerate(circles)
    ]

    logger.info(
        f"Starting parallel processing of {len(tasks)} GUV(s) "
        f"with {cfg.N_WORKERS} worker(s) ..."
    )

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=cfg.N_WORKERS) as pool:
        results = pool.starmap(process_func, tasks)

    logger.info("... Parallel processing complete.")

    # ── Unpack ────────────────────────────────────────────────────────────
    all_intensity  = []
    all_background = []
    all_actin_data = []     
    quality_log    = []
    valid_indices  = []
    tracking_rows  = []

    for i, (result, qe) in enumerate(results):
        quality_log.append(qe)
        gid = qe['guv_id']
        rup = qe.get('ruptured_at_frame')
        exit_reason = qe.get('exit_reason')

        if rup is not None:
            reason_label = 'OUT OF FRAME (tracking lost, not a rupture)' \
                if exit_reason == 'OUT_OF_FRAME' else 'RUPTURED'
            logger.warning(
                f"  GUV {gid}: {reason_label} at frame {rup}  "
                f"({qe.get('n_valid_frames','?')} valid frames, "
                f"mean score={qe.get('mean_ring_score',0):.1f})"
            )
        elif qe['comments'] != 'OK':
            logger.warning(f"  GUV {gid}: {qe['comments']}")
        else:
            logger.info(
                f"  GUV {gid}: OK  "
                f"({qe.get('n_valid_frames','?')} valid frames, "
                f"mean score={qe.get('mean_ring_score',0):.1f})"
            )

        if result is not None:
            all_intensity .append(result['intensity_trace'])
            all_background.append(result['background_trace'])
            all_actin_data.append(result.get('actin_data'))   # None if not analysed
            valid_indices .append(i)

            tr = result.get('tracking', {})
            for fi, (c, r, s, el) in enumerate(zip(
                    tr.get('centers', []),
                    tr.get('radii', []),
                    tr.get('ring_scores', []),
                    tr.get('ellipses', []))):
                if c is not None and r is not None and el is not None:
                    a, b = el['axes']
                    major = max(a, b)
                    minor = min(a, b)
                    eccentricity = np.sqrt(1.0 - (minor / major)**2) if major > 0 else 0
                    circularity = (2.0 * major * minor) / (major**2 + minor**2) if (major**2 + minor**2) > 0 else 0
                    
                    tracking_rows.append({
                        'guv_id': gid, 'frame': fi,
                        'x': c[0], 'y': c[1],
                        'radius': r, 
                        'semi_major': round(major, 2), 
                        'semi_minor': round(minor, 2),
                        'eccentricity': round(eccentricity, 4),
                        'circularity': round(circularity, 4),
                        'ring_score': round(s, 2)
                    })
        else:
            logger.warning(f"  GUV {gid}: no data returned.")

    ruptured_ids = {
        str(qe['guv_id'])
        for qe in quality_log
        if qe.get('ruptured_at_frame') is not None and qe.get('exit_reason') == 'RUPTURED'
    }
    out_of_frame_ids = {
        str(qe['guv_id'])
        for qe in quality_log
        if qe.get('ruptured_at_frame') is not None and qe.get('exit_reason') == 'OUT_OF_FRAME'
    }

    logger.info(
        f"Processed {len(all_intensity)} / {len(circles)} GUV(s) successfully."
    )
    if ruptured_ids or out_of_frame_ids:
        logger.info(
            f"  Of these: {len(ruptured_ids)} ruptured (biological), "
            f"{len(out_of_frame_ids)} lost to FOV edge (tracking limitation, "
            f"excluded from rupture statistics)."
        )

    # Save quality log
    pd.DataFrame(quality_log).to_csv(
        os.path.join(cfg.FOLDER_TRACKING,
                     f"{cfg.EXPERIMENT_BASE_NAME}_detection_quality.csv"),
        index=False
    )

    # Save tracking data
    df_track = None
    if tracking_rows and getattr(cfg, 'EXPORT_TRACKING_DATA', True):
        df_track = pd.DataFrame(tracking_rows)
        df_track.to_csv(
            os.path.join(cfg.FOLDER_TRACKING,
                         f"{cfg.EXPERIMENT_BASE_NAME}_tracking_data.csv"),
            index=False
        )
        logger.info("Saved per-frame tracking data.")

    return {
        'raw_results':           results,
        'all_intensity_curves':  all_intensity,
        'all_background_curves': all_background,
        'all_actin_data':        all_actin_data,
        'valid_guv_indices':     valid_indices,
        'tracking_dataframe':    df_track,
        'ruptured_guv_ids':      ruptured_ids,      # set of str IDs that genuinely ruptured
        'out_of_frame_guv_ids':  out_of_frame_ids,   # set of str IDs lost to the FOV edge (not a rupture)
    }


# -------------------------------------------------------------------
# --- 4. NORMALISATION & ALIGNMENT ---
# -------------------------------------------------------------------

def normalize_and_align_curves(guv_results: dict,
                                time_array: np.ndarray,
                                circles: list[dict],
                                pulse_frame: int,
                                logger: logging.Logger) -> dict | None:
    """
    Normalises using the efflux-mirrored form of the published SRB uptake
    method (Idye,t - Idye,0)/(Ibackground,t - Idye,0):

        I_retained(t) = (I_bg,t - I_dye,t) / (I_bg,t - I_dye,0)

    Derivation: the published formula measures progress toward equilibrium
    with the background, I_uptake(t) = (I_dye,t - I_dye,0)/(I_bg,t - I_dye,0),
    which runs 0 (baseline) -> 1 (fully equilibrated with the bath). For
    efflux the natural quantity is *retention*, 1 -> 0, so this uses
    1 - I_uptake(t), which simplifies to the expression above. At t=0,
    I_dye,t = I_dye,0 so I_retained = 1; as efflux completes, I_dye,t -> I_bg,t
    so I_retained -> 0 — the same shape the EFFLUX-1EXP/2EXP models expect.

    Unlike the previous (I_dye,t - I_bg,t)/(I_dye,0 - I_bg,0) form, the
    denominator here uses I_bg,t (background AT EACH FRAME) rather than a
    fixed pre-pulse bg0, so it tracks background photobleaching drift over
    the course of the movie instead of assuming it's constant. I_dye,0 is
    the mean of the (already per-frame, mask-averaged) intensity trace over
    the 5 frames immediately preceding the detected pulse_frame — no
    per-pixel statistics involved.

    Frames where I_bg,t sits too close to I_dye,0 for this frame's division
    to be numerically meaningful are set to NaN individually rather than
    discarding the whole GUV.

    Dead-GUV filter
    ----------------
    Separately, a whole GUV is excluded up front if its pre-pulse interior
    signal is not meaningfully separated from background at all:
        |I_dye,0 - I_bg,0|  <  MIN_PREPULSE_SEPARATION_SIGMA * bg_std0
    bg_std0 is the RAW per-pixel background std (not divided by sqrt(n) —
    that SEM-scaled version is what over-excluded everything previously).
    This is a plain "is the average brightness jump at least a couple of
    typical pixel-noise-widths" check, not a significance test on the mean,
    so it doesn't get stricter just because the background ring happens to
    contain more pixels.

    Aligns: t = 0 at *pulse_frame* (auto-detected or overridden in config).
    """
    logger.info(f"Normalising intensity curves (pulse frame = {pulse_frame}) ...")

    dye_list    = guv_results['all_intensity_curves']
    bg_list     = guv_results['all_background_curves']
    bg_std_list = guv_results.get('all_background_std_curves', [None] * len(dye_list))
    min_sep_sigma = getattr(cfg, 'MIN_PREPULSE_SEPARATION_SIGMA', 2.0)

    # Baseline window = the 5 frames immediately preceding the ACTUAL
    # detected/configured pulse_frame — not a hardcoded "first 5 frame
    # indices of the movie". These only coincide when pulse_frame happens
    # to equal 5 (true under the default FRAME_INTERVAL_SCHEDULE, but not
    # guaranteed under PULSE_FRAME_OVERRIDE, a different acquisition
    # schedule, or ordinary auto-detection variation). This matches how
    # normalize_actin_curves already defines its baseline window, so the
    # dye and actin channels are now consistent.
    safe_pre_end   = pulse_frame if pulse_frame > 0 else 1
    safe_pre_start = max(0, safe_pre_end - 5)

    normalised          = []
    kept_local_indices  = []
    excluded_dead_ids    = []

    for local_i, (dye, bg, bg_std) in enumerate(zip(dye_list, bg_list, bg_std_list)):
        if pulse_frame >= len(dye):
            continue

        dye0 = float(np.nanmean(dye[safe_pre_start:safe_pre_end]))
        bg0  = float(np.nanmean(bg[safe_pre_start:safe_pre_end]))

        # Dead-GUV filter (see docstring): plain amplitude check against the
        # background's own pixel-noise scale, not a sample-size-scaled test.
        orig_idx = guv_results['valid_guv_indices'][local_i]
        gid      = str(circles[orig_idx]['id'])

        bg_std0 = float(np.nanmean(bg_std[safe_pre_start:safe_pre_end])) if bg_std is not None else np.nan
        if np.isfinite(bg_std0) and bg_std0 > 0:
            min_sep = min_sep_sigma * bg_std0
            sep0 = abs(bg0 - dye0)
            if sep0 < min_sep:
                excluded_dead_ids.append(gid)
                logger.warning(
                    f"  GUV {gid}: excluded — pre-pulse interior signal "
                    f"indistinguishable from background "
                    f"(|Idye,0 - Ibg,0| = {sep0:.2f} AU < "
                    f"{min_sep_sigma}x background std = {min_sep:.2f} AU)."
                )
                continue

        den = bg - dye0   # time-varying: I_bg,t - I_dye,0, one value per frame
        eps = 1e-9
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.where(np.abs(den) > eps, (bg - dye) / den, np.nan)

        normalised.append(norm)
        kept_local_indices.append(local_i)

    if not normalised:
        logger.error("Normalisation failed for all GUVs.")
        return None

    # Align curves to start at t=0 on the pulse frame
    arr   = np.array(normalised)[:, pulse_frame:]
    t_al  = time_array[pulse_frame:] - time_array[pulse_frame]
    avg   = np.nanmean(arr, axis=0)

    valid_ids = [
        str(circles[guv_results['valid_guv_indices'][i]]['id'])
        for i in kept_local_indices
    ]

    return {
        't_aligned':                t_al,
        'average_curve':            avg,
        'all_curves_aligned':       arr,
        'valid_guv_ids':            valid_ids,
        'median_jump_frame':        pulse_frame,
        'excluded_dead_guv_ids':    excluded_dead_ids,
        # Un-truncated normalised curves (pre-pulse frames INCLUDED) and the
        # matching absolute time base. 'all_curves_aligned' starts at the
        # pulse and so cannot express a pre-pulse baseline; the model-free
        # pre-vs-final endpoint figure needs both sides of the pulse.
        'all_curves_full':          np.array(normalised),
        'time_full':                time_array,
        'pre_window':               (safe_pre_start, safe_pre_end),
    }


# -------------------------------------------------------------------
# --- 5. FITTING ---
# -------------------------------------------------------------------

def _get_kinetic_model_spec(model_name: str, y_start: float, y_end: float,
                             A_est: float, tau_e: float):
    """
    Returns (fn, p0, bounds, param_names) for a named kinetic model, using
    the same initial-guess/bounds conventions for all four model variants.
    Returns None for an unrecognised model_name.
    """
    if model_name == 'EFFLUX-1EXP':
        fn = utils.efflux_1exp
        p0 = (np.clip(y_start, 0.01, 1.99),
              np.clip(y_end,   -1.99, 1.99),
              np.clip(tau_e,   0.02, 4999.0),
              0.0)
        bnds  = ((0, -2, 0.01, -0.1), (2, 2, 5000, 0.1))
        names = ['I0', 'Iinf', 'tau', 'D']
    elif model_name == 'INFLUX-1EXP':
        fn = utils.influx_1exp
        p0 = (np.clip(y_start, -1.99, 1.99),
              np.clip(y_end,    0.01, 1.99),
              np.clip(tau_e,   0.02, 4999.0),
              0.0)
        bnds  = ((-2, 0, 0.01, -0.1), (2, 2, 5000, 0.1))
        names = ['I0', 'Iinf', 'tau', 'D']
    elif model_name == 'EFFLUX-2EXP':
        fn = utils.efflux_2exp
        p0    = (y_end, A_est*0.5, tau_e*0.5, A_est*0.5, tau_e*2.0, 0.0)
        bnds  = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
        names = ['Iinf', 'a1', 'tau1', 'a2', 'tau2', 'D']
    elif model_name == 'INFLUX-2EXP':
        fn = utils.influx_2exp
        p0    = (y_start, A_est*0.5, tau_e*0.5, A_est*0.5, tau_e*2.0, 0.0)
        bnds  = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
        names = ['I0', 'a1', 'tau1', 'a2', 'tau2', 'D']
    else:
        return None
    return fn, p0, bnds, names


def _fit_kinetic_model(model_name: str, t_c: np.ndarray, y_c: np.ndarray,
                        y_start: float, y_end: float, A_est: float, tau_e: float):
    """
    Fits one named kinetic model to (t_c, y_c) and returns a dict with the
    fitted params, their SEs, RSS, and AIC/AICc/BIC — or None if the model
    is unrecognised or the fit fails to converge.
    """
    spec = _get_kinetic_model_spec(model_name, y_start, y_end, A_est, tau_e)
    if spec is None:
        return None
    fn, p0, bnds, names = spec

    try:
        params, pcov = sc.curve_fit(fn, t_c, y_c, p0=p0, bounds=bnds, maxfev=10000)
    except RuntimeError:
        return None

    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    y_fit = fn(t_c, *params)
    rss = float(np.sum((y_c - y_fit) ** 2))
    aic, aicc, bic = utils.compute_information_criteria(rss, n=len(t_c), k=len(names))

    return {
        'model_name': model_name, 'fn': fn, 'params': params, 'names': names,
        'perr': perr, 'y_fit': y_fit, 'rss': rss, 'n_points': len(t_c),
        'aic': aic, 'aicc': aicc, 'bic': bic,
    }


def fit_individual_curves(aligned: dict, guv_results: dict, t: np.ndarray, logger: logging.Logger) -> pd.DataFrame:
    arr      = aligned['all_curves_aligned']
    ids      = aligned['valid_guv_ids']
    df_track = guv_results['tracking_dataframe']

    direction = 'EFFLUX' if cfg.MODEL_TO_USE.startswith('EFFLUX') else 'INFLUX'
    name_1exp = f'{direction}-1EXP'
    name_2exp = f'{direction}-2EXP'

    fit_records = []
    fit_plot_data = []   # accumulated for the grid figure
    comparison_records = []   # AICc/BIC comparison, 1EXP vs 2EXP, per GUV

    for i, curve in enumerate(arr):
        guv_id = ids[i]

        guv_data = df_track[df_track['guv_id'] == guv_id]
        if guv_data.empty: continue
        r_px = guv_data['radius'].iloc[0]
        r_um = r_px * getattr(cfg, 'MICRONS_PER_PIXEL', 1.0)

        ok = np.isfinite(curve)
        t_c, y_c = t[ok], curve[ok]

        if len(t_c) < 5:
            continue

        y_start, y_end = y_c[0], y_c[-1]
        A_est = abs(y_end - y_start)

        try:
            target_y = y_start - 0.63 * A_est if cfg.MODEL_TO_USE.startswith('EFFLUX') else y_start + 0.63 * A_est
            idx63 = np.argmax(y_c < target_y) if cfg.MODEL_TO_USE.startswith('EFFLUX') else np.argmax(y_c > target_y)
            tau_e = t_c[idx63] if idx63 > 0 else t_c[-1] / 3.0
        except Exception:
            tau_e = t_c[-1] / 3.0

        tau_e = max(tau_e, 1.0)

        # --- Fit BOTH the 1EXP and 2EXP variants (same efflux/influx
        # direction as MODEL_TO_USE) so AICc/BIC can compare them directly.
        # Only the variant matching cfg.MODEL_TO_USE feeds the main
        # fit_records/plot outputs below; the pair always feeds the
        # comparison table.
        fit_1exp = _fit_kinetic_model(name_1exp, t_c, y_c, y_start, y_end, A_est, tau_e)
        fit_2exp = _fit_kinetic_model(name_2exp, t_c, y_c, y_start, y_end, A_est, tau_e)

        comp_row = {'guv_id': guv_id, 'n_points': len(t_c)}
        for tag, f in ((name_1exp, fit_1exp), (name_2exp, fit_2exp)):
            comp_row[f'RSS_{tag}']  = f['rss']  if f else np.nan
            comp_row[f'AICc_{tag}'] = f['aicc'] if f else np.nan
            comp_row[f'BIC_{tag}']  = f['bic']  if f else np.nan
        if fit_1exp and fit_2exp and np.isfinite(fit_1exp['aicc']) and np.isfinite(fit_2exp['aicc']):
            comp_row['delta_AICc_1EXPminus2EXP'] = fit_1exp['aicc'] - fit_2exp['aicc']
            comp_row['preferred_model_AICc'] = '2EXP' if fit_2exp['aicc'] < fit_1exp['aicc'] else '1EXP'
        else:
            comp_row['delta_AICc_1EXPminus2EXP'] = np.nan
            comp_row['preferred_model_AICc'] = None
        if fit_1exp and fit_2exp and np.isfinite(fit_1exp['bic']) and np.isfinite(fit_2exp['bic']):
            comp_row['delta_BIC_1EXPminus2EXP'] = fit_1exp['bic'] - fit_2exp['bic']
            comp_row['preferred_model_BIC'] = '2EXP' if fit_2exp['bic'] < fit_1exp['bic'] else '1EXP'
        else:
            comp_row['delta_BIC_1EXPminus2EXP'] = np.nan
            comp_row['preferred_model_BIC'] = None
        comparison_records.append(comp_row)

        primary = fit_1exp if cfg.MODEL_TO_USE == name_1exp else fit_2exp
        if primary is None:
            logger.warning(f"Fitting failed for GUV {guv_id}")
            continue

        params, names, perr, fn = primary['params'], primary['names'], primary['perr'], primary['fn']

        res = {'guv_id': guv_id, 'radius_um': r_um}
        res.update(dict(zip(names, params)))
        res.update({f'{n}_SE': e for n, e in zip(names, perr)})
        fit_records.append(res)

        # Build a short parameter label for the subplot title
        if 'tau2' in names:
            param_str = f"τ₁={params[2]:.1f} s,  τ₂={params[4]:.1f} s"
        else:
            param_str = f"τ = {params[2]:.1f} s"

        fit_plot_data.append({
            'guv_id':    guv_id,
            't_data':    t_c,
            'y_data':    y_c,
            'y_fit':     fn(t_c, *params),
            'r_um':      r_um,
            'param_str': param_str,
        })

    # ── Grid figure (replaces individual PNGs) ────────────────────────────
    if fit_plot_data:
        grid_path = utils.plot_dye_fits_grid(
            fit_plot_data, cfg.FOLDER_DYE_FITTING,
            cfg.EXPERIMENT_BASE_NAME, cfg.MODEL_TO_USE,
            ruptured_guv_ids=guv_results.get('ruptured_guv_ids_refined',
                                              guv_results.get('ruptured_guv_ids')),
            out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
            shrunk_guv_ids=guv_results.get('shrunk_guv_ids'),
        )
        logger.info(f"Saved dye fits grid → {grid_path}")

    # ── Model comparison export (AICc/BIC, 1EXP vs 2EXP) ──────────────────
    if comparison_records:
        df_comp = pd.DataFrame(comparison_records)
        comp_path = os.path.join(
            cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_model_comparison.csv"
        )
        df_comp.to_csv(comp_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved model comparison (AICc/BIC, 1EXP vs 2EXP) → {comp_path}")

        n_valid = df_comp['preferred_model_AICc'].notna().sum()
        if n_valid > 0:
            n_2exp_aicc = (df_comp['preferred_model_AICc'] == '2EXP').sum()
            n_2exp_bic  = (df_comp['preferred_model_BIC']  == '2EXP').sum()
            mean_d_aicc = df_comp['delta_AICc_1EXPminus2EXP'].mean()
            mean_d_bic  = df_comp['delta_BIC_1EXPminus2EXP'].mean()
            logger.info(
                f"Model comparison summary ({n_valid} GUV(s) with both fits): "
                f"2EXP preferred by AICc in {n_2exp_aicc}/{n_valid}, "
                f"by BIC in {n_2exp_bic}/{n_valid}. "
                f"Mean ΔAICc (1EXP−2EXP) = {mean_d_aicc:.2f}, "
                f"mean ΔBIC (1EXP−2EXP) = {mean_d_bic:.2f}. "
                f"(Positive Δ favours 2EXP; a difference >~2 is usually "
                f"considered meaningful for AICc, >~6 for BIC.)"
            )
        else:
            logger.warning("Model comparison: no GUV had both 1EXP and 2EXP fits converge.")

    df_fits = pd.DataFrame(fit_records)
    return df_fits

def generate_parameter_boxplots(df_fits: pd.DataFrame, logger: logging.Logger):
    if df_fits.empty: return
    
    bins = [0, 5.0, 8.0, 11.0, 100.0]
    labels = ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']
    df_fits['size_group'] = pd.cut(df_fits['radius_um'], bins=bins, labels=labels)
    
    df_fits.to_csv(os.path.join(cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_fit_parameters.csv"), index=False)
    
    params_to_plot = [
        col for col in df_fits.columns
        if col not in ['guv_id', 'radius_um', 'size_group'] and not col.endswith('_SE')
    ]
    
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    if len(params_to_plot) == 1: axes = [axes]
    
    for ax, param in zip(axes, params_to_plot):
        sns.boxplot(data=df_fits, x='size_group', y=param, ax=ax)
        sns.stripplot(data=df_fits, x='size_group', y=param, ax=ax, color='black', alpha=0.5)
        ax.set_title(param)
        ax.set_xlabel(r'Radius ($\mu$m)')
        
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_parameter_boxplots.png"), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# --- 6. EXPORT ---
# -------------------------------------------------------------------

def export_results(aligned: dict, dye_mmap_info: tuple, time_array: np.ndarray, logger: logging.Logger):
    t    = aligned['t_aligned']
    avg  = aligned['average_curve']
    arr  = aligned['all_curves_aligned']
    ids  = aligned['valid_guv_ids']
    mjf  = aligned['median_jump_frame']

    # Extract 5 dynamic snapshots directly from the memmap
    if dye_mmap_info[0] is not None:
        dye_path, dye_shape, dye_dtype = dye_mmap_info
        dye_stack = np.memmap(dye_path, dtype=dye_dtype, mode='r', shape=dye_shape)
        
        total_frames = dye_shape[0]
        if mjf < total_frames:
            # Generate 5 evenly spaced indices from the pulse frame to the end
            export_indices = np.linspace(mjf, total_frames - 1, 5).astype(int)
            
            for i, fi in enumerate(export_indices):
                img_raw = dye_stack[fi]
                rel_time = time_array[fi] - time_array[mjf]
                
                img_bright = utils._convert_to_8bit_gray(img_raw)
                styled = utils.style_image(img_bright, f"{int(round(rel_time))} S",
                                           cfg.MICRONS_PER_PIXEL, cfg.SCALE_BAR_LENGTH_MICRONS)
                
                cv2.imwrite(os.path.join(cfg.FOLDER_DYE,
                            f"snapshot_{i+1}_frame{fi}_at_{int(round(rel_time))}s.png"), styled)
        del dye_stack

    # Export CSV data
    try:
        hdrs  = ['Time (s)'] + [f'GUV_{g}_I_uptake' for g in ids] + ['Average_I_uptake']
        data  = np.hstack([t.reshape(-1,1), arr.T, avg.reshape(-1,1)])
        pd.DataFrame(data, columns=hdrs).to_csv(
            os.path.join(cfg.FOLDER_DYE_INTENSITY,
                         f"{cfg.EXPERIMENT_BASE_NAME}_normalized_curves.csv"),
            index=False, float_format='%.6f', na_rep='NaN')
    except Exception as e:
        logger.warning(f"Could not save normalised CSV: {e}")

    logger.info("Export complete.")    

# -------------------------------------------------------------------
# --- 7. VALIDATION ---
# -------------------------------------------------------------------

def validate_config(logger: logging.Logger) -> bool:
    ok = True
    checks = [
        (os.path.exists(cfg.DATA_FOLDER), f"DATA_FOLDER not found: {cfg.DATA_FOLDER}"),
        (cfg.MODEL_TO_USE in ('EFFLUX-1EXP', 'INFLUX-1EXP', 'EFFLUX-2EXP', 'INFLUX-2EXP'),  
         "MODEL_TO_USE must be an EFFLUX or INFLUX exponential model"),
        (0.1 <= getattr(cfg,'TRACKING_SEARCH_WINDOW_FACTOR',0.7) <= 3.0,
         "TRACKING_SEARCH_WINDOW_FACTOR should be 0.1–3.0"),
        (0.05 <= getattr(cfg,'TRACKING_MAX_RADIUS_CHANGE_FACTOR',0.2) <= 1.0,
         "TRACKING_MAX_RADIUS_CHANGE_FACTOR should be 0.05–1.0"),
        (1 <= getattr(cfg,'RUPTURE_DETECTION_CONSECUTIVE_FAILS',3) <= 20,
         "RUPTURE_DETECTION_CONSECUTIVE_FAILS should be 1–20"),
    ]
    for passed, msg in checks:
        if not passed:
            logger.error(f"Config error: {msg}"); ok = False
    if not ok:
        logger.critical("Configuration validation failed.")
    return ok

# -------------------------------------------------------------------
# --- 8. MAIN ---
# -------------------------------------------------------------------

def main():
    folders = [
        cfg.OUTPUT_IMAGE_FOLDER,
        cfg.FOLDER_TRACKING,
        cfg.FOLDER_MASKS,
        cfg.FOLDER_DYE,
        cfg.FOLDER_DYE_FITTING,
        cfg.FOLDER_DYE_INTENSITY,
    ]
    
    if getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False):
        folders.append(cfg.FOLDER_ACTIN)
        
    for _folder in folders:
        os.makedirs(_folder, exist_ok=True)
    
    logger = utils.setup_logging(cfg.OUTPUT_IMAGE_FOLDER, cfg.EXPERIMENT_BASE_NAME)
    logger.info("=== GUV Analysis Pipeline ===")

    if not validate_config(logger):
        return
    
    # Initialize data as None so the finally block can safely check it
    data = None

    try:
        # Step 1 – get GUV starting circles (launches GUI if JSON missing)
        circles = get_guv_circles(logger)
        if not circles:
            return

        # Step 2 – load image stacks and timestamps
        data = load_input_data(circles, logger)
        if data is None:
            return

        # Step 3 – track & extract intensities
        guv_results = process_all_guvs(
            data['circles'], data['roi_mmap_info'], data['dye_mmap_info'], logger,
            actin_mmap_info=data.get('actin_mmap_info', (None, None, None))
        )

        # Step 3b – recompute background (median + std) excluding neighboring
        # GUVs from each background annulus. This replaces the naive
        # per-worker background estimate, which has no knowledge of where
        # other tracked GUVs are and can be biased by a neighbor sitting
        # inside the ring. Runs before pulse detection so the corrected
        # background feeds every downstream step (pulse detection,
        # normalisation, fitting).
        dye_path0 = data['dye_mmap_info'][0]
        if not getattr(cfg, 'TRACKING_ONLY_MODE', False) and dye_path0 is not None \
                and guv_results['all_intensity_curves']:
            dye_path, dye_shape, dye_dtype = data['dye_mmap_info']
            dye_stack_bg = np.memmap(dye_path, dtype=dye_dtype, mode='r', shape=dye_shape)
            img_shape_bg = dye_stack_bg[0].shape[:2]
            n_frames_bg  = data['roi_mmap_info'][1][0]

            all_ellipses = [
                guv_results['raw_results'][idx][0]['tracking']['ellipses']
                for idx in guv_results['valid_guv_indices']
            ]

            new_bg_curves     = []
            new_bg_std_curves = []
            new_bg_n_curves   = []
            diagnostics_per_guv = {}
            for local_i in range(len(all_ellipses)):
                bg_val, bg_std, n_excl, diag = utils.recompute_background_traces(
                    dye_stack_bg, n_frames_bg, all_ellipses, local_i,
                    cfg.MEMBRANE_FIXED_HALF_WIDTH, cfg.BG_BUFFER_PIXELS, cfg.BG_RING_WIDTH_PIXELS,
                    img_shape_bg,
                    exclusion_padding=getattr(cfg, 'BG_EXCLUSION_PADDING_PX', 2),
                    min_bg_pixels=getattr(cfg, 'BG_EXCLUSION_MIN_PIXELS', 20),
                    neighbor_hold_frames=getattr(cfg, 'BG_NEIGHBOR_HOLD_FRAMES', 5),
                    sigma_clip=getattr(cfg, 'BG_SIGMA_CLIP', 3.0),
                    percentile=getattr(cfg, 'BG_PERCENTILE', 50.0),
                )
                new_bg_curves.append(bg_val)
                new_bg_std_curves.append(bg_std)
                # Final pixel count actually used for bg_val/bg_std (after
                # neighbor-position exclusion AND the statistical clip) —
                # needed downstream to convert the per-pixel std into a
                # standard error of the mean.
                new_bg_n_curves.append(diag['n_pixels_total'] - diag['n_excluded_stats'])

                gid = str(data['circles'][guv_results['valid_guv_indices'][local_i]]['id'])
                diagnostics_per_guv[gid] = diag

                frames_with_overlap = int(np.sum(n_excl > 0))
                if frames_with_overlap > 0:
                    logger.info(
                        f"  GUV {gid}: neighbor overlap excluded from background "
                        f"in {frames_with_overlap} frame(s)."
                    )

            guv_results['all_background_curves']     = new_bg_curves
            guv_results['all_background_std_curves'] = new_bg_std_curves
            guv_results['all_background_n_curves']   = new_bg_n_curves

            diag_csv = utils.export_background_diagnostics_csv(
                diagnostics_per_guv, data['time_array'], cfg.FOLDER_DYE,
                cfg.EXPERIMENT_BASE_NAME,
                sigma_clip_active=getattr(cfg, 'BG_SIGMA_CLIP', 3.0),
                logger=logger,
            )
            logger.info(f"Saved background diagnostics → {diag_csv}")

            del dye_stack_bg

        if getattr(cfg, 'EXPORT_CONSOLIDATED_TRACK_VIDEO', False) or \
           getattr(cfg, 'EXPORT_CONSOLIDATED_MASK_VIDEO', False):
            
            # Re-open the ROI stack for reading (the main process needs it now)
            roi_path, roi_shape, roi_dtype = data['roi_mmap_info']
            roi_stack = np.memmap(roi_path, dtype=roi_dtype, mode='r', shape=roi_shape)
            
            utils.export_full_stack_videos(
                cfg.FOLDER_TRACKING,
                cfg.FOLDER_MASKS,
                cfg.EXPERIMENT_BASE_NAME,
                roi_stack,
                [r[0] for r in guv_results['raw_results']],
                data['circles'],
                fps=getattr(cfg, 'VIDEO_EXPORT_FPS', 10.0)
            )
            
            # free the file handle
            del roi_stack
        
        # ── Detect (or override) the pulse frame ──────────────────────────
        override = getattr(cfg, 'PULSE_FRAME_OVERRIDE', None)
        if override is not None:
            PULSE_FRAME = int(override)
            logger.info(f"Pulse frame: {PULSE_FRAME}  (manual override from config)")
        elif not getattr(cfg, 'TRACKING_ONLY_MODE', False) and guv_results['all_intensity_curves']:
            sigma = getattr(cfg, 'PULSE_DETECT_SMOOTH_SIGMA', 2.0)
            PULSE_FRAME = utils.detect_pulse_frame_efflux(
                guv_results['all_intensity_curves'],
                guv_results['all_background_curves'],
                smooth_sigma=sigma,
                output_folder=cfg.FOLDER_DYE,
                experiment_name=cfg.EXPERIMENT_BASE_NAME,
                time_array=data['time_array'],
            )
            # Cross-check against the filename tag if present
            fn_match = re.search(r'frame(\d+)', cfg.EXPERIMENT_BASE_NAME)
            if fn_match:
                fn_frame = int(fn_match.group(1))
                delta = abs(PULSE_FRAME - fn_frame)
                if delta > 2:
                    logger.warning(
                        f"Auto-detected pulse frame ({PULSE_FRAME}) differs from "
                        f"filename tag ({fn_frame}) by {delta} frames — "
                        f"check *_pulse_frame_detection.png or set PULSE_FRAME_OVERRIDE."
                    )
                else:
                    logger.info(
                        f"Pulse frame: {PULSE_FRAME}  "
                        f"(auto-detected; filename tag = {fn_frame} ✓)"
                    )
            else:
                logger.info(f"Pulse frame: {PULSE_FRAME}  (auto-detected)")
        else:
            # Fallback: filename tag or 0
            fn_match = re.search(r'frame(\d+)', cfg.EXPERIMENT_BASE_NAME)
            PULSE_FRAME = int(fn_match.group(1)) if fn_match else 0
            logger.info(f"Pulse frame: {PULSE_FRAME}  (from filename tag)")

        t_pulse = data['time_array'][PULSE_FRAME]

        # Generate the Size and Circularity plots and update tracking CSV
        df_track = guv_results.get('tracking_dataframe')
        
        if df_track is not None:
            # 1. Normalize Tracking Data based on detected PULSE_FRAME
            df_track = utils.normalize_tracking_data(df_track, PULSE_FRAME)
            
            # 2. Re-save updated main tracking CSV
            df_track.to_csv(
                os.path.join(cfg.FOLDER_TRACKING, f"{cfg.EXPERIMENT_BASE_NAME}_tracking_data.csv"),
                index=False
            )
            
            # 3. Export Summary CSV
            utils.export_tracking_summary_csv(
                df_track, data['time_array'], cfg.FOLDER_TRACKING, cfg.EXPERIMENT_BASE_NAME
            )
            
            # 4. Generate 1x2 shape plots for each GUV
            try:
                utils.plot_guv_shape_metrics(
                    df_track, 
                    data['time_array'], 
                    cfg.FOLDER_TRACKING, 
                    cfg.EXPERIMENT_BASE_NAME, 
                    t_pulse
                )
                logger.info("Saved normalized size and circularity plots + summary CSV.")
            except Exception as e:
                logger.warning(f"Failed to generate shape metrics plots: {e}")

            # 5. Refine RUPTURED / OUT_OF_FRAME into RUPTURED / SHRUNK /
            # OUT_OF_FRAME / SURVIVED, now that PULSE_FRAME (needed for the
            # pre-pulse radius baseline) is known. A vesicle that deflates
            # gradually and only then drops below the ring-detector's
            # minimum resolvable size will ALSO trip the tracker's raw
            # rupture-score exit — that's a detection-limit artifact of
            # shrinkage, not a membrane burst, so it's reclassified SHRUNK
            # here rather than left counted as a rupture.
            fate_map, df_fate = utils.classify_guv_fates(
                df_track,
                guv_results.get('ruptured_guv_ids', set()),
                guv_results.get('out_of_frame_guv_ids', set()),
                shrinkage_fraction_threshold=getattr(cfg, 'SHRINKAGE_FRACTION_THRESHOLD', 0.15),
                terminal_n_frames=getattr(cfg, 'SHRINKAGE_TERMINAL_N_FRAMES', 3),
            )
            df_fate.to_csv(
                os.path.join(cfg.FOLDER_TRACKING,
                             f"{cfg.EXPERIMENT_BASE_NAME}_guv_fate_classification.csv"),
                index=False, float_format='%.4f'
            )
            ruptured_guv_ids = {g for g, f in fate_map.items() if f == 'RUPTURED'}
            shrunk_guv_ids   = {g for g, f in fate_map.items() if f == 'SHRUNK'}
            out_of_frame_ids_refined = {g for g, f in fate_map.items() if f == 'OUT_OF_FRAME'}
            n_survived = sum(1 for f in fate_map.values() if f == 'SURVIVED')
            logger.info(
                f"GUV fate classification: {len(ruptured_guv_ids)} ruptured, "
                f"{len(shrunk_guv_ids)} shrunk (>= "
                f"{getattr(cfg, 'SHRINKAGE_FRACTION_THRESHOLD', 0.15)*100:.0f}% radius loss), "
                f"{len(out_of_frame_ids_refined)} lost out of frame, "
                f"{n_survived} survived intact → "
                f"{cfg.EXPERIMENT_BASE_NAME}_guv_fate_classification.csv"
            )
            # Stored back on guv_results so any function downstream that
            # only receives guv_results (e.g. fit_individual_curves) can
            # pick up the shrinkage-aware classification without needing an
            # extra parameter threaded through every call.
            guv_results['ruptured_guv_ids_refined'] = ruptured_guv_ids
            guv_results['shrunk_guv_ids']            = shrunk_guv_ids
        else:
            # No tracking dataframe: fall back to the tracker's raw
            # RUPTURED/OUT_OF_FRAME sets with no shrinkage refinement.
            ruptured_guv_ids = guv_results.get('ruptured_guv_ids', set())
            shrunk_guv_ids   = set()
            guv_results['ruptured_guv_ids_refined'] = ruptured_guv_ids
            guv_results['shrunk_guv_ids']            = shrunk_guv_ids
        
        if getattr(cfg, 'TRACKING_ONLY_MODE', False):
            logger.info("=== TRACKING_ONLY_MODE Active: Halting before intensity analysis ===")
            return
        
        if not guv_results['all_intensity_curves']:
            logger.error("No GUVs produced usable data."); return

        # Step 4 – normalise & align
        aligned = normalize_and_align_curves(
            guv_results, data['time_array'], data['circles'], PULSE_FRAME, logger
        )
        if aligned is None:
            return

        excluded_dead = aligned.get('excluded_dead_guv_ids', [])
        if excluded_dead:
            logger.info(
                f"Excluded {len(excluded_dead)} GUV(s) from dye-channel "
                f"analysis for pre-pulse signal indistinguishable from "
                f"background: {excluded_dead}"
            )

        # guv_id -> raw_results index, built fresh here rather than relying on
        # positional alignment with aligned['valid_guv_ids'] — the dead-GUV
        # exclusion above (and other drops) can leave gaps relative to
        # guv_results['valid_guv_indices'].
        id_to_raw_idx = {
            str(data['circles'][gi]['id']): gi
            for gi in guv_results['valid_guv_indices']
        }

        # Step 4b – dye summary (all traces + mean)
        summary_path = utils.plot_dye_summary(
            aligned, cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
            ruptured_guv_ids=ruptured_guv_ids,
            out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
            shrunk_guv_ids=shrunk_guv_ids,
        )
        logger.info(f"Saved dye summary plot → {summary_path}")

        # Step 4c – model-free endpoint: pre-pulse vs final frame, per GUV.
        # Kept separate from (and computed before) the kinetic fitting so it
        # remains valid when the post-pulse sampling is too coarse for tau to
        # be identifiable.
        if getattr(cfg, 'EXPORT_PREPOST_INTENSITY_PLOT', True):
            prepost_png, prepost_csv, df_prepost = utils.plot_dye_prepulse_vs_final(
                aligned['all_curves_full'],
                aligned['time_full'],
                aligned['median_jump_frame'],
                aligned['valid_guv_ids'],
                cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
                n_pre=getattr(cfg, 'PREPOST_N_PRE_FRAMES', 5),
                n_final=getattr(cfg, 'PREPOST_N_FINAL_FRAMES', 3),
                ruptured_guv_ids=ruptured_guv_ids,
                out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                shrunk_guv_ids=shrunk_guv_ids,
            )
            logger.info(f"Saved pre-pulse vs final plot → {prepost_png}")
            logger.info(f"Saved pre-pulse vs final CSV  → {prepost_csv}")
            _ok = df_prepost['released_pct'].dropna()
            if len(_ok) > 0:
                logger.info(
                    f"  Released fraction across {len(_ok)} GUV(s): "
                    f"{_ok.mean():.1f} ± {_ok.std():.1f} %"
                )

            # Matching single-vesicle crops (membrane + dye [+ actin]) at the
            # last pre-pulse frame and each GUV's own final frame. Keyed off
            # id_to_raw_idx rather than positional order, since the dead-GUV
            # filter can leave gaps relative to valid_guv_indices.
            if getattr(cfg, 'EXPORT_GUV_CROPS', True):
                gid_to_tracking = {
                    gid: guv_results['raw_results'][gi][0]['tracking']
                    for gid, gi in id_to_raw_idx.items()
                    if guv_results['raw_results'][gi][0] is not None
                }
                crops_png = utils.export_guv_crops_prepost(
                    df_prepost,
                    gid_to_tracking,
                    aligned['median_jump_frame'],
                    aligned['time_full'],
                    data['roi_mmap_info'],
                    data['dye_mmap_info'],
                    data.get('actin_mmap_info', (None, None, None)),
                    cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
                    crop_factor=getattr(cfg, 'CROP_FACTOR', 2.0),
                    microns_per_pixel=getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
                    scale_bar_microns=getattr(cfg, 'SCALE_BAR_LENGTH_MICRONS', 10),
                    analyze_actin=getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False),
                    save_individual=getattr(cfg, 'CROP_SAVE_INDIVIDUAL', True),
                    uniform_box=getattr(cfg, 'CROP_UNIFORM_BOX', True),
                    channel_colors=getattr(cfg, 'CROP_CHANNEL_COLORS', None),
                    ruptured_guv_ids=ruptured_guv_ids,
                    out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                    shrunk_guv_ids=shrunk_guv_ids,
                )
                if crops_png:
                    logger.info(f"Saved single-vesicle crops → {crops_png}")

        # Step 5 – fit
        df_fits = fit_individual_curves(aligned, guv_results, aligned['t_aligned'], logger)
        generate_parameter_boxplots(df_fits, logger)
        
        # Step 6 – export
        export_results(aligned, data['dye_mmap_info'], data['time_array'], logger)

        # Step 7 – Actin cortex analysis (C2 channel)
        actin_data_list = [ad for ad in guv_results.get('all_actin_data', [])
                           if ad is not None]
        if actin_data_list and getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False):
            logger.info("Running actin cortex analysis ...")
            aligned_actin = utils.normalize_actin_curves(
                actin_data_list, PULSE_FRAME, data['time_array']
            )
            valid_ids = aligned['valid_guv_ids']  # GUVs that passed the pre-pulse SNR gate

            if getattr(cfg, 'EXPORT_ACTIN_TRACES', True):
                csv_path = utils.export_actin_csv(
                    aligned_actin, valid_ids,
                    cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME
                )
                logger.info(f"Saved actin traces → {csv_path}")

            plot_path = utils.plot_actin_analysis(
                aligned_actin, valid_ids,
                cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                smooth_sigma=getattr(cfg, 'ACTIN_PLOT_SMOOTH_SIGMA', 1.5),
                ruptured_guv_ids=ruptured_guv_ids,
                out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                shrunk_guv_ids=shrunk_guv_ids,
            )
            logger.info(f"Saved actin analysis plot → {plot_path}")

            # Combined dye + actin overlay (only when both pipelines ran)
            if aligned is not None:
                overlay_path = utils.plot_dye_actin_overlay(
                    aligned, aligned_actin, valid_ids,
                    cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                    smooth_sigma=getattr(cfg, 'ACTIN_PLOT_SMOOTH_SIGMA', 1.5),
                    ruptured_guv_ids=ruptured_guv_ids,
                    out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                    shrunk_guv_ids=shrunk_guv_ids,
                )
                logger.info(f"Saved dye/actin overlay → {overlay_path}")

            # Actin angular kymographs (directionality of cortex breakdown).
            # Built from a fresh guv_id -> actin_data mapping keyed off
            # valid_guv_indices (the order all_actin_data was appended in),
            # rather than reusing valid_ids from the dye-channel dead-GUV
            # filter above — those two lists can diverge in length/order
            # whenever a GUV is excluded by the dye SNR gate but still has
            # usable actin data, or vice versa.
            if getattr(cfg, 'EXPORT_ACTIN_KYMOGRAPH', True):
                n_angles_cfg = getattr(cfg, 'ACTIN_N_ANGLES', 72)
                electrode_angle = getattr(cfg, 'ELECTRODE_ANGLE_DEG', 0.0)
                display_offset  = getattr(cfg, 'ANGLE_DISPLAY_OFFSET_DEG', 90.0)

                actin_gid_ad_pairs = [
                    (str(data['circles'][gi]['id']), ad)
                    for gi, ad in zip(guv_results['valid_guv_indices'],
                                      guv_results.get('all_actin_data', []))
                    if ad is not None
                ]

                kymo_ids, kymo_data_list = [], []
                for gid, ad in actin_gid_ad_pairs:
                    kd = utils.build_actin_angular_kymograph(
                        ad, PULSE_FRAME, data['time_array'], n_angles=n_angles_cfg
                    )
                    if kd is None:
                        continue
                    kymo_path = utils.plot_actin_angular_kymograph(
                        kd, gid, cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                        electrode_angle_deg=electrode_angle,
                        angle_display_offset_deg=display_offset,
                    )
                    logger.info(f"  GUV {gid}: saved actin angular kymograph → {kymo_path}")

                    pe_data = utils.compute_pole_vs_equator_trace(
                        kd, electrode_angle_deg=electrode_angle,
                        angular_half_width_deg=getattr(cfg, 'POLE_EQUATOR_HALF_WIDTH_DEG', 22.5),
                    )
                    pe_path = utils.plot_pole_vs_equator(
                        pe_data, gid, cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME
                    )
                    logger.info(f"  GUV {gid}: saved pole-vs-equator trace → {pe_path}")

                    kymo_ids.append(gid)
                    kymo_data_list.append(kd)

                if kymo_data_list:
                    group_path = utils.plot_actin_angular_kymograph_group(
                        kymo_data_list, kymo_ids,
                        cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                        electrode_angle_deg=electrode_angle,
                        angle_display_offset_deg=display_offset,
                    )
                    logger.info(f"Saved population-average actin kymograph → {group_path}")
                else:
                    logger.info("Actin kymograph: no GUV had a detected cortex peak to plot.")

            # Radial profile snapshot at the last pre-pulse frame
            pre_frame = max(0, PULSE_FRAME - 1)
            snapshots = []
            
            for gid in valid_ids:
                raw_idx = id_to_raw_idx.get(gid)
                if raw_idx is None:
                    continue
                raw = guv_results['raw_results'][raw_idx][0]
                if raw is None:
                    continue
                ellipses_list = raw.get('tracking', {}).get('ellipses', [])
                if pre_frame < len(ellipses_list) and ellipses_list[pre_frame] is not None:
                    snapshots.append((gid, pre_frame, ellipses_list[pre_frame]))

            if snapshots and data.get('actin_mmap_info', (None,))[0] is not None:
                ap, ashape, adtype = data['actin_mmap_info']
                actin_stack_main = np.memmap(ap, dtype=adtype, mode='r', shape=ashape)
                
                # Single Radial profile  (pre-pulse frame, per GUV)
                utils.plot_actin_spatial_snapshot(
                    actin_stack_main, snapshots,
                    cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                    microns_per_pixel=getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
                )

                # Radial profile evolution (all frames, per GUV)
                evolution_input = []
                for gid in valid_ids:
                    raw_idx = id_to_raw_idx.get(gid)
                    if raw_idx is None:
                        continue
                    raw = guv_results['raw_results'][raw_idx][0]
                    if raw is None:
                        continue
                    el_list = raw.get('tracking', {}).get('ellipses', [])
                    evolution_input.append((gid, el_list))

                if evolution_input:
                    evo_path = utils.plot_actin_radial_evolution(
                        actin_stack_main, evolution_input,
                        cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                        microns_per_pixel=getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
                    )
                    logger.info(f"Saved actin radial evolution plot → {evo_path}")

                del actin_stack_main
                logger.info("Saved actin radial profile snapshots.")
        elif getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False):
            logger.warning("Actin analysis enabled but no actin data was extracted.")

        logger.info("=== Pipeline finished successfully ===")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        
    finally:
        # Force garbage collection to release lingering Windows memmap file locks
        gc.collect()
        
        # Guarantee removal of memory-mapped files from local disk
        if data is not None:
            for path_key in ('roi_mmap_path', 'dye_mmap_path', 'actin_mmap_path'):
                mmap_path = data.get(path_key)
                if mmap_path and os.path.exists(mmap_path):
                    try:
                        os.remove(mmap_path)
                        logger.info(f"Cleaned up temporary file: {mmap_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {mmap_path}: {e}")

if __name__ == "__main__":
    main()