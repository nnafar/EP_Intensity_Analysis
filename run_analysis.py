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

import config as cfg
import guv_analysis_utils as utils


# -------------------------------------------------------------------
# --- 1. GUV CIRCLE DEFINITIONS
# -------------------------------------------------------------------

def get_guv_circles(logger: logging.Logger) -> list[dict]:
    """
    Returns the list of user-drawn GUV circles.

    If CIRCLES_JSON_PATH already exists, loads it.
    Otherwise, launches the interactive circle selector so the user
    can draw circles on the first ROI frame, then saves the result.

    Each circle is a dict: {"id": N, "x": int, "y": int, "r": int}
    """
    path = cfg.CIRCLES_JSON_PATH

    if os.path.exists(path):
        with open(path) as f:
            circles = json.load(f)
        logger.info(f"Loaded {len(circles)} GUV circle(s) from: {path}")
        return circles

    # ── No JSON found → run interactive selector ──────────────────────────
    logger.info("No circle definitions found — launching interactive selector.")

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
        logger.error(f"Could not load first ROI frame: {roi_files[0]}")
        return []

    # Import here so the rest of the pipeline works even without a display
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
    
    first_frame = cv2.imread(files[0], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    dtype = first_frame.dtype
    shape = (len(files), first_frame.shape[0], first_frame.shape[1])
    
    mmap_arr = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=shape)
    
    # Wrap the enumerator in tqdm for the progress bar
    for i, f in enumerate(tqdm(files, desc=desc, unit="frame")):
        mmap_arr[i] = cv2.imread(f, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    mmap_arr.flush()
    
    return mmap_path, shape, str(dtype)

def load_input_data(circles: list[dict], logger: logging.Logger) -> dict | None:
    """
    Loads ROI frames, dye frames, and timestamps.

    Parameters
    ----------
    circles : list of {"id", "x", "y", "r"} dicts
    """
    dye_files = natsorted(glob.glob(os.path.join(
        cfg.DATA_FOLDER,
        cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
    )))
    roi_files = natsorted(glob.glob(os.path.join(
        cfg.DATA_FOLDER,
        cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX
    )))

    if not dye_files:
        logger.error("No dye files found.")
        return None
    if not roi_files:
        logger.error("No ROI files found.")
        return None

    if len(dye_files) != len(roi_files):
        n = min(len(dye_files), len(roi_files))
        logger.warning(f"Frame count mismatch — trimming to {n} frames.")
        dye_files = dye_files[:n]
        roi_files = roi_files[:n]

    logger.info(f"Found {len(dye_files)} frame(s), {len(circles)} GUV(s).")

    # Timestamps
    time_array, frame_interval = utils.extract_timestamps_from_metadata(dye_files)
    if time_array is None:
        logger.warning("Metadata timestamps not found — using fallback FPS.")
        time_array, frame_interval = utils.create_manual_timestamps(
            len(dye_files), getattr(cfg, 'FALLBACK_FPS', 1.0)
        )
    logger.info(f"Frame interval: {frame_interval:.3f} s  ({len(time_array)} frames)")
    
    
    # Reroute the memory maps to a shorter, dedicated local path
    mmap_dir = r"C:\temp_guv_data" 
    if not os.path.exists(mmap_dir):
        try:
            os.makedirs(mmap_dir)
        except Exception:
            # Fallback to the experiment folder if C: is restricted
            mmap_dir = cfg.DATA_FOLDER
   
    # Use a generic name to keep the path length short
    roi_mmap_path = os.path.join(mmap_dir, "roi_temp.dat")
    dye_mmap_path = os.path.join(mmap_dir, "dye_temp.dat")
    
    tracking_only = getattr(cfg, 'TRACKING_ONLY_MODE', False)
    
    logger.info(f"Compiling TIFFs into local memory-mapped stacks at {mmap_dir}...")
    roi_mmap_info = create_memmap_stack(roi_files, roi_mmap_path, desc="ROI Stack")
    
    if tracking_only:
        logger.info("TRACKING_ONLY_MODE is True. Skipping Dye Stack compilation.")
        dye_mmap_info = (None, None, None)
    else:
        dye_mmap_info = create_memmap_stack(dye_files, dye_mmap_path, desc="Dye Stack")

    return {
        'dye_files':      dye_files,
        'roi_mmap_info':  roi_mmap_info,
        'dye_mmap_info':  dye_mmap_info,
        'roi_mmap_path':  roi_mmap_path,  
        'dye_mmap_path':  dye_mmap_path,  
        'time_array':     time_array,
        'frame_interval': frame_interval,
        'circles':        circles,
    }

# -------------------------------------------------------------------
# --- 3. PARALLEL GUV PROCESSING ---
# -------------------------------------------------------------------

def process_all_guvs(circles: list[dict],
                     roi_mmap_info: tuple,
                     dye_mmap_info: tuple,
                     logger: logging.Logger) -> dict:
    """
    Runs process_single_guv for each circle using multiprocessing.
    """
    # Extract the total number of frames from the shape tuple
    n_frames = roi_mmap_info[1][0]

    process_func = partial(
        utils.process_single_guv,
        roi_mmap_info=roi_mmap_info,
        dye_mmap_info=dye_mmap_info,
        n_frames=n_frames
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
    quality_log    = []
    valid_indices  = []
    tracking_rows  = []

    for i, (result, qe) in enumerate(results):
        quality_log.append(qe)
        gid = qe['guv_id']
        rup = qe.get('ruptured_at_frame')

        if rup is not None:
            logger.warning(
                f"  GUV {gid}: RUPTURED at frame {rup}  "
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
                    eccentricity = np.sqrt(1.0 - (minor / major)**2)
                    
                    tracking_rows.append({
                        'guv_id': gid, 'frame': fi,
                        'x': c[0], 'y': c[1],
                        'radius': r, 
                        'semi_major': round(major, 2), 
                        'semi_minor': round(minor, 2),
                        'eccentricity': round(eccentricity, 4),
                        'ring_score': round(s, 2)
                    })
        else:
            logger.warning(f"  GUV {gid}: no data returned.")

    logger.info(
        f"Processed {len(all_intensity)} / {len(circles)} GUV(s) successfully."
    )

    # Save quality log
    pd.DataFrame(quality_log).to_csv(
        os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                     f"{cfg.EXPERIMENT_BASE_NAME}_detection_quality.csv"),
        index=False
    )

    # Save tracking data
    df_track = None
    if tracking_rows and getattr(cfg, 'EXPORT_TRACKING_DATA', True):
        df_track = pd.DataFrame(tracking_rows)
        df_track.to_csv(
            os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                         f"{cfg.EXPERIMENT_BASE_NAME}_tracking_data.csv"),
            index=False
        )
        logger.info("Saved per-frame tracking data.")

    return {
        'raw_results':           results, 
        'all_intensity_curves':  all_intensity,
        'all_background_curves': all_background,
        'valid_guv_indices':     valid_indices,
        'tracking_dataframe':    df_track,
    }


# -------------------------------------------------------------------
# --- 4. NORMALISATION & ALIGNMENT ---
# -------------------------------------------------------------------

# run_analysis.py

def normalize_and_align_curves(guv_results: dict,
                                time_array: np.ndarray,
                                circles: list[dict],
                                logger: logging.Logger) -> dict | None:
    """
    Normalises: Fractional Retention = (I_dye - I_bg) / (I0 - bg0)
    Aligns:     t = 0 at the pulse frame extracted from filename.
    """
    logger.info("Normalising intensity curves for dye efflux...")
    
    # Extract pulse frame from experiment name (e.g., frame7)
    match = re.search(r'frame(\d+)', cfg.EXPERIMENT_BASE_NAME)
    PULSE_FRAME = int(match.group(1)) if match else 0
    
    normalised = []
    for dye, bg in zip(guv_results['all_intensity_curves'],
                       guv_results['all_background_curves']):
        if PULSE_FRAME >= len(dye):
            continue

        # Baseline is everything before the pulse
        I0  = float(np.nanmean(dye[:PULSE_FRAME]))
        bg0 = float(np.nanmean(bg[:PULSE_FRAME]))
        den = I0 - bg0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            if den == 0 or np.isnan(den):
                norm = np.full_like(dye, np.nan)
            else:
                norm = (dye - bg) / den
        normalised.append(norm)

    if not normalised:
        logger.error("Normalisation failed for all GUVs.")
        return None

    # Align curves to start at t=0 on the pulse frame
    arr   = np.array(normalised)[:, PULSE_FRAME:]
    t_al  = time_array[PULSE_FRAME:] - time_array[PULSE_FRAME]
    avg   = np.nanmean(arr, axis=0)

    valid_ids = [str(circles[i]['id']) for i in guv_results['valid_guv_indices']]

    return {
        't_aligned':          t_al,
        'average_curve':      avg,
        'all_curves_aligned': arr,
        'valid_guv_ids':      valid_ids,
        'median_jump_frame':  PULSE_FRAME,
    }


# -------------------------------------------------------------------
# --- 5. FITTING ---
# -------------------------------------------------------------------

def fit_individual_curves(aligned: dict, guv_results: dict, t: np.ndarray, logger: logging.Logger) -> pd.DataFrame:
    arr = aligned['all_curves_aligned']
    ids = aligned['valid_guv_ids']
    df_track = guv_results['tracking_dataframe']
    
    fit_records = []
    
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
        
        if cfg.MODEL_TO_USE == 'EFFLUX-1EXP':
            fn = utils.efflux_1exp
            p0 = (y_start, y_end, tau_e, 0.0)
            bnds = ((0, -2, 0.01, -0.1), (2, 2, 5000, 0.1))
            names = ['I0', 'Iinf', 'tau', 'D']
        elif cfg.MODEL_TO_USE == 'INFLUX-1EXP':
            fn = utils.influx_1exp
            p0 = (y_start, y_end, tau_e, 0.0)
            bnds = ((-2, 0, 0.01, -0.1), (2, 2, 5000, 0.1))
            names = ['I0', 'Iinf', 'tau', 'D']
        elif cfg.MODEL_TO_USE == 'EFFLUX-2EXP':
            fn = utils.efflux_2exp
            p0 = (y_end, A_est * 0.5, tau_e * 0.5, A_est * 0.5, tau_e * 2.0, 0.0)
            bnds = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
            names = ['Iinf', 'a1', 'tau1', 'a2', 'tau2', 'D']
        elif cfg.MODEL_TO_USE == 'INFLUX-2EXP':
            fn = utils.influx_2exp
            p0 = (y_start, A_est * 0.5, tau_e * 0.5, A_est * 0.5, tau_e * 2.0, 0.0)
            bnds = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
            names = ['I0', 'a1', 'tau1', 'a2', 'tau2', 'D']
        else:
            logger.error("Unknown model selected.")
            continue
        
        try:
            params, _ = sc.curve_fit(fn, t_c, y_c, p0=p0, bounds=bnds, maxfev=10000)
            res = {'guv_id': guv_id, 'radius_um': r_um}
            res.update(dict(zip(names, params)))
            fit_records.append(res)
            
            fig, ax = plt.subplots()
            ax.plot(t_c, y_c, 'k.', alpha=0.5, label='Data')
            ax.plot(t_c, fn(t_c, *params), 'r-', label='Fit')
            # Add Pulse line at t=0 (since data is aligned to pulse)
            ax.axvline(0, color='gray', ls='--', lw=1.5, label='Pulse', zorder=0)
            
            if 'tau2' in names:
                ax.set_title(f"GUV {guv_id} (R={r_um:.1f} $\mu$m)\nTau1: {params[2]:.2f} s, Tau2: {params[4]:.2f} s")
            else:
                ax.set_title(f"GUV {guv_id} (R={r_um:.1f} $\mu$m)\nTau: {params[2]:.2f} s")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Intensity")
            ax.legend()
            fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_fit_GUV_{guv_id}.png"))
            plt.close(fig)
            
        except RuntimeError:
            logger.warning(f"Fitting failed for GUV {guv_id}")
            
    df_fits = pd.DataFrame(fit_records)
    return df_fits

def generate_parameter_boxplots(df_fits: pd.DataFrame, logger: logging.Logger):
    if df_fits.empty: return
    
    bins = [0, 5.0, 8.0, 11.0, 100.0]
    labels = ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']
    df_fits['size_group'] = pd.cut(df_fits['radius_um'], bins=bins, labels=labels)
    
    df_fits.to_csv(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_fit_parameters.csv"), index=False)
    
    params_to_plot = [col for col in df_fits.columns if col not in ['guv_id', 'radius_um', 'size_group']]
    
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    if len(params_to_plot) == 1: axes = [axes]
    
    for ax, param in zip(axes, params_to_plot):
        sns.boxplot(data=df_fits, x='size_group', y=param, ax=ax)
        sns.stripplot(data=df_fits, x='size_group', y=param, ax=ax, color='black', alpha=0.5)
        ax.set_title(param)
        ax.set_xlabel('Radius ($\mu$m)')
        
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_parameter_boxplots.png"), dpi=300)
    plt.close(fig)

# -------------------------------------------------------------------
# --- 6. EXPORT ---
# -------------------------------------------------------------------

def export_results(aligned: dict, dye_files: list, logger: logging.Logger):
    t    = aligned['t_aligned']
    avg  = aligned['average_curve']
    arr  = aligned['all_curves_aligned']
    ids  = aligned['valid_guv_ids']
    mjf  = aligned['median_jump_frame']

    dye_al = dye_files[mjf:]

    for i, tp in enumerate(cfg.EXPORT_TIME_POINTS_S):
        fi, _ = utils.find_closest_frame(t, tp)
        if fi >= len(dye_al): continue
        img_raw = cv2.imread(dye_al[fi], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img_raw is None: continue
        
    
        # USE THE BOOSTED CONTRAST FUNCTION HERE
        img_bright = utils._convert_to_8bit_gray(img_raw)
        
        styled = utils.style_image(img_bright, f"{int(round(tp))} S",
                                   cfg.MICRONS_PER_PIXEL, cfg.SCALE_BAR_LENGTH_MICRONS)
        cv2.imwrite(os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                    f"frame_{i+1}_at_{int(round(tp))}s.png"), styled)

    try:
        hdrs  = ['Time (s)'] + [f'GUV_{g}_I_uptake' for g in ids] + ['Average_I_uptake']
        data  = np.hstack([t.reshape(-1,1), arr.T, avg.reshape(-1,1)])
        pd.DataFrame(data, columns=hdrs).to_csv(
            os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
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
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
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
            data['circles'], data['roi_mmap_info'], data['dye_mmap_info'], logger
        )
        
        
        if getattr(cfg, 'EXPORT_CONSOLIDATED_TRACK_VIDEO', False) or \
           getattr(cfg, 'EXPORT_CONSOLIDATED_MASK_VIDEO', False):
            
            # Re-open the ROI stack for reading (the main process needs it now)
            roi_path, roi_shape, roi_dtype = data['roi_mmap_info']
            roi_stack = np.memmap(roi_path, dtype=roi_dtype, mode='r', shape=roi_shape)
            
            utils.export_full_stack_videos(
                cfg.OUTPUT_IMAGE_FOLDER, 
                cfg.EXPERIMENT_BASE_NAME,
                roi_stack,
                [r[0] for r in guv_results['raw_results']], # List of 'result' dicts
                data['circles'],
                fps=getattr(cfg, 'VIDEO_EXPORT_FPS', 10.0)
            )
        
        # Calculate pulse time for the absolute-time tracking plots
        match = re.search(r'frame(\d+)', cfg.EXPERIMENT_BASE_NAME)
        PULSE_FRAME = int(match.group(1)) if match else 7
        t_pulse = data['time_array'][PULSE_FRAME]
        
        # Generate the Size, Deformation, and MSD plots
        df_track = guv_results.get('tracking_dataframe')
        if df_track is not None:
            try:
                utils.plot_tracking_metrics(
                    df_track, 
                    data['time_array'], 
                    cfg.OUTPUT_IMAGE_FOLDER, 
                    cfg.EXPERIMENT_BASE_NAME, 
                    getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
                    pulse_time=t_pulse
                )
                logger.info("Saved size, deformation, and MSD tracking plots.")
            except Exception as e:
                logger.warning(f"Failed to generate tracking metrics plots: {e}")
        
        if getattr(cfg, 'TRACKING_ONLY_MODE', False):
            logger.info("=== TRACKING_ONLY_MODE Active: Halting before intensity analysis ===")
            return
        
        if not guv_results['all_intensity_curves']:
            logger.error("No GUVs produced usable data."); return

        # Step 4 – normalise & align
        aligned = normalize_and_align_curves(
            guv_results, data['time_array'], data['circles'], logger
        )
        if aligned is None:
            return

        # Step 5 – fit
        df_fits = fit_individual_curves(aligned, guv_results, aligned['t_aligned'], logger)
        generate_parameter_boxplots(df_fits, logger)
        
        # Step 6 – export
        export_results(aligned, data['dye_files'], logger)

        logger.info("=== Pipeline finished successfully ===")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        
    finally:
        # Guarantee removal of memory-mapped files from local disk
        if data is not None:
            for path_key in ('roi_mmap_path', 'dye_mmap_path'):
                mmap_path = data.get(path_key)
                if mmap_path and os.path.exists(mmap_path):
                    try:
                        os.remove(mmap_path)
                        logger.info(f"Cleaned up temporary file: {mmap_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {mmap_path}: {e}")

if __name__ == "__main__":
    main()