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

    logger.info(
        f"Processed {len(all_intensity)} / {len(circles)} GUV(s) successfully."
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

    ruptured_ids = {
        str(qe['guv_id'])
        for qe in quality_log
        if qe.get('ruptured_at_frame') is not None
    }

    return {
        'raw_results':           results,
        'all_intensity_curves':  all_intensity,
        'all_background_curves': all_background,
        'all_actin_data':        all_actin_data,
        'valid_guv_indices':     valid_indices,
        'tracking_dataframe':    df_track,
        'ruptured_guv_ids':      ruptured_ids,   # set of str IDs that ruptured
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
    Normalises: Fractional Retention = (I_dye - I_bg) / (I0 - bg0)
    Aligns:     t = 0 at *pulse_frame* (auto-detected or overridden in config).
    """
    logger.info(f"Normalising intensity curves (pulse frame = {pulse_frame}) ...")
    
    normalised = []
    for dye, bg in zip(guv_results['all_intensity_curves'],
                       guv_results['all_background_curves']):
        if pulse_frame >= len(dye):
            continue

        # Strictly use the first 5 frames for a clean pre-pulse baseline
        safe_baseline_frames = min(5, len(dye))
        I0  = float(np.nanmean(dye[:safe_baseline_frames]))
        bg0 = float(np.nanmean(bg[:safe_baseline_frames]))
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
    arr   = np.array(normalised)[:, pulse_frame:]
    t_al  = time_array[pulse_frame:] - time_array[pulse_frame]
    avg   = np.nanmean(arr, axis=0)

    valid_ids = [str(circles[i]['id']) for i in guv_results['valid_guv_indices']]

    return {
        't_aligned':          t_al,
        'average_curve':      avg,
        'all_curves_aligned': arr,
        'valid_guv_ids':      valid_ids,
        'median_jump_frame':  pulse_frame,
    }


# -------------------------------------------------------------------
# --- 5. FITTING ---
# -------------------------------------------------------------------

def fit_individual_curves(aligned: dict, guv_results: dict, t: np.ndarray, logger: logging.Logger) -> pd.DataFrame:
    arr      = aligned['all_curves_aligned']
    ids      = aligned['valid_guv_ids']
    df_track = guv_results['tracking_dataframe']

    fit_records = []
    fit_plot_data = []   # accumulated for the grid figure

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
            p0 = (np.clip(y_start, 0.01, 1.99),
                  np.clip(y_end,   -1.99, 1.99),
                  np.clip(tau_e,   0.02, 4999.0),
                  0.0)
            bnds  = ((0, -2, 0.01, -0.1), (2, 2, 5000, 0.1))
            names = ['I0', 'Iinf', 'tau', 'D']
        elif cfg.MODEL_TO_USE == 'INFLUX-1EXP':
            fn = utils.influx_1exp
            p0 = (np.clip(y_start, -1.99, 1.99),
                  np.clip(y_end,    0.01, 1.99),
                  np.clip(tau_e,   0.02, 4999.0),
                  0.0)
            bnds  = ((-2, 0, 0.01, -0.1), (2, 2, 5000, 0.1))
            names = ['I0', 'Iinf', 'tau', 'D']
        elif cfg.MODEL_TO_USE == 'EFFLUX-2EXP':
            fn = utils.efflux_2exp
            p0    = (y_end, A_est*0.5, tau_e*0.5, A_est*0.5, tau_e*2.0, 0.0)
            bnds  = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
            names = ['Iinf', 'a1', 'tau1', 'a2', 'tau2', 'D']
        elif cfg.MODEL_TO_USE == 'INFLUX-2EXP':
            fn = utils.influx_2exp
            p0    = (y_start, A_est*0.5, tau_e*0.5, A_est*0.5, tau_e*2.0, 0.0)
            bnds  = ((-2, 0, 0.01, 0, 0.01, -0.1), (2, 2, 5000, 2, 5000, 0.1))
            names = ['I0', 'a1', 'tau1', 'a2', 'tau2', 'D']
        else:
            logger.error("Unknown model selected.")
            continue

        try:
            params, _ = sc.curve_fit(fn, t_c, y_c, p0=p0, bounds=bnds, maxfev=10000)
            res = {'guv_id': guv_id, 'radius_um': r_um}
            res.update(dict(zip(names, params)))
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

        except RuntimeError:
            logger.warning(f"Fitting failed for GUV {guv_id}")

    # ── Grid figure (replaces individual PNGs) ────────────────────────────
    if fit_plot_data:
        grid_path = utils.plot_dye_fits_grid(
            fit_plot_data, cfg.FOLDER_DYE,
            cfg.EXPERIMENT_BASE_NAME, cfg.MODEL_TO_USE,
            ruptured_guv_ids=guv_results.get('ruptured_guv_ids'),
        )
        logger.info(f"Saved dye fits grid → {grid_path}")

    df_fits = pd.DataFrame(fit_records)
    return df_fits

def generate_parameter_boxplots(df_fits: pd.DataFrame, logger: logging.Logger):
    if df_fits.empty: return
    
    bins = [0, 5.0, 8.0, 11.0, 100.0]
    labels = ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']
    df_fits['size_group'] = pd.cut(df_fits['radius_um'], bins=bins, labels=labels)
    
    df_fits.to_csv(os.path.join(cfg.FOLDER_DYE, f"{cfg.EXPERIMENT_BASE_NAME}_fit_parameters.csv"), index=False)
    
    params_to_plot = [col for col in df_fits.columns if col not in ['guv_id', 'radius_um', 'size_group']]
    
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    if len(params_to_plot) == 1: axes = [axes]
    
    for ax, param in zip(axes, params_to_plot):
        sns.boxplot(data=df_fits, x='size_group', y=param, ax=ax)
        sns.stripplot(data=df_fits, x='size_group', y=param, ax=ax, color='black', alpha=0.5)
        ax.set_title(param)
        ax.set_xlabel(r'Radius ($\mu$m)')
        
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.FOLDER_DYE, f"{cfg.EXPERIMENT_BASE_NAME}_parameter_boxplots.png"), dpi=300)
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
            os.path.join(cfg.FOLDER_DYE,
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

        # Step 4b – dye summary (all traces + mean)
        summary_path = utils.plot_dye_summary(
            aligned, cfg.FOLDER_DYE, cfg.EXPERIMENT_BASE_NAME,
            ruptured_guv_ids=guv_results.get('ruptured_guv_ids'),
        )
        logger.info(f"Saved dye summary plot → {summary_path}")

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
            valid_ids = aligned['valid_guv_ids']  # same order as GUV results

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
                ruptured_guv_ids=guv_results.get('ruptured_guv_ids'),
            )
            logger.info(f"Saved actin analysis plot → {plot_path}")

            # Combined dye + actin overlay (only when both pipelines ran)
            if aligned is not None:
                overlay_path = utils.plot_dye_actin_overlay(
                    aligned, aligned_actin, valid_ids,
                    cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                    smooth_sigma=getattr(cfg, 'ACTIN_PLOT_SMOOTH_SIGMA', 1.5),
                    ruptured_guv_ids=guv_results.get('ruptured_guv_ids'),
                )
                logger.info(f"Saved dye/actin overlay → {overlay_path}")

            # Radial profile snapshot at the last pre-pulse frame
            pre_frame = max(0, PULSE_FRAME - 1)
            snapshots = []
            
            for i, gid in enumerate(valid_ids):
                raw_idx = guv_results['valid_guv_indices'][i]
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
                for i, gid in enumerate(valid_ids):
                    raw_idx = guv_results['valid_guv_indices'][i]
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