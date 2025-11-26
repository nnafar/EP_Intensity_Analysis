# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS ---
======================================================
Main execution script.

Pipeline Overview:
1. Load Data: Images, CSV coordinates, Timestamps.
2. Process GUVs (Parallel): 
   - Detect membrane boundaries.
   - Extract intensity traces.
   - Detect "Jump" (Dye entry) time.
3. Normalize:
   - Calculate Uptake = (Dye - Dye_0) / (Background - Dye_0).
   - Align all curves so t=0 is the jump time.
4. Fit: Apply kinetic model to the average curve.
5. Export: Save plots, CSVs, and images.
"""

# --- Standard Library Imports ---
import glob
import os
import logging
import multiprocessing
from functools import partial
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as sc
import pandas as pd 

# --- Our Custom Imports ---
import config as cfg
import guv_analysis_utils as utils


# -------------------------------------------------------------------
# --- 1. DATA LOADING FUNCTIONS ---
# -------------------------------------------------------------------

def load_input_data(paths: dict, logger: logging.Logger) -> dict:
    """
    Loads all required input files (images, CSV, timestamps).
    """
    
    # --- 1a. Load File Names ---
    dye_files = natsorted(glob.glob(paths['dye_tifs']))
    roi_guide_files = natsorted(glob.glob(paths['roi_tifs']))
    
    if not dye_files:
        logger.error(f"No dye files found matching: {paths['dye_tifs']}")
        return None
    if not roi_guide_files:
        logger.error(f"No ROI files found matching: {paths['roi_tifs']}")
        return None
        
    roi_guide_file = roi_guide_files[0] # Only need the first frame for ROI detection
    
    logger.info(f"Loading Dye (measurement) files from: {paths['dye_tifs']}")
    logger.info(f"Loading ROI (guide) files from: {paths['roi_tifs']}")
    logger.info(f"Loading CSV from: {paths['csv']}")
    logger.info(f"Found {len(dye_files)} dye image files.")

    # --- 1b. Extract Timestamps ---
    logger.info("Trying ImageJ metadata extraction...")
    time_array, frame_interval = utils.extract_timestamps_from_metadata(dye_files)
    
    if time_array is None or frame_interval is None:
        logger.warning("Falling back to manual timestamps...")
        fallback_fps = getattr(cfg, "FALLBACK_FPS", 1.0)
        time_array, frame_interval = utils.create_manual_timestamps(len(dye_files), fallback_fps)
        
    if time_array is None:
        logger.error("Could not extract or generate timestamps.")
        return None
        
    logger.info(f"Found frame interval: {frame_interval} seconds. {len(time_array)} timestamps.")
    
    # --- 1c. Load Image Stacks and ROI Data ---
    logger.info("Loading ROI (C1) guide image...")
    roi_guide_frame = cv2.imread(roi_guide_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    
    if roi_guide_frame is None:
        logger.error(f"Could not load ROI guide frame: {roi_guide_file}")
        return None
    
    logger.info(f"Loading GUV coordinates from: {paths['csv']}")
    try:
        df_vesicles = pd.read_csv(paths['csv'], comment='#', header=None)
        # Expecting columns: [id, xc, yc, size, score]
        df_vesicles = df_vesicles.iloc[:, :5]
        df_vesicles.columns = ['id', 'xc', 'yc', 'size', 'score']
        
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return None
        
    logger.info(f"Found {len(df_vesicles)} GUV(s) in CSV file.")

    return {
        "dye_files": dye_files,
        "roi_frame": roi_guide_frame,
        "guv_data": df_vesicles,
        "time_array": time_array,
        "frame_interval": frame_interval
    }

# -------------------------------------------------------------------
# --- 2. GUV PROCESSING FUNCTIONS ---
# -------------------------------------------------------------------

def process_all_guvs(guv_data: pd.DataFrame, dye_files: list, 
                     roi_frame: np.ndarray, logger: logging.Logger) -> dict:
    """
    Iterates over all GUVs and runs 'process_single_guv' for each
    using Multiprocessing for speed.
    """
    
    guv_coords = guv_data[['xc', 'yc']].values.astype(int)
    guv_sizes = guv_data['size'].values.astype(int) 
    
    # Partial allows us to freeze the 'constant' arguments
    process_func = partial(
        utils.process_single_guv,
        roi_frame=roi_frame,
        dye_files=dye_files
    )

    # Prepare arguments for each worker
    tasks = []
    for i, (xc, yc) in enumerate(guv_coords):
        tasks.append((
            guv_data.loc[i, 'id'],
            (xc, yc),
            guv_sizes[i],
            (i == 0) # Flag to force mask visualization for the first GUV
        ))

    logger.info(f"Starting parallel processing of {len(tasks)} GUVs using {cfg.N_WORKERS} workers...")

    # 'spawn' context is safer for Windows/Mac compatibility with OpenCV
    mp_context = multiprocessing.get_context('spawn')
    with mp_context.Pool(processes=cfg.N_WORKERS) as pool:
        results = pool.starmap(process_func, tasks)
        
    logger.info("...Parallel processing complete.")

    # Unpack Results
    all_intensity_curves = []
    all_background_curves = [] 
    all_jump_frames = []
    detection_quality_log = []
    valid_guv_indices = []

    for i, (result, quality_entry) in enumerate(results):
        guv_id = quality_entry['guv_id']
        detection_quality_log.append(quality_entry)
        
        if quality_entry['failed'] or quality_entry['comments'] != 'OK':
            logger.warning(f"  - GUV {guv_id}: Detection failed or has warnings: {quality_entry['comments']}")
        else:
            logger.info(f"  - GUV {guv_id}: Processed successfully.")

        if result is not None:
            all_intensity_curves.append(result['intensity_trace'])
            all_background_curves.append(result['background_trace'])
            all_jump_frames.append(result['jump_frame'])
            valid_guv_indices.append(i)
        else:
            logger.warning(f"  - GUV {guv_id} returned no data (likely empty mask).")

    logger.info(f"--- Analysis Complete: {len(all_intensity_curves)} / {len(guv_coords)} GUVs processed ---")
    
    # Save a log file of which GUVs were detected/failed
    df_quality = pd.DataFrame(detection_quality_log)
    quality_log_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_detection_quality.csv")
    df_quality.to_csv(quality_log_path, index=False)
    
    return {
        "all_intensity_curves": all_intensity_curves,
        "all_background_curves": all_background_curves,
        "all_jump_frames": all_jump_frames,
        "valid_guv_indices": valid_guv_indices
    }

# -------------------------------------------------------------------
# --- 3. NORMALIZATION & FITTING FUNCTIONS ---
# -------------------------------------------------------------------

def normalize_and_align_curves(guv_results: dict, time_array: np.ndarray, 
                               all_guv_ids: np.ndarray, logger: logging.Logger) -> dict:
    """
    Normalizes raw intensity using the background and baseline, 
    then aligns all curves in time so the 'Jump' happens at t=0.
    """
    
    logger.info("Normalizing intensity curves using $I_{uptake} = (I_{dye,t} - I_{dye,0}) / (I_{background,t} - I_{dye,0})$...")
    
    normalized_curves = []
    
    for idx, (dye_trace, bg_trace, jump_f) in enumerate(zip(
        guv_results['all_intensity_curves'], 
        guv_results['all_background_curves'], 
        guv_results['all_jump_frames']
    )):
        # Determine baseline window (must be before the jump)
        baseline_frames_end = max(cfg.MIN_BASELINE_FRAMES, jump_f)
        if baseline_frames_end >= len(dye_trace):
             logger.warning(f"  - Skipping GUV {idx+1} due to insufficient baseline frames.")
             continue
             
        # Calculate initial dye intensity (I_dye_0)
        I_dye_0 = np.mean(dye_trace[:baseline_frames_end])
        
        # Calculate Normalization
        denominator = bg_trace - I_dye_0
        numerator = dye_trace - I_dye_0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_trace = np.divide(numerator, denominator)
            normalized_trace = np.nan_to_num(normalized_trace, nan=0.0, posinf=0.0, neginf=0.0)
            
        normalized_curves.append(normalized_trace)

    if not normalized_curves:
        logger.error("Normalization failed for all curves. Cannot continue.")
        return None

    all_intensity_curves = normalized_curves
    
    # --- ALIGNMENT ---
    curve_array_original = np.array(all_intensity_curves)
    valid_jump_frames = np.array(guv_results['all_jump_frames'])
    
    # We align to the MEDIAN jump frame to keep the timeline consistent for all
    valid_jumps_nonzero = valid_jump_frames[valid_jump_frames > 0]
    if len(valid_jumps_nonzero) > 0:
        median_jump_frame = int(np.median(valid_jumps_nonzero))
    else:
        median_jump_frame = 0
        
    logger.info(f"Aligning all data using the median jump frame index: {median_jump_frame}")
    
    total_frames_original = len(time_array)
    if median_jump_frame >= total_frames_original:
        median_jump_frame = 0
    
    # Reset Time: The median jump frame becomes t=0
    t_aligned = time_array[median_jump_frame:]
    t_aligned = t_aligned - t_aligned[0]
    
    # Slice Curves: Remove pre-jump data
    average_curve = np.mean(curve_array_original[:, median_jump_frame:], axis=0)
    curve_array = curve_array_original[:, median_jump_frame:]
    
    valid_guv_ids = all_guv_ids[guv_results['valid_guv_indices']]

    return {
        "t_aligned": t_aligned,
        "average_curve": average_curve,
        "all_curves_aligned": curve_array,
        "valid_guv_ids": valid_guv_ids,
        "median_jump_frame": median_jump_frame
    }


def fit_average_curve(t_aligned: np.ndarray, average_curve: np.ndarray, 
                      logger: logging.Logger) -> dict:
    """
    Fits the averaged normalized curve to the kinetic model specified in config.
    """

    logger.info(f"Fitting average curve to model: {cfg.MODEL_TO_USE}...")
    
    # Slice data for fitting (e.g., ignore last 10%)
    total_frames = len(t_aligned)
    fit_slice_index = int(np.round(total_frames * cfg.FIT_DATA_PERCENTAGE))
    fit_slice_index = min(fit_slice_index, total_frames) 
    
    t_fit = t_aligned[:fit_slice_index]
    y_fit = average_curve[:fit_slice_index]
        
    logger.info(f"Fitting first {cfg.FIT_DATA_PERCENTAGE*100:.0f}% of aligned data ({fit_slice_index} frames).")

    # --- DYNAMIC INITIAL GUESS ESTIMATION ---
    # Helps the optimizer converge by guessing reasonable starting values
    y_min = np.min(y_fit)
    y_max = np.max(y_fit)
    A_est = y_max - y_min
    offset_est = y_min
    
    # Guess Tau (time to reach ~63% of rise)
    try:
        threshold = y_min + 0.63 * A_est
        idx_tau = np.argmax(y_fit > threshold)
        if idx_tau == 0 and y_fit[0] < threshold:
             tau_est = t_fit[-1] / 2.0
        else:
             tau_est = t_fit[idx_tau]
    except Exception:
        tau_est = t_fit[-1] / 5.0
    if tau_est <= 0: tau_est = 10.0

    # --- CONFIGURE MODEL & BOUNDS ---
    if cfg.MODEL_TO_USE == '5-PARAM':
        fit_function = utils.dyn_model_5param
        # Guess: (Af, A1, tau1, A2, tau2)
        p0_guess = (y_max, A_est * 0.5, tau_est * 0.5, A_est * 0.5, tau_est * 2.0)
        bounds = ((-1.0, -10.0, 0.01, -10.0, 0.01), (10.0, 10.0, 5000.0, 10.0, 5000.0))
        
    elif cfg.MODEL_TO_USE == '4-PARAM':
        fit_function = utils.dyn_model_4param
        # Guess: (Offset, A, tau, drift)
        p0_guess = (offset_est, A_est, tau_est, 0.0)
        bounds = ((-2.0, -0.2, 0.01, -0.1), (2.0, 10.0, 5000.0, 0.1))
    else:
        raise ValueError("Invalid MODEL_TO_USE specified in config.py.")

    logger.info(f"Initial Guesses: {p0_guess}")

    try:
        params, covariance = sc.curve_fit(fit_function, t_fit, y_fit, p0=p0_guess, bounds=bounds, maxfev=10000)
        fit_failed = False
    except RuntimeError as e:
        logger.error(f"Curve fitting failed: {e}. Using initial guess parameters for output.")
        params = np.array(p0_guess)
        fit_failed = True
        
    # Generate the fitted curve for plotting
    if cfg.MODEL_TO_USE == '5-PARAM':
        param_names = ['Af', 'A1', 'tau1', 'A2', 'tau2']
        fit_curve = utils.dyn_model_5param(t_aligned, *params)
        logger.info(f"Fit Complete. Af={params[0]:.2f}, A1={params[1]:.2f}, tau1={params[2]:.2f} s, A2={params[3]:.2f}, tau2={params[4]:.2f} s")
    elif cfg.MODEL_TO_USE == '4-PARAM':
        param_names = ['I_offset', 'A', 'tau', 'D']
        fit_curve = utils.dyn_model_4param(t_aligned, *params)
        logger.info(f"Fit Complete. I_offset={params[0]:.2f}, A={params[1]:.2f}, tau={params[2]:.2f} s, D={params[3]:.4f}")

    fit_params_dict = dict(zip(param_names, params))

    return {
        "params": fit_params_dict,
        "fit_curve": fit_curve,
        "fit_failed": fit_failed,
        "fit_slice_index": fit_slice_index
    }

# -------------------------------------------------------------------
# --- 4. EXPORT & PLOTTING FUNCTIONS ---
# -------------------------------------------------------------------

def export_results(aligned_data: dict, fit_results: dict, dye_files: list, 
                   logger: logging.Logger):
    """
    Exports all final plots, images, and CSV files to the output folder.
    """
    
    t = aligned_data['t_aligned']
    average_curve = aligned_data['average_curve']
    curve_array = aligned_data['all_curves_aligned']
    valid_guv_ids = aligned_data['valid_guv_ids']
    
    fit_curve = fit_results['fit_curve']
    fit_failed = fit_results['fit_failed']
    fit_params = fit_results['params']
    fit_slice_index = fit_results['fit_slice_index']
    
    # Get the file list corresponding to the aligned timeframe
    dye_files_aligned = dye_files[aligned_data['median_jump_frame']:]
    
    # --- EXPORT SNAPSHOT FRAMES ---
    logger.info("Exporting representative frames...")
    for i, time_point in enumerate(cfg.EXPORT_TIME_POINTS_S):
        frame_idx, actual_time = utils.find_closest_frame(t, time_point)
        
        if frame_idx >= len(dye_files_aligned): continue
            
        frame_path = dye_files_aligned[frame_idx]
        frame_data = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        
        if frame_data is None: continue
            
        label = f"{int(np.round(time_point))} S" 
        styled_frame = utils.style_image(frame_data, label, cfg.MICRONS_PER_PIXEL, cfg.SCALE_BAR_LENGTH_MICRONS)
        cv2.imwrite(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"frame_{i+1}_at_{int(np.round(time_point))}s.png"), styled_frame)
        
    logger.info("Frame export complete.")

    # --- PLOT KINETIC RESULTS ---
    logger.info("Generating summary plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 6))
    
    # Plot faint lines for individual traces
    for curve in curve_array:
        ax.plot(t[:len(curve)], curve, color='gray', alpha=0.2)
        
    # Plot Average and Fit
    ax.plot(t, average_curve, 'r.', markersize=3, label=f'Average (n={len(curve_array)})')
    ax.plot(t, fit_curve, 'k-', linewidth=2, label='Fitted Curve')
    
    # Mark where the fit data ended
    if fit_slice_index > 0 and fit_slice_index <= len(t):
        ax.plot(t[fit_slice_index-1], average_curve[fit_slice_index-1], 'b*', markersize=10, label='End of Fit Data')

    # Build informative title with parameters
    if cfg.MODEL_TO_USE == '5-PARAM':
        p = fit_params
        fig_title = f"Fit: $A_f = {p['Af']:.2f}$, $A_1 = {p['A1']:.2f}$, $\\tau_1 = {p['tau1']:.2f}$ s, $A_2 = {p['A2']:.2f}$, $\\tau_2 = {p['tau2']:.2f}$ s"
    else:
        p = fit_params
        fig_title = f"Fit: $I_{{offset}} = {p['I_offset']:.2f}$, $A = {p['A']:.2f}$, $\\tau = {p['tau']:.2f}$ s, $D = {p['D']:.4f}$"
    
    if fit_failed: fig_title = "FIT FAILED: Using Initial Guess\n" + fig_title
    ax.set_title(fig_title)
    ax.set_xlabel("Time (s) relative to Electroporation Pulse")
    ax.set_ylabel("Normalized Intensity ($I_{uptake}$)") 
    ax.legend(loc='lower right')
    plt.grid(True)
    
    fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_kinetic_fit.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved kinetic plot.")

    # --- EXPORT NORMALIZED CURVE DATA ---
    logger.info("Exporting normalized curve data to CSV...")
    try:
        headers = ['Time (s)'] + [f'GUV_{gid}_I_uptake' for gid in valid_guv_ids] + ['Average_I_uptake']
        t_col = t.reshape(-1, 1)
        avg_col = average_curve.reshape(-1, 1)
        data_to_save = np.hstack((t_col, curve_array.T, avg_col))
        df_curves = pd.DataFrame(data_to_save, columns=headers)
        df_curves.to_csv(os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_normalized_curves.csv"), index=False, float_format='%.6f')
    except Exception as e:
        logger.warning(f"  - Could not export normalized curve CSV: {e}")

# -------------------------------------------------------------------
# --- 5. VALIDATION ---
# -------------------------------------------------------------------

def validate_config(logger: logging.Logger) -> bool:
    """
    Validates parameters from config.py.
    """
    is_valid = True
    if not os.path.exists(cfg.DATA_FOLDER):
        logger.error(f"Config Error: DATA_FOLDER does not exist: {cfg.DATA_FOLDER}")
        is_valid = False
    if not (0.1 <= cfg.MEMBRANE_SEARCH_FACTOR <= 1.0):
        logger.error(f"Config Error: MEMBRANE_SEARCH_FACTOR invalid.")
        is_valid = False
    if cfg.MODEL_TO_USE not in ['4-PARAM', '5-PARAM']:
        logger.error(f"Config Error: MODEL_TO_USE must be '4-PARAM' or '5-PARAM'")
        is_valid = False
    
    # Validate Threshold
    if not (0.05 <= getattr(cfg, 'JUMP_THRESHOLD_PERCENT', 0.20) <= 0.90):
        logger.error("Config Error: JUMP_THRESHOLD_PERCENT should be between 0.05 and 0.90")
        is_valid = False

    if not is_valid: logger.critical("Configuration validation failed.")
    return is_valid

# -------------------------------------------------------------------
# --- 6. MAIN ORCHESTRATOR ---
# -------------------------------------------------------------------

def main():
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    logger = utils.setup_logging(cfg.OUTPUT_IMAGE_FOLDER, cfg.EXPERIMENT_BASE_NAME)
    logger.info("--- GUV Analysis Pipeline Started ---")
    
    if not validate_config(logger): return 
    
    try:
        paths = {
            'dye_tifs': os.path.join(cfg.DATA_FOLDER, cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX),
            'roi_tifs': os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX),
            'csv': os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.CSV_SUFFIX)
        }

        input_data = load_input_data(paths, logger)
        if input_data is None: return

        guv_results = process_all_guvs(
            input_data['guv_data'], 
            input_data['dye_files'], 
            input_data['roi_frame'],
            logger
        )
        
        if not guv_results['all_intensity_curves']: return

        aligned_data = normalize_and_align_curves(
            guv_results, 
            input_data['time_array'], 
            input_data['guv_data']['id'].values,
            logger
        )
        if aligned_data is None: return

        fit_results = fit_average_curve(aligned_data['t_aligned'], aligned_data['average_curve'], logger)
        export_results(aligned_data, fit_results, input_data['dye_files'], logger)
        
        logger.info("--- GUV Analysis Pipeline Finished Successfully ---")
        
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in main: {e}", exc_info=True)

if __name__ == "__main__":
    main()