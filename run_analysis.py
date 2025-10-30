# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS (v3 - Fractional I_uptake) ---
======================================================
This script analyzes all GUVs found in the CSV file, applying background
correction and scaling the kinetic traces to a 0-to-1 fractional uptake.

It will:
1.  Loop over all GUVs to get their individual uptake curves.
2.  Apply background correction using an annulus mask.
3.  Calculate an average uptake curve.
4.  Fit the average curve to the kinetic model.
5.  Export time-lapse frames.
6.  Plot all individual GUV curves (gray) and the
    average curve with its fit (red/black).
"""

# --- Standard Library Imports ---
import glob
import os
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as sc
import pandas as pd 

# --- Our Custom Imports ---
import config as cfg
import guv_analysis_utils as utils


def main():
    """Main analysis function."""
    
    # --- 1. Build File Paths ---
    path_to_dye_tifs = os.path.join(cfg.DATA_FOLDER, cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    path_to_roi_tifs = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    path_to_csv = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.CSV_SUFFIX)
    
    # --- 2. Load File Names ---
    # ... (unchanged)
    dye_files = natsorted(glob.glob(path_to_dye_tifs))
    roi_guide_files = natsorted(glob.glob(path_to_roi_tifs))
    
    if not dye_files:
        print(f"Error: No dye files found matching: {path_to_dye_tifs}")
        return
        
    if not roi_guide_files:
        print(f"Error: No ROI files found matching: {path_to_roi_tifs}")
        return
        
    roi_guide_file = roi_guide_files[0] # Only need the first frame for ROI
    
    print(f"Loading Dye (measurement) files from: {path_to_dye_tifs}")
    print(f"Loading ROI (guide) files from: {path_to_roi_tifs}")
    print(f"Loading CSV from: {path_to_csv}")
    print(f"Found {len(dye_files)} dye image files.")

    # --- 3. Extract Timestamps ---
    # ... (unchanged)
    print("Trying ImageJ metadata extraction...")
    time_array_original, frame_interval = utils.extract_timestamps_from_metadata(dye_files)
    
    if time_array_original is None or frame_interval is None:
        print("Falling back to manual timestamps...")
        fallback_fps = getattr(cfg, "FALLBACK_FPS", 1.0)
        time_array_original, frame_interval = utils.create_manual_timestamps(len(dye_files), fallback_fps)
        
    total_frames_original = len(dye_files)

    if time_array_original is not None and frame_interval is not None:
        print(f"      Found frame interval: {frame_interval} seconds")
        print(f"Successfully extracted {len(time_array_original)} timestamps")
    else:
        print("Error: Could not extract or generate timestamps.")
        return
    
    
    # --- 4. Load Image Stacks and ROI Data ---
    print("Loading DYE (C3) image stack (this may take a moment)...")
    im_stack = utils.tifflist_to_numpy(dye_files)
    print("Dye stack loaded.")
    
    print("Loading ROI (C1) guide image...")
    roi_guide_frame = cv2.imread(roi_guide_file, cv2.IMREAD_ANYDEPTH)
    
    print(f"Loading GUV coordinates from: {path_to_csv}")
    try:
        df_vesicles = pd.read_csv(path_to_csv, comment='#', header=None)
        
        # --- FIX 2: Flexible CSV Column Handling ---
        num_cols = df_vesicles.shape[1]
        if num_cols >= 5:
             # Use the first 5 columns explicitly
            df_vesicles = df_vesicles.iloc[:, :5] 
            df_vesicles.columns = ['id', 'xc', 'yc', 'size', 'score']
        else:
            raise ValueError(f"CSV file must contain at least 5 columns (id, xc, yc, yc, size, score), found {num_cols}.")
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    guv_coords = df_vesicles[['xc', 'yc']].values.astype(int)
    guv_sizes = df_vesicles['size'].values.astype(int)
    
    print(f"Found {len(guv_coords)} GUV(s) in CSV file.")

    # --- 5. Process Each GUV ---
    all_corrected_curves = [] # Store I_dye - I_bg, baseline subtracted (but not scaled 0-1)
    all_jump_frames = [] # Store jump frames for alignment
    
    for i, (xc, yc) in enumerate(guv_coords):
        guv_id = df_vesicles.loc[i, 'id']
        guv_size = guv_sizes[i]
        
        print(f"\n--- Processing GUV {guv_id} at ({xc}, {yc}) ---")
        
        # 5a. Define Masks
        radius = int(np.sqrt(guv_size / np.pi))
        guv_mask = utils.create_circular_mask(roi_guide_frame, (xc, yc), radius=radius)
        bg_mask = utils.create_background_mask(roi_guide_frame, (xc, yc), radius=radius)
        
        # 5b. Get Intensity Traces
        guv_trace = utils.get_intensity_trace(im_stack, guv_mask)
        bg_trace = utils.get_intensity_trace(im_stack, bg_mask)
        
        # 5c. Background Correction
        corrected_trace = guv_trace - bg_trace
        
        # 5d. Detect Jump (Use corrected trace for robust detection)
        try:
            jump_frame = utils.detect_intensity_jump(
                corrected_trace, 
                sensitivity=cfg.JUMP_SENSITIVITY,
                baseline_frames=cfg.JUMP_DETECTION_BASELINE_FRAMES
            )
            if jump_frame == -1:
                jump_frame = 0 
                print(f"No clear jump detected for GUV {guv_id}, assuming start at frame 0.")
            else:
                print(f"Fluorescence jump detected at frame {jump_frame}")
                
        except Exception as e:
            print(f"Warning: Jump detection failed for GUV {guv_id}: {e}")
            jump_frame = 0 
            
        # 5e. Baseline Subtraction (I_t - I_0)
        baseline_frames = max(0, jump_frame - 5)
        # Use mean of corrected trace before the jump as baseline
        baseline_I_corrected = np.mean(corrected_trace[baseline_frames:jump_frame]) if jump_frame > 0 else corrected_trace[0]
        
        # Final trace is background-corrected and baseline-subtracted, but NOT scaled (0-1) yet
        final_trace = corrected_trace - baseline_I_corrected

        # 5f. Store Trace and Jump Index
        all_corrected_curves.append(final_trace)
        all_jump_frames.append(jump_frame)

    print(f"\n--- Analysis Complete: {len(all_corrected_curves)} / {len(guv_coords)} GUVs processed ---")
    print("All traces are background-corrected and baseline-subtracted.")

    # --- 6. Calculate Average Curve (and perform alignment) ---
    if not all_corrected_curves:
        print("Error: No valid intensity curves were processed. Cannot continue to fitting.")
        return
        
    curve_array_unscaled_original = np.array(all_corrected_curves)
    
    # Calculate the median jump frame index to use as the common alignment point
    valid_jump_frames = np.array(all_jump_frames)
    median_jump_frame = int(np.median(valid_jump_frames))
    print(f"Aligning all data using the median jump frame index: {median_jump_frame}")
    
    # Slice the time array from the median jump frame and re-zero it
    t_aligned = time_array_original[median_jump_frame:]
    t = t_aligned - t_aligned[0]
    
    # Slice the curve data 
    curve_array_unscaled = curve_array_unscaled_original[:, median_jump_frame:]
    average_curve_unscaled = np.mean(curve_array_unscaled, axis=0)

    # --- 7. Final Scaling to 0-to-1 Fractional Uptake (Fix for Inversion) ---
    
    # I_start is the maximum value in the mean curve (at t=0, or slightly after)
    I_start = np.max(average_curve_unscaled)
    # I_end is the final value (steady state)
    I_end = average_curve_unscaled[-1]
    
    # The total dynamic range is the difference between start and end
    I_total_amplitude = I_start - I_end
    
    if I_total_amplitude <= 0:
        print("Error: Total amplitude is zero or negative (no decay/rise detected). Cannot scale to 0-1.")
        return
        
    # INVERSION: Calculate the rise relative to the total amplitude.
    # The desired I_uptake rises from 0 to 1.
    # Formula: I_uptake = (I_start - I(t)) / I_total_amplitude
    
    # 1. Calculate the inverted, baseline-shifted average curve (should start at 0)
    inverted_average_curve_unscaled = I_start - average_curve_unscaled
    
    # 2. Apply scaling to the average curve and all individual curves
    average_curve = inverted_average_curve_unscaled / I_total_amplitude
    curve_array = (I_start - curve_array_unscaled) / I_total_amplitude
    
    print(f"Final normalization applied: data scaled by 1/{I_total_amplitude:.2f} after inversion.")
    print(f"I_start: {I_start:.2f}, I_end: {I_end:.2f}, Total Amplitude: {I_total_amplitude:.2f}")

    # I_max_amplitude (now I_total_amplitude) is used only for printout below.
    I_max_amplitude = I_total_amplitude 
    
    # Slice image stack for export
    im_stack_aligned = im_stack[median_jump_frame:] 
    total_frames = len(t)
    
    # --- 8. Fit Average Curve to Selected Model ---
    print(f"\nFitting average curve to model: {cfg.MODEL_TO_USE}...")
    
    # Calculate the number of frames to use for fitting based on the configuration
    fit_slice_index = int(np.round(total_frames * cfg.FIT_DATA_PERCENTAGE))
    if fit_slice_index > total_frames:
        fit_slice_index = total_frames
        
    print(f"Fitting first {cfg.FIT_DATA_PERCENTAGE*100:.0f}% of aligned data ({fit_slice_index} frames).")

    if cfg.MODEL_TO_USE == '5-PARAM':
        p0_guess = cfg.FIT_INITIAL_GUESS
        fit_function = utils.dyn_model
    elif cfg.MODEL_TO_USE == '4-PARAM':
        p0_guess = cfg.FIT_INITIAL_GUESS_4PARAM
        fit_function = utils.dyn_model_4param
    else:
        raise ValueError("Invalid MODEL_TO_USE specified in config.py. Must be '4-PARAM' or '5-PARAM'.")

    # --- FIX 1: Robust Curve Fitting with Error Handling ---
    try:
        params, covariance = sc.curve_fit(
            fit_function, 
            t[:fit_slice_index], 
            average_curve[:fit_slice_index], 
            p0=p0_guess,
            maxfev=5000 
        )
        print("Curve fitting completed successfully.")
        fit_successful = True
    
    except RuntimeError as e:
        print(f"WARNING: Curve fit failed to converge. {e}")
        print("Using initial guess parameters for output and plotting.")
        params = p0_guess # Fallback to initial guess
        fit_successful = False
    
    # --- 9. Extract Parameters and Generate Outputs ---
    if cfg.MODEL_TO_USE == '5-PARAM':
        Af, A1, tau1, A2, tau2 = params
        print(f"Fit Complete. Af={Af:.2f}, A1={A1:.2f}, tau1={tau1:.2f} s, A2={A2:.2f}, tau2={tau2:.2f} s")
        fig_title = f"Fit Parameters: $A_f = {Af:.2f}$, $A_1 = {A1:.2f}$, $\\tau_1 = {tau1:.2f}$ s, $A_2 = {A2:.2f}$, $\\tau_2 = {tau2:.2f}$ s"
        fit_curve = utils.dyn_model(t, Af, A1, tau1, A2, tau2)
        
    elif cfg.MODEL_TO_USE == '4-PARAM':
        I_offset, A, tau, D = params
        print(f"Fit Complete. I_offset={I_offset:.2f}, A={A:.2f}, tau={tau:.2f} s, D={D:.4f}")
        fig_title = f"Fit Parameters: $I_{{offset}} = {I_offset:.2f}$, $A = {A:.2f}$, $\\tau = {tau:.2f}$ s, $D = {D:.4f}$"
        fit_curve = utils.dyn_model_4param(t, I_offset, A, tau, D)

    # --- 10. Export Representative Frames ---
    # ... (unchanged)
    print("\nExporting representative frames...")
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    
    for i, time_point in enumerate(cfg.EXPORT_TIME_POINTS_S):
        frame_idx, actual_time = utils.find_closest_frame(t, time_point)
        frame_data = im_stack_aligned[frame_idx]
        
        # Image styling uses the raw (un-normalized) image data
        label = f"{int(np.round(time_point))} S" 
        
        styled_frame = utils.style_image(
            frame_data, 
            label, 
            cfg.MICRONS_PER_PIXEL, 
            cfg.SCALE_BAR_LENGTH_MICRONS
            )
        out_name = f"frame_{i+1}_at_{int(np.round(time_point))}s.png"
        out_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, out_name)
        cv2.imwrite(out_path, styled_frame)
        print(f"  - Saved {out_name} (actual time: {actual_time:.3f}s relative to pulse)")
        
    print("Frame export complete.")

    # --- 11. PLOT KINETIC RESULTS (Summary Plot) ---
    print("\nGenerating summary plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 6))
    
    # Plot all individual GUV traces
    for curve in curve_array:
        ax.plot(t[:len(curve)], curve, color='gray', alpha=0.2)
        
    # Plot the average curve
    ax.plot(t, average_curve, 'r.', markersize=3, label=f'Average (n={len(guv_coords)})')

    # Plot the fitted curve
    fit_label = 'Fitted Curve' if fit_successful else 'Initial Guess (Fit Failed)'
    ax.plot(t[:fit_slice_index], fit_curve[:fit_slice_index], 'k-', linewidth=2, label=fit_label)

    # Add fit parameters to the title
    ax.set_title(fig_title)
    
    # Final plot styling
    ax.set_xlabel("Time (s) relative to Electroporation Pulse")
    ax.set_ylabel("Fractional Intensity Uptake ($I_{\\text{uptake}}$)")
    ax.legend(loc='lower right')
    plt.ylim(bottom=-0.1, top=1.5) # Set limits for 0-1 fractional plot
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()