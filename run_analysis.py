# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS (v2 - Multi-GUV) ---
======================================================
This script analyzes all GUVs found in the CSV file.

It will:
1.  Loop over all GUVs to get their individual uptake curves.
2.  Calculate an average uptake curve.
3.  Fit the average curve to the kinetic model.
4.  Export time-lapse frames.
5.  Plot all individual GUV curves (gray) and the
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
    print("Trying ImageJ metadata extraction...")
    # Function call is now expecting two values (time_array, frame_interval)
    time_array, frame_interval = utils.extract_timestamps_from_metadata(dye_files)
    
    if time_array is None or frame_interval is None:
        print("Falling back to manual timestamps...")
        # Assuming FALLBACK_FPS is set in config.py or use a default of 1.0
        fallback_fps = getattr(cfg, "FALLBACK_FPS", 1.0)
        time_array, frame_interval = utils.create_manual_timestamps(len(dye_files), fallback_fps)
        
    
    total_frames = len(dye_files)

    if time_array is not None and frame_interval is not None:
        print(f"      Found frame interval: {frame_interval} seconds")
        print(f"Successfully extracted {len(time_array)} timestamps using ImageJ metadata")
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
        # Assuming columns: id_vesicle, xc (pix), yc (pix), size (pix), matching score
        df_vesicles.columns = ['id', 'xc', 'yc', 'size', 'score']
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    guv_coords = df_vesicles[['xc', 'yc']].values.astype(int)
    guv_sizes = df_vesicles['size'].values.astype(int)
    
    print(f"Found {len(guv_coords)} GUV(s) in CSV file.")

    # --- 5. Process Each GUV ---
    all_intensity_curves = []
    
    for i, (xc, yc) in enumerate(guv_coords):
        guv_id = df_vesicles.loc[i, 'id']
        guv_size = guv_sizes[i]
        
        print(f"\n--- Processing GUV {guv_id} at ({xc}, {yc}) ---")
        
        # 5a. Define Mask
        radius = int(np.sqrt(guv_size / np.pi))
        mask = utils.create_circular_mask(roi_guide_frame, (xc, yc), radius=radius)
        
        # 5b. Get Intensity Trace
        intensity_trace = utils.get_intensity_trace(im_stack, mask)
        
        # 5c. Detect Jump (optional, but useful for aligning the curves if needed)
        try:
            jump_frame = utils.detect_intensity_jump(intensity_trace, sensitivity=cfg.JUMP_SENSITIVITY)
            if jump_frame != -1:
                print(f"Fluorescence jump detected at frame {jump_frame}")
        except Exception as e:
            print(f"Warning: Jump detection failed for GUV {guv_id}: {e}")
            jump_frame = 0 # Assume no jump or start at 0
            
        # 5d. Store Trace
        all_intensity_curves.append(intensity_trace)

    print(f"\n--- Analysis Complete: {len(all_intensity_curves)} / {len(guv_coords)} GUVs processed ---")

    # --- 6. Calculate Average Curve ---
    if not all_intensity_curves:
        print("Error: No valid intensity curves were processed. Cannot continue to fitting.")
        return
        
    curve_array = np.array(all_intensity_curves)
    average_curve = np.mean(curve_array, axis=0)

    # --- 7. Fit Average Curve to Model (Double Exponential) ---
    print("Fitting average curve to model...")
    t = time_array[:total_frames]
    
    # Calculate the number of frames to use for fitting based on the configuration
    total_frames_after_pulse = int(np.round(total_frames * cfg.FIT_DATA_PERCENTAGE))
    time_stamp = total_frames_after_pulse 
    
    if time_stamp > total_frames_after_pulse: time_stamp = total_frames_after_pulse
    print(f"Fitting first {cfg.FIT_DATA_PERCENTAGE*100:.0f}% of data ({time_stamp} frames).")
    
    # Perform the curve fit
    params, covariance = sc.curve_fit(
        utils.dyn_model, 
        t[:time_stamp], 
        average_curve[:time_stamp], 
        p0=cfg.FIT_INITIAL_GUESS,
        maxfev=5000 # Increased maxfev for double exponential stability
    )
    
    # Extract the 5 fitted parameters: Af, A1, tau1, A2, tau2
    Af = params[0] # Final normalized intensity (Af)
    A1 = params[1] # Amplitude of the fast component (A1)
    tau1 = params[2] # Fast time constant (tau1)
    A2 = params[3] # Amplitude of the slow component (A2)
    tau2 = params[4] # Slow time constant (tau2)

    print(f"Fit Complete. Af={Af:.2f}, A1={A1:.2f}, tau1={tau1:.2f} s, A2={A2:.2f}, tau2={tau2:.2f} s")

    # --- 8. Export Representative Frames ---
    print("\nExporting representative frames...")
    
    # Ensure the output directory exists
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    
    for i, time_point in enumerate(cfg.EXPORT_TIME_POINTS_S):
        frame_idx, actual_time = utils.find_closest_frame(t, time_point)
        frame_data = im_stack[frame_idx] # Get frame from the C3 (Dye) stack
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
        print(f"  - Saved {out_name} (actual time: {actual_time:.3f}s)")
        
    print("Frame export complete.")

    # --- 9. PLOT KINETIC RESULTS (Summary Plot) ---
    print("\nGenerating summary plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 6))
    
    # Plot all individual GUV traces
    for curve in curve_array:
        ax.plot(t, curve, color='gray', alpha=0.2)
        
    # Plot the average curve
    ax.plot(t, average_curve, 'r.', markersize=3, label=f'Average (n={len(all_intensity_curves)})')

    # Plot the fitted curve
    fit_curve = utils.dyn_model(t, Af, A1, tau1, A2, tau2)
    ax.plot(t[:time_stamp], fit_curve[:time_stamp], 'k-', linewidth=2, label='Fitted Curve')

    # Add fit parameters to the title - UPDATED FOR 5-PARAMETER MODEL
    fig_title = f"Fit Parameters: $A_f = {Af:.2f}$, $A_1 = {A1:.2f}$, $\\tau_1 = {tau1:.2f}$ s, $A_2 = {A2:.2f}$, $\\tau_2 = {tau2:.2f}$ s"
    ax.set_title(fig_title)
    
    # Final plot styling
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Intensity ($I_{uptake}$)")
    ax.legend(loc='lower right')
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()