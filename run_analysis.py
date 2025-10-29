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

# --- Our Custom Imports ---
import config as cfg
import guv_analysis_utils as utils


def main():
    """Main analysis function."""
    
    # --- 1. Build File Paths ---
    path_to_dye_tifs = os.path.join(cfg.DATA_FOLDER, cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    path_to_roi_tifs = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    path_to_csv = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.CSV_SUFFIX)
    
    print(f"Loading Dye (measurement) files from: {path_to_dye_tifs}")
    print(f"Loading ROI (guide) files from: {path_to_roi_tifs}")
    print(f"Loading CSV from: {path_to_csv}")

    sorted_dye_files = natsorted(glob.glob(path_to_dye_tifs))
    if not sorted_dye_files:
        raise FileNotFoundError(f"No DYE .tif files found at: {path_to_dye_tifs}")
    print(f"Found {len(sorted_dye_files)} dye image files.")

    # --- 2. Extract Time Data ---
    time_data = utils.extract_timestamps_from_metadata(sorted_dye_files)
    if time_data is None:
        time_data = utils.create_manual_timestamps(len(sorted_dye_files), cfg.FALLBACK_FPS)
    print(f"Successfully extracted {len(time_data)} timestamps using {utils.metadata_method_used}")

    # --- 3. Load DYE Image Stack ---
    print("Loading DYE (C3) image stack (this may take a moment)...")
    im_stack0 = [cv2.imread(f, cv2.IMREAD_ANYDEPTH) for f in sorted_dye_files]
    im_stack = np.asarray(im_stack0)
    print("Dye stack loaded.")

    # --- 4. Load C1 Guide Image ---
    print("Loading ROI (C1) guide image...")
    sorted_roi_files = natsorted(glob.glob(path_to_roi_tifs))
    if not sorted_roi_files:
        raise FileNotFoundError(f"No ROI .tif files found at: {path_to_roi_tifs}")
    guide_image_path = sorted_roi_files[-1]
    guide_image = cv2.imread(guide_image_path, cv2.IMREAD_UNCHANGED)
    if guide_image is None:
        raise FileNotFoundError(f"C1 Guide Image not found at path: {guide_image_path}")

    # --- 5. Process All GUVs ---
    # Load the full list of GUVs
    detected_vesicles = utils.load_vesicles_dataframe(path_to_csv)
    
    all_intensity_curves = []
    all_start_frames = []

    # Loop over every GUV in the file
    for index, vesicle in detected_vesicles.iterrows():
        # Assumes X, Y, Radius are the first 3 columns
        xc = int(vesicle.iloc[0])
        yc = int(vesicle.iloc[1])
        radius = int(vesicle.iloc[2])
        print(f"\n--- Processing GUV {index + 1} at ({xc}, {yc}) ---")

        # Create the mask for THIS GUV using the C1 guide image
        ideal_mask = utils.create_ideal_mask(guide_image.shape[:2], xc, yc, radius)
        mask_guided = utils.apply_guided_filter(guide_image, ideal_mask.astype(np.float32))

        # Calculate Intensity (from C3 stack)
        ROI_stack = np.multiply(im_stack, mask_guided)
        intensity_total_stack = np.sum(ROI_stack,axis=(1,2)) 
        
        # Check if mask is valid
        num_pixels_in_mask = np.count_nonzero(mask_guided)
        if num_pixels_in_mask == 0:
            print(f"Skipping GUV {index + 1}: Mask is empty.")
            continue
            
        intensity_mean_stack = intensity_total_stack / num_pixels_in_mask

        # Find Fluorescence Jump
        b = -1 
        for i in range(10, len(intensity_mean_stack)):
            if intensity_mean_stack[i] > np.average(intensity_mean_stack[i-10:i]) + cfg.JUMP_SENSITIVITY:
                background = np.array([intensity_mean_stack[i-1]])
                b = i-1
                print(f"Fluorescence jump detected at frame {i}")
                break
                
        if b == -1:
            print(f"Skipping GUV {index + 1}: No jump detected.")
            continue
            
        intensity_mean_curve = intensity_mean_stack[b:] - background
        all_intensity_curves.append(intensity_mean_curve)
        all_start_frames.append(b)

    # --- End of GUV Loop ---

    if not all_intensity_curves:
        raise ValueError("Analysis failed: No valid GUVs with a detected jump were found.")
        
    print(f"\n--- Analysis Complete: {len(all_intensity_curves)} / {len(detected_vesicles)} GUVs processed ---")

    # --- 6. Prepare Data for Averaging & Fitting ---
    
    # Find the shortest curve length to standardize all curves
    min_len = min(len(curve) for curve in all_intensity_curves)
    
    # Truncate all curves to the minimum length
    truncated_curves = [curve[:min_len] for curve in all_intensity_curves]
    
    # Create a 2D numpy array (rows=GUVs, cols=time)
    curve_array = np.array(truncated_curves)
    
    # Calculate the average curve
    average_curve = np.mean(curve_array, axis=0)
    
    # Get the average start frame (for time array and frame export)
    avg_start_frame = int(np.mean(all_start_frames))
    exp_start_time = time_data[avg_start_frame]
    
    # Create the master time array 't'
    time_data_sliced = time_data[avg_start_frame:]
    t = np.array(time_data_sliced) - time_data_sliced[0]
    t = t[:min_len] # Truncate time array to match curves

    # --- 7. Fit Kinetic Model to the AVERAGE curve ---
    print("Fitting average curve to model...")
    total_frames_after_pulse = len(t)
    time_stamp = int(total_frames_after_pulse * cfg.FIT_DATA_PERCENTAGE)
    if time_stamp < 10: time_stamp = 10
    if time_stamp > total_frames_after_pulse: time_stamp = total_frames_after_pulse
    print(f"Fitting first {cfg.FIT_DATA_PERCENTAGE*100:.0f}% of data ({time_stamp} frames).")
    
    params, covariance = sc.curve_fit(
        utils.dyn_model, 
        t[:time_stamp], 
        average_curve[:time_stamp], 
        p0=cfg.FIT_INITIAL_GUESS
    )
    A = params[0] # Amplitude
    tau = params[1] # Time constant
    print(f"Fit complete. A = {A:.2f}, tau = {tau:.3f} s")

    # --- 8. EXPORT TIME-LAPSE FRAMES (from C3) ---
    print("\n--- Exporting Frames ---")
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    total_duration = t[-1]
    times_to_export = np.linspace(0, total_duration, num=cfg.NUM_FRAMES_TO_EXPORT)
    print(f"Generating {cfg.NUM_FRAMES_TO_EXPORT} frames for {total_duration:.1f}s duration...")
    
    for i, time_point in enumerate(times_to_export):
        target_time_abs = exp_start_time + time_point
        frame_idx, actual_time = utils.find_closest_frame(time_data, target_time_abs)
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
    print("Generating summary plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 6))
    
    # Plot all individual GUV traces
    for curve in curve_array:
        ax.plot(t, curve, color='gray', alpha=0.2)
        
    # Plot the average curve
    ax.plot(t, average_curve, 'r.', markersize=3, label=f'Average (n={len(all_intensity_curves)})')
    
    # Plot the fit to the average
    ax.plot(t[:time_stamp], utils.dyn_model(t[:time_stamp],A,tau), 'k--', 
            label=f'Fit (τ = {tau:.3f} s)', linewidth=2)
            
    ax.set_title("GUV Resealing Kinetics (Dye Channel)")
    ax.set_xlabel("Time after pulse (s)")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()