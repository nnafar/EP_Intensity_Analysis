# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS ---
======================================================
This is the main script to run the analysis.
It imports settings from 'config.py' and functions
from 'guv_analysis_utils.py'.
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
    # Build path for the DYE (C3) TIF stack
    path_to_dye_tifs = os.path.join(cfg.DATA_FOLDER, cfg.DYE_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    
    # Build path for the ROI (C1) TIF stack
    path_to_roi_tifs = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.TIF_SUFFIX)
    
    # Build path for the ROI-based (C1) CSV
    path_to_csv = os.path.join(cfg.DATA_FOLDER, cfg.ROI_CHANNEL_PREFIX + cfg.EXPERIMENT_BASE_NAME + cfg.CSV_SUFFIX)
    
    print(f"Loading Dye (measurement) files from: {path_to_dye_tifs}")
    print(f"Loading ROI (guide) files from: {path_to_roi_tifs}")
    print(f"Loading CSV from: {path_to_csv}")

    sorted_dye_files = natsorted(glob.glob(path_to_dye_tifs))
    if not sorted_dye_files:
        raise FileNotFoundError(f"No DYE .tif files found at: {path_to_dye_tifs}")
    print(f"Found {len(sorted_dye_files)} dye image files.")

    # --- 2. Extract Time Data ---
    # Time data is based on the DYE stack, as that's what we're measuring
    time_data = utils.extract_timestamps_from_metadata(sorted_dye_files)
    if time_data is None:
        print("Metadata extraction failed. Using manual FPS calculation.")
        time_data = utils.create_manual_timestamps(len(sorted_dye_files), cfg.FALLBACK_FPS)
    
    print(f"Successfully extracted {len(time_data)} timestamps using {utils.metadata_method_used}")

    # --- 3. Load DYE Image Stack ---
    print("Loading DYE (C3) image stack (this may take a moment)...")
    im_stack0=[]
    for i in range(len(sorted_dye_files)):
        img_load = cv2.imread(sorted_dye_files[i],cv2.IMREAD_ANYDEPTH)
        im_stack0.append(img_load)
    
    # This 'im_stack' is now the C3 (Dye) data
    im_stack = np.asarray(im_stack0)
    print("Dye stack loaded.")

    # --- 4. Create Automated ROI from C1 ---
    print("Loading ROI (C1) guide image...")
    sorted_roi_files = natsorted(glob.glob(path_to_roi_tifs))
    if not sorted_roi_files:
        raise FileNotFoundError(f"No ROI .tif files found at: {path_to_roi_tifs}")

    # Load the last frame of the C1 (Membrane) channel
    guide_image_path = sorted_roi_files[-1]
    guide_image = cv2.imread(guide_image_path, cv2.IMREAD_UNCHANGED)
    if guide_image is None:
        raise FileNotFoundError(f"C1 Guide Image not found at path: {guide_image_path}")

    # Load C1-based CSV
    xc, yc, radius = utils.load_vesicle_coords_from_csv(path_to_csv)
    
    # Create the mask using the C1 guide image
    ideal_mask = utils.create_ideal_mask(guide_image.shape[:2], xc, yc, radius)
    mask_guided = utils.apply_guided_filter(guide_image, ideal_mask.astype(np.float32))
    print("Guided filter applied to C1 mask.")

    # Optional: Display the final C1-based mask
    cv2.imshow("Automated Guided Mask (from C1)", mask_guided * 255)
    print("Showing final mask. Press any key to continue analysis...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 5. Calculate Intensity (from C3) ---
    # This is now correct:
    # We multiply the DYE (C3) stack by the C1-defined mask
    ROI_stack = np.multiply(im_stack, mask_guided)
    intensity_total_stack = np.sum(ROI_stack,axis=(1,2)) 
    intensity_mean_stack = intensity_total_stack / np.count_nonzero(mask_guided)

    # --- 6. Find Fluorescence Jump (from C3) ---
    b = -1 
    for i in range(10, len(intensity_mean_stack)):
        if intensity_mean_stack[i] > np.average(intensity_mean_stack[i-10:i]) + cfg.JUMP_SENSITIVITY:
            background = np.array([intensity_mean_stack[i-1]])
            b = i-1
            print(f"Fluorescence jump detected at frame {i} (time {time_data[i]:.3f}s)")
            break
    if b == -1:
        raise ValueError("Could not detect fluorescence jump. Check 'JUMP_SENSITIVITY' or data.")
    intensity_mean_curve = intensity_mean_stack[b:] - background

    # --- 7. Fit Kinetic Model ---
    time_data_sliced = time_data[b:]
    if not time_data_sliced:
        raise ValueError("Failed to slice time data; jump detection might be off.")
    t = np.array(time_data_sliced) - time_data_sliced[0]
    exp_start_time = time_data[b]
    print("Fitting data to model...")
    total_frames_after_pulse = len(t)
    time_stamp = int(total_frames_after_pulse * cfg.FIT_DATA_PERCENTAGE)
    if time_stamp < 10: time_stamp = 10
    if time_stamp > total_frames_after_pulse: time_stamp = total_frames_after_pulse
    print(f"Fitting first {cfg.FIT_DATA_PERCENTAGE*100:.0f}% of data ({time_stamp} frames).")
    params, covariance = sc.curve_fit(
        utils.dyn_model, 
        t[:time_stamp], 
        intensity_mean_curve[:time_stamp], 
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
        
        # Get frame from the DYE (C3) stack
        frame_data = im_stack[frame_idx]
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

    # --- 9. PLOT KINETIC RESULTS ---
    print("Generating plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(t[:time_stamp], utils.dyn_model(t[:time_stamp],A,tau), 'r-', label=f'Fit (τ = {tau:.3f} s)')
    ax.plot(t, intensity_mean_curve, 'b.', markersize=2, label='Data (all)')
    ax.set_title("GUV Resealing Kinetics (Dye Channel)")
    ax.set_xlabel("Time after pulse (s)")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()