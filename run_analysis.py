# -*- coding: utf-8 -*-
"""
======================================================
--- GUV INTENSITY ANALYSIS ---
======================================================
This enhanced version uses radial profile analysis to accurately detect
the membrane location, adapted from the skeleton.py approach.

Key improvements:
1. Membrane detection from intensity profiles instead of fixed CSV radius
2. Better handling of irregular vesicles
3. Quality metrics for each detection
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
import guv_analysis_utils as utils  # Use enhanced version


def main():
    """Main analysis function with enhanced membrane detection."""
    
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
        # Explicitly take the first 5 columns to handle trailing comma/extra columns.
        df_vesicles = df_vesicles.iloc[:, :5]
        df_vesicles.columns = ['id', 'xc', 'yc', 'size', 'score']
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    guv_coords = df_vesicles[['xc', 'yc']].values.astype(int)
    # 'guv_sizes' now correctly holds the radius values (as estimates)
    guv_sizes = df_vesicles['size'].values.astype(int) 
    
    print(f"Found {len(guv_coords)} GUV(s) in CSV file.")

    # --- 5. Process Each GUV with Enhanced Detection ---
    all_intensity_curves = []
    all_background_curves = [] 
    all_jump_frames = []
    detection_quality_log = []  # Store detection quality info
    
    for i, (xc, yc) in enumerate(guv_coords):
        guv_id = df_vesicles.loc[i, 'id']
        guv_radius_from_csv_raw = guv_sizes[i]
        
        print(f"\n--- Processing GUV {guv_id} at ({xc}, {yc}) ---")
        
        # --- HOTFIX: Apply 10x correction to radius ---
        guv_radius_estimate = int(np.round(guv_radius_from_csv_raw / 10.0))
        if i == 0:
             print(f"  - Applying 10x correction to radius from CSV: {guv_radius_from_csv_raw}px -> {guv_radius_estimate}px")
        # --- End Hotfix ---

        
        # 5a. ENHANCED: Define Masks using Membrane Detection
        print(f"  - Detecting membrane from radial intensity profile...")
        inner_mask, membrane_mask, background_mask, detection_info = utils.create_guv_masks_with_detection(
            roi_guide_frame, 
            (xc, yc), 
            radius_estimate=guv_radius_estimate,
            bg_buffer=cfg.BG_BUFFER_PIXELS,
            bg_width=cfg.BG_RING_WIDTH_PIXELS,
            num_angles=360,  # Use lots of angles for accurate detection
            length_excess=1.5
        )
        
        # Log detection quality
        quality_entry = {
            'guv_id': guv_id,
            'estimated_radius': guv_radius_estimate,
            'detected_inner': detection_info['detected_inner_radius'],
            'detected_outer': detection_info['detected_outer_radius'],
            'failed': detection_info['detection_failed'],
            'comments': ', '.join(detection_info['comments']) if detection_info['comments'] else 'OK'
        }
        detection_quality_log.append(quality_entry)
        
        # Print quality warnings if any
        if detection_info['comments']:
            print(f"  - Quality flags: {', '.join(detection_info['comments'])}")

        # Export mask visualization for the first GUV (i==0) OR if detection had issues
        should_export_viz = (cfg.EXPORT_MASK_VISUALIZATION and i == 0) or \
                           (detection_info['detection_failed'] or detection_info['comments'])
        
        if should_export_viz:
            print(f"  - Generating mask visualization for GUV {guv_id}...")
            viz_image = utils.create_mask_visualization(
                roi_guide_frame,
                inner_mask,
                membrane_mask,
                background_mask,
                alpha=cfg.MASK_VIZ_OVERLAY_ALPHA
            )
            
            # Ensure output directory exists
            os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
            
            # Build a unique name
            viz_name = f"{cfg.EXPERIMENT_BASE_NAME}_mask_viz_GUV_{guv_id}.png"
            viz_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, viz_name)
            
            try:
                cv2.imwrite(viz_path, viz_image)
                print(f"    Saved mask visualization to: {viz_path}")
            except Exception as e:
                print(f"    Warning: Could not save mask visualization: {e}")
                
            # ALSO save the radial profile plot for problematic detections
            if detection_info['detection_failed'] or detection_info['comments']:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(detection_info['along_radius'], detection_info['radial_profile'], 'b-', linewidth=2)
                ax.axvline(x=detection_info['detected_inner_radius'], color='green', linestyle='--', 
                          label=f"Inner border ({detection_info['detected_inner_radius']:.1f}px)")
                ax.axvline(x=detection_info['detected_outer_radius'], color='red', linestyle='--', 
                          label=f"Outer border ({detection_info['detected_outer_radius']:.1f}px)")
                ax.axvline(x=guv_radius_estimate, color='orange', linestyle=':', 
                          label=f"CSV estimate ({guv_radius_estimate}px)")
                ax.set_xlabel('Radius (pixels)')
                ax.set_ylabel('Intensity (a.u.)')
                ax.set_title(f'Radial Intensity Profile - GUV {guv_id}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                profile_name = f"{cfg.EXPERIMENT_BASE_NAME}_radial_profile_GUV_{guv_id}.png"
                profile_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, profile_name)
                fig.savefig(profile_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"    Saved radial profile to: {profile_path}")
        
        # 5b. Get Intensity Traces
        intensity_trace = utils.get_intensity_trace(im_stack, inner_mask)
        background_trace = utils.get_intensity_trace(im_stack, background_mask)
        
        # 5c. Detect Jump 
        try:
            jump_frame = utils.detect_intensity_jump(intensity_trace, sensitivity=cfg.JUMP_SENSITIVITY)
            baseline_frames_end = max(5, jump_frame) 
            
            if jump_frame == -1:
                jump_frame = 0 
                print(f"  - No clear jump detected for GUV {guv_id}, assuming start at frame 0.")
            else:
                print(f"  - Fluorescence jump detected at frame {jump_frame}")
                
        except Exception as e:
            print(f"  - Warning: Jump detection failed for GUV {guv_id}: {e}")
            jump_frame = 0 
            baseline_frames_end = 5
            
        # 5d. Store Traces and Jump Index
        all_intensity_curves.append(intensity_trace)
        all_background_curves.append(background_trace) 
        all_jump_frames.append(jump_frame)

    print(f"\n--- Analysis Complete: {len(all_intensity_curves)} / {len(guv_coords)} GUVs processed ---")
    
    # Save detection quality log
    df_quality = pd.DataFrame(detection_quality_log)
    quality_log_path = os.path.join(cfg.OUTPUT_IMAGE_FOLDER, f"{cfg.EXPERIMENT_BASE_NAME}_detection_quality.csv")
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    df_quality.to_csv(quality_log_path, index=False)
    print(f"\nSaved detection quality log to: {quality_log_path}")
    
    # Print summary statistics
    num_failed = df_quality['failed'].sum()
    num_with_warnings = (df_quality['comments'] != 'OK').sum()
    print(f"\nDetection Summary:")
    print(f"  - Total GUVs: {len(df_quality)}")
    print(f"  - Failed detections: {num_failed}")
    print(f"  - Detections with warnings: {num_with_warnings}")
    print(f"  - Clean detections: {len(df_quality) - num_with_warnings}")
    
    # --- Three-Term Normalization ---
    normalized_curves = []
    print("\nNormalizing intensity curves using $I_{uptake} = (I_{dye,t} - I_{dye,0}) / (I_{background,t} - I_{dye,0})$...")
    
    for dye_trace, bg_trace, jump_f in zip(all_intensity_curves, all_background_curves, all_jump_frames):
        # Determine baseline frames for I_dye,0
        baseline_frames_end = max(5, jump_f)
        
        # Calculate I_dye,0 (initial intensity before pulse)
        I_dye_0 = np.mean(dye_trace[:baseline_frames_end])
        
        # Calculate the denominator (I_background,t - I_dye,0)
        denominator = bg_trace - I_dye_0
        
        # Calculate the numerator (I_dye,t - I_dye,0)
        numerator = dye_trace - I_dye_0
        
        # Perform division. Use np.divide and np.nan_to_num for safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_trace = np.divide(numerator, denominator)
            # Set inf/NaN values (e.g., where denominator is near zero) to 0.0
            normalized_trace = np.nan_to_num(normalized_trace, nan=0.0, posinf=0.0, neginf=0.0)
            
        normalized_curves.append(normalized_trace)

    all_intensity_curves = normalized_curves # Replace original list with normalized
    # --- END Normalization ---


    # --- 6. Calculate Average Curve (and perform alignment) ---
    if not all_intensity_curves:
        print("Error: No valid intensity curves were processed. Cannot continue to fitting.")
        return
        
    curve_array_original = np.array(all_intensity_curves)
    
    # --- Aligning the data to the median jump frame ---
    valid_jump_frames = np.array(all_jump_frames)
    
    # Calculate the median jump frame index to use as the common alignment point
    median_jump_frame = int(np.median(valid_jump_frames))
    print(f"Aligning all data using the median jump frame index: {median_jump_frame}")
    
    # Slice the time array from the median jump frame and re-zero it
    t_aligned = time_array_original[median_jump_frame:]
    t = t_aligned - t_aligned[0]
    
    # Slice the curve data and image stack
    average_curve = np.mean(curve_array_original[:, median_jump_frame:], axis=0)
    curve_array = curve_array_original[:, median_jump_frame:]
    im_stack_aligned = im_stack[median_jump_frame:] # For image export
    
    # Update variables for subsequent steps
    total_frames = len(t)
    
    # --- 7. Fit Average Curve to Selected Model ---
    print(f"Fitting average curve to model: {cfg.MODEL_TO_USE}...")
    
    # Calculate the number of frames to use for fitting based on the configuration
    fit_slice_index = int(np.round(total_frames * cfg.FIT_DATA_PERCENTAGE))
    
    # Ensure the count does not exceed the total available frames after alignment
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

    # Perform the curve fit
    try:
        params, covariance = sc.curve_fit(
            fit_function, 
            t[:fit_slice_index], 
            average_curve[:fit_slice_index], 
            p0=p0_guess,
            maxfev=5000 
        )
    except RuntimeError as e:
        print(f"FATAL WARNING: Curve fitting failed: {e}. Using initial guess parameters for output.")
        params = np.array(p0_guess)
        
    # --- 8. Extract Parameters and Generate Outputs ---
    if cfg.MODEL_TO_USE == '5-PARAM':
        Af = params[0]
        A1 = params[1]
        tau1 = params[2]
        A2 = params[3]
        tau2 = params[4]

        print(f"Fit Complete. Af={Af:.2f}, A1={A1:.2f}, tau1={tau1:.2f} s, A2={A2:.2f}, tau2={tau2:.2f} s")
        fig_title = f"Fit Parameters: $A_f = {Af:.2f}$, $A_1 = {A1:.2f}$, $\\tau_1 = {tau1:.2f}$ s, $A_2 = {A2:.2f}$, $\\tau_2 = {tau2:.2f}$ s"
        fit_curve = utils.dyn_model(t, Af, A1, tau1, A2, tau2)
        
    elif cfg.MODEL_TO_USE == '4-PARAM':
        I_offset = params[0]
        A = params[1]
        tau = params[2]
        D = params[3]

        print(f"Fit Complete. I_offset={I_offset:.2f}, A={A:.2f}, tau={tau:.2f} s, D={D:.4f}")
        fig_title = f"Fit Parameters: $I_{{offset}} = {I_offset:.2f}$, $A = {A:.2f}$, $\\tau = {tau:.2f}$ s, $D = {D:.4f}$"
        fit_curve = utils.dyn_model_4param(t, I_offset, A, tau, D)

    # --- 9. Export Representative Frames ---
    print("\nExporting representative frames...")
    
    # Ensure the output directory exists
    os.makedirs(cfg.OUTPUT_IMAGE_FOLDER, exist_ok=True)
    
    # Use the aligned time array (t) and image stack (im_stack_aligned)
    for i, time_point in enumerate(cfg.EXPORT_TIME_POINTS_S):
        frame_idx, actual_time = utils.find_closest_frame(t, time_point)
        frame_data = im_stack_aligned[frame_idx]
        
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

    # --- 10. PLOT KINETIC RESULTS (Summary Plot) ---
    print("\nGenerating summary plot...")
    fig,ax = plt.subplots(nrows=1,ncols=1, figsize=(10, 6))
    
    # Plot all individual GUV traces
    for curve in curve_array:
        ax.plot(t[:len(curve)], curve, color='gray', alpha=0.2)
        
    # Plot the average curve
    ax.plot(t, average_curve, 'r.', markersize=3, label=f'Average (n={len(guv_coords)})')

    # Plot the fitted curve
    ax.plot(t, fit_curve, 'k-', linewidth=2, label='Fitted Curve')
    
    # Highlight the fitted slice for clarity
    ax.plot(t[fit_slice_index-1], average_curve[fit_slice_index-1], 'b*', markersize=10, label='End of Fit Data')


    # Add fit parameters to the title
    ax.set_title(fig_title)
    
    # Final plot styling
    ax.set_xlabel("Time (s) relative to Electroporation Pulse")
    ax.set_ylabel("Normalized Intensity ($I_{uptake}$)") 
    ax.legend(loc='lower right')
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()