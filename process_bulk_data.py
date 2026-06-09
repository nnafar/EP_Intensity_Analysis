import pandas as pd
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = r"C:\GitHub\EP_Intensity_Analysis\Outputs"
BULK_OUT_DIR = os.path.join(BASE_DIR, "Bulk_Summary")
R2_THRESHOLD = 0.87
SHRINKAGE_THRESHOLD = 0.90 

def compute_r_squared(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def process_bulk_data(base_dir, bulk_out_dir):
    os.makedirs(bulk_out_dir, exist_ok=True)
    all_guvs = []
    
    exp_folders = []
    for root, dirs, files in os.walk(base_dir):
        if "Bulk_Summary" in root:
            continue
        if all(sub in dirs for sub in ["dye", "actin", "tracking"]):
            exp_folders.append(root)
    
    for folder in exp_folders:
        exp_name = os.path.basename(folder)
        condition = "BranchedCortex" if "BranchedCortex" in exp_name else "Empty" if "Empty" in exp_name else "Unknown"
        
        dye_dir = os.path.join(folder, "dye")
        actin_dir = os.path.join(folder, "actin")
        track_dir = os.path.join(folder, "tracking")
        
        fit_file = glob.glob(os.path.join(dye_dir, "*_fit_parameters.csv"))
        quality_file = glob.glob(os.path.join(track_dir, "*_detection_quality.csv"))
        actin_file = glob.glob(os.path.join(actin_dir, "*_actin_cortex_traces.csv"))
        track_file = glob.glob(os.path.join(track_dir, "*_tracking_data.csv"))
        curve_file = glob.glob(os.path.join(dye_dir, "*_normalized_curves.csv"))
        fit_grid_img = glob.glob(os.path.join(dye_dir, "*_dye_fits_grid.png"))
        
        if not all([fit_file, quality_file, actin_file, track_file, curve_file]):
            continue
            
        if fit_grid_img:
            shutil.copy2(fit_grid_img[0], os.path.join(bulk_out_dir, os.path.basename(fit_grid_img[0])))
            
        df_fit = pd.read_csv(fit_file[0])
        df_qual = pd.read_csv(quality_file[0])
        df_actin = pd.read_csv(actin_file[0])
        df_track = pd.read_csv(track_file[0])
        df_curves = pd.read_csv(curve_file[0])
        
        # Determine pulse frame using the actin timeframe (t=0)
        pulse_idx_arr = df_actin.index[df_actin['time_s'] >= 0].tolist()
        pulse_frame = pulse_idx_arr[0] if pulse_idx_arr else 0
        
        for _, row in df_fit.iterrows():
            guv_id = int(row['guv_id'])
            
            # Basic validation
            qual_row = df_qual[df_qual['guv_id'] == guv_id].iloc[0]
            is_ruptured = qual_row['failed']
            
            # Tracking & Morphology
            guv_track = df_track[df_track['guv_id'] == guv_id]
            initial_radius = guv_track['radius'].iloc[0] if not guv_track.empty else row['radius_um']
            final_radius = guv_track['radius'].iloc[-1] if not guv_track.empty else initial_radius
            is_shrinking = (final_radius / initial_radius) < SHRINKAGE_THRESHOLD
            
            guv_track_pre = guv_track[guv_track['frame'] < pulse_frame]
            guv_track_post = guv_track[guv_track['frame'] >= pulse_frame]
            pre_circ = guv_track_pre['circularity'].mean() if not guv_track_pre.empty else np.nan
            min_post_circ = guv_track_post['circularity'].min() if not guv_track_post.empty else np.nan
            max_circ_drop = pre_circ - min_post_circ if pd.notnull(pre_circ) and pd.notnull(min_post_circ) else np.nan
            
            # Kinetics
            y_data = df_curves[f'GUV_{guv_id}_I_uptake'].values
            t_data = df_curves['Time (s)'].values
            y_fit = row['Iinf'] + (row['I0'] - row['Iinf']) * np.exp(-t_data / row['tau']) + row['D'] * t_data
            r2 = compute_r_squared(y_data, y_fit)
            
            # Actin Extraction
            peak_det_col = f'GUV_{guv_id}_peak_detected'
            cortex_mean_col = f'GUV_{guv_id}_cortex_mean'
            cortex_peak_col = f'GUV_{guv_id}_cortex_peak'
            lumen_col = f'GUV_{guv_id}_lumen'
            fwhm_col = f'GUV_{guv_id}_fwhm_um'
            gini_col = f'GUV_{guv_id}_gini'
            
            has_cortex = False
            lumenal_actin_intensity = np.nan
            pre_fwhm = np.nan
            pre_gini = np.nan
            deg_mean_pct = np.nan
            deg_peak_pct = np.nan
            actin_half_life = np.nan
            
            if peak_det_col in df_actin.columns and cortex_mean_col in df_actin.columns:
                pre_mask = df_actin['time_s'] < 0
                post_mask = df_actin['time_s'] >= 0
                post_t = df_actin.loc[post_mask, 'time_s'].values
                
                if lumen_col in df_actin.columns:
                    lumenal_actin_intensity = df_actin.loc[pre_mask, lumen_col].mean()

                if df_actin.loc[pre_mask, peak_det_col].mean() > 0.5:
                    has_cortex = True
                    pre_fwhm = df_actin.loc[pre_mask, fwhm_col].mean()
                    pre_gini = df_actin.loc[pre_mask, gini_col].mean()
                    
                    pre_cortex_mean = df_actin.loc[pre_mask, cortex_mean_col].mean()
                    pre_cortex_peak = df_actin.loc[pre_mask, cortex_peak_col].mean()
                    
                    # Extract raw post-pulse data
                    raw_post_mean = df_actin.loc[post_mask, cortex_mean_col].copy()
                    raw_post_peak = df_actin.loc[post_mask, cortex_peak_col].copy()
                    post_peak_det = df_actin.loc[post_mask, peak_det_col].values
                    post_lumen    = df_actin.loc[post_mask, lumen_col].values
                    
                    # Substitute missing cortex data with lumenal baseline when the structure is completely lost
                    lost_mask = (post_peak_det == 0)
                    raw_post_mean[lost_mask] = post_lumen[lost_mask]
                    raw_post_peak[lost_mask] = post_lumen[lost_mask]
                    
                    # Smooth post-pulse signals to prevent noise from defining the absolute minimum
                    post_cortex_mean = raw_post_mean.rolling(window=3, center=True, min_periods=1).mean().values
                    post_cortex_peak = raw_post_peak.rolling(window=3, center=True, min_periods=1).mean().values
                    
                    if pre_cortex_mean > 0 and len(post_cortex_mean) > 0:
                        min_post_mean = np.nanmin(post_cortex_mean)
                        deg_mean_pct = (1 - (min_post_mean / pre_cortex_mean)) * 100
                        
                        # Calculate Actin Disassembly Half-Life (Time to reach 50% of its maximum drop)
                        if deg_mean_pct > 10: # Only calculate kinetics if significant degradation occurred
                            half_drop_val = pre_cortex_mean - 0.5 * (pre_cortex_mean - min_post_mean)
                            cross_idx = np.where(post_cortex_mean < half_drop_val)[0]
                            if len(cross_idx) > 0:
                                actin_half_life = post_t[cross_idx[0]]
                                
                    if pre_cortex_peak > 0 and len(post_cortex_peak) > 0:
                        deg_peak_pct = (1 - (np.nanmin(post_cortex_peak) / pre_cortex_peak)) * 100
            
            valid_for_stats = (not is_ruptured) and (not is_shrinking) and (r2 >= R2_THRESHOLD)
            
            # Update the size grouping logic
            all_guvs.append({
                'experiment': exp_name,
                'condition': condition,
                'guv_id': guv_id,
                'radius_um': row['radius_um'],
                'size_group': 'Small (<5µm)' if row['radius_um'] < 5 else 'Medium (5-10µm)' if row['radius_um'] <= 10 else 'Large (10-15µm)' if row['radius_um'] <= 15 else 'Extra Large (>15µm)',
                'tau': row['tau'],
                'I0': row['I0'],
                'Iinf': row['Iinf'],
                'r_squared': r2,
                'is_ruptured': is_ruptured,
                'is_shrinking': is_shrinking,
                'has_cortex': has_cortex,
                'max_circ_drop': max_circ_drop,
                'lumenal_actin_intensity': lumenal_actin_intensity,
                'pre_fwhm_um': pre_fwhm,
                'pre_gini': pre_gini,
                'degradation_mean_pct': deg_mean_pct,
                'degradation_peak_pct': deg_peak_pct,
                'actin_half_life': actin_half_life,
                'valid_for_stats': valid_for_stats
            })
            
    return pd.DataFrame(all_guvs)

def plot_analysis(df, bulk_out_dir):
    df_valid = df[df['valid_for_stats'] == True]
    
    # ---------------------------------------------------------
    # FIGURE 1: Morphology & Core Kinetics
    # ---------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    
    # Update the order array in Figure 1
    sns.boxplot(data=df_valid, x='size_group', y='tau', hue='condition', ax=axes1[0, 0], 
                order=['Small (<5µm)', 'Medium (5-10µm)', 'Large (10-15µm)', 'Extra Large (>15µm)'])
    axes1[0, 0].set_title('Uptake Rate (τ) vs. GUV Size')
    axes1[0, 0].set_ylabel('Tau (s)')
    
    sns.boxplot(data=df_valid, x='has_cortex', y='tau', hue='condition', ax=axes1[0, 1])
    axes1[0, 1].set_title('Uptake Rate (τ) by Cortex Presence')
    axes1[0, 1].set_xticklabels(['Lumenal Actin Only', 'Cortex Present'])
    axes1[0, 1].set_ylabel('Tau (s)')
    
    sns.scatterplot(data=df_valid, x='lumenal_actin_intensity', y='tau', hue='condition', ax=axes1[1, 0], alpha=0.7)
    axes1[1, 0].set_title('Uptake Rate (τ) vs. Lumenal Actin')
    axes1[1, 0].set_xlabel('Lumenal Actin Baseline (AU)')
    axes1[1, 0].set_ylabel('Tau (s)')
    
    sns.scatterplot(data=df_valid, x='max_circ_drop', y='tau', hue='condition', ax=axes1[1, 1], alpha=0.7)
    axes1[1, 1].set_title('Uptake Rate (τ) vs. Membrane Deformation')
    axes1[1, 1].set_xlabel('Max Circularity Drop Post-Pulse')
    axes1[1, 1].set_ylabel('Tau (s)')
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(bulk_out_dir, "bulk_kinetics_morphology.png"), dpi=300)
    
    # ---------------------------------------------------------
    # FIGURE 2: Actin Cortex Deep Dive
    # ---------------------------------------------------------
    df_cortex = df_valid[df_valid['has_cortex'] == True].copy()
    
    if not df_cortex.empty:
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
        
        # Reshape data to plot Mean vs Peak degradation side-by-side
        df_melted_deg = df_cortex.melt(id_vars=['condition', 'guv_id'], 
                                       value_vars=['degradation_mean_pct', 'degradation_peak_pct'],
                                       var_name='Measurement_Type', value_name='Degradation_Pct')
        df_melted_deg['Measurement_Type'] = df_melted_deg['Measurement_Type'].map({
            'degradation_mean_pct': 'Spatial Mean', 'degradation_peak_pct': 'Apical Peak'
        })
        
        sns.violinplot(data=df_melted_deg, x='Measurement_Type', y='Degradation_Pct', hue='condition', ax=axes2[0, 0], split=True, inner="quart")
        axes2[0, 0].set_title('Cortex Degradation Profile')
        axes2[0, 0].set_ylabel('Max Loss Post-Pulse (%)')
        axes2[0, 0].set_xlabel('')
        
        sns.scatterplot(data=df_cortex, x='pre_fwhm_um', y='tau', hue='condition', ax=axes2[0, 1], alpha=0.7)
        axes2[0, 1].set_title('Uptake Rate (τ) vs. Cortex Thickness')
        axes2[0, 1].set_xlabel('Pre-Pulse FWHM (µm)')
        axes2[0, 1].set_ylabel('Tau (s)')
        
        sns.scatterplot(data=df_cortex, x='pre_gini', y='tau', hue='condition', ax=axes2[1, 0], alpha=0.7)
        axes2[1, 0].set_title('Uptake Rate (τ) vs. Cortex Heterogeneity')
        axes2[1, 0].set_xlabel('Pre-Pulse Gini Index')
        axes2[1, 0].set_ylabel('Tau (s)')
        
        sns.scatterplot(data=df_cortex, x='actin_half_life', y='tau', hue='condition', ax=axes2[1, 1], alpha=0.7)
        axes2[1, 1].plot([0, df_cortex['tau'].max()], [0, df_cortex['tau'].max()], 'k--', alpha=0.3, label='1:1 Ratio')
        axes2[1, 1].set_title('Kinetic Coupling: Disassembly vs. Influx')
        axes2[1, 1].set_xlabel('Actin Disassembly Half-Life (s)')
        axes2[1, 1].set_ylabel('Dye Uptake Tau (s)')
        axes2[1, 1].legend()
        
        fig2.tight_layout()
        fig2.savefig(os.path.join(bulk_out_dir, "bulk_actin_deepdive.png"), dpi=300)

    plt.show()

# Execution
df_summary = process_bulk_data(BASE_DIR, BULK_OUT_DIR)
df_summary.to_csv(os.path.join(BULK_OUT_DIR, "bulk_analysis_summary.csv"), index=False)
plot_analysis(df_summary, BULK_OUT_DIR)