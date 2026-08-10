import pandas as pd
import numpy as np
import os
import re
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR     = r"D:\EP\Outputs"
BULK_OUT_DIR = os.path.join(BASE_DIR, "Bulk_Summary")

# A GUV is 'responding' if amplitude >= this fraction of pre-pulse baseline
RESPONSE_AMPLITUDE_THRESHOLD = 0.05
# tau above this (s) means fit hit the boundary — no real decay detected
TAU_MAX_PHYSICAL     = 4500.0
# Extreme radius loss flagged as tracking failure (not biology)
SHRINKAGE_THRESHOLD  = 0.50

# Output subfolders inside BULK_OUT_DIR
FOLDER_DYE   = os.path.join(BULK_OUT_DIR, "dye")
FOLDER_MORPH = os.path.join(BULK_OUT_DIR, "morphology")
FOLDER_ACTIN = os.path.join(BULK_OUT_DIR, "actin")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_experiment_name(exp_name: str) -> dict:
    """
    Extract condition labels from folder names such as:
        260331_DOPC_BranchedCortex_Experiment1-400V-500us-frame6
        260507_DOPC_Empty_Experiment2-360V-500us-frame5

    Returns keys: condition, voltage_v, pulse_dur_us, ep_condition
    """
    if "BranchedCortex" in exp_name:
        condition = "BranchedCortex"
    elif "Empty" in exp_name:
        condition = "Empty"
    else:
        condition = "Unknown"

    volt_match  = re.search(r'(\d+)V',  exp_name)
    pulse_match = re.search(r'(\d+)us', exp_name, re.IGNORECASE)

    voltage_v    = int(volt_match.group(1))  if volt_match  else None
    pulse_dur_us = int(pulse_match.group(1)) if pulse_match else None

    if voltage_v is not None and pulse_dur_us is not None:
        ep_condition = f"{voltage_v}V / {pulse_dur_us}\u00b5s"
    elif voltage_v is not None:
        ep_condition = f"{voltage_v}V"
    elif pulse_dur_us is not None:
        ep_condition = f"{pulse_dur_us}\u00b5s"
    else:
        ep_condition = "Unknown"

    return dict(
        condition=condition,
        voltage_v=voltage_v,
        pulse_dur_us=pulse_dur_us,
        ep_condition=ep_condition,
    )


def compute_r_squared(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def _size_group(r_um: float) -> str:
    if r_um < 5:
        return "Small (<5\u00b5m)"
    elif r_um <= 10:
        return "Medium (5-10\u00b5m)"
    elif r_um <= 15:
        return "Large (10-15\u00b5m)"
    else:
        return "Extra Large (>15\u00b5m)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def process_bulk_data(base_dir: str, bulk_out_dir: str) -> pd.DataFrame:
    for _d in (bulk_out_dir, FOLDER_DYE, FOLDER_MORPH, FOLDER_ACTIN):
        os.makedirs(_d, exist_ok=True)
    all_guvs = []

    # Discover experiment folders — only dye + tracking are required;
    # actin is optional (Empty vesicles won't have it).
    exp_folders = []
    for root, dirs, files in os.walk(base_dir):
        if "Bulk_Summary" in root:
            continue
        if all(sub in dirs for sub in ["dye", "tracking"]):
            exp_folders.append(root)

    if not exp_folders:
        print(f"[WARNING] No experiment folders found under: {base_dir}")
        return pd.DataFrame()

    for folder in exp_folders:
        exp_name = os.path.basename(folder)
        meta     = parse_experiment_name(exp_name)

        dye_dir   = os.path.join(folder, "dye")
        actin_dir = os.path.join(folder, "actin")
        track_dir = os.path.join(folder, "tracking")

        fit_file     = glob.glob(os.path.join(dye_dir,   "*_fit_parameters.csv"))
        quality_file = glob.glob(os.path.join(track_dir, "*_detection_quality.csv"))
        track_file   = glob.glob(os.path.join(track_dir, "*_tracking_data.csv"))
        curve_file   = glob.glob(os.path.join(dye_dir,   "*_normalized_curves.csv"))
        # Actin is optional
        actin_file   = glob.glob(os.path.join(actin_dir, "*_actin_cortex_traces.csv"))
        fit_grid_img = glob.glob(os.path.join(dye_dir,   "*_dye_fits_grid.png"))

        # Only the four core files are required
        if not all([fit_file, quality_file, track_file, curve_file]):
            print(f"  [SKIP] Missing required files in: {exp_name}")
            continue

        if fit_grid_img:
            shutil.copy2(fit_grid_img[0],
                         os.path.join(FOLDER_DYE, os.path.basename(fit_grid_img[0])))

        df_fit    = pd.read_csv(fit_file[0])
        df_qual   = pd.read_csv(quality_file[0])
        df_track  = pd.read_csv(track_file[0])
        df_curves = pd.read_csv(curve_file[0])

        # Actin CSV: load if present, otherwise None
        df_actin = pd.read_csv(actin_file[0]) if actin_file else None

        # ---------------------------------------------------------------------------
        # Pulse frame for tracking pre/post split.
        #
        # The actin CSV is already pulse-aligned (t=0 is the pulse), so it has
        # NO negative-time rows — we cannot derive pulse_frame from it.
        # Instead we infer it from the normalised-curves CSV: the pulse frame is
        # the row index where Time(s) == 0 (i.e. the column header alignment point).
        # If that isn't present, fall back to the frame where the dye signal first
        # starts dropping (min of the average curve), or just use frame 0.
        # ---------------------------------------------------------------------------
        pulse_frame = 0
        if 'Time (s)' in df_curves.columns:
            t_vals = df_curves['Time (s)'].values
            zero_idx = np.where(t_vals == 0)[0]
            if len(zero_idx) > 0:
                pulse_frame = int(zero_idx[0])
            else:
                # t=0 not present: find the frame closest to 0
                pulse_frame = int(np.argmin(np.abs(t_vals)))

        for _, row in df_fit.iterrows():
            guv_id = int(row['guv_id'])

            # --- Quality / rupture ---
            qual_rows = df_qual[df_qual['guv_id'] == guv_id]
            if qual_rows.empty:
                continue
            qual_row    = qual_rows.iloc[0]
            is_ruptured = bool(qual_row['failed'])

            # --- Tracking / morphology ---
            guv_track      = df_track[df_track['guv_id'] == guv_id]
            initial_radius = guv_track['radius'].iloc[0]  if not guv_track.empty else row['radius_um']
            final_radius   = guv_track['radius'].iloc[-1] if not guv_track.empty else initial_radius
            is_shrinking   = (final_radius / initial_radius) < SHRINKAGE_THRESHOLD  # now 0.50

            guv_track_pre  = guv_track[guv_track['frame'] < pulse_frame]
            guv_track_post = guv_track[guv_track['frame'] >= pulse_frame]
            pre_circ       = guv_track_pre['circularity'].mean()  if not guv_track_pre.empty  else np.nan
            min_post_circ  = guv_track_post['circularity'].min()  if not guv_track_post.empty else np.nan
            max_circ_drop  = (pre_circ - min_post_circ
                              if pd.notnull(pre_circ) and pd.notnull(min_post_circ)
                              else np.nan)

            # --- Kinetic R² ---
            guv_col = f'GUV_{guv_id}_I_uptake'
            if guv_col not in df_curves.columns:
                continue
            y_data = df_curves[guv_col].values
            t_data = df_curves['Time (s)'].values
            y_fit  = (row['Iinf']
                      + (row['I0'] - row['Iinf']) * np.exp(-t_data / row['tau'])
                      + row['D'] * t_data)
            r2 = compute_r_squared(y_data, y_fit)

            # --- Response classification (replaces R² gate) ---
            # Parameters cached for reuse
            _tau  = float(row['tau'])
            _I0   = float(row['I0'])
            _Iinf = float(row['Iinf'])

            # is_bad_fit: optimizer returned unphysical values — exclude entirely
            is_bad_fit = (
                not np.isfinite(_tau)  or
                not np.isfinite(_I0)   or
                not np.isfinite(_Iinf) or
                _tau  >= 4999.0        or   # hit curve_fit upper bound
                _I0   >  2.0           or   # baseline > 2× pre-pulse is unphysical
                _I0   <= 0.0           or   # non-positive baseline is unphysical
                _Iinf < -0.20               # influx behaviour — efflux model inapplicable
            )

            # fit_amplitude: fractional drop (I0-Iinf)/I0
            fit_amplitude = (_I0 - _Iinf) / max(abs(_I0), 1e-6)

            # is_responding: clean efflux with meaningful amplitude and tau
            is_responding = (
                not is_bad_fit                                and
                fit_amplitude >= RESPONSE_AMPLITUDE_THRESHOLD and
                _tau < TAU_MAX_PHYSICAL
            )

            # --- Actin (optional) ---
            has_cortex              = False
            lumenal_actin_intensity = np.nan
            pre_fwhm                = np.nan
            pre_gini                = np.nan
            deg_mean_pct            = np.nan
            deg_peak_pct            = np.nan
            actin_half_life         = np.nan

            if df_actin is not None:
                peak_det_col    = f'GUV_{guv_id}_peak_detected'
                cortex_mean_col = f'GUV_{guv_id}_cortex_mean'
                cortex_peak_col = f'GUV_{guv_id}_cortex_peak'
                lumen_col       = f'GUV_{guv_id}_lumen'
                fwhm_col        = f'GUV_{guv_id}_fwhm_um'
                gini_col        = f'GUV_{guv_id}_gini'

                if peak_det_col in df_actin.columns and cortex_mean_col in df_actin.columns:
                    # All traces are already normalised to the pre-pulse baseline
                    # by the pipeline (pre-pulse mean = 1.0 by construction).
                    # The entire CSV is the post-pulse window; there are no
                    # negative-time rows.  We therefore:
                    #   · determine has_cortex from peak_detected across all frames
                    #   · read pre-pulse FWHM / Gini from the avg_ columns in the CSV
                    #     (those were computed from the true pre-pulse frames inside
                    #      the pipeline and exported alongside the aligned traces)
                    #   · compute degradation directly as (1 - min_post_norm) * 100
                    #     since the pre-pulse baseline is 1.0
                    all_t       = df_actin['time_s'].values
                    post_t      = all_t                       # entire window is post-pulse
                    peak_det    = df_actin[peak_det_col].values

                    # has_cortex: cortex was present in the majority of frames
                    has_cortex = peak_det.mean() > 0.5

                    # Lumenal actin: mean of the normalised lumen trace
                    if lumen_col in df_actin.columns:
                        lumenal_actin_intensity = float(np.nanmean(df_actin[lumen_col].values))

                    if has_cortex:
                        # Pre-pulse FWHM and Gini: use the first frames where
                        # cortex was actually detected (peak_detected==1), up to
                        # 20 frames in.  Avoids NaN-only slices for GUVs where
                        # cortex is not detectable right at t=0.
                        det_idx = np.where(peak_det[:20] == 1)[0]
                        if len(det_idx) > 0:
                            early = det_idx[:5]
                            if fwhm_col in df_actin.columns:
                                pre_fwhm = float(np.nanmean(df_actin[fwhm_col].values[early]))
                            if gini_col in df_actin.columns:
                                pre_gini = float(np.nanmean(df_actin[gini_col].values[early]))

                        # Post-pulse cortex traces (normalised; pre-pulse baseline = 1.0)
                        raw_post_mean = df_actin[cortex_mean_col].copy()
                        raw_post_peak = df_actin[cortex_peak_col].copy()
                        post_lumen    = df_actin[lumen_col].values if lumen_col in df_actin.columns \
                                        else np.zeros(len(df_actin))

                        # When cortex is no longer detectable, substitute the
                        # normalised lumenal value as the floor
                        lost_mask = peak_det == 0
                        raw_post_mean.values[lost_mask] = post_lumen[lost_mask]
                        raw_post_peak.values[lost_mask] = post_lumen[lost_mask]

                        post_cortex_mean = raw_post_mean.rolling(
                            window=3, center=True, min_periods=1).mean().values
                        post_cortex_peak = raw_post_peak.rolling(
                            window=3, center=True, min_periods=1).mean().values

                        # Degradation = fractional drop below the normalised baseline (1.0)
                        min_post_mean = float(np.nanmin(post_cortex_mean))
                        min_post_peak = float(np.nanmin(post_cortex_peak))
                        deg_mean_pct  = (1.0 - min_post_mean) * 100
                        deg_peak_pct  = (1.0 - min_post_peak) * 100

                        # Disassembly half-life: time to reach 50 % of max drop
                        if deg_mean_pct > 10:
                            half_drop_val = 1.0 - 0.5 * (1.0 - min_post_mean)
                            cross_idx = np.where(post_cortex_mean < half_drop_val)[0]
                            if len(cross_idx) > 0:
                                actin_half_life = float(post_t[cross_idx[0]])

            # Keep flat/non-responding GUVs — they are valid biology (sub-threshold).
            # Only exclude tracking failures, ruptures, and unphysical fits.
            valid_for_stats = (not is_ruptured) and (not is_shrinking) and (not is_bad_fit)

            all_guvs.append({
                'experiment':    exp_name,
                'condition':     meta['condition'],
                'voltage_v':     meta['voltage_v'],
                'pulse_dur_us':  meta['pulse_dur_us'],
                'ep_condition':  meta['ep_condition'],
                'guv_id':        guv_id,
                'radius_um':     row['radius_um'],
                'size_group':    _size_group(row['radius_um']),
                'tau':           row['tau'],
                'I0':            row['I0'],
                'Iinf':          row['Iinf'],
                'r_squared':     r2,
                'fit_amplitude': fit_amplitude,
                'is_responding': is_responding,
                'is_bad_fit':    is_bad_fit,
                'is_ruptured':   is_ruptured,
                'is_shrinking':  is_shrinking,
                'has_cortex':    has_cortex,
                'valid_for_stats': valid_for_stats,
                'max_circ_drop': max_circ_drop,
                'lumenal_actin_intensity': lumenal_actin_intensity,
                'pre_fwhm_um':             pre_fwhm,
                'pre_gini':                pre_gini,
                'degradation_mean_pct':    deg_mean_pct,
                'degradation_peak_pct':    deg_peak_pct,
                'actin_half_life':         actin_half_life,
            })

    df = pd.DataFrame(all_guvs)
    if not df.empty and 'voltage_v' in df.columns:
        df = df.sort_values(['voltage_v', 'pulse_dur_us'], na_position='last')
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SIZE_ORDER = ["Small (<5\u00b5m)", "Medium (5-10\u00b5m)", "Large (10-15\u00b5m)", "Extra Large (>15\u00b5m)"]


def _ep_order(df: pd.DataFrame) -> list:
    return (
        df[['ep_condition', 'voltage_v', 'pulse_dur_us']]
        .drop_duplicates()
        .sort_values(['voltage_v', 'pulse_dur_us'], na_position='last')
        ['ep_condition']
        .tolist()
    )


def _resolve_palette(palette, hue_order):
    """
    Convert a dict palette to a list keyed by hue_order, or build a list
    from a named colormap.  This avoids the seaborn UnboundLocalError that
    fires when a dict palette contains keys not present in the current data
    subset (a known bug in seaborn ≥0.13 with older-style dict palettes).
    """
    n = max(len(hue_order), 1)  # guard against empty list → 0-colour palette
    if isinstance(palette, dict):
        resolved = [palette[h] for h in hue_order if h in palette]
        # Fall back to a generic palette if nothing matched
        return resolved if resolved else sns.color_palette('tab10', n_colors=n)
    if isinstance(palette, str):
        return sns.color_palette(palette, n_colors=n)
    return palette  # already a list/sequence


def _strip_box(ax, data, x, y, hue=None, order=None, hue_order=None,
               palette='tab10', ylabel='', title=''):
    """Combined boxplot + stripplot, deduplicating the legend."""

    def _no_data(msg='No data'):
        ax.text(0.5, 0.5, msg, ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('')

    # Drop rows where y is NaN — seaborn ≥0.13 crashes when all y-values
    # within a hue group are NaN (leaves boxprops uninitialised).
    data = data.dropna(subset=[y])

    if data.empty:
        _no_data()
        return

    # Filter order/hue_order to only levels present in the cleaned data.
    if order is not None:
        order = [o for o in order if o in data[x].values]
    if hue_order is not None and hue is not None:
        hue_order = [h for h in hue_order if h in data[hue].values]

    # Guard: if filtering left hue_order empty seaborn would crash.
    if hue_order is not None and len(hue_order) == 0:
        _no_data()
        return

    # Resolve palette to a plain list so seaborn never sees a dict with
    # missing keys (triggers UnboundLocalError in seaborn ≥0.13)
    pal = _resolve_palette(palette, hue_order if hue_order is not None else [])

    sns.boxplot(data=data, x=x, y=y, hue=hue,
                order=order, hue_order=hue_order,
                palette=pal, width=0.55, fliersize=0,
                linewidth=0.9, ax=ax)
    sns.stripplot(data=data, x=x, y=y, hue=hue,
                  order=order, hue_order=hue_order,
                  palette=pal, dodge=True, alpha=0.55,
                  size=4, jitter=True, ax=ax)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel('')
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='x', labelrotation=25, labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    n = len(handles) // 2
    if n > 0:
        ax.legend(handles[:n], labels[:n], fontsize=7,
                  title=hue, title_fontsize=7)
    else:
        ax.legend().remove() if ax.get_legend() else None
    ax.grid(axis='y', alpha=0.25)


def plot_analysis(df: pd.DataFrame, bulk_out_dir: str):
    df_valid      = df[df['valid_for_stats']].copy()
    df_responding = df_valid[df_valid['is_responding']].copy()

    if df_valid.empty:
        print("[WARNING] No valid GUVs — no plots generated.")
        return

    ep_order  = _ep_order(df_valid)
    cond_pal  = {'BranchedCortex': '#E76F51', 'Empty': '#457B9D', 'Unknown': '#888888'}

    # -----------------------------------------------------------------------
    # FIGURE 1 – Dye kinetics vs EP parameters
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Dye Kinetics \u2014 EP Parameter Comparison', fontsize=12, y=1.01)

    _strip_box(axes1[0, 0], df_responding,
               x='ep_condition', y='tau', hue='condition',
               order=ep_order, hue_order=['BranchedCortex', 'Empty'],
               palette=cond_pal,
               ylabel='\u03c4 (s)', title='Uptake \u03c4 vs EP condition')

    df_responding['pulse_label'] = df_valid['pulse_dur_us'].apply(
        lambda x: f"{int(x)}\u00b5s" if pd.notnull(x) else 'Unknown')
    df_responding['volt_label'] = df_valid['voltage_v'].apply(
        lambda x: f"{int(x)}V" if pd.notnull(x) else 'Unknown')
    volt_strs  = sorted(df_responding['volt_label'].unique().tolist(),
                        key=lambda s: int(re.search(r'\d+', s).group()))
    pulse_strs = sorted(df_responding['pulse_label'].unique().tolist(),
                        key=lambda s: int(re.search(r'\d+', s).group()))

    _strip_box(axes1[0, 1], df_responding,
               x='volt_label', y='tau', hue='pulse_label',
               order=volt_strs, hue_order=pulse_strs,
               palette='viridis',
               ylabel='\u03c4 (s)', title='Uptake \u03c4: voltage \u00d7 pulse duration')

    # Rupture rate per EP condition (uses full df, not just valid)
    rupt_rate = (df.groupby('ep_condition')['is_ruptured']
                   .agg(['sum', 'count'])
                   .rename(columns={'sum': 'ruptured', 'count': 'total'}))
    rupt_rate['rate'] = rupt_rate['ruptured'] / rupt_rate['total'] * 100
    ep_in_rupt = [e for e in ep_order if e in rupt_rate.index]
    rupt_rate  = rupt_rate.reindex(ep_in_rupt)
    colors = ['#E76F51' if 'Branched' in str(e) else '#457B9D' for e in ep_in_rupt]
    axes1[1, 0].bar(range(len(rupt_rate)), rupt_rate['rate'],
                    color=colors, alpha=0.8, edgecolor='white')
    axes1[1, 0].set_xticks(range(len(rupt_rate)))
    axes1[1, 0].set_xticklabels(ep_in_rupt, rotation=25, ha='right', fontsize=8)
    axes1[1, 0].set_ylabel('Rupture rate (%)', fontsize=9)
    axes1[1, 0].set_title('Rupture rate per EP condition', fontsize=9)
    axes1[1, 0].grid(axis='y', alpha=0.25)
    for i, (idx, r) in enumerate(rupt_rate.iterrows()):
        axes1[1, 0].text(i, r['rate'] + 0.5, f"n={int(r['total'])}",
                         ha='center', fontsize=7)

    for ep in ep_order:
        sub = df_responding[df_responding['ep_condition'] == ep]
        axes1[1, 1].scatter(sub['radius_um'], sub['tau'], label=ep, alpha=0.65, s=30)
    axes1[1, 1].set_xlabel('Radius (\u00b5m)', fontsize=9)
    axes1[1, 1].set_ylabel('\u03c4 (s)', fontsize=9)
    axes1[1, 1].set_title('\u03c4 vs GUV size, coloured by EP condition', fontsize=9)
    axes1[1, 1].legend(fontsize=7, title='EP condition', title_fontsize=7)
    axes1[1, 1].grid(alpha=0.25)

    fig1.tight_layout()
    fig1.savefig(os.path.join(FOLDER_DYE, "bulk_kinetics_ep_comparison.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {FOLDER_DYE}/bulk_kinetics_ep_comparison.png")

    # -----------------------------------------------------------------------
    # FIGURE 2 – Morphology
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Morphology \u2014 EP Parameter Comparison', fontsize=12, y=1.01)

    _strip_box(axes2[0, 0], df_valid,
               x='ep_condition', y='max_circ_drop', hue='condition',
               order=ep_order, hue_order=['BranchedCortex', 'Empty'],
               palette=cond_pal,
               ylabel='Max circularity drop',
               title='Membrane deformation vs EP condition')

    _strip_box(axes2[0, 1], df_valid,
               x='size_group', y='tau', hue='ep_condition',
               order=SIZE_ORDER, hue_order=ep_order,
               palette='tab10',
               ylabel='\u03c4 (s)', title='\u03c4 vs GUV size, split by EP condition')

    v_min = df_valid['voltage_v'].min()
    v_max = df_valid['voltage_v'].max()
    sc = axes2[1, 0].scatter(df_valid['max_circ_drop'], df_valid['tau'],
                              c=df_valid['voltage_v'], cmap='plasma',
                              vmin=v_min, vmax=v_max, alpha=0.65, s=30)
    fig2.colorbar(sc, ax=axes2[1, 0], label='Voltage (V)')
    axes2[1, 0].set_xlabel('Max circularity drop', fontsize=9)
    axes2[1, 0].set_ylabel('\u03c4 (s)', fontsize=9)
    axes2[1, 0].set_title('\u03c4 vs membrane deformation (coloured by voltage)', fontsize=9)
    axes2[1, 0].grid(alpha=0.25)

    # Only show has_cortex panel if actin data exists for at least some GUVs
    if df_valid['has_cortex'].any():
        _strip_box(axes2[1, 1], df_valid,
                   x='has_cortex', y='tau', hue='ep_condition',
                   order=[False, True], hue_order=ep_order,
                   palette='tab10',
                   ylabel='\u03c4 (s)', title='\u03c4: lumenal actin only vs cortex present')
        axes2[1, 1].set_xticks(axes2[1, 1].get_xticks())
        axes2[1, 1].set_xticks(axes2[1, 1].get_xticks())
        axes2[1, 1].set_xticklabels(['Lumenal actin only', 'Cortex present'])
    else:
        axes2[1, 1].text(0.5, 0.5, 'No actin data available',
                         ha='center', va='center', transform=axes2[1, 1].transAxes,
                         fontsize=10, color='gray')
        axes2[1, 1].set_title('\u03c4: lumenal actin only vs cortex present', fontsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(FOLDER_MORPH, "bulk_morphology_ep_comparison.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {FOLDER_MORPH}/bulk_morphology_ep_comparison.png")

    # -----------------------------------------------------------------------
    # FIGURE 3 – Actin cortex (BranchedCortex GUVs with detected cortex only)
    # -----------------------------------------------------------------------
    df_cortex = df_valid[df_valid['has_cortex']].copy()

    if df_cortex.empty:
        print("No GUVs with detected cortex \u2014 skipping Figure 3.")
    else:
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
        fig3.suptitle('Actin Cortex \u2014 EP Parameter Comparison', fontsize=12, y=1.01)

        _strip_box(axes3[0, 0], df_cortex,
                   x='ep_condition', y='degradation_mean_pct', hue='condition',
                   order=ep_order, hue_order=['BranchedCortex', 'Empty'],
                   palette=cond_pal,
                   ylabel='Max loss (%)', title='Cortex mean degradation vs EP condition')

        _strip_box(axes3[0, 1], df_cortex,
                   x='ep_condition', y='degradation_peak_pct', hue='condition',
                   order=ep_order, hue_order=['BranchedCortex', 'Empty'],
                   palette=cond_pal,
                   ylabel='Max loss (%)', title='Cortex peak degradation vs EP condition')

        for ep in ep_order:
            sub = df_cortex[df_cortex['ep_condition'] == ep]
            axes3[0, 2].scatter(sub['degradation_mean_pct'], sub['degradation_peak_pct'],
                                label=ep, alpha=0.65, s=30)
        valid_deg = df_cortex[['degradation_mean_pct', 'degradation_peak_pct']].dropna()
        if not valid_deg.empty:
            lim = valid_deg.max().max() * 1.05
            axes3[0, 2].plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4, label='1:1')
        axes3[0, 2].set_xlabel('Mean degradation (%)', fontsize=9)
        axes3[0, 2].set_ylabel('Peak degradation (%)', fontsize=9)
        axes3[0, 2].set_title('Mean vs peak cortex loss', fontsize=9)
        axes3[0, 2].legend(fontsize=7)
        axes3[0, 2].grid(alpha=0.25)

        _strip_box(axes3[1, 0], df_cortex,
                   x='ep_condition', y='pre_fwhm_um', hue='condition',
                   order=ep_order, hue_order=['BranchedCortex', 'Empty'],
                   palette=cond_pal,
                   ylabel='FWHM (\u00b5m)', title='Pre-pulse cortex thickness vs EP condition')

        _strip_box(axes3[1, 1], df_cortex,
                   x='ep_condition', y='pre_gini', hue='condition',
                   order=ep_order, hue_order=['BranchedCortex', 'Empty'],
                   palette=cond_pal,
                   ylabel='Gini index',
                   title='Pre-pulse cortex heterogeneity vs EP condition')

        for ep in ep_order:
            sub = df_cortex[df_cortex['ep_condition'] == ep].dropna(
                subset=['actin_half_life', 'tau'])
            if sub.empty:
                continue
            axes3[1, 2].scatter(sub['actin_half_life'], sub['tau'],
                                label=ep, alpha=0.65, s=30)
        valid_kin = df_cortex[['actin_half_life', 'tau']].dropna()
        max_val = valid_kin.max().max() if not valid_kin.empty else 1.0
        axes3[1, 2].plot([0, max_val], [0, max_val], 'k--', lw=0.8, alpha=0.4, label='1:1')
        axes3[1, 2].set_xlabel('Actin disassembly half-life (s)', fontsize=9)
        axes3[1, 2].set_ylabel('Dye uptake \u03c4 (s)', fontsize=9)
        axes3[1, 2].set_title('Kinetic coupling: disassembly vs influx', fontsize=9)
        axes3[1, 2].legend(fontsize=7)
        axes3[1, 2].grid(alpha=0.25)

        fig3.tight_layout()
        fig3.savefig(os.path.join(FOLDER_ACTIN, "bulk_actin_ep_comparison.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved: {FOLDER_ACTIN}/bulk_actin_ep_comparison.png")

    plt.show()


# ---------------------------------------------------------------------------
# Multigrid dye-efflux overview
# ---------------------------------------------------------------------------

def plot_dye_multigrid(base_dir: str, bulk_out_dir: str):
    """
    One panel per (condition × ep_condition) combination, styled like the
    per-experiment dye_summary plot:
      · Individual GUV traces — grey (survived) or red (ruptured)
      · Bold black population mean
      · Light-grey ±1 SD band
      · Crimson dashed pulse line at t = 0

    Traces are loaded directly from each experiment's
    *_normalized_curves.csv and *_detection_quality.csv files so no
    re-running of the main pipeline is needed.

    Panels are sorted by voltage (rows) × actin condition (columns) so
    BranchedCortex and Empty sit side-by-side at each voltage.
    """
    # ── Collect data ──────────────────────────────────────────────────────
    # Each entry: {ep_condition, condition, voltage_v, pulse_dur_us,
    #              t, curves (n×T), ruptured_ids (set of str)}
    panel_data = {}   # key = (voltage_v, pulse_dur_us, condition)

    for root, dirs, files in os.walk(base_dir):
        if "Bulk_Summary" in root:
            continue
        if "dye" not in dirs or "tracking" not in dirs:
            continue

        exp_name = os.path.basename(root)
        meta     = parse_experiment_name(exp_name)

        curve_file   = glob.glob(os.path.join(root, "dye",      "*_normalized_curves.csv"))
        quality_file = glob.glob(os.path.join(root, "tracking", "*_detection_quality.csv"))
        if not curve_file or not quality_file:
            continue

        df_curves = pd.read_csv(curve_file[0])
        df_qual   = pd.read_csv(quality_file[0])

        if 'Time (s)' not in df_curves.columns:
            continue

        t = df_curves['Time (s)'].values

        # Collect ruptured GUV ids
        ruptured_ids = set(
            str(int(r['guv_id']))
            for _, r in df_qual.iterrows()
            if bool(r.get('failed', False))
        )

        # Extract individual GUV columns (all *_I_uptake columns)
        guv_cols = [c for c in df_curves.columns if c.endswith('_I_uptake')]
        if not guv_cols:
            continue

        curves = df_curves[guv_cols].values.T.astype(float)  # shape (n_guvs, n_frames)

        key = (meta['voltage_v'], meta['pulse_dur_us'], meta['condition'])
        if key not in panel_data:
            panel_data[key] = {
                'ep_condition': meta['ep_condition'],
                'condition':    meta['condition'],
                'voltage_v':    meta['voltage_v'],
                'pulse_dur_us': meta['pulse_dur_us'],
                't':            t,
                'curves':       [],
                'ruptured_ids': set(),
                'guv_cols_per_exp': [],
            }

        # Offset ruptured ids by the running GUV count so they stay unique
        offset = sum(len(c) for c in panel_data[key]['curves'])
        for i, col in enumerate(guv_cols):
            gid = re.search(r'GUV_(\d+)', col)
            if gid and gid.group(1) in ruptured_ids:
                panel_data[key]['ruptured_ids'].add(offset + i)

        # Store (t, curve) pairs so curves from different experiments
        # with different frame counts can be pooled without shape mismatch.
        for curve in curves:
            panel_data[key]['curves'].append((t, curve))

    if not panel_data:
        print("[multigrid] No normalized_curves files found — skipping.")
        return

    # ── Layout: rows = voltage (ascending), cols = condition order ────────
    COND_ORDER = ['BranchedCortex', 'Empty', 'Unknown']

    voltages   = sorted({k[0] for k in panel_data if k[0] is not None})
    conditions = [c for c in COND_ORDER
                  if any(k[2] == c for k in panel_data)]

    ncols = len(conditions)
    nrows = len(voltages)

    if nrows == 0 or ncols == 0:
        print("[multigrid] Nothing to plot.")
        return

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
        sharey=False,
    )

    COND_TITLE = {'BranchedCortex': 'Branched cortex', 'Empty': 'Empty', 'Unknown': 'Unknown'}
    C_SURV = '#888888'
    C_RUPT = '#CC2222'
    C_SD   = '#BBBBBB'

    for row_i, volt in enumerate(voltages):
        for col_i, cond in enumerate(conditions):
            ax = axes[row_i][col_i]

            # Find the matching panel (pulse duration assumed uniform per volt)
            matching = [v for k, v in panel_data.items()
                        if k[0] == volt and k[2] == cond]

            if not matching:
                ax.set_visible(False)
                continue

            pd_entry  = matching[0]
            rupt_set  = pd_entry['ruptured_ids']
            ep_label  = pd_entry['ep_condition']
            raw_pairs = pd_entry['curves']   # list of (t_i, curve_i)

            # Build a common time axis covering the union of all experiments
            # in this panel, then interpolate every curve onto it so curves
            # from runs with different frame counts can be stacked safely.
            t_min = min(tc[0].min() for tc in raw_pairs)
            t_max = max(tc[0].max() for tc in raw_pairs)
            n_pts = max(len(tc[0]) for tc in raw_pairs)
            t     = np.linspace(t_min, t_max, n_pts)
            all_curves = np.array(
                [np.interp(t, tc[0], tc[1], left=np.nan, right=np.nan)
                 for tc in raw_pairs],
                dtype=float,
            )  # shape (n_guvs, n_pts)
            n_guvs = len(all_curves)

            # Survivor mask: indices NOT in rupt_set
            surv_mask   = np.array([i not in rupt_set for i in range(n_guvs)])
            surv_curves = all_curves[surv_mask]

            # Plot individual traces — survivors only
            for idx, curve in enumerate(all_curves):
                if idx in rupt_set:
                    continue
                ax.plot(t, curve, color=C_SURV, lw=0.8, alpha=0.55)

            # Mean +/- SD -- survivors only.
            # Mask out time points where fewer than 30% of survivors contribute
            # (interpolation edge effects cause wild oscillations there).
            if len(surv_curves) > 0:
                min_n = max(1, int(np.ceil(len(surv_curves) * 0.30)))
                n_valid_per_t = np.sum(np.isfinite(surv_curves), axis=0)
                mean_curve = np.where(n_valid_per_t >= min_n,
                                      np.nanmean(surv_curves, axis=0), np.nan)
                sd_curve   = np.where(n_valid_per_t >= min_n,
                                      np.nanstd(surv_curves,  axis=0), np.nan)
                ax.fill_between(t,
                                mean_curve - sd_curve,
                                mean_curve + sd_curve,
                                color=C_SD, alpha=0.50, zorder=1)
                ax.plot(t, mean_curve, color='black', lw=2.5, zorder=2,
                        label='Mean (survivors)')

            # Reference lines
            ax.axvline(0,   color='crimson', ls='--', lw=1.5, zorder=3)
            ax.axhline(1.0, color='gray',    ls=':',  lw=0.8)

            # Counts
            n_rupt = len(rupt_set)
            n_surv = n_guvs - n_rupt

            ax.set_title(
                f"{ep_label}  ·  {COND_TITLE.get(cond, cond)}\n"
                f"n={n_guvs}  (survived={n_surv}, ruptured={n_rupt})",
                fontsize=9,
            )
            ax.set_xlabel('Time relative to pulse (s)', fontsize=8)
            ax.set_ylabel('Normalised dye intensity (a.u.)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.20)

            # Clip y-axis so a few wild outlier traces don't crush the mean
            finite_vals = all_curves[np.isfinite(all_curves)]
            if len(finite_vals) > 0:
                p_lo = max(-1.0, np.percentile(finite_vals, 1))
                p_hi = min(np.percentile(finite_vals, 99) * 1.10, 20.0)
                ax.set_ylim(p_lo, p_hi)

    # Column headers (condition label at the top of each column)
    for col_i, cond in enumerate(conditions):
        axes[0][col_i].set_title(
            axes[0][col_i].get_title(),   # keep existing title
            fontsize=9,
        )

    plt.suptitle('Dye efflux overview — all EP conditions', fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(FOLDER_DYE, "bulk_dye_multigrid.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FOLDER_DYE}/bulk_dye_multigrid.png")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_summary = process_bulk_data(BASE_DIR, BULK_OUT_DIR)

    if df_summary.empty:
        print("No data loaded — check BASE_DIR and folder structure.")
    else:
        out_csv = os.path.join(BULK_OUT_DIR, "bulk_analysis_summary.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"Saved summary CSV \u2192 {out_csv}")
        n_total      = len(df_summary)
        n_valid      = df_summary['valid_for_stats'].sum()
        n_responding = df_summary['is_responding'].sum()
        n_ruptured   = df_summary['is_ruptured'].sum()
        n_shrink     = df_summary['is_shrinking'].sum()
        n_bad_fit    = df_summary['is_bad_fit'].sum()
        print(f"Total GUVs: {n_total}  |  Valid: {n_valid}  |  Responding: {n_responding}")
        print(f"  Excluded — ruptured:             {n_ruptured}")
        print(f"  Excluded — shrinking (>50%):     {n_shrink}")
        print(f"  Excluded — bad fit (unphysical): {n_bad_fit}")
        print(f"  Valid non-responding (flat):     {n_valid - n_responding}")
        print(f"  Valid responding (real efflux):  {n_responding}")
        amp  = df_summary['fit_amplitude']
        tau  = df_summary['tau']
        print(f"  Amplitude (I0-Iinf)/I0 — median: {amp.median():.3f},  mean: {amp.mean():.3f},  p25: {amp.quantile(0.25):.3f},  p75: {amp.quantile(0.75):.3f}")
        print(f"  tau (s)               — median: {tau.median():.1f},  p25: {tau.quantile(0.25):.1f},  p75: {tau.quantile(0.75):.1f},  at_boundary(>=4999): {(tau >= 4999).sum()}")
        print(f"  Bad fit breakdown: tau>=4999: {(df_summary['tau']>=4999).sum()},  I0>2: {(df_summary['I0']>2).sum()},  I0<=0: {(df_summary['I0']<=0).sum()},  Iinf<-0.20: {(df_summary['Iinf']<-0.20).sum()}")
        print("\nEP conditions found:")
        print(df_summary.groupby(['ep_condition', 'condition'])['guv_id']
                        .count().rename('n_guvs').to_string())
        plot_dye_multigrid(BASE_DIR, BULK_OUT_DIR)
        plot_analysis(df_summary, BULK_OUT_DIR)