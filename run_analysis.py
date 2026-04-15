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
   - Detect dye-entry jump.
4. Normalise & align curves (t = 0 at jump).
5. Fit kinetic model to the population average.
6. Export plots, tracking data, and per-GUV CSVs.
"""

import glob
import os
import json
import logging
import multiprocessing
import sys
from functools import partial
from natsort import natsorted

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.optimize as sc
import pandas as pd

import config as cfg
import guv_analysis_utils as utils


# -------------------------------------------------------------------
# --- 1. GUV CIRCLE DEFINITIONS  (replaces CSV loading)
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

    return {
        'dye_files':      dye_files,
        'roi_files':      roi_files,
        'time_array':     time_array,
        'frame_interval': frame_interval,
        'circles':        circles,
    }


# -------------------------------------------------------------------
# --- 3. PARALLEL GUV PROCESSING ---
# -------------------------------------------------------------------

def process_all_guvs(circles: list[dict],
                     dye_files: list,
                     roi_files: list,
                     logger: logging.Logger) -> dict:
    """
    Runs process_single_guv for each circle using multiprocessing.
    """
    process_func = partial(
        utils.process_single_guv,
        roi_files=roi_files,
        dye_files=dye_files,
    )

    tasks = [
        (str(c['id']),          # guv_id
         (c['x'], c['y']),      # initial_center
         c['r'],                 # initial_radius  (already in pixels, no /2)
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
    all_jumps      = []
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
            all_jumps     .append(result['jump_frame'])
            valid_indices .append(i)

            tr = result.get('tracking', {})
            for fi, (c, r, s) in enumerate(zip(
                    tr.get('centers', []),
                    tr.get('radii', []),
                    tr.get('ring_scores', []))):
                if c is not None and r is not None:
                    tracking_rows.append({
                        'guv_id': gid, 'frame': fi,
                        'x': c[0], 'y': c[1],
                        'radius': r, 'ring_score': round(s, 2)
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
    if tracking_rows and getattr(cfg, 'EXPORT_TRACKING_DATA', True):
        pd.DataFrame(tracking_rows).to_csv(
            os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                         f"{cfg.EXPERIMENT_BASE_NAME}_tracking_data.csv"),
            index=False
        )
        logger.info("Saved per-frame tracking data.")

    return {
        'all_intensity_curves':  all_intensity,
        'all_background_curves': all_background,
        'all_jump_frames':       all_jumps,
        'valid_guv_indices':     valid_indices,
    }


# -------------------------------------------------------------------
# --- 4. NORMALISATION & ALIGNMENT ---
# -------------------------------------------------------------------

def normalize_and_align_curves(guv_results: dict,
                                time_array: np.ndarray,
                                circles: list[dict],
                                logger: logging.Logger) -> dict | None:
    """
    Normalises: I_uptake = (I_dye − I₀) / (I_bg − I₀)
    Aligns:     t = 0 at the median jump frame.
    Uses nanmean/nanmedian so post-rupture NaN frames are excluded.
    """
    logger.info("Normalising intensity curves ...")

    normalised = []
    for dye, bg, jf in zip(guv_results['all_intensity_curves'],
                            guv_results['all_background_curves'],
                            guv_results['all_jump_frames']):
        baseline_end = max(cfg.MIN_BASELINE_FRAMES, jf)
        if baseline_end >= len(dye):
            continue
        I0   = float(np.nanmean(dye[:baseline_end]))
        num  = dye - I0
        den  = bg  - I0
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.where(np.isfinite(den) & (den != 0), num / den, np.nan)
        normalised.append(norm)

    if not normalised:
        logger.error("Normalisation failed for all GUVs.")
        return None

    jumps = np.array(guv_results['all_jump_frames'])
    nz    = jumps[jumps > 0]
    med_j = int(np.median(nz)) if len(nz) > 0 else 0
    med_j = min(med_j, len(time_array) - 1)
    logger.info(f"Median jump frame: {med_j}")

    arr   = np.array(normalised)[:, med_j:]
    t_al  = time_array[med_j:] - time_array[med_j]
    avg   = np.nanmean(arr, axis=0)

    valid_ids = [str(circles[i]['id']) for i in guv_results['valid_guv_indices']]

    return {
        't_aligned':          t_al,
        'average_curve':      avg,
        'all_curves_aligned': arr,
        'valid_guv_ids':      valid_ids,
        'median_jump_frame':  med_j,
    }


# -------------------------------------------------------------------
# --- 5. FITTING ---
# -------------------------------------------------------------------

def fit_average_curve(t: np.ndarray, avg: np.ndarray,
                      logger: logging.Logger) -> dict:

    logger.info(f"Fitting model: {cfg.MODEL_TO_USE}")

    end = min(int(round(len(t) * cfg.FIT_DATA_PERCENTAGE)), len(t))
    t_f = t[:end]
    y_f = avg[:end]

    ok = np.isfinite(y_f)
    t_c, y_c = t_f[ok], y_f[ok]

    if len(t_c) < 5:
        logger.error("Too few valid points for fitting.")
        p0  = (cfg.FIT_INITIAL_GUESS_4PARAM if cfg.MODEL_TO_USE == '4-PARAM'
               else cfg.FIT_INITIAL_GUESS_5PARAM)
        fc  = (utils.dyn_model_4param(t, *p0) if cfg.MODEL_TO_USE == '4-PARAM'
               else utils.dyn_model_5param(t, *p0))
        names = (['I_offset','A','tau','D'] if cfg.MODEL_TO_USE == '4-PARAM'
                 else ['Af','A1','tau1','A2','tau2'])
        return {'params': dict(zip(names, p0)), 'fit_curve': fc,
                'fit_failed': True, 'fit_slice_index': end}

    mn, mx = y_c.min(), y_c.max()
    A_est  = mx - mn
    try:
        idx63 = np.argmax(y_c > mn + 0.63 * A_est)
        tau_e = t_c[idx63] if idx63 > 0 else t_c[-1] / 2
    except Exception:
        tau_e = t_c[-1] / 5
    tau_e = max(tau_e, 1.0)

    if cfg.MODEL_TO_USE == '5-PARAM':
        fn   = utils.dyn_model_5param
        p0   = (mx, A_est*0.5, tau_e*0.5, A_est*0.5, tau_e*2)
        bnds = ((-1,-10,0.01,-10,0.01),(10,10,5000,10,5000))
        names = ['Af','A1','tau1','A2','tau2']
    else:
        fn   = utils.dyn_model_4param
        p0   = (mn, A_est, tau_e, 0.0)
        bnds = ((-2,-0.2,0.01,-0.1),(2,10,5000,0.1))
        names = ['I_offset','A','tau','D']

    try:
        params, _ = sc.curve_fit(fn, t_c, y_c, p0=p0, bounds=bnds, maxfev=10000)
        failed = False
    except RuntimeError as e:
        logger.error(f"Fitting failed: {e}")
        params = np.array(p0); failed = True

    fc = fn(t, *params)
    logger.info(f"Fit: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, params)))

    return {'params': dict(zip(names, params)), 'fit_curve': fc,
            'fit_failed': failed, 'fit_slice_index': end}


# -------------------------------------------------------------------
# --- 6. EXPORT ---
# -------------------------------------------------------------------

def export_results(aligned: dict, fit: dict,
                   dye_files: list, logger: logging.Logger):

    t    = aligned['t_aligned']
    avg  = aligned['average_curve']
    arr  = aligned['all_curves_aligned']
    ids  = aligned['valid_guv_ids']
    mjf  = aligned['median_jump_frame']
    fc   = fit['fit_curve']
    fp   = fit['fit_params'] if 'fit_params' in fit else fit['params']
    fsi  = fit['fit_slice_index']
    fail = fit['fit_failed']

    dye_al = dye_files[mjf:]

    # Snapshot frames
    for i, tp in enumerate(cfg.EXPORT_TIME_POINTS_S):
        fi, _ = utils.find_closest_frame(t, tp)
        if fi >= len(dye_al): continue
        img = cv2.imread(dye_al[fi], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        styled = utils.style_image(img, f"{int(round(tp))} S",
                                   cfg.MICRONS_PER_PIXEL, cfg.SCALE_BAR_LENGTH_MICRONS)
        cv2.imwrite(os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                    f"frame_{i+1}_at_{int(round(tp))}s.png"), styled)

    # Kinetic plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in arr:
        ax.plot(t[:len(curve)], np.where(np.isfinite(curve), curve, np.nan),
                color='gray', alpha=0.2, lw=0.8)
    ax.plot(t, avg, 'r.', ms=3, label=f'Average (n={len(arr)})')
    ax.plot(t, fc,  'k-', lw=2, label='Fit')
    if 0 < fsi <= len(t):
        ax.plot(t[fsi-1], avg[fsi-1], 'b*', ms=10, label='End of fit')

    if cfg.MODEL_TO_USE == '5-PARAM':
        title = (f"$A_f={fp['Af']:.2f}$  $A_1={fp['A1']:.2f}$  "
                 f"$\\tau_1={fp['tau1']:.1f}$ s  $A_2={fp['A2']:.2f}$  "
                 f"$\\tau_2={fp['tau2']:.1f}$ s")
    else:
        title = (f"$I_{{off}}={fp['I_offset']:.2f}$  $A={fp['A']:.2f}$  "
                 f"$\\tau={fp['tau']:.1f}$ s  $D={fp['D']:.4f}$")
    if fail: title = "FIT FAILED — Initial Guess\n" + title

    ax.set_title(title); ax.set_xlabel("Time (s) from pulse")
    ax.set_ylabel(r"$I_{uptake}$"); ax.legend(loc='lower right')
    plt.grid(True, alpha=0.4)
    fig.savefig(os.path.join(cfg.OUTPUT_IMAGE_FOLDER,
                f"{cfg.EXPERIMENT_BASE_NAME}_kinetic_fit.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Normalised curve CSV
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
        (cfg.MODEL_TO_USE in ('4-PARAM','5-PARAM'),  "MODEL_TO_USE must be '4-PARAM' or '5-PARAM'"),
        (0.05 <= cfg.JUMP_THRESHOLD_PERCENT <= 0.90, "JUMP_THRESHOLD_PERCENT out of range"),
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
    logger.info("=== GUV Analysis Pipeline (ring-score tracking, no CSV) ===")

    if not validate_config(logger):
        return

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
            data['circles'], data['dye_files'], data['roi_files'], logger
        )
        
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
        fit = fit_average_curve(aligned['t_aligned'], aligned['average_curve'], logger)

        # Step 6 – export
        export_results(aligned, fit, data['dye_files'], logger)

        logger.info("=== Pipeline finished successfully ===")

    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)


if __name__ == "__main__":
    main()