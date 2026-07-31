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
import warnings
import scipy.optimize as sc
from scipy.stats import norm as _scipy_norm


def _norm_ppf(q: float) -> float:
    """Gaussian quantile function; wrapped so the import stays local to use."""
    return float(_scipy_norm.ppf(q))
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
            idx_mem = cfg.ND2_CHANNEL_IDX_MEMBRANE   # no default: 0 would be ACTIN
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

            idx_mem = cfg.ND2_CHANNEL_IDX_MEMBRANE   # no default: 0 would be ACTIN
            idx_dye = cfg.ND2_CHANNEL_IDX_DYE
            idx_act = cfg.ND2_CHANNEL_IDX_ACTIN      # old default of 2 was MEMBRANE

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

            # Empirical sanity check on the channel assignment: warns if the
            # channel nominated as MEMBRANE is not the one with the strongest
            # ring-like structure. Catches a transposed acquisition order
            # before it costs a full run.
            if getattr(cfg, 'VALIDATE_CHANNEL_ASSIGNMENT', True):
                try:
                    _probe = {'membrane': np.memmap(roi_mmap_info[0], dtype=roi_mmap_info[2],
                                                    mode='r', shape=roi_mmap_info[1])}
                    if dye_mmap_info[0] is not None:
                        _probe['dye'] = np.memmap(dye_mmap_info[0], dtype=dye_mmap_info[2],
                                                  mode='r', shape=dye_mmap_info[1])
                    if actin_mmap_info[0] is not None:
                        _probe['actin'] = np.memmap(actin_mmap_info[0], dtype=actin_mmap_info[2],
                                                    mode='r', shape=actin_mmap_info[1])
                    utils.check_channel_assignment(_probe, logger)
                    del _probe
                except Exception as _e:
                    logger.warning(f"Channel assignment check skipped: {_e}")

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
        exit_reason = qe.get('exit_reason')

        if rup is not None:
            reason_label = 'OUT OF FRAME (tracking lost, not a rupture)' \
                if exit_reason == 'OUT_OF_FRAME' else 'RUPTURED'
            logger.warning(
                f"  GUV {gid}: {reason_label} at frame {rup}  "
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

    ruptured_ids = {
        str(qe['guv_id'])
        for qe in quality_log
        if qe.get('ruptured_at_frame') is not None and qe.get('exit_reason') == 'RUPTURED'
    }
    out_of_frame_ids = {
        str(qe['guv_id'])
        for qe in quality_log
        if qe.get('ruptured_at_frame') is not None and qe.get('exit_reason') == 'OUT_OF_FRAME'
    }

    logger.info(
        f"Processed {len(all_intensity)} / {len(circles)} GUV(s) successfully."
    )
    if ruptured_ids or out_of_frame_ids:
        logger.info(
            f"  Of these: {len(ruptured_ids)} ruptured (biological), "
            f"{len(out_of_frame_ids)} lost to FOV edge (tracking limitation, "
            f"excluded from rupture statistics)."
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

    return {
        'raw_results':           results,
        'all_intensity_curves':  all_intensity,
        'all_background_curves': all_background,
        'all_actin_data':        all_actin_data,
        'valid_guv_indices':     valid_indices,
        'tracking_dataframe':    df_track,
        'ruptured_guv_ids':      ruptured_ids,      # set of str IDs that genuinely ruptured
        'out_of_frame_guv_ids':  out_of_frame_ids,   # set of str IDs lost to the FOV edge (not a rupture)
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
    Normalises using the efflux-mirrored form of the published SRB uptake
    method (Idye,t - Idye,0)/(Ibackground,t - Idye,0):

        I_retained(t) = (I_bg,t - I_dye,t) / (I_bg,t - I_dye,0)

    Derivation: the published formula measures progress toward equilibrium
    with the background, I_uptake(t) = (I_dye,t - I_dye,0)/(I_bg,t - I_dye,0),
    which runs 0 (baseline) -> 1 (fully equilibrated with the bath). For
    efflux the natural quantity is *retention*, 1 -> 0, so this uses
    1 - I_uptake(t), which simplifies to the expression above. At t=0,
    I_dye,t = I_dye,0 so I_retained = 1; as efflux completes, I_dye,t -> I_bg,t
    so I_retained -> 0 — the same shape the EFFLUX-1EXP/2EXP models expect.

    Unlike the previous (I_dye,t - I_bg,t)/(I_dye,0 - I_bg,0) form, the
    denominator here uses I_bg,t (background AT EACH FRAME) rather than a
    fixed pre-pulse bg0, so it tracks background photobleaching drift over
    the course of the movie instead of assuming it's constant. I_dye,0 is
    the mean of the (already per-frame, mask-averaged) intensity trace over
    the 5 frames immediately preceding the detected pulse_frame — no
    per-pixel statistics involved.

    Frames where I_bg,t sits too close to I_dye,0 for this frame's division
    to be numerically meaningful are set to NaN individually rather than
    discarding the whole GUV.

    Dead-GUV filter
    ----------------
    Separately, a whole GUV is excluded up front if its pre-pulse interior
    signal is not meaningfully separated from background at all:
        |I_dye,0 - I_bg,0|  <  MIN_PREPULSE_SEPARATION_SIGMA * bg_std0
    bg_std0 is the RAW per-pixel background std (not divided by sqrt(n) —
    that SEM-scaled version is what over-excluded everything previously).
    This is a plain "is the average brightness jump at least a couple of
    typical pixel-noise-widths" check, not a significance test on the mean,
    so it doesn't get stricter just because the background ring happens to
    contain more pixels.

    Aligns: t = 0 at *pulse_frame* (auto-detected or overridden in config).
    """
    logger.info(f"Normalising intensity curves (pulse frame = {pulse_frame}) ...")

    dye_list    = guv_results['all_intensity_curves']
    bg_list     = guv_results['all_background_curves']
    bg_std_list = guv_results.get('all_background_std_curves', [None] * len(dye_list))
    bg_n_list   = guv_results.get('all_background_n_curves',   [None] * len(dye_list))
    min_sep_sigma = getattr(cfg, 'MIN_PREPULSE_SEPARATION_SIGMA', None)
    if min_sep_sigma is None:
        logger.info(
            "Dead-GUV filter DISABLED (MIN_PREPULSE_SEPARATION_SIGMA = None): "
            "all GUVs retained, contrast reported only. Calibrate from "
            "{experiment}_dye_qc.csv before enabling — see config notes."
        )

    # Baseline window = the 5 frames immediately preceding the ACTUAL
    # detected/configured pulse_frame — not a hardcoded "first 5 frame
    # indices of the movie". These only coincide when pulse_frame happens
    # to equal 5 (true under the default FRAME_INTERVAL_SCHEDULE, but not
    # guaranteed under PULSE_FRAME_OVERRIDE, a different acquisition
    # schedule, or ordinary auto-detection variation). This matches how
    # normalize_actin_curves already defines its baseline window, so the
    # dye and actin channels are now consistent.
    safe_pre_end   = pulse_frame if pulse_frame > 0 else 1
    safe_pre_start = max(0, safe_pre_end - 5)

    normalised          = []
    kept_local_indices  = []
    excluded_dead_ids    = []
    # QC rows for EVERY GUV, including those the dead-GUV filter rejects.
    # Without this the excluded vesicles never reach any export, so the
    # sep0/sigma distribution that MIN_PREPULSE_SEPARATION_SIGMA is supposed
    # to be calibrated against is invisible — you can only ever see the half
    # that already passed whatever threshold was in force.
    qc_rows: list = []

    for local_i, (dye, bg, bg_std, bg_n) in enumerate(zip(dye_list, bg_list, bg_std_list, bg_n_list)):
        if pulse_frame >= len(dye):
            continue

        dye0 = float(np.nanmean(dye[safe_pre_start:safe_pre_end]))
        bg0  = float(np.nanmean(bg[safe_pre_start:safe_pre_end]))

        # Dead-GUV filter (see docstring): plain amplitude check against the
        # background's own pixel-noise scale, not a sample-size-scaled test.
        orig_idx = guv_results['valid_guv_indices'][local_i]
        gid      = str(circles[orig_idx]['id'])

        bg_std0 = float(np.nanmean(bg_std[safe_pre_start:safe_pre_end])) if bg_std is not None else np.nan
        sep0_all = abs(bg0 - dye0)
        # Pre-pulse contrast expressed in units of the raw per-pixel
        # background std — the quantity MIN_PREPULSE_SEPARATION_SIGMA is
        # compared against, and the one the QC CSV is sorted and quantiled
        # on downstream. Computed for EVERY GUV, before the filter branch,
        # so both the kept and the excluded QC rows can report it: the
        # excluded half is precisely the part of the distribution the
        # threshold needs to be calibrated against.
        # NaN when there is no usable noise scale (bg_std missing or zero),
        # which is also exactly the case in which the dead-GUV filter below
        # cannot fire — so such GUVs are retained by default and show up as
        # NaN in the QC CSV rather than being silently scored as high
        # contrast.
        ratio_all = (sep0_all / bg_std0
                     if np.isfinite(bg_std0) and bg_std0 > 0 else np.nan)

        if min_sep_sigma is not None and np.isfinite(bg_std0) and bg_std0 > 0:
            min_sep = min_sep_sigma * bg_std0
            if sep0_all < min_sep:
                qc_rows.append({'guv_id': gid, 'dye0': dye0, 'bg0': bg0,
                                'bg_std0': bg_std0, 'sep0': sep0_all,
                                'sep0_over_sigma': ratio_all,
                                'kept': False})
                excluded_dead_ids.append(gid)
                logger.warning(
                    f"  GUV {gid}: excluded — pre-pulse interior signal "
                    f"indistinguishable from background "
                    f"(|Idye,0 - Ibg,0| = {sep0_all:.2f} AU < "
                    f"{min_sep_sigma}x background std = {min_sep:.2f} AU, "
                    f"sep0/sigma = {ratio_all:.2f})."
                )
                continue

        den = bg - dye0   # time-varying: I_bg,t - I_dye,0, one value per frame

        # Denominator guard.
        # den is a DIFFERENCE OF TWO SIMILAR NUMBERS: the per-frame background
        # and the pre-pulse lumen level. When the two are within a few counts
        # of each other — which is exactly the low-contrast regime the
        # dead-GUV filter is meant to catch — noise in bg,t can push den
        # through zero, flipping the sign of the normalised value and sending
        # it to +/-infinity. The previous guard was |den| > 1e-9, which only
        # catches exact zero and let those excursions through as real data
        # points (visible as spikes above 1.2 and dips near 0 in the summary
        # plot). Require den to be a genuine multiple of the background noise.
        # The relevant scale is the standard error of the BACKGROUND ESTIMATE,
        # not the per-pixel noise. bg is a percentile of ~700 annulus pixels;
        # its sampling error is
        #     SE = sqrt(p(1-p)/n) / phi(z_p) * sigma
        # which for p=0.15 and n=696 is 0.058*sigma — about 17x smaller than
        # sigma itself. Guarding against per-pixel sigma (as an earlier version
        # of this code did) blanks frames whose denominator is in fact tens of
        # standard errors away from zero, and destroys most of the dataset
        # whenever the contrast happens to be a few sigma.
        k_guard = float(getattr(cfg, 'NORM_DENOM_MIN_SE', 5.0))
        if bg_std is not None:
            sigma_px = np.where(np.isfinite(bg_std) & (bg_std > 0), bg_std, np.nan)
            n_px = np.asarray(bg_n, dtype=float) if bg_n is not None else None
            if n_px is None or not np.any(np.isfinite(n_px)):
                n_px = np.full_like(sigma_px, 100.0)
            n_px = np.where(np.isfinite(n_px) & (n_px > 1), n_px, 100.0)
            # The sampling error of a quantile estimate depends on WHICH
            # quantile was actually used for bg. With BG_LEVEL_ESTIMATOR =
            # 'median' (the default) bg is the annulus median, not the
            # BG_PERCENTILE quantile, so plugging p = 0.15 in here computes
            # the SE of an estimator that was never used — it overstates the
            # true SE by ~20% and makes this guard correspondingly stricter
            # than intended. Match p to the estimator in force.
            if str(getattr(cfg, 'BG_LEVEL_ESTIMATOR', 'median')).lower() == 'median':
                p_q = 0.5
            else:
                p_q = float(getattr(cfg, 'BG_PERCENTILE', 15.0)) / 100.0
            phi_z = float(_scipy_norm.pdf(_norm_ppf(p_q)))
            se_bg = np.sqrt(p_q * (1 - p_q) / n_px) / max(phi_z, 1e-9) * sigma_px
            floor = k_guard * se_bg
        else:
            floor = np.full_like(den, 1e-9, dtype=float)
        floor = np.where(np.isfinite(floor) & (floor > 0), floor, 1e-9)

        with np.errstate(divide='ignore', invalid='ignore'):
            norm = np.where(np.abs(den) > floor, (bg - dye) / den, np.nan)

        n_blanked = int(np.sum(~np.isfinite(norm) & np.isfinite(dye)))
        if n_blanked:
            logger.info(
                f"  GUV {gid}: {n_blanked} frame(s) blanked — "
                f"|I_bg,t - I_dye,0| fell below {k_guard}x the background "
                f"noise, so the normalisation denominator was not resolvable."
            )

        qc_rows.append({'guv_id': gid, 'dye0': dye0, 'bg0': bg0,
                        'bg_std0': bg_std0, 'sep0': sep0_all,
                        'sep0_over_sigma': ratio_all,
                        'kept': True})

        normalised.append(norm)
        kept_local_indices.append(local_i)

    if qc_rows:
        df_qc = pd.DataFrame(qc_rows).sort_values('sep0_over_sigma')
        qc_path = os.path.join(cfg.FOLDER_DYE,
                               f"{cfg.EXPERIMENT_BASE_NAME}_dye_qc.csv")
        df_qc.to_csv(qc_path, index=False, float_format='%.6f', na_rep='NaN')
        logger.info(f"Saved pre-pulse contrast QC (all GUVs) -> {qc_path}")

        rr = df_qc['sep0_over_sigma'].dropna()
        if len(rr) >= 3:
            qs = rr.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            logger.info(
                "Pre-pulse contrast sep0/sigma: "
                f"min={rr.min():.1f}, p05={qs[0.05]:.1f}, p25={qs[0.25]:.1f}, "
                f"median={qs[0.5]:.1f}, p75={qs[0.75]:.1f}, max={rr.max():.1f}"
            )
            # Largest gap in the sorted ratios = candidate natural threshold.
            srt = np.sort(rr.values)
            if len(srt) > 3:
                gaps = np.diff(srt)
                k = int(np.argmax(gaps))
                logger.info(
                    f"  Largest gap in the sorted distribution: {srt[k]:.1f} -> "
                    f"{srt[k+1]:.1f}. If that gap is clean, a threshold in "
                    f"between separates loaded from unloaded vesicles; if the "
                    f"distribution is smooth there is no natural cut and the "
                    f"threshold is a judgement call."
                )

    if not normalised:
        logger.error("Normalisation failed for all GUVs.")
        return None

    # Align curves so t=0 is the pulse frame. Fitting still uses only the
    # post-pulse segment, but the FULL curve (including the pre-pulse frames
    # at negative t) is carried through as well, so the baseline that dye0,
    # bg0 and bg_std0 were derived from is visible in the exports instead of
    # being discarded at this line.
    full_arr = np.array(normalised)
    t_full   = time_array - time_array[pulse_frame]

    arr   = full_arr[:, pulse_frame:]
    t_al  = t_full[pulse_frame:]
    avg   = np.nanmean(arr, axis=0)

    valid_ids = [
        str(circles[guv_results['valid_guv_indices'][i]]['id'])
        for i in kept_local_indices
    ]

    return {
        't_aligned':                t_al,
        'average_curve':            avg,
        'all_curves_aligned':       arr,
        # Full record including pre-pulse frames (negative times). Used for
        # the CSV export and baseline QC; NOT passed to the fitter.
        't_full':                   t_full,
        'all_curves_full':          full_arr,
        'pulse_frame':              pulse_frame,
        'valid_guv_ids':            valid_ids,
        'median_jump_frame':        pulse_frame,
        'excluded_dead_guv_ids':    excluded_dead_ids,
    }


# -------------------------------------------------------------------
# --- 5. FITTING ---
# -------------------------------------------------------------------

def _get_kinetic_model_spec(model_name: str, y_start: float, y_end: float,
                             A_est: float, tau_e: float):
    """
    Returns (fn, p0, bounds, param_names) for a named kinetic model, using
    the same initial-guess/bounds conventions for all four model variants.
    Returns None for an unrecognised model_name.
    """
    IINF_MAX = float(getattr(cfg, 'FIT_IINF_MAX', 1.2))
    AMP_MAX  = float(getattr(cfg, 'FIT_AMPLITUDE_MAX', 2.0))
    TAU_MIN  = float(getattr(cfg, 'FIT_TAU_MIN', 0.01))
    TAU_MAX  = float(getattr(cfg, 'FIT_TAU_MAX', 5000.0))

    use_D  = bool(getattr(cfg, 'FIT_INCLUDE_DRIFT', False))
    D_MAX  = float(getattr(cfg, 'FIT_DRIFT_ABS_MAX', 5e-5))
    # curve_fit needs a strictly-positive-width interval on every parameter.
    # When drift is disabled we keep D in the signature (the model functions
    # default it to 0) but pin it to a numerically negligible window.
    d_lo, d_hi = (-D_MAX, D_MAX) if use_D else (-1e-12, 1e-12)

    tau_g = float(np.clip(tau_e, TAU_MIN * 2, TAU_MAX * 0.999))
    amp_g = float(np.clip(abs(A_est), 1e-3, AMP_MAX * 0.999))

    if model_name == 'EFFLUX-1EXP':
        # I(t) = Iinf + A*exp(-t/tau) [+ D*t],  A >= 0, Iinf >= 0.
        # A >= 0 forbids a *rising* efflux; Iinf >= 0 forbids negative
        # fluorescence. Together they close the degenerate branch in which
        # the fit rose toward Iinf = 2 while a negative D dragged it back
        # down — the branch that pinned 21/41 GUVs at the old bound.
        fn = utils.efflux_1exp_amp
        p0 = (float(np.clip(y_end, 0.0, IINF_MAX)),
              amp_g, tau_g, 0.0)
        bnds  = ((0.0, 0.0, TAU_MIN, d_lo), (IINF_MAX, AMP_MAX, TAU_MAX, d_hi))
        names = ['Iinf', 'A', 'tau', 'D']
    elif model_name == 'EFFLUX-2EXP':
        # I(t) = Iinf + a1*exp(-t/tau1) + a2*exp(-t/tau2) [+ D*t]
        # Same bounds on Iinf and on each amplitude as 1EXP, so setting
        # a2 = 0 reproduces 1EXP exactly. The 1EXP feasible set is now a
        # strict SUBSET of the 2EXP feasible set and RSS_2EXP <= RSS_1EXP
        # holds by construction.
        fn = utils.efflux_2exp
        p0    = (float(np.clip(y_end, 0.0, IINF_MAX)),
                 amp_g * 0.5, max(tau_g * 0.25, TAU_MIN * 2),
                 amp_g * 0.5, min(tau_g * 4.0, TAU_MAX * 0.999),
                 0.0)
        bnds  = ((0.0, 0.0, TAU_MIN, 0.0, TAU_MIN, d_lo),
                 (IINF_MAX, AMP_MAX, TAU_MAX, AMP_MAX, TAU_MAX, d_hi))
        names = ['Iinf', 'a1', 'tau1', 'a2', 'tau2', 'D']
    elif model_name == 'INFLUX-1EXP':
        # I(t) = Iinf - A*exp(-t/tau) [+ D*t],  A >= 0, Iinf >= 0.
        fn = utils.influx_1exp_amp
        p0 = (float(np.clip(y_end, 0.0, IINF_MAX)),
              amp_g, tau_g, 0.0)
        bnds  = ((0.0, 0.0, TAU_MIN, d_lo), (IINF_MAX, AMP_MAX, TAU_MAX, d_hi))
        names = ['Iinf', 'A', 'tau', 'D']
    elif model_name == 'INFLUX-2EXP':
        fn = utils.influx_2exp_amp
        p0    = (float(np.clip(y_end, 0.0, IINF_MAX)),
                 amp_g * 0.5, max(tau_g * 0.25, TAU_MIN * 2),
                 amp_g * 0.5, min(tau_g * 4.0, TAU_MAX * 0.999),
                 0.0)
        bnds  = ((0.0, 0.0, TAU_MIN, 0.0, TAU_MIN, d_lo),
                 (IINF_MAX, AMP_MAX, TAU_MAX, AMP_MAX, TAU_MAX, d_hi))
        names = ['Iinf', 'a1', 'tau1', 'a2', 'tau2', 'D']
    else:
        return None
    return fn, p0, bnds, names


def _fit_kinetic_model(model_name: str, t_c: np.ndarray, y_c: np.ndarray,
                        y_start: float, y_end: float, A_est: float, tau_e: float):
    """
    Fits one named kinetic model to (t_c, y_c) and returns a dict with the
    fitted params, their SEs, RSS, and AIC/AICc/BIC — or None if the model
    is unrecognised or the fit fails to converge.
    """
    spec = _get_kinetic_model_spec(model_name, y_start, y_end, A_est, tau_e)
    if spec is None:
        return None
    fn, p0, bnds, names = spec

    lo = np.asarray(bnds[0], dtype=float)
    hi = np.asarray(bnds[1], dtype=float)
    use_drift = bool(getattr(cfg, 'FIT_INCLUDE_DRIFT', False))

    # --- Multi-start ---------------------------------------------------------
    # curve_fit's trust-region solver can stall when a starting value sits on
    # a bound, which previously left a subset of 2EXP fits returning their
    # initial guess unchanged (identical RSS to 1EXP, to the last bit). We try
    # the deterministic heuristic guess plus a handful of randomised starts
    # drawn log-uniformly for the tau parameters, and keep the best.
    n_extra = int(getattr(cfg, 'FIT_N_MULTISTART', 6))
    seed    = getattr(cfg, 'FIT_MULTISTART_SEED', 0)
    rng     = np.random.default_rng(seed)

    t_span = float(t_c[-1] - t_c[0]) if len(t_c) > 1 else 1.0
    starts = [np.clip(np.asarray(p0, dtype=float), lo, hi)]
    for _ in range(max(0, n_extra)):
        cand = np.empty_like(lo)
        for j, nm in enumerate(names):
            if nm.startswith('tau'):
                # log-uniform between a fast fraction of the record and the record
                a = max(lo[j], t_span * 1e-3)
                b = min(hi[j], max(t_span, a * 10))
                cand[j] = float(np.exp(rng.uniform(np.log(a), np.log(b))))
            elif nm == 'D':
                cand[j] = 0.0
            else:
                cand[j] = float(rng.uniform(lo[j], hi[j]))
        starts.append(np.clip(cand, lo, hi))

    best = None
    for start in starts:
        try:
            # A multistart point can land somewhere the Jacobian is singular;
            # curve_fit still returns usable parameters but warns about the
            # covariance. That start is simply discarded if its RSS is not the
            # best, so the warning is noise rather than information.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sc.OptimizeWarning)
                params, pcov = sc.curve_fit(fn, t_c, y_c, p0=start,
                                            bounds=(lo, hi), maxfev=200000)
        except (RuntimeError, ValueError):
            continue
        resid = y_c - fn(t_c, *params)
        rss = float(np.sum(resid ** 2))
        if not np.isfinite(rss):
            continue
        if best is None or rss < best[0]:
            best = (rss, params, pcov)

    if best is None:
        return None

    rss, params, pcov = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    y_fit = fn(t_c, *params)
    aic, aicc, bic = utils.compute_information_criteria(rss, n=len(t_c), k=len(names))

    # --- Is the decline actually EXPONENTIAL, or merely a decline? -----------
    # An exponential whose tau greatly exceeds the record is, over the observed
    # window, a straight line: exp(-t/tau) ~ 1 - t/tau. The fit still returns a
    # tau, and the panel still looks like "a decay", but nothing in the data
    # distinguishes it from linear leakage. Comparing the model's RSS against a
    # plain straight line fitted to the SAME window separates a resolved
    # exponential relaxation from a slow drift that an exponential was merely
    # draped over.
    #
    #   exp_vs_linear >> 1  : curvature is real and resolved within the record
    #   exp_vs_linear ~  1  : indistinguishable from a straight line
    #   exp_vs_linear <  1  : a straight line fits BETTER than the model
    #
    # This is a shape descriptor, not a model-selection test: the two have
    # different parameter counts, so it is not an F-test and no p-value is
    # implied. It is used only to bin traces by phenotype.
    if len(t_c) >= 3:
        lin_coef = np.polyfit(t_c, y_c, 1)
        rss_linear = float(np.sum((y_c - np.polyval(lin_coef, t_c)) ** 2))
    else:
        rss_linear = np.nan
    exp_vs_linear = (rss_linear / rss
                     if (np.isfinite(rss_linear) and rss > 0) else np.nan)

    # --- Identifiability -----------------------------------------------------
    # tau cannot be recovered from a record that ends long before the decay
    # completes: the data then constrain only the initial slope, and Iinf/tau
    # trade off freely along a flat valley. Such fits are kept (they still
    # carry an initial-rate estimate) but flagged so they can be excluded from
    # population statistics on tau.
    tau_idx = names.index('tau') if 'tau' in names else (
        names.index('tau1') if 'tau1' in names else None)
    tau_val = float(params[tau_idx]) if tau_idx is not None else np.nan
    tau_se  = float(perr[tau_idx])   if tau_idx is not None else np.nan

    se_ratio_max = float(getattr(cfg, 'TAU_SE_RATIO_MAX', 0.5))
    frac_max     = float(getattr(cfg, 'TAU_MAX_FRACTION_OF_RECORD', 1/3))

    se_ratio = tau_se / tau_val if (np.isfinite(tau_se) and tau_val > 0) else np.inf
    tau_too_long = tau_val > frac_max * t_span if np.isfinite(tau_val) else True
    # Bound check ignores D when drift is disabled: it is pinned to a
    # degenerate +/-1e-12 window by design, so it is trivially "at bound"
    # and would otherwise flag every single fit.
    check = [j for j, nm in enumerate(names)
             if not (nm == 'D' and not use_drift)]
    at_bound = bool(
        any(np.isclose(params[j], lo[j], rtol=0, atol=1e-9) for j in check) or
        any(np.isclose(params[j], hi[j], rtol=0, atol=1e-9) for j in check)
    )

    # --- Responding? ---------------------------------------------------------
    # A flat trace (a GUV below the permeabilisation threshold) has no decay to
    # measure. Such a GUV is a perfectly valid DATA POINT -- finding the
    # voltage threshold is the whole experiment -- but it does not carry a tau.
    #
    # This test is deliberately MODEL-FREE: whether a vesicle responded is an
    # observation, not a fit output. The previous version gated on the fitted
    # amplitude sum, sum|A|, which fails in two ways:
    #
    #   (i)  sum|A| is the amplitude the model would express over INFINITE
    #        time, not the drop realised inside the record. With tau free up
    #        to FIT_TAU_MAX (5000 s) and a record of ~700 s, the optimiser can
    #        park tau at the upper bound and carry a large A while the fitted
    #        curve is, over the observed window, a flat line: exp(-t/tau) only
    #        moves from 1 to 0.87. On the 260422 Empty 400 V run this called
    #        GUV 5 (realised drop 0.03) and GUV 10 (realised drop 0.01, trace
    #        dead flat) 'responding'.
    #   (ii) the absolute value discards direction, so a trace drifting the
    #        WRONG way scored the same as one that emptied. The EFFLUX bounds
    #        (A >= 0) mean a rising trace cannot even be represented, so its
    #        fitted A is an artefact of the feasible set, not a measurement.
    #
    # Instead: compare the SIGNED decline actually observed in the data to
    # that trace's own high-frequency noise. A robust median of the first and
    # last few samples sets the endpoints (so one bad frame cannot create or
    # destroy a response), and the noise scale comes from the MAD of
    # successive differences, which is insensitive to the slow trend itself.
    resp_drop, resp_noise = _observed_response(
        t_c, y_c, efflux=model_name.startswith('EFFLUX'))
    k_resp = float(getattr(cfg, 'FIT_RESPONSE_AMPLITUDE_SIGMA', 3.0))
    min_amp = float(getattr(cfg, 'FIT_MIN_RESPONSE_AMPLITUDE', 0.05))
    responding = bool(np.isfinite(resp_drop) and np.isfinite(resp_noise)
                      and resp_drop > max(min_amp, k_resp * resp_noise))

    # Retained purely as a diagnostic, no longer a gate: a LARGE amp_total
    # next to a SMALL resp_drop is the signature of the tau-at-bound
    # degeneracy described above, so keeping both columns makes that
    # failure visible in the parameter table instead of silent.
    amp_idx = [j for j, nm in enumerate(names) if nm in ('A', 'a1', 'a2')]
    amp_total = float(np.sum(np.abs(params[amp_idx]))) if amp_idx else np.nan
    resid_std = float(np.std(y_c - y_fit)) if len(y_c) > 2 else np.nan

    # tau is only meaningful for a GUV that actually responded AND whose decay
    # resolved inside the record.
    identifiable = bool(responding
                        and np.isfinite(se_ratio) and se_ratio <= se_ratio_max
                        and not tau_too_long)

    return {
        'model_name': model_name, 'fn': fn, 'params': params, 'names': names,
        'perr': perr, 'y_fit': y_fit, 'rss': rss, 'n_points': len(t_c),
        'aic': aic, 'aicc': aicc, 'bic': bic,
        'tau_se_ratio': float(se_ratio) if np.isfinite(se_ratio) else np.nan,
        'tau_over_record': float(tau_val / t_span) if t_span > 0 else np.nan,
        'tau_identifiable': identifiable,
        'is_responding': responding,
        'response_drop': resp_drop,
        'response_noise': resp_noise,
        'rss_linear': rss_linear,
        'exp_vs_linear': exp_vs_linear,
        'amplitude_total': amp_total,
        'residual_std': resid_std,
        'param_at_bound': at_bound,
        'n_starts_tried': len(starts),
    }


def _observed_response(t_c: np.ndarray, y_c: np.ndarray,
                       efflux: bool = True) -> tuple:
    """
    Model-free measure of whether a dye trace actually moved, and by how much.

    Returns (change, noise), both in normalised-intensity units:

      change : SIGNED response amplitude, oriented so that POSITIVE always
               means "in the expected direction" — a fall for EFFLUX, a rise
               for INFLUX. A flat trace gives ~0; a trace drifting the wrong
               way gives a negative value and can therefore never be scored
               as responding, which a magnitude-only test cannot achieve.
               Endpoints are MEDIANS of the first and last few samples rather
               than single points, so one bad frame at either end cannot
               manufacture or erase a response.

      noise  : robust high-frequency scatter, from the MAD of SUCCESSIVE
               DIFFERENCES divided by sqrt(2). Differencing removes the slow
               trend before the scale is estimated, so a genuine smooth decay
               does not inflate its own rejection threshold — which is what
               happens if the fit's residual standard deviation is used, since
               that conflates measurement noise with model misfit.
    """
    fin = np.isfinite(y_c)
    y = np.asarray(y_c, dtype=float)[fin]
    if y.size < 6:
        return np.nan, np.nan

    k = int(max(3, min(5, y.size // 10)))
    y_start = float(np.median(y[:k]))
    y_end   = float(np.median(y[-k:]))
    change  = (y_start - y_end) if efflux else (y_end - y_start)

    d = np.diff(y)
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(d) / np.sqrt(2.0)) if d.size else np.nan
    return change, noise


def _model_free_metrics(t_c: np.ndarray, y_c: np.ndarray) -> dict:
    """
    Fit-independent descriptors of a dye trace. These are always reported,
    including for GUVs whose tau is not identifiable, because they are
    constrained directly by the data rather than by an extrapolated model.

      t50               : first time the trace falls to <= 0.5 (linear
                          interpolation between bracketing samples); NaN if
                          it never does within the record.
      frac_remaining_*  : normalised intensity at fixed times post-pulse.
      initial_rate      : slope over the first 10% of the record (per second),
                          from a straight-line fit. This is the one quantity a
                          non-identifiable exponential still pins down.
    """
    out = {}
    finite = np.isfinite(y_c)
    t, y = t_c[finite], y_c[finite]
    if len(t) < 3:
        return {'t50_s': np.nan, 't50_censored': False,
                't50_censored_below_s': np.nan, 'initial_rate_per_s': np.nan,
                'has_step': False, 'n_steps': 0,
                'first_step_t_s': np.nan, 'largest_step_drop': np.nan,
                'drop_t10_s': np.nan, 'drop_t50_s': np.nan,
                'drop_t90_s': np.nan, 'onset_lag_frac': np.nan}

    # t50 via first downward crossing of 0.5.
    #
    # LEFT CENSORING: a GUV that is already below 0.5 at the first fitted
    # frame crossed 50% during or before the pulse frame itself. The fast
    # acquisition segment is only ~1.6 s long, so at higher field strengths
    # a growing fraction of vesicles empty inside it. Reporting t50 = 0 for
    # those would drag any t50 distribution toward zero with values that are
    # not measurements — the crossing time is unobserved, only bounded above
    # by the first sample. Report NaN and flag it; the flag lets a survival
    # analysis treat these correctly rather than discarding them.
    t50 = np.nan
    censored = False
    below = np.where(y <= 0.5)[0]
    if below.size > 0:
        k = int(below[0])
        if k == 0:
            # Already below threshold at first observation: crossing time is
            # left-censored at t[0], not equal to it.
            censored = True
        else:
            y0, y1 = y[k - 1], y[k]
            if y0 != y1:
                frac = (y0 - 0.5) / (y0 - y1)
                t50 = float(t[k - 1] + frac * (t[k] - t[k - 1]))
            else:
                t50 = float(t[k])
    out['t50_s'] = t50
    out['t50_censored'] = censored
    # Upper bound on the true crossing time when censored; NaN otherwise.
    out['t50_censored_below_s'] = float(t[0]) if censored else np.nan

    for target in getattr(cfg, 'MODEL_FREE_REPORT_TIMES_S', [10, 30, 60, 120, 300, 600]):
        if t[-1] >= target >= t[0]:
            out[f'frac_remaining_{int(target)}s'] = float(np.interp(target, t, y))
        else:
            out[f'frac_remaining_{int(target)}s'] = np.nan

    # --- step detection ----------------------------------------------------
    # The efflux formula assumes ONE permeabilisation event followed by a
    # monotonic relaxation. A trace that declines slowly and then drops
    # abruptly partway through has had a second event (or a tracking/mask
    # artefact); no single- or double-exponential can describe that, and the
    # fit will return a tau belonging to neither segment. Flagged, not fitted
    # around.
    if len(t) > 8:
        dy = np.diff(y); dt_ = np.diff(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = np.where(dt_ > 0, dy / dt_, np.nan)
        med_rate = np.nanmedian(rate)
        scale = np.nanmedian(np.abs(rate - med_rate)) * 1.4826
        k_step = float(getattr(cfg, 'STEP_DETECT_SIGMA', 6.0))
        min_drop = float(getattr(cfg, 'STEP_DETECT_MIN_DROP', 0.10))
        if np.isfinite(scale) and scale > 0:
            cand = np.where((rate < med_rate - k_step * scale) & (dy < -min_drop))[0]
        else:
            cand = np.where(dy < -min_drop)[0]
        out['has_step'] = bool(cand.size > 0)
        out['n_steps'] = int(cand.size)
        out['first_step_t_s'] = float(t[cand[0] + 1]) if cand.size else np.nan
        out['largest_step_drop'] = float(-dy[cand].min()) if cand.size else np.nan
    else:
        out['has_step'] = False; out['n_steps'] = 0
        out['first_step_t_s'] = np.nan; out['largest_step_drop'] = np.nan

    # --- response timing: milestones of the GUV'S OWN total decline ---------
    # t50_s above is an ABSOLUTE crossing of 0.5, so a vesicle that only ever
    # empties to 0.6 never reaches it and reports NaN. These milestones are
    # relative to each trace's own start and end levels instead, so a partial
    # responder still gets a timescale, and the shapes of large and small
    # responses become comparable.
    #
    # onset_lag_frac = t10 / t90 is the discriminator between a decline that
    # begins AT the pulse and one that begins some time after it. A vesicle
    # permeabilised by the pulse itself starts leaking immediately, so t10 is
    # a small fraction of t90 (~0.00 in practice). A trace that sits flat and
    # only then falls gives a large ratio. That lag is not what STEP_DETECT
    # finds: a step is one abrupt frame-to-frame drop, whereas a delayed onset
    # can be a perfectly smooth decline that simply starts late, and the two
    # flag almost disjoint sets of GUVs.
    k = int(max(3, min(5, y.size // 10)))
    y_hi = float(np.median(y[:k]))
    y_lo = float(np.median(y[-k:]))
    total = y_hi - y_lo
    ms = {0.10: np.nan, 0.50: np.nan, 0.90: np.nan}
    if np.isfinite(total) and total > 0 and y.size >= 6:
        # Median-smooth before reading milestones so a single noisy frame
        # cannot set the onset time.
        ys = pd.Series(y).rolling(5, center=True, min_periods=1).median().values
        for f in ms:
            hit = np.where(ys <= y_hi - f * total)[0]
            if hit.size:
                ms[f] = float(t[hit[0]])
    out['drop_t10_s'] = ms[0.10]
    out['drop_t50_s'] = ms[0.50]
    out['drop_t90_s'] = ms[0.90]
    out['onset_lag_frac'] = (ms[0.10] / ms[0.90]
                             if (np.isfinite(ms[0.10]) and np.isfinite(ms[0.90])
                                 and ms[0.90] > 0) else np.nan)

    span = t[-1] - t[0]
    early = t <= (t[0] + max(0.10 * span, 1e-9))
    if early.sum() >= 3:
        out['initial_rate_per_s'] = float(np.polyfit(t[early], y[early], 1)[0])
    else:
        out['initial_rate_per_s'] = np.nan
    return out


def fit_individual_curves(aligned: dict, guv_results: dict, t: np.ndarray, logger: logging.Logger) -> pd.DataFrame:
    arr      = aligned['all_curves_aligned']
    ids      = aligned['valid_guv_ids']
    df_track = guv_results['tracking_dataframe']

    direction = 'EFFLUX' if cfg.MODEL_TO_USE.startswith('EFFLUX') else 'INFLUX'
    name_1exp = f'{direction}-1EXP'
    name_2exp = f'{direction}-2EXP'

    fit_records = []
    fit_plot_data = []   # accumulated for the grid figure
    comparison_records = []   # AICc/BIC comparison, 1EXP vs 2EXP, per GUV

    for i, curve in enumerate(arr):
        guv_id = ids[i]

        guv_data = df_track[df_track['guv_id'] == guv_id]
        if guv_data.empty: continue
        mpp  = getattr(cfg, 'MICRONS_PER_PIXEL', 1.0)
        radii_px = pd.to_numeric(guv_data['radius'], errors='coerce').dropna()
        if radii_px.empty:
            continue
        # radius_um is the INITIAL radius (used for size grouping). The
        # terminal radius is exported alongside it so shrinkage is visible
        # in the parameter table rather than only in the fate labels — a
        # vesicle that deflates changes its own surface-to-volume ratio over
        # the record, which matters for any permeability calculation.
        r_um      = float(radii_px.iloc[0]) * mpp
        r_um_end  = float(radii_px.iloc[-1]) * mpp
        r_um_min  = float(radii_px.min()) * mpp

        ok = np.isfinite(curve)
        t_c, y_c = t[ok], curve[ok]

        if len(t_c) < 5:
            continue

        # Honour FIT_DATA_PERCENTAGE, measured in TIME rather than frame
        # count so it means the same thing across the variable-rate
        # schedule. Previously this config value was never read at all and
        # every fit silently used the whole record.
        fit_frac = float(getattr(cfg, 'FIT_DATA_PERCENTAGE', 1.0))
        if 0 < fit_frac < 1.0 and len(t_c) > 1:
            t_cut = t_c[0] + fit_frac * (t_c[-1] - t_c[0])
            keep = t_c <= t_cut
            if keep.sum() >= 5:
                t_c, y_c = t_c[keep], y_c[keep]

        metrics_free = _model_free_metrics(t_c, y_c)

        # Optionally restrict the fit to the first relaxation, so the efflux
        # formula is applied to data it can actually describe rather than
        # averaged across two distinct events.
        if getattr(cfg, 'FIT_TRUNCATE_AT_FIRST_STEP', False) and metrics_free.get('has_step'):
            t_step = metrics_free.get('first_step_t_s', np.nan)
            if np.isfinite(t_step):
                keep_pre = t_c < t_step
                if keep_pre.sum() >= 5:
                    t_c, y_c = t_c[keep_pre], y_c[keep_pre]

        y_start, y_end = y_c[0], y_c[-1]
        A_est = abs(y_end - y_start)

        try:
            target_y = y_start - 0.63 * A_est if cfg.MODEL_TO_USE.startswith('EFFLUX') else y_start + 0.63 * A_est
            idx63 = np.argmax(y_c < target_y) if cfg.MODEL_TO_USE.startswith('EFFLUX') else np.argmax(y_c > target_y)
            tau_e = t_c[idx63] if idx63 > 0 else t_c[-1] / 3.0
        except Exception:
            tau_e = t_c[-1] / 3.0

        tau_e = max(tau_e, 1.0)

        # --- Fit BOTH the 1EXP and 2EXP variants (same efflux/influx
        # direction as MODEL_TO_USE) so AICc/BIC can compare them directly.
        # Only the variant matching cfg.MODEL_TO_USE feeds the main
        # fit_records/plot outputs below; the pair always feeds the
        # comparison table.
        fit_1exp = _fit_kinetic_model(name_1exp, t_c, y_c, y_start, y_end, A_est, tau_e)
        fit_2exp = _fit_kinetic_model(name_2exp, t_c, y_c, y_start, y_end, A_est, tau_e)

        comp_row = {'guv_id': guv_id, 'n_points': len(t_c)}
        for tag, f in ((name_1exp, fit_1exp), (name_2exp, fit_2exp)):
            comp_row[f'RSS_{tag}']  = f['rss']  if f else np.nan
            comp_row[f'AICc_{tag}'] = f['aicc'] if f else np.nan
            comp_row[f'BIC_{tag}']  = f['bic']  if f else np.nan

        # --- NESTING ASSERTION ------------------------------------------------
        # Under the bounds in _get_kinetic_model_spec, 1EXP is a strict special
        # case of 2EXP (a2 = 0). RSS_2EXP therefore CANNOT exceed RSS_1EXP at a
        # true optimum. If it does, the 2EXP optimiser failed, and reporting
        # "1EXP preferred" would be an artefact of that failure rather than
        # evidence about the kinetics. Flag it and emit NaN instead.
        tol = float(getattr(cfg, 'FIT_NESTING_TOLERANCE', 1e-6))
        nesting_ok = True
        if fit_1exp and fit_2exp:
            nesting_ok = fit_2exp['rss'] <= fit_1exp['rss'] + tol
        comp_row['nesting_ok'] = nesting_ok

        if fit_1exp and fit_2exp and not nesting_ok:
            logger.warning(
                f"  GUV {guv_id}: 2EXP fit failed to converge "
                f"(RSS_2EXP={fit_2exp['rss']:.5f} > RSS_1EXP={fit_1exp['rss']:.5f}). "
                f"2EXP nests 1EXP, so this is an optimiser failure, not a model "
                f"result. Model comparison suppressed for this GUV."
            )

        comparable = bool(fit_1exp and fit_2exp and nesting_ok)

        if comparable and np.isfinite(fit_1exp['aicc']) and np.isfinite(fit_2exp['aicc']):
            comp_row['delta_AICc_1EXPminus2EXP'] = fit_1exp['aicc'] - fit_2exp['aicc']
            comp_row['preferred_model_AICc'] = '2EXP' if fit_2exp['aicc'] < fit_1exp['aicc'] else '1EXP'
        else:
            comp_row['delta_AICc_1EXPminus2EXP'] = np.nan
            comp_row['preferred_model_AICc'] = None if comparable else 'FIT_FAILED'
        if comparable and np.isfinite(fit_1exp['bic']) and np.isfinite(fit_2exp['bic']):
            comp_row['delta_BIC_1EXPminus2EXP'] = fit_1exp['bic'] - fit_2exp['bic']
            comp_row['preferred_model_BIC'] = '2EXP' if fit_2exp['bic'] < fit_1exp['bic'] else '1EXP'
        else:
            comp_row['delta_BIC_1EXPminus2EXP'] = np.nan
            comp_row['preferred_model_BIC'] = None if comparable else 'FIT_FAILED'
        comparison_records.append(comp_row)

        primary = fit_1exp if cfg.MODEL_TO_USE == name_1exp else fit_2exp
        if primary is None:
            logger.warning(f"Fitting failed for GUV {guv_id}")
            continue

        params, names, perr, fn = primary['params'], primary['names'], primary['perr'], primary['fn']

        res = {'guv_id': guv_id,
               'radius_um': r_um,
               'radius_um_final': r_um_end,
               'radius_um_min': r_um_min}
        res.update(dict(zip(names, params)))
        res.update({f'{n}_SE': e for n, e in zip(names, perr)})

        # Quality flags. tau_identifiable == False means the record ended
        # before the decay resolved, so tau and Iinf are extrapolations and
        # must be excluded from population statistics — use the model-free
        # columns for those GUVs instead.
        res['tau_identifiable'] = primary['tau_identifiable']
        res['is_responding']    = primary['is_responding']
        res['response_drop']    = primary['response_drop']
        res['response_noise']   = primary['response_noise']
        res['exp_vs_linear']    = primary['exp_vs_linear']
        res['rss_linear']       = primary['rss_linear']

        # --- response phenotype ------------------------------------------
        # Four mutually exclusive classes, assigned in this order:
        #
        #   NON_RESPONDING - no decline resolvable above the trace's own
        #                    noise. Retained as a data point: locating the
        #                    voltage threshold is the point of the experiment.
        #   EXPONENTIAL    - curvature resolved INSIDE the record, i.e. the
        #                    exponential beats a straight line on the same
        #                    window by RESPONSE_SHAPE_MIN_RSS_RATIO. Only
        #                    these carry a tau that means anything.
        #   DELAYED        - the decline begins measurably AFTER the pulse
        #                    (onset_lag_frac >= RESPONSE_ONSET_LAG_FRAC),
        #                    rather than at it.
        #   GRADUAL        - declines from the pulse onward, but too slowly
        #                    for the record to distinguish from a straight
        #                    line. A tau is still fitted and reported; it is
        #                    an extrapolation and tau_identifiable will
        #                    almost always be False.
        #
        # EXPONENTIAL is tested before DELAYED deliberately. The exponential
        # is fitted from the pulse, so a trace that sits flat and only then
        # decays fits it poorly and falls through to DELAYED on its own —
        # the ordering does not hide late-onset exponentials, it routes them
        # to the class that describes the more salient feature.
        _shape_min = float(getattr(cfg, 'RESPONSE_SHAPE_MIN_RSS_RATIO', 3.0))
        _lag_min   = float(getattr(cfg, 'RESPONSE_ONSET_LAG_FRAC', 0.07))
        _evl = primary['exp_vs_linear']
        _lag = metrics_free.get('onset_lag_frac', np.nan)
        if not primary['is_responding']:
            res['response_class'] = 'NON_RESPONDING'
        elif np.isfinite(_evl) and _evl >= _shape_min:
            res['response_class'] = 'EXPONENTIAL'
        elif np.isfinite(_lag) and _lag >= _lag_min:
            res['response_class'] = 'DELAYED'
        else:
            res['response_class'] = 'GRADUAL'
        res['amplitude_total']  = primary['amplitude_total']
        res['residual_std']     = primary['residual_std']
        # Fourth agreed flag. Sourced from the fate classification rather
        # than recomputed here, so "grew" means exactly one thing across the
        # whole pipeline (same pre-pulse baseline, same terminal-frame
        # averaging, same threshold) instead of two definitions that can
        # drift apart. NaN/False when tracking produced no fate map.
        res['radius_growth_flag'] = bool(
            str(guv_id) in guv_results.get('grown_guv_ids', set()))
        res['radius_change_frac'] = float(
            guv_results.get('radius_change_frac', {}).get(str(guv_id), np.nan))
        res['tau_se_ratio']     = primary['tau_se_ratio']
        res['tau_over_record']  = primary['tau_over_record']
        res['nesting_ok']       = nesting_ok
        res['rss']              = primary['rss']
        res['n_points']         = primary['n_points']

        res.update(metrics_free)
        fit_records.append(res)

        # Build a short parameter label for the subplot title. A tau that
        # failed the identifiability gate is shown parenthesised so the grid
        # figure cannot be read as though every panel reported a measurement.
        if 'tau2' in names:
            param_str = f"τ₁={params[2]:.1f} s,  τ₂={params[4]:.1f} s"
        else:
            param_str = f"τ = {params[2]:.1f} s"
        if not primary['tau_identifiable']:
            param_str = f"({param_str})  τ NOT IDENTIFIABLE"

        fit_plot_data.append({
            'guv_id':    guv_id,
            't_data':    t_c,
            'y_data':    y_c,
            'y_fit':     fn(t_c, *params),
            'r_um':      r_um,
            'param_str': param_str,
        })

    # ── Grid figure (replaces individual PNGs) ────────────────────────────
    if fit_plot_data:
        grid_path = utils.plot_dye_fits_grid(
            fit_plot_data, cfg.FOLDER_DYE_FITTING,
            cfg.EXPERIMENT_BASE_NAME, cfg.MODEL_TO_USE,
            ruptured_guv_ids=guv_results.get('ruptured_guv_ids_refined',
                                              guv_results.get('ruptured_guv_ids')),
            out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
            shrunk_guv_ids=guv_results.get('shrunk_guv_ids'),
            grown_guv_ids=guv_results.get('grown_guv_ids'),
        )
        logger.info(f"Saved dye fits grid → {grid_path}")

    # ── Model comparison export (AICc/BIC, 1EXP vs 2EXP) ──────────────────
    if comparison_records:
        df_comp = pd.DataFrame(comparison_records)
        comp_path = os.path.join(
            cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_model_comparison.csv"
        )
        df_comp.to_csv(comp_path, index=False, float_format='%.4f', na_rep='NaN')
        logger.info(f"Saved model comparison (AICc/BIC, 1EXP vs 2EXP) → {comp_path}")

        n_failed = int((df_comp['preferred_model_AICc'] == 'FIT_FAILED').sum())
        if n_failed:
            logger.warning(
                f"Model comparison: {n_failed}/{len(df_comp)} GUV(s) suppressed "
                f"because the 2EXP fit did not converge (RSS_2EXP > RSS_1EXP, "
                f"which is impossible for a nested model). These are optimiser "
                f"failures — raise FIT_N_MULTISTART if the count is large."
            )

        valid_mask = ~df_comp['preferred_model_AICc'].isin([None, 'FIT_FAILED'])
        df_comp_ok = df_comp[valid_mask & df_comp['preferred_model_AICc'].notna()]
        n_valid = len(df_comp_ok)
        if n_valid > 0:
            n_2exp_aicc = (df_comp_ok['preferred_model_AICc'] == '2EXP').sum()
            n_2exp_bic  = (df_comp_ok['preferred_model_BIC']  == '2EXP').sum()
            # Report the MEDIAN. The mean is dominated by a handful of GUVs
            # with very large positive deltas: the 260422 run had mean
            # dAICc = +12.4 (nominally "favours 2EXP") while only 5 of 41
            # GUVs individually preferred 2EXP. A skewed mean says the
            # opposite of what the per-GUV verdicts say.
            mean_d_aicc = df_comp_ok['delta_AICc_1EXPminus2EXP'].median()
            mean_d_bic  = df_comp_ok['delta_BIC_1EXPminus2EXP'].median()
            logger.info(
                f"Model comparison summary ({n_valid} GUV(s) with both fits): "
                f"2EXP preferred by AICc in {n_2exp_aicc}/{n_valid}, "
                f"by BIC in {n_2exp_bic}/{n_valid}. "
                f"Median ΔAICc (1EXP−2EXP) = {mean_d_aicc:.2f}, "
                f"median ΔBIC (1EXP−2EXP) = {mean_d_bic:.2f}. "
                f"(Positive Δ favours 2EXP; >~2 is usually taken as meaningful "
                f"for AICc, >~6 for BIC. NOTE: these criteria assume independent "
                f"residuals; GUV traces are strongly autocorrelated, which "
                f"inflates Δ, so treat the threshold as optimistic.)"
            )
        else:
            logger.warning("Model comparison: no GUV had both 1EXP and 2EXP fits converge.")

    df_fits = pd.DataFrame(fit_records)

    # Size grouping lives here, with the fit table, rather than inside the
    # plotting routine. Bin edges are configurable and the resulting counts
    # are logged, because a bin holding a single vesicle is not a population
    # and should not be silently boxplotted as one.
    if not df_fits.empty:
        bins   = list(getattr(cfg, 'SIZE_GROUP_BINS', [0, 5.0, 8.0, 11.0, 100.0]))
        labels = list(getattr(cfg, 'SIZE_GROUP_LABELS', ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']))
        df_fits['size_group'] = pd.cut(df_fits['radius_um'], bins=bins, labels=labels)
        counts = df_fits['size_group'].value_counts().reindex(labels).fillna(0).astype(int)
        logger.info("Size group counts: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
        thin = [k for k, v in counts.items() if 0 < v < int(getattr(cfg, 'SIZE_GROUP_MIN_N', 3))]
        if thin:
            logger.warning(
                f"Size group(s) {thin} contain fewer than "
                f"{getattr(cfg, 'SIZE_GROUP_MIN_N', 3)} GUV(s). Per-experiment "
                f"size binning is unreliable — pool across experiments before "
                f"drawing size-dependence conclusions, or regress against "
                f"continuous radius instead of binning."
            )

        if 'tau_identifiable' in df_fits.columns:
            n_id = int(df_fits['tau_identifiable'].sum())
            logger.info(
                f"tau identifiability: {n_id}/{len(df_fits)} GUV(s) passed "
                f"(tau_SE/tau <= {getattr(cfg, 'TAU_SE_RATIO_MAX', 0.5)} and "
                f"tau <= {getattr(cfg, 'TAU_MAX_FRACTION_OF_RECORD', 1/3):.2f} x record). "
                f"The remaining {len(df_fits) - n_id} have tau/Iinf reported for "
                f"completeness only — use the model-free columns (t50_s, "
                f"frac_remaining_*, initial_rate_per_s) for those."
            )

    # Primary data export. This used to happen as a side effect inside
    # generate_parameter_boxplots(), so any failure in plotting also lost
    # the parameter table.
    if not df_fits.empty:
        fits_path = os.path.join(
            cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_fit_parameters.csv")
        df_fits.to_csv(fits_path, index=False)
        if 'response_class' in df_fits.columns:
            vc = df_fits['response_class'].value_counts()
            logger.info(
                "Response classes: "
                + ", ".join(f"{k}={int(v)}" for k, v in vc.items())
                + f"  (EXPONENTIAL requires exp_vs_linear >= "
                  f"{getattr(cfg, 'RESPONSE_SHAPE_MIN_RSS_RATIO', 3.0)}, "
                  f"DELAYED requires onset_lag_frac >= "
                  f"{getattr(cfg, 'RESPONSE_ONSET_LAG_FRAC', 0.07)})"
            )
            n_exp = int((df_fits['response_class'] == 'EXPONENTIAL').sum())
            n_tid = int(df_fits['tau_identifiable'].fillna(False).sum())
            if n_exp != n_tid:
                logger.info(
                    f"  Note: {n_exp} GUV(s) classed EXPONENTIAL vs "
                    f"{n_tid} with identifiable tau. The two gates ask "
                    f"different questions — shape resolved within the record "
                    f"vs tau constrained by it — so a mismatch is expected, "
                    f"not an error."
                )

        logger.info(f"Saved fit parameters → {fits_path}")

    return df_fits

def generate_parameter_boxplots(df_fits: pd.DataFrame, logger: logging.Logger):
    if df_fits.empty: return

    if 'size_group' not in df_fits.columns:
        bins   = list(getattr(cfg, 'SIZE_GROUP_BINS', [0, 5.0, 8.0, 11.0, 100.0]))
        labels = list(getattr(cfg, 'SIZE_GROUP_LABELS', ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']))
        df_fits = df_fits.copy()
        df_fits['size_group'] = pd.cut(df_fits['radius_um'], bins=bins, labels=labels)

    # Boxplot only the identifiable fits. Boxplotting a tau that the record
    # cannot constrain produces a distribution of optimiser artefacts.
    if 'tau_identifiable' in df_fits.columns:
        n_before = len(df_fits)
        df_fits = df_fits[df_fits['tau_identifiable'].fillna(False)]
        if df_fits.empty:
            logger.warning(
                f"Parameter boxplots skipped: none of {n_before} fit(s) passed "
                f"the tau identifiability gate. This usually means the record "
                f"is too short for the kinetics present, not that the fits are "
                f"broken — report the model-free metrics instead."
            )
            return
        if len(df_fits) < n_before:
            logger.info(
                f"Parameter boxplots use {len(df_fits)}/{n_before} fit(s) with "
                f"identifiable tau."
            )

    # Explicit shortlist, not "every numeric column that isn't excluded".
    # The sweep produced a panel for each of ~24 columns — flags, backing
    # diagnostics, rss, model-free metrics — giving a 36000 px figure in which
    # the three quantities the model actually estimates were indistinguishable
    # from the bookkeeping. Everything else remains in fit_parameters.csv,
    # which is where per-GUV numbers belong; the boxplot exists to show how
    # the FITTED PARAMETERS vary with vesicle size, nothing more.
    params_to_plot = [
        col for col in getattr(cfg, 'BOXPLOT_PARAMS', ['Iinf', 'A', 'tau'])
        if col in df_fits.columns and pd.api.types.is_numeric_dtype(df_fits[col])
    ]
    missing = [c for c in getattr(cfg, 'BOXPLOT_PARAMS', ['Iinf', 'A', 'tau'])
               if c not in df_fits.columns]
    if missing:
        logger.info(
            f"Boxplot parameters not present for model {cfg.MODEL_TO_USE} and "
            f"skipped: {missing}. (A 2EXP model names them a1/tau1/a2/tau2 — "
            f"set BOXPLOT_PARAMS accordingly in config.py.)"
        )
    if not params_to_plot:
        return
    
    fig, axes = plt.subplots(1, len(params_to_plot), figsize=(5 * len(params_to_plot), 5))
    if len(params_to_plot) == 1: axes = [axes]
    
    # seaborn >= 0.13 raises "boxplot statistics and positions must have the
    # same length" if the categorical carries levels with no rows left after
    # filtering. Drop unused levels, and drop NaN rows per-parameter.
    if hasattr(df_fits['size_group'], 'cat'):
        df_fits = df_fits.copy()
        df_fits['size_group'] = df_fits['size_group'].cat.remove_unused_categories()

    for ax, param in zip(axes, params_to_plot):
        sub = df_fits[['size_group', param]].dropna()
        if hasattr(sub['size_group'], 'cat'):
            sub['size_group'] = sub['size_group'].cat.remove_unused_categories()
        if sub.empty or sub['size_group'].nunique() == 0:
            ax.text(0.5, 0.5, f'no data\n({param})', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='0.5')
            ax.set_title(param); ax.set_xticks([])
            continue
        order = list(sub['size_group'].cat.categories) if hasattr(sub['size_group'], 'cat') \
            else sorted(sub['size_group'].unique())
        sns.boxplot(data=sub, x='size_group', y=param, ax=ax, order=order)
        sns.stripplot(data=sub, x='size_group', y=param, ax=ax, order=order,
                      color='black', alpha=0.5)
        # Put n ON the axis. After the tau-identifiability gate a box can rest
        # on two points, which draws a full box-and-whisker that looks like a
        # distribution. The count belongs in the figure, not only in the log.
        counts = sub.groupby('size_group', observed=True)[param].size()
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f'{g}\n(n={int(counts.get(g, 0))})' for g in order])
        if int(counts.min()) < 3:
            ax.set_title(f'{param}  [n<3 in a group]', color='#B22222')
        else:
            ax.set_title(param)
        ax.set_xlabel(r'Radius ($\mu$m)')
        
    grp_n = df_fits.groupby('size_group', observed=True).size()
    thin = grp_n[grp_n < 3]
    if len(thin):
        logger.warning(
            f"Parameter boxplots: size group(s) {list(thin.index)} have "
            f"n<3 ({dict(thin)}). A box drawn on two points is not a "
            f"distribution — pool across experiments before interpreting "
            f"any size dependence in this figure."
        )
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.FOLDER_DYE_FITTING, f"{cfg.EXPERIMENT_BASE_NAME}_parameter_boxplots.png"), dpi=300)
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

    # Export CSV data.
    # Columns are named I_retained, not I_uptake. The quantity computed in
    # normalise_and_align_curves is
    #     I_retained(t) = (I_bg,t - I_dye,t) / (I_bg,t - I_dye,0) = 1 - I_uptake(t)
    # which runs 1 -> 0 as dye leaves. The old I_uptake header named the
    # complement of what was actually written.
    #
    # The export now spans the FULL record, including the pre-pulse frames at
    # negative time. Those frames are where dye0, bg0 and bg_std0 come from,
    # so without them the baseline underlying every normalised value — and
    # the dead-GUV filter threshold — cannot be checked from the outputs.
    # A `phase` column marks which rows the fitter actually saw.
    try:
        t_out   = aligned.get('t_full')
        arr_out = aligned.get('all_curves_full')
        if t_out is None or arr_out is None:
            t_out, arr_out = t, arr
        avg_out = np.nanmean(arr_out, axis=0)

        phase = np.where(t_out < 0, 'pre_pulse',
                         np.where(t_out == 0, 'pulse', 'post_pulse'))

        hdrs = ['Time (s)'] + [f'GUV_{g}_I_retained' for g in ids] + ['Average_I_retained']
        data = np.hstack([t_out.reshape(-1, 1), arr_out.T, avg_out.reshape(-1, 1)])
        df_out = pd.DataFrame(data, columns=hdrs)
        df_out.insert(1, 'phase', phase)
        df_out.to_csv(
            os.path.join(cfg.FOLDER_DYE_INTENSITY,
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
    checks += [
        (0 < float(getattr(cfg, 'FIT_DATA_PERCENTAGE', 1.0)) <= 1.0,
         "FIT_DATA_PERCENTAGE must be in (0, 1]"),
        (float(getattr(cfg, 'FIT_IINF_MAX', 1.2)) > 0,
         "FIT_IINF_MAX must be positive (negative fluorescence is unphysical)"),
        (float(getattr(cfg, 'TAU_SE_RATIO_MAX', 0.5)) > 0,
         "TAU_SE_RATIO_MAX must be positive"),
        (0 < float(getattr(cfg, 'TAU_MAX_FRACTION_OF_RECORD', 1/3)) <= 1.0,
         "TAU_MAX_FRACTION_OF_RECORD must be in (0, 1]"),
        (0 < float(getattr(cfg, 'BG_PERCENTILE', 15.0)) < float(getattr(cfg, 'BG_SIGMA_UPPER_PERCENTILE', 40.0)) < 50.0,
         "Require 0 < BG_PERCENTILE < BG_SIGMA_UPPER_PERCENTILE < 50"),
        (int(getattr(cfg, 'PHOTOMETRY_BLUR_KERNEL', 5)) % 2 == 1,
         "PHOTOMETRY_BLUR_KERNEL must be odd"),
    ]

    for passed, msg in checks:
        if not passed:
            logger.error(f"Config error: {msg}"); ok = False

    # Channel roles: raises rather than falling back to a default index.
    try:
        utils.validate_channel_config(logger)
    except Exception as e:
        logger.error(f"Config error: {e}")
        ok = False

    if getattr(cfg, 'FIT_INCLUDE_DRIFT', False):
        logger.warning(
            "FIT_INCLUDE_DRIFT is enabled. The linear D*t term is only "
            "justified if bleaching has been demonstrated independently in "
            "this dataset — check that non-responding GUVs actually drift. "
            "An unconstrained D can cancel real curvature and make tau "
            "unidentifiable."
        )
    if getattr(cfg, 'PHOTOMETRY_BLUR', False):
        logger.warning(
            "PHOTOMETRY_BLUR is enabled. Blurring before photometry corrupts "
            "the per-pixel background std that MIN_PREPULSE_SEPARATION_SIGMA "
            "divides by; the dead-GUV filter threshold will need recalibrating."
        )

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
        cfg.FOLDER_DYE_FITTING,
        cfg.FOLDER_DYE_INTENSITY,
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

        # Step 3b – recompute background (median + std) excluding neighboring
        # GUVs from each background annulus. This replaces the naive
        # per-worker background estimate, which has no knowledge of where
        # other tracked GUVs are and can be biased by a neighbor sitting
        # inside the ring. Runs before pulse detection so the corrected
        # background feeds every downstream step (pulse detection,
        # normalisation, fitting).
        dye_path0 = data['dye_mmap_info'][0]
        if not getattr(cfg, 'TRACKING_ONLY_MODE', False) and dye_path0 is not None \
                and guv_results['all_intensity_curves']:
            dye_path, dye_shape, dye_dtype = data['dye_mmap_info']
            dye_stack_bg = np.memmap(dye_path, dtype=dye_dtype, mode='r', shape=dye_shape)
            img_shape_bg = dye_stack_bg[0].shape[:2]
            n_frames_bg  = data['roi_mmap_info'][1][0]

            all_ellipses = [
                guv_results['raw_results'][idx][0]['tracking']['ellipses']
                for idx in guv_results['valid_guv_indices']
            ]

            new_bg_curves     = []
            new_bg_std_curves = []
            new_bg_n_curves   = []
            diagnostics_per_guv = {}
            for local_i in range(len(all_ellipses)):
                bg_med, bg_std, n_excl, diag = utils.recompute_background_traces(
                    dye_stack_bg, n_frames_bg, all_ellipses, local_i,
                    cfg.MEMBRANE_FIXED_HALF_WIDTH, cfg.BG_BUFFER_PIXELS, cfg.BG_RING_WIDTH_PIXELS,
                    img_shape_bg,
                )
                new_bg_curves.append(bg_med)
                new_bg_std_curves.append(bg_std)
                # Final pixel count actually used for bg_med/bg_std (after
                # neighbor-position exclusion AND the statistical clip) —
                # needed downstream to convert the per-pixel std into a
                # standard error of the mean.
                new_bg_n_curves.append(diag['n_pixels_total'])

                gid = str(data['circles'][guv_results['valid_guv_indices'][local_i]]['id'])
                diagnostics_per_guv[gid] = diag

                # Annulus crowding diagnostic. The low-quantile anchor does
                # not exclude neighbours -- it tolerates them -- so instead of
                # an exclusion count we report how far the annulus MEDIAN has
                # been pulled above the quantile anchor. A large gap means
                # neighbours are filling the ring; the anchor should still be
                # sound, but it is worth knowing which GUVs sit in crowded
                # neighbourhoods.
                with np.errstate(divide='ignore', invalid='ignore'):
                    gap = (diag['bg_median_raw'] - diag['bg_estimate']) / diag['bg_std_raw']
                gap = gap[np.isfinite(gap)]
                if gap.size:
                    expected = -float(_norm_ppf(getattr(cfg, 'BG_PERCENTILE', 15.0) / 100.0))
                    if float(np.median(gap)) > expected + 1.0:
                        logger.info(
                            f"  GUV {gid}: crowded annulus — (median−quantile)/sigma "
                            f"= {np.median(gap):.2f} vs {expected:.2f} expected for a "
                            f"clean ring — neighbours are filling the ring."
                        )

            guv_results['all_background_curves']     = new_bg_curves
            guv_results['all_background_std_curves'] = new_bg_std_curves
            guv_results['all_background_n_curves']   = new_bg_n_curves

            # Optional one-off QC: bath level far from any vesicle. Off by
            # default — it costs ~50 s per run and tells you nothing about
            # kinetics. Worth running once per new imaging condition to
            # confirm the annulus is not sitting in a locally bright region.
            if getattr(cfg, 'FAR_FIELD_DIAGNOSTIC', False):
                try:
                    utils.measure_far_field_background(
                        dye_stack_bg, n_frames_bg, all_ellipses, img_shape_bg,
                        exclusion_factor=getattr(cfg, 'FAR_FIELD_EXCLUSION_FACTOR', 2.5),
                        logger=logger,
                    )
                except Exception as _e:
                    logger.warning(f"Far-field diagnostic skipped: {_e}")

            diag_csv = utils.export_background_diagnostics_csv(
                diagnostics_per_guv, data['time_array'], cfg.FOLDER_DYE,
                cfg.EXPERIMENT_BASE_NAME,
                logger=logger,
            )
            logger.info(f"Saved background diagnostics → {diag_csv}")

            del dye_stack_bg

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

            # 5. Refine the tracker's raw RUPTURED / OUT_OF_FRAME exit tags
            # into five fates — OUT_OF_FRAME / SHRUNK / RUPTURED / GROWN /
            # SURVIVED — now that PULSE_FRAME (needed for the pre-pulse
            # radius baseline) is known. Two asymmetries in the priority
            # order are deliberate:
            #   * SHRUNK OVERRIDES a raw RUPTURED tag. A vesicle that
            #     deflates gradually and only then drops below the
            #     ring-detector's minimum resolvable size will ALSO trip the
            #     rupture-score exit — a detection-limit artifact of
            #     shrinkage, not a membrane burst.
            #   * GROWN DOES NOT override RUPTURED. Swelling cannot
            #     masquerade as a rupture-score exit the way deflation can,
            #     and swell-then-burst is the canonical electroporation
            #     failure mode, so an expanding vesicle that then bursts is
            #     scored as the rupture it is.
            # OUT_OF_FRAME outranks everything: the vesicle's eventual fate
            # is simply unobserved once it leaves the field of view.
            fate_map, df_fate = utils.classify_guv_fates(
                df_track,
                guv_results.get('ruptured_guv_ids', set()),
                guv_results.get('out_of_frame_guv_ids', set()),
                shrinkage_fraction_threshold=getattr(cfg, 'SHRINKAGE_FRACTION_THRESHOLD', 0.15),
                growth_fraction_threshold=getattr(cfg, 'GROWTH_FRACTION_THRESHOLD', 0.15),
                terminal_n_frames=getattr(cfg, 'SHRINKAGE_TERMINAL_N_FRAMES', 3),
                growth_use_peak_radius=getattr(cfg, 'GROWTH_USE_PEAK_RADIUS', False),
                pulse_frame=PULSE_FRAME,
            )
            df_fate.to_csv(
                os.path.join(cfg.FOLDER_TRACKING,
                             f"{cfg.EXPERIMENT_BASE_NAME}_guv_fate_classification.csv"),
                index=False, float_format='%.4f'
            )
            ruptured_guv_ids = {g for g, f in fate_map.items() if f == 'RUPTURED'}
            shrunk_guv_ids   = {g for g, f in fate_map.items() if f == 'SHRUNK'}
            grown_guv_ids    = {g for g, f in fate_map.items() if f == 'GROWN'}
            out_of_frame_ids_refined = {g for g, f in fate_map.items() if f == 'OUT_OF_FRAME'}
            n_survived = sum(1 for f in fate_map.values() if f == 'SURVIVED')
            logger.info(
                f"GUV fate classification: {len(ruptured_guv_ids)} ruptured, "
                f"{len(shrunk_guv_ids)} shrunk (>= "
                f"{getattr(cfg, 'SHRINKAGE_FRACTION_THRESHOLD', 0.15)*100:.0f}% radius loss), "
                f"{len(grown_guv_ids)} grown (>= "
                f"{getattr(cfg, 'GROWTH_FRACTION_THRESHOLD', 0.15)*100:.0f}% radius gain, "
                f"{'peak' if getattr(cfg, 'GROWTH_USE_PEAK_RADIUS', False) else 'terminal'} radius), "
                f"{len(out_of_frame_ids_refined)} lost out of frame, "
                f"{n_survived} survived intact → "
                f"{cfg.EXPERIMENT_BASE_NAME}_guv_fate_classification.csv"
            )
            if grown_guv_ids:
                # A >15% radius gain is ~32% area, which a taut bilayer
                # cannot supply — so GROWN is either a floppy vesicle
                # rounding up (circularity rises toward 1) or a tracking /
                # fusion artifact. Flag it for manual triage rather than
                # letting it pass silently into the fate counts.
                logger.warning(
                    f"{len(grown_guv_ids)} GUV(s) classified GROWN "
                    f"({', '.join(sorted(grown_guv_ids))}). Radius gains of this "
                    f"size exceed what bilayer stretching allows, so check "
                    f"terminal_norm_circularity in the fate CSV: a rise toward 1 "
                    f"indicates a floppy vesicle tensing up (real), a flat or "
                    f"falling value suggests vesicle fusion or the ring detector "
                    f"jumping to a neighbouring object (artifact)."
                )
            # Stored back on guv_results so any function downstream that
            # only receives guv_results (e.g. fit_individual_curves) can
            # pick up the shrinkage-aware classification without needing an
            # extra parameter threaded through every call.
            guv_results['ruptured_guv_ids_refined'] = ruptured_guv_ids
            guv_results['shrunk_guv_ids']            = shrunk_guv_ids
            guv_results['grown_guv_ids']             = grown_guv_ids
            # Signed fractional radius change per GUV, so fit_individual_curves
            # can report radius_growth_flag together with the NUMBER behind it
            # without recomputing (and possibly redefining) growth.
            guv_results['fate_map'] = fate_map
            guv_results['radius_change_frac'] = dict(
                zip(df_fate['guv_id'].astype(str), df_fate['frac_radius_change'])
            )
        else:
            # No tracking dataframe: fall back to the tracker's raw
            # RUPTURED/OUT_OF_FRAME sets with no shrinkage/growth refinement.
            ruptured_guv_ids = guv_results.get('ruptured_guv_ids', set())
            shrunk_guv_ids   = set()
            grown_guv_ids    = set()
            guv_results['ruptured_guv_ids_refined'] = ruptured_guv_ids
            guv_results['shrunk_guv_ids']            = shrunk_guv_ids
            guv_results['grown_guv_ids']             = grown_guv_ids
            guv_results['fate_map'] = {}
            guv_results['radius_change_frac'] = {}
        
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

        excluded_dead = aligned.get('excluded_dead_guv_ids', [])
        if excluded_dead:
            logger.info(
                f"Excluded {len(excluded_dead)} GUV(s) from dye-channel "
                f"analysis for pre-pulse signal indistinguishable from "
                f"background: {excluded_dead}"
            )

        # guv_id -> raw_results index, built fresh here rather than relying on
        # positional alignment with aligned['valid_guv_ids'] — the dead-GUV
        # exclusion above (and other drops) can leave gaps relative to
        # guv_results['valid_guv_indices'].
        id_to_raw_idx = {
            str(data['circles'][gi]['id']): gi
            for gi in guv_results['valid_guv_indices']
        }

        # Step 4b – dye summary (all traces + mean)
        summary_path = utils.plot_dye_summary(
            aligned, cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
            ruptured_guv_ids=ruptured_guv_ids,
            out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
            shrunk_guv_ids=shrunk_guv_ids,
            grown_guv_ids=grown_guv_ids,
        )
        logger.info(f"Saved dye summary plot → {summary_path}")

        # Step 4c – model-free endpoint: pre-pulse vs final frame, per GUV.
        # Deliberately computed BEFORE the fitting step and kept independent
        # of it, so it stays valid when the post-pulse sampling is too coarse
        # for tau to be identifiable.
        if getattr(cfg, 'EXPORT_PREPOST_INTENSITY_PLOT', True):
            prepost_png, prepost_csv, df_prepost = utils.plot_dye_prepulse_vs_final(
                aligned['all_curves_full'],
                aligned['t_full'],
                aligned['median_jump_frame'],
                aligned['valid_guv_ids'],
                cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
                n_pre=getattr(cfg, 'PREPOST_N_PRE_FRAMES', 5),
                n_final=getattr(cfg, 'PREPOST_N_FINAL_FRAMES', 3),
                min_final_time_frac=getattr(
                    cfg, 'PREPOST_MIN_FINAL_TIME_FRAC', 0.5),
                max_endpoint_rise_sigma=getattr(
                    cfg, 'PREPOST_MAX_ENDPOINT_RISE_SIGMA', 5.0),
                ruptured_guv_ids=ruptured_guv_ids,
                out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                shrunk_guv_ids=shrunk_guv_ids,
                grown_guv_ids=grown_guv_ids,
            )
            logger.info(f"Saved pre-pulse vs final plot → {prepost_png}")
            logger.info(f"Saved pre-pulse vs final CSV  → {prepost_csv}")
            # Report the QC-passing population, and name every GUV left out
            # with its reason — an exclusion that is not logged is a silent
            # one, and these rows remain in the CSV for inspection.
            _pass = df_prepost[df_prepost['qc_pass'].astype(bool)]
            _ok   = _pass['released_pct'].dropna()
            if len(_ok) > 0:
                logger.info(
                    f"  Released fraction across {len(_ok)} GUV(s) "
                    f"passing endpoint QC: "
                    f"{_ok.mean():.1f} ± {_ok.std():.1f} %"
                )
            _fail = df_prepost[~df_prepost['qc_pass'].astype(bool)]
            if len(_fail) > 0:
                logger.warning(
                    f"  {len(_fail)} GUV(s) excluded from the endpoint mean "
                    f"(flagged, not removed from the CSV):"
                )
                for _, _r in _fail.iterrows():
                    logger.warning(
                        f"    GUV {_r['guv_id']}: {_r['endpoint_qc']} "
                        f"(I_final = {_r['I_final']:.2f}, "
                        f"t_final = {_r['t_final_s']:.0f} s)"
                    )

            # Matching single-vesicle crops. Keyed off id_to_raw_idx rather
            # than positional order, since the dead-GUV filter can leave gaps
            # relative to valid_guv_indices.
            if getattr(cfg, 'EXPORT_GUV_CROPS', True):
                gid_to_tracking = {
                    gid: guv_results['raw_results'][gi][0]['tracking']
                    for gid, gi in id_to_raw_idx.items()
                    if guv_results['raw_results'][gi][0] is not None
                }
                crops_png = utils.export_guv_crops_prepost(
                    df_prepost,
                    gid_to_tracking,
                    aligned['median_jump_frame'],
                    aligned['t_full'],
                    data['roi_mmap_info'],
                    data['dye_mmap_info'],
                    data.get('actin_mmap_info', (None, None, None)),
                    cfg.FOLDER_DYE_INTENSITY, cfg.EXPERIMENT_BASE_NAME,
                    crop_factor=getattr(cfg, 'CROP_FACTOR', 2.0),
                    microns_per_pixel=getattr(cfg, 'MICRONS_PER_PIXEL', 1.0),
                    scale_bar_microns=getattr(cfg, 'SCALE_BAR_LENGTH_MICRONS', 10),
                    analyze_actin=getattr(cfg, 'ANALYZE_ACTIN_CHANNEL', False),
                    save_individual=getattr(cfg, 'CROP_SAVE_INDIVIDUAL', True),
                    uniform_box=getattr(cfg, 'CROP_UNIFORM_BOX', True),
                    channel_colors=getattr(cfg, 'CROP_CHANNEL_COLORS', None),
                    ruptured_guv_ids=ruptured_guv_ids,
                    out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                    shrunk_guv_ids=shrunk_guv_ids,
                    grown_guv_ids=grown_guv_ids,
                )
                if crops_png:
                    logger.info(f"Saved single-vesicle crops → {crops_png}")

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
            valid_ids = aligned['valid_guv_ids']  # GUVs that passed the pre-pulse SNR gate

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
                ruptured_guv_ids=ruptured_guv_ids,
                out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                shrunk_guv_ids=shrunk_guv_ids,
                grown_guv_ids=grown_guv_ids,
            )
            logger.info(f"Saved actin analysis plot → {plot_path}")

            # Combined dye + actin overlay (only when both pipelines ran)
            if aligned is not None:
                overlay_path = utils.plot_dye_actin_overlay(
                    aligned, aligned_actin, valid_ids,
                    cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                    smooth_sigma=getattr(cfg, 'ACTIN_PLOT_SMOOTH_SIGMA', 1.5),
                    ruptured_guv_ids=ruptured_guv_ids,
                    out_of_frame_guv_ids=guv_results.get('out_of_frame_guv_ids'),
                    shrunk_guv_ids=shrunk_guv_ids,
                    grown_guv_ids=grown_guv_ids,
                )
                logger.info(f"Saved dye/actin overlay → {overlay_path}")

            # Actin angular kymographs (directionality of cortex breakdown).
            # Built from a fresh guv_id -> actin_data mapping keyed off
            # valid_guv_indices (the order all_actin_data was appended in),
            # rather than reusing valid_ids from the dye-channel dead-GUV
            # filter above — those two lists can diverge in length/order
            # whenever a GUV is excluded by the dye SNR gate but still has
            # usable actin data, or vice versa.
            if getattr(cfg, 'EXPORT_ACTIN_KYMOGRAPH', True):
                n_angles_cfg = getattr(cfg, 'ACTIN_N_ANGLES', 72)
                electrode_angle = getattr(cfg, 'ELECTRODE_ANGLE_DEG', 0.0)
                display_offset  = getattr(cfg, 'ANGLE_DISPLAY_OFFSET_DEG', 90.0)

                actin_gid_ad_pairs = [
                    (str(data['circles'][gi]['id']), ad)
                    for gi, ad in zip(guv_results['valid_guv_indices'],
                                      guv_results.get('all_actin_data', []))
                    if ad is not None
                ]

                kymo_ids, kymo_data_list = [], []
                for gid, ad in actin_gid_ad_pairs:
                    kd = utils.build_actin_angular_kymograph(
                        ad, PULSE_FRAME, data['time_array'], n_angles=n_angles_cfg
                    )
                    if kd is None:
                        continue
                    kymo_path = utils.plot_actin_angular_kymograph(
                        kd, gid, cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                        electrode_angle_deg=electrode_angle,
                        angle_display_offset_deg=display_offset,
                    )
                    logger.info(f"  GUV {gid}: saved actin angular kymograph → {kymo_path}")

                    pe_data = utils.compute_pole_vs_equator_trace(
                        kd, electrode_angle_deg=electrode_angle,
                        angular_half_width_deg=getattr(cfg, 'POLE_EQUATOR_HALF_WIDTH_DEG', 22.5),
                    )
                    pe_path = utils.plot_pole_vs_equator(
                        pe_data, gid, cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME
                    )
                    logger.info(f"  GUV {gid}: saved pole-vs-equator trace → {pe_path}")

                    kymo_ids.append(gid)
                    kymo_data_list.append(kd)

                if kymo_data_list:
                    group_path = utils.plot_actin_angular_kymograph_group(
                        kymo_data_list, kymo_ids,
                        cfg.FOLDER_ACTIN, cfg.EXPERIMENT_BASE_NAME,
                        electrode_angle_deg=electrode_angle,
                        angle_display_offset_deg=display_offset,
                    )
                    logger.info(f"Saved population-average actin kymograph → {group_path}")
                else:
                    logger.info("Actin kymograph: no GUV had a detected cortex peak to plot.")

            # Radial profile snapshot at the last pre-pulse frame
            pre_frame = max(0, PULSE_FRAME - 1)
            snapshots = []
            
            for gid in valid_ids:
                raw_idx = id_to_raw_idx.get(gid)
                if raw_idx is None:
                    continue
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
                for gid in valid_ids:
                    raw_idx = id_to_raw_idx.get(gid)
                    if raw_idx is None:
                        continue
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