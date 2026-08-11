# =============================================================================
# PATCH 1 of 4 — guv_analysis_utils.py
# Paste this function at the end of the file (or next to classify_guv_fates).
# =============================================================================

def check_synchronized_exit(df_track,
                            fate_map,
                            n_frames,
                            logger,
                            pulse_frame=None,
                            min_fraction=0.5,
                            window_frames=3,
                            min_guvs=3):
    """Flag fields of view where the tracker lost every GUV at the same instant.

    Genuine electroporative rupture is stochastic across vesicles: each one
    fails at its own time, so exit frames scatter. A focal-plane excursion,
    stage bump or illumination dropout removes every vesicle from the image
    simultaneously, and the tracker emits RUPTURED for all of them within a
    frame or two. The vesicles are usually intact when the plane returns, but
    RUPTURED is a terminal exit state, so the tracker never re-acquires and
    the fate table records a field-wide lysis event that did not happen.

    The test is deliberately on exit-frame clustering rather than on the
    rupture count. A high rupture fraction can be real at high field strength;
    a rupture fraction that is high AND concentrated in one frame cannot be.

    Exits inside the pulse window are reported but not failed: mass poration
    at the pulse is the experiment working, not an artefact.

    Returns a dict. Also emits one machine-readable '[QC:SYNC_EXIT]' line so
    batch_run.py can surface it without parsing the log file.
    """
    import numpy as np
    import pandas as pd

    result = {'status': 'PASS', 'n_guv': 0, 'cluster_n': 0,
              'cluster_frac': 0.0, 'exit_frame': None, 'reason': ''}

    if df_track is None or df_track.empty or 'frame' not in df_track.columns:
        result['reason'] = 'no tracking data'
        print("[QC:SYNC_EXIT] SKIP  no tracking data", flush=True)
        return result

    radius_col = 'radius' if 'radius' in df_track.columns else None
    valid = df_track if radius_col is None else \
        df_track[pd.to_numeric(df_track[radius_col], errors='coerce').notna()]
    if valid.empty:
        result['reason'] = 'no valid radii'
        print("[QC:SYNC_EXIT] SKIP  no valid radii", flush=True)
        return result

    last_frame = valid.groupby('guv_id')['frame'].max()
    n_guv = int(last_frame.size)
    result['n_guv'] = n_guv

    # A GUV tracked to the final frame did not exit; it is part of the
    # denominator but cannot be part of a cluster.
    exited = last_frame[last_frame < (int(n_frames) - 1)]
    if n_guv < min_guvs or exited.empty:
        print(f"[QC:SYNC_EXIT] PASS  {len(exited)}/{n_guv} exited early", flush=True)
        return result

    frames = np.sort(exited.to_numpy(dtype=float))
    best_n, best_f = 0, np.nan
    for f in frames:
        n_in = int(((frames >= f) & (frames <= f + window_frames)).sum())
        if n_in > best_n:
            best_n, best_f = n_in, f

    frac = best_n / n_guv
    result.update(cluster_n=best_n, cluster_frac=round(float(frac), 3),
                  exit_frame=int(best_f))

    at_pulse = (pulse_frame is not None
                and best_f <= float(pulse_frame) + window_frames)

    if best_n >= min_guvs and frac >= min_fraction and not at_pulse:
        result['status'] = 'FAIL'
        result['reason'] = 'synchronized exit away from the pulse'
        msg = (f"{best_n}/{n_guv} GUV(s) ({frac:.0%}) stopped tracking within "
               f"{window_frames} frame(s) of frame {int(best_f)}")
        logger.warning(
            f"SYNCHRONIZED EXIT: {msg}. Vesicles do not rupture in unison — "
            f"this is the signature of a focal-plane excursion, stage bump or "
            f"illumination dropout, and every affected GUV will have been "
            f"labelled RUPTURED. Open the ALL_TRACKING video around frame "
            f"{int(best_f)} before trusting any fate from this experiment. If "
            f"the vesicles are intact after the event, truncate the record via "
            f"cfg.FRAME_TRUNCATION rather than discarding the field of view."
        )
        print(f"[QC:SYNC_EXIT] FAIL  {msg}", flush=True)
    else:
        if at_pulse and best_n >= min_guvs and frac >= min_fraction:
            logger.info(
                f"{best_n}/{n_guv} GUV(s) exited within {window_frames} frame(s) "
                f"of frame {int(best_f)}, at the pulse. Clustering at the pulse "
                f"is expected for mass poration and is not flagged."
            )
            result['reason'] = 'cluster at pulse, not flagged'
        print(f"[QC:SYNC_EXIT] PASS  largest cluster {best_n}/{n_guv} "
              f"at frame {int(best_f)}", flush=True)

    return result


# =============================================================================
# PATCH 2 of 4 — config.py
# Paste next to the fate-classification constants (around line 540).
# =============================================================================

# Synchronized-exit QC. Fails a field of view when this fraction or more of
# its GUVs stop tracking within QC_SYNC_EXIT_WINDOW_FRAMES of each other,
# away from the pulse. Vesicles rupture independently, so simultaneous exit
# is an imaging failure. 0.5 is deliberately permissive: the failure mode it
# catches produces fractions near 1.0, and a lower gate would start flagging
# genuine high-field poration.
QC_SYNC_EXIT_MIN_FRACTION  = 0.5
QC_SYNC_EXIT_WINDOW_FRAMES = 3
QC_SYNC_EXIT_MIN_GUVS      = 3


# =============================================================================
# PATCH 3 of 4 — run_analysis.py
# Paste immediately AFTER the logger.info(...) block that reports
# "GUV fate classification: ... survived intact -> ..." (around line 1981),
# INSIDE the same `if` block, at the same indentation as that logger.info.
# =============================================================================

            utils.check_synchronized_exit(
                df_track,
                fate_map,
                n_frames=data['roi_mmap_info'][1][0],
                logger=logger,
                pulse_frame=PULSE_FRAME,
                min_fraction=getattr(cfg, 'QC_SYNC_EXIT_MIN_FRACTION', 0.5),
                window_frames=getattr(cfg, 'QC_SYNC_EXIT_WINDOW_FRAMES', 3),
                min_guvs=getattr(cfg, 'QC_SYNC_EXIT_MIN_GUVS', 3),
            )


# =============================================================================
# PATCH 4 of 4 — batch_run.py
# Three edits. Replace the marked blocks with the versions below.
# =============================================================================

# --- 4a. In run_one(), replace the stdout loop and the return -----------------
#     OLD:
#         for line in proc.stdout:
#             print(line.rstrip(), flush=True)
#         proc.wait()
#         return proc.returncode
#     NEW:

  qc_lines = []
  for line in proc.stdout:
      line = line.rstrip()
      print(line, flush=True)
      if "[QC:" in line:
          qc_lines.append(line[line.index("[QC:"):])
  proc.wait()
  return proc.returncode, qc_lines


# --- 4b. In main(), the execute loop -----------------------------------------
#     Replace the three lines that call run_one and append to results.
#     NEW:

    try:
      code, qc = run_one(data_folder, base_name, repo_dir)
    except KeyboardInterrupt:
      print("\nInterrupted.")
      break
    except Exception as e:  # a crash here must not kill the remaining queue
      print(f"  launcher error: {e}")
      code, qc = -1, []
    results.append((base_name, code, time.time() - started, qc))


# --- 4c. Replace the batch summary block at the end of main() ----------------

  print(f"\n{'=' * 70}\nBatch summary\n{'=' * 70}")
  for base_name, code, elapsed, qc in results:
    status = "ok  " if code == 0 else f"FAIL({code})"
    flag = "  <-- QC FLAG" if any("FAIL" in q for q in qc) else ""
    print(f"  {status}  {elapsed / 60:6.1f} min  {base_name}{flag}")
  n_failed = sum(1 for _, code, _, _ in results if code != 0)
  print(f"\n{len(results) - n_failed} succeeded, {n_failed} failed.")
  if n_failed:
    print("Failed experiments left their logs in their own output folders.")

  flagged = [(b, q) for b, _, _, qc in results for q in qc if "FAIL" in q]
  n_checked = sum(1 for _, _, _, qc in results if qc)
  print(f"\n{'=' * 70}\nQC: synchronized exit\n{'=' * 70}")
  if flagged:
    print(f"  {len(flagged)} of {n_checked} experiment(s) flagged. Open the")
    print("  ALL_TRACKING video for each before using its fate labels:")
    for base_name, q in flagged:
      print(f"    {base_name}")
      print(f"        {q}")
  else:
    print(f"  No flags across {n_checked} experiment(s) checked.")