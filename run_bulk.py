"""Run the post-batch analysis: bulk intensity figures and the tau report.

Order matters only in that both read the per-experiment CSVs written by
run_analysis.py, so batch_run.py must have finished first. The two halves are
otherwise independent and either can be switched off below.

    INTENSITY  process.py  -> per-GUV bulk summary, size/intensity figures,
                              released fraction vs voltage, cortex split,
                              cortex-breakdown and area-loss onset fields
    TAU        tau_identifiability_report.py
                           -> why tau is not reportable, pooled over all
                              experiments, with the per-constraint breakdown
    MONTAGE    cortex_montage.py
                           -> actin-channel crops of GUVs with and without a
                              cortex. The only stage that reopens the raw ND2s,
                              so it needs DATA_ROOT and the nd2 package, takes
                              minutes rather than seconds, and is the one most
                              likely to fail on its own -- hence its own toggle.

They are deliberately NOT merged into one figure script. The intensity
analysis aggregates one row per GUV from three different CSVs and gates on
size category; the tau report reads the raw fit parameters and gates on
nothing, because its subject is precisely the GUVs the other gates discard.
Sharing a loader would mean one of them silently inherits the other's filter.

In Spyder: open this file and press F5.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# --- SPYDER SETTINGS ---
# -----------------------------------------------------------------------------

RUN_INTENSITY = True    # process.py: bulk summary + intensity figures
RUN_TAU       = True    # tau identifiability diagnostic
RUN_MONTAGE   = True    # actin montages; needs the raw ND2s, slowest stage
RUN_SUSCEPT   = True    # size-matched Bare vs Branched comparison

# Only the montage needs this: the tree holding the .nd2 files.
MONTAGE_DATA_ROOT = r"D:\Data\EP"

# None = take PARENT_OUTPUT_FOLDER from config.py. Override only to point at
# a copied or archived Outputs tree.
OUTPUTS_ROOT = None


def resolve_root() -> Path:
  if OUTPUTS_ROOT is not None:
    return Path(OUTPUTS_ROOT)
  import config as cfg
  return Path(cfg.PARENT_OUTPUT_FOLDER)


def results_dir_for(root: Path) -> Path:
  """Where the bulk outputs go: a subfolder of the per-experiment tree."""
  import process
  out = root / process.RESULTS_SUBFOLDER
  out.mkdir(parents=True, exist_ok=True)
  return out


def run_intensity(root: Path):
  import process

  print(f"\n{'=' * 70}\nBulk intensity analysis\n{'=' * 70}")
  results = results_dir_for(root)
  # Two populations, deliberately.
  #
  # df_all keeps every tracked vesicle. df keeps only those inside
  # process.SIZE_WINDOW_UM, where the two preparations are matched on radius.
  #
  # Which one a stage gets depends on whether it compares the preparations.
  # Anything that puts Bare against Branched has to run inside the window, or
  # the comparison confounds cortex presence with vesicle size. Anything that
  # describes the cortex-bearing population on its own -- how the actin shell
  # loses height, whether it fails at the poles -- makes no such comparison,
  # and inside the window it would lose three quarters of its vesicles and
  # the 30 V reference it is read against, in exchange for removing a
  # confound it does not contain.
  #
  # The cost of doing it this way is that the endpoint is read for every
  # vesicle including the ones the window discards. Filtering earlier would
  # save that work but would leave the cortex stages with nothing to run on.
  df_all = process.aggregate_pipeline_results(str(root))
  df = process.apply_size_window(df_all)
  if df.empty:
    print("No GUV records found -- has batch_run.py been run?")
    return None

  summary_csv = results / "guv_bulk_summary.csv"
  df.to_csv(summary_csv, index=False)
  print(f"Saved {summary_csv}  ({len(df)} GUV rows, "
        f"{df['experiment'].nunique()} experiments)")
  print(f"  This file is the SIZE-MATCHED population and is what "
        f"susceptibility.py reads. The cortex-breakdown stages below run on "
        f"all {len(df_all)} vesicles instead; see the note in run_intensity.")
  process.write_layout_readme(results)

  # Aggregated once and passed to each consumer, rather than letting
  # process.py's own __main__ re-read every CSV per figure.
  # Calibration figures for the thresholds the split depends on. Produced
  # from the unsplit frame, since they are what the split is read from.
  process.plot_lumen_abs_histogram(df_all, str(results))
  # Actin-side metric: how much radial cortex peak height was lost.
  df_peak = process.add_cortex_peak_metrics(df_all, str(root))
  process.plot_cortex_peak_breakdown(df_peak, str(results))
  # Onset fields: the sigmoid midpoint in volts for cortex breakdown and for
  # area loss, each with a bootstrap CI. Fitted on df merged with
  # peak_drop_final rather than on df_peak itself, because df_peak is an inner
  # merge and so contains only vesicles with an actin trace -- Bare GUVs have
  # none, and dropping them would leave the area-loss onset with one
  # population to fit instead of two. The merge is left, so bare vesicles
  # carry NaN for the cortex measure and are simply absent from that fit.
  #
  # One thing this does NOT yet do, an open decision rather than an
  # oversight: the two measures do not have to rest on the same vesicles.
  # Area loss is available for every GUV with a terminal radius, the cortex
  # drop only for those whose actin trace reached ENDPOINT_MATCHED_T_S. Check
  # n_guv per measure in onset_fields.csv before reading the two midpoints as
  # an ordering. The cortex split is applied inside report_onset_fields, so
  # the frame passed here is deliberately the unsplit one.
  process.report_onset_fields(
      df_all.merge(df_peak[["experiment", "guv_id", "peak_drop_final"]],
                   on=["experiment", "guv_id"], how="left")
      if not df_peak.empty else df_all,
      str(results))
  process.report_bleach_comparability(str(root))
  process.plot_cortex_peak_timecourse(df_all, str(root), str(results))
  # Directionality of that loss, pooled across vesicles. Reads the
  # per-experiment *_pole_vs_equator_traces.csv files, so it needs
  # EXPORT_ACTIN_KYMOGRAPH = True on the run that produced them.
  #
  # Given df_peak, not df: the conditioned test ("of the cortices that broke
  # down, did they break down directionally?") needs peak_drop_final, which
  # only exists after the actin merge. Falls back to df so the pooled test
  # still runs when no actin traces were found.
  process.plot_pole_equator_pooled(
      df_peak if not df_peak.empty else df_all, str(root), str(results))
  process.plot_cortex_contrast_histogram(df_all, str(results))
  process.report_cortex_status(df_all)

  # --- primary figure set: BranchedCortex split by phenotype ---------------
  df_split = process.apply_cortex_split(df)
  # Printed before the figures that use both metrics, so the overlap between
  # the response call and the endpoint category is in the log above them.
  process.report_response_agreement(df_split)
  process.generate_separate_pdf_plots(df_split, str(results))
  process.plot_released_fraction_by_voltage(df_split, str(results))
  process.plot_drop_by_cortex_status(df, str(results))
  process.report_tau_yield(df_split)
  process.plot_tau_by_size_and_voltage(df_split, str(results))

  # --- pooled reference set ------------------------------------------------
  # Same figures with BranchedCortex undivided. Kept because the split rests
  # on provisional thresholds: if a difference between phenotypes is real, it
  # should not appear or vanish depending on where those thresholds sit, and
  # the pooled version is what that gets checked against.
  pooled_dir = process.sub_dir(results, "pooled")
  print(f"\n--- pooled (unsplit) reference figures -> {pooled_dir.name}/ ---")
  process.generate_separate_pdf_plots(df, str(pooled_dir))
  process.plot_released_fraction_by_voltage(df, str(pooled_dir))
  process.report_tau_yield(df)
  process.plot_tau_by_size_and_voltage(df, str(pooled_dir))
  return df


def run_tau(root: Path):
  import tau_identifiability_report as tau

  print(f"\n{'=' * 70}\nTau identifiability\n{'=' * 70}")
  return tau.main(root, results_dir_for(root))


def run_susceptibility(root: Path):
  import susceptibility

  print(f"\n{'=' * 70}\nSusceptibility: bare, cortex and lumenal-only "
        f"at matched dV_m\n{'=' * 70}")
  # Reads guv_bulk_summary.csv, so it must follow the intensity stage.
  return susceptibility.main(results_dir_for(root), root)


def run_montage(root: Path):
  import cortex_montage

  print(f"\n{'=' * 70}\nActin montages\n{'=' * 70}")
  return cortex_montage.main(MONTAGE_DATA_ROOT, results_dir_for(root))


def main():
  root = resolve_root()
  if not root.is_dir():
    raise SystemExit(f"Outputs folder does not exist: {root}")
  print(f"Outputs root: {root}")

  # Each half is wrapped so a failure in one still leaves the other's figures
  # on disk -- a long batch should not be re-run because a plot raised.
  for enabled, name, fn in ((RUN_INTENSITY, "intensity", run_intensity),
                            (RUN_TAU, "tau", run_tau),
                            (RUN_SUSCEPT, "susceptibility",
                             run_susceptibility),
                            (RUN_MONTAGE, "montage", run_montage)):
    if not enabled:
      print(f"\nSkipping {name} (disabled in the settings block).")
      continue
    try:
      fn(root)
    except (Exception, SystemExit) as e:
      # SystemExit is caught deliberately: the stages raise it for missing
      # paths, and it is not an Exception, so it would otherwise skip this
      # guard and abort the remaining stages along with the final summary.
      print(f"\n{name} analysis FAILED: {type(e).__name__}: {e}")
      import traceback
      traceback.print_exc()

  print(f"\n{'=' * 70}\nDone. Figures and CSVs are in:"
        f"\n  {results_dir_for(root)}\n{'=' * 70}")


if __name__ == "__main__":
  main()