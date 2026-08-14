import re
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import config as cfg

POPULATIONS = ["Empty", "Bare", "BranchedCortex",
               "Branched, cortex", "Branched, lumenal only",
               "Branched, ambiguous", "Branched, unclassified"]

POPULATION_LABELS = {"Empty": "Bare", "Bare": "Bare",
                     "BranchedCortex": "Branched cortex"}

EXCLUDE_GROUPS = ["Branched, ambiguous", "Branched, unclassified"]
from palette import (
    PALETTE,
    POPULATION_COLORS,
    GROUP_COLORS,
    GROUP_ORDER,
    SIZE_CATEGORY_COLORS,
    SIZE_BIN_COLORS,
    CORTEX_STATUS_COLORS,
    INDIVIDUAL_TRACE,
    SUMMARY_LINE,
    ANNOTATION_TEXT,
)


# -----------------------------------------------------------------------------
# --- FIGURE STYLE ---
# Per-figure size and font control, so a figure can be resized for a thesis
# page without editing the plotting function it lives in.
#
# Every entry defaults to None, which means "leave whatever the plotting code
# already set". Nothing changes until a value is filled in, so existing figures
# come out of a re-run byte-identical to before. Fill in only the fields you
# want to take over.
#
# Sizes are inches, font sizes are points.
#
#   width            total figure width. Overrides everything else.
#   width_per_panel  width of ONE panel; the figure is this times the number
#                    of panels drawn. Only meaningful for the multi-panel
#                    figures (noted below) -- ignored elsewhere. Use this
#                    rather than `width` if you want the size to keep tracking
#                    how many populations are present.
#   height           total figure height.
#   axis_label       x and y axis label text.
#   tick_label       numbers and category names along both axes.
#   legend           legend entries.
#   legend_title     the legend's own title, where one is set.
#   title            per-axes title.
#   suptitle         the figure-wide title, where one is set.
#   annotation       text drawn inside the axes (n = counts, percentages).
#
# Example -- a single-column figure for the thesis:
#
#   "lumen_abs_histogram": dict(
#       width=3.4, height=2.6,
#       axis_label=8, tick_label=7, legend=7, annotation=6,
#   ),
#
# The keys below are the eleven figures this script writes, named after their
# PDF. Panel counts scale with the number of populations present unless noted.
# -----------------------------------------------------------------------------

_FIG_FIELDS = ("width", "width_per_panel", "height", "axis_label",
               "tick_label", "legend", "legend_title", "title",
               "suptitle", "annotation")


def _fig(**kwargs):
    """One figure's overrides. Unspecified fields stay None (= don't touch)."""
    bad = set(kwargs) - set(_FIG_FIELDS)
    if bad:
        raise ValueError(f"unknown figure style field(s): {sorted(bad)}. "
                         f"Valid fields: {list(_FIG_FIELDS)}")
    return {f: kwargs.get(f) for f in _FIG_FIELDS}


FIGURE_STYLE = {
    # 01_dye_release
    "released_fraction_by_voltage":  _fig(),   # single panel
    "intensity_drop_trajectories":   _fig(),   # one panel per population
    "stagnate_intensity_counts":     _fig(),   # single panel
    "drop_by_size_per_voltage":      _fig(),   # one panel per population
    # 02_size_and_fate
    "size_category_distribution":    _fig(),   # single panel
    # 03_cortex_classification
    "lumen_abs_histogram":           _fig(),   # single panel
    "cortex_contrast_histogram":     _fig(),   # single panel
    "drop_by_cortex_status":         _fig(),   # one panel per population
    # 04_cortex_breakdown
    "onset_fields":                  _fig(),   # one panel per population
    "cortex_peak_breakdown":         _fig(),   # single panel
    "cortex_peak_timecourse":        _fig(),   # fixed at 2 panels
    "pole_equator_pooled":           _fig(),   # fixed at 2 panels
    # 05_kinetics
    "tau_by_size_voltage":           _fig(),   # one panel per population
}


def _style(key: str) -> dict:
    return FIGURE_STYLE.get(key) or {f: None for f in _FIG_FIELDS}


def fig_size(key: str, default_w: float, default_h: float, n_panels: int = 1):
    """Figure size for `key`, falling back to what the plotting code computed.

    `default_w` is the width the code would have used, already multiplied out
    for the panel count, so an untouched entry reproduces the old figure
    exactly.
    """
    s = _style(key)
    if s["width"] is not None:
        w = float(s["width"])
    elif s["width_per_panel"] is not None:
        w = float(s["width_per_panel"]) * max(1, int(n_panels))
    else:
        w = float(default_w)
    h = float(default_h) if s["height"] is None else float(s["height"])
    return (w, h)


def style_figure(key: str) -> None:
    """Apply the font sizes for `key` to the current figure.

    Called immediately before tight_layout so the new sizes are what the
    layout is computed against; calling it after would leave labels clipped
    or floating. Applied at the end rather than at each draw call because the
    inline fontsize= arguments are scattered through the plotting functions,
    and overriding them in one place keeps the knob in one place too.

    Anything left as None is not touched, so a figure with no overrides keeps
    the sizes its plotting function chose.
    """
    s = _style(key)
    fig = plt.gcf()

    for ax in fig.get_axes():
        if s["axis_label"] is not None:
            ax.xaxis.label.set_size(s["axis_label"])
            ax.yaxis.label.set_size(s["axis_label"])
        if s["tick_label"] is not None:
            ax.tick_params(axis="both", which="both",
                           labelsize=s["tick_label"])
        if s["title"] is not None and ax.get_title():
            ax.title.set_size(s["title"])
        if s["annotation"] is not None:
            for txt in ax.texts:
                txt.set_fontsize(s["annotation"])
        leg = ax.get_legend()
        if leg is not None:
            if s["legend"] is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(s["legend"])
            if s["legend_title"] is not None and leg.get_title() is not None:
                leg.get_title().set_fontsize(s["legend_title"])

    if s["suptitle"] is not None:
        sup = getattr(fig, "_suptitle", None)
        if sup is not None:
            sup.set_size(s["suptitle"])


RESULTS_SUBFOLDER = "Bulk_Analysis_Results"

LAYOUT = {
    "dye":              "01_dye_release",
    "dye_by_voltage":   "01_dye_release/by_voltage",
    "fate":             "02_size_and_fate",
    "cortex_class":     "03_cortex_classification",
    "cortex_breakdown": "04_cortex_breakdown",
    "kinetics":         "05_kinetics",
    "pooled":           "06_pooled_reference",
}

README = """Bulk_Analysis_Results
=====================

guv_bulk_summary.csv      one row per GUV, every experiment. Start here.

01_dye_release/           what the dye did: released fraction per vesicle
                          against voltage, drop trajectories, and the
                          stagnate lose/flatline breakdown.
   by_voltage/            per-voltage size-vs-drop panels, one file each.

02_size_and_fate/         what became of the vesicles: grow / stagnate /
                          reduce / rupture.

03_cortex_classification/ whether a cortex was there BEFORE the pulse, and
                          how well that call can be trusted: the contrast and
                          lumen-level histograms the thresholds are read from,
                          and the dye response split by that call.
   montages/              actin crops of GUVs called cortex vs lumenal-only.
                          The visual check on the two histograms.

04_cortex_breakdown/      what happened to the cortex after the pulse: peak
                          height lost, its full time course, and whether the
                          loss is directional (poles vs equator). All three
                          carry the 30 V control as a reference; the
                          directional one also carries the pre-pulse index,
                          which is zero by construction and so measures its
                          own noise floor.

05_kinetics/              time constants. Last because the records mostly
                          cannot constrain them -- tau_identifiability.pdf
                          shows why, and is the justification for reporting
                          release magnitude instead.

06_pooled_reference/      the same figures with BranchedCortex NOT split by
                          cortex presence. The split rests on provisional
                          thresholds; a real difference between phenotypes
                          should not appear or vanish with where those
                          thresholds sit, and this is what that is checked
                          against.
"""

def write_layout_readme(results_dir) -> None:
  path = Path(results_dir) / "README.txt"
  path.write_text(README, encoding="utf-8")
  print(f"Saved layout guide: {path}")

def sub_dir(output_dir, key: str) -> Path:
  out = Path(output_dir) / LAYOUT[key]
  out.mkdir(parents=True, exist_ok=True)
  return out

ENDPOINT_N_FRAMES = 3
PRE_PULSE_TOLERANCE = 0.05
INTENSITY_DIFF_THRESHOLD = 0.05

# -----------------------------------------------------------------------------
# --- WHERE THE ENDPOINT IS READ ---
#
# None (default) reads i_final, the cortex peak drop and the response call at
# each record's OWN last frames. That is the behaviour every earlier run had.
#
# It is only safe while every experiment runs for the same length, and these
# do not: the 260422 and 260428 sessions record for about 721 s, the 260507
# ones for about 431 s. Dye keeps leaking and fluorophores keep bleaching for
# the whole record, so a vesicle watched for 67% longer has 67% longer to
# accumulate both before its endpoint is taken. Voltage is not randomised
# across sessions here -- 400 V comes almost entirely from the long records --
# so record length, session and voltage are confounded with each other, and a
# difference between conditions cannot be attributed to the field.
#
# Setting this to a number in seconds reads every endpoint at that same
# elapsed time after the pulse instead, using the last ENDPOINT_N_FRAMES
# samples at or before it. report_bleach_comparability already prints the
# instruction to compare at matched elapsed time; this is what carries it out.
#
# 431.0 is the natural value for this dataset: the shortest full-length
# record. Experiments that end before the target cannot be read there at all,
# so their GUVs get NaN rather than a substituted earlier endpoint -- a
# shorter record is censored, and filling it in with its own end is exactly
# the bias being removed. Which experiments those are, and how many vesicles
# they cost, is printed at aggregation time.
#
# Run it both ways. A result that only exists in one of them is a result about
# the recording schedule.
# -----------------------------------------------------------------------------
ENDPOINT_MATCHED_T_S = 431.0

# How far short of the target a record may end and still be read there.
#
# The test was `t_final_s < ENDPOINT_MATCHED_T_S` with nothing to spare, so a
# record ending at 430.6 s was censored on the same footing as one ending at
# 100 s. Frames arrive every 5 s and the target is the observed length of the
# shortest full record, which is itself one of these samples -- so whether a
# session clears its own nominal length comes down to where the last frame
# happens to land, and a run can lose half its experiments to a rounding
# difference.
#
# One frame is the honest allowance: it is the resolution at which the record
# length is known at all. It is NOT a licence to read a materially shorter
# record at the target -- the endpoint is still taken from the last samples at
# or before ENDPOINT_MATCHED_T_S, so a record admitted under the tolerance is
# read a few seconds early, not extrapolated. Set to 0.0 to restore the strict
# test.
ENDPOINT_MATCHED_TOLERANCE_S = 5.0

# Fates dropped before any vesicle enters the bulk table.
#
# LOST_PREPULSE: the tracker lost the vesicle before the pulse was delivered,
# so there is no post-pulse observation at all. Without this guard the row
# falls through to the "Stagnate" default below and is counted as a survivor
# of a pulse it never received.
#
# OUT_OF_FRAME: the vesicle left the measurable window. An early exit is the
# same problem as LOST_PREPULSE; a late exit leaves a partial trace that is
# real but censored at an unknown point, and the endpoint metrics (i_final,
# frac_remaining_Xs) would read that censoring as a measurement. Both are
# dropped, which is the conservative denominator: a vesicle that was not
# watched to the end is not evidence that it survived.
#
# Dropping OUT_OF_FRAME here also decouples the Stagnate population from
# OUT_OF_FRAME_CLEARANCE_R, which only decides RUPTURED vs OUT_OF_FRAME.
# Moving that threshold can no longer shift the dye, tau, or susceptibility
# results -- it is confined to the rupture fraction in 02_size_and_fate/.
EXCLUDE_FATES = ("LOST_PREPULSE", "OUT_OF_FRAME")
CORTEX_STATUS_ORDER = ["CORTEX", "AMBIGUOUS", "NO_CORTEX", "UNKNOWN"]
TAU_REQUIRE_IDENTIFIABLE = True
TAU_REQUIRE_RESPONDING = True

def extract_voltage(path_str: str) -> str:
  match = re.search(r"(\d+)\s*V", path_str, re.IGNORECASE)
  return f"{match.group(1)}V" if match else "Unknown"

# Folder-name variants that are the same preparation under two names. The
# protein-free population was recorded as DOPC_Empty in some sessions and
# DOPC_Bare in others; left alone, extract_population returns two different
# strings and _present_populations then draws them as two panels, both titled
# "Bare" by POPULATION_LABELS. apply_cortex_split already merges them, so the
# split figures were right and the pooled reference and the onset fits were
# not -- the bare population was halved in exactly the place a sigmoid needs
# every vesicle it can get. Folding them here means one population name
# reaches the table and every downstream consumer agrees by construction.
POPULATION_ALIASES = {"Empty": "Bare"}


def extract_population(path_str: str) -> str:
  for pop in POPULATIONS:
    if re.search(pop, path_str, re.IGNORECASE):
      return POPULATION_ALIASES.get(pop, pop)
  return "Unknown"

def assign_size_bin(r_um: float) -> str:
  if pd.isna(r_um):
    return "Unknown"
  elif r_um < 5.0:
    return "< 5 µm"
  elif 5.0 <= r_um <= 8.0:
    return "5-8 µm"
  elif 8.0 < r_um <= 12.0:
    return "9-12 µm"
  else:
    return "> 13 µm"

def _post_pulse_mask(df_norm: pd.DataFrame) -> pd.Series:
  if "phase" in df_norm.columns:
    return df_norm["phase"] == "post_pulse"
  elif "Time (s)" in df_norm.columns:
    return df_norm["Time (s)"] > 0
  return df_norm.index >= 5

def _pre_pulse_mask(df_norm: pd.DataFrame) -> pd.Series:
  if "phase" in df_norm.columns:
    return df_norm["phase"] == "pre_pulse"
  elif "Time (s)" in df_norm.columns:
    return df_norm["Time (s)"] < 0
  return df_norm.index < 5

def load_cortex_status(exp_dir: Path) -> pd.DataFrame:
  files = list(exp_dir.glob("**/*_actin_cortex_status.csv"))
  if not files:
    return None
  try:
    return pd.read_csv(files[0])
  except Exception as e:
    print(f"Skipping cortex status file {files[0]}: {e}")
    return None

_RESPONDING_IMPORT_FAILED = []


def _responding_at(t_c: np.ndarray, y_c: np.ndarray):
  """run_analysis's own response call, applied to a truncated trace.

  _observed_response and the two thresholds are imported rather than
  reimplemented. A second copy of this rule in this file is precisely the
  drift that would make the matched-time and whole-record numbers
  incomparable for a reason that had nothing to do with the data.

  Returns True/False, or None if the rule could not be applied (in which case
  the caller leaves the original whole-record call in place).
  """
  try:
    from run_analysis import _observed_response
  except Exception as e:
    if not _RESPONDING_IMPORT_FAILED:
      _RESPONDING_IMPORT_FAILED.append(e)
      print(f"  WARNING: cannot import run_analysis._observed_response ({e}). "
            "is_responding is left as computed over the whole record, so it "
            "does not match the truncated i_final. Do not read the responding "
            "fraction from this run.")
    return None

  efflux = str(getattr(cfg, "MODEL_TO_USE", "EFFLUX")).startswith("EFFLUX")
  drop, noise = _observed_response(np.asarray(t_c, float),
                                   np.asarray(y_c, float), efflux=efflux)
  k = float(getattr(cfg, "FIT_RESPONSE_AMPLITUDE_SIGMA", 3.0))
  min_amp = float(getattr(cfg, "FIT_MIN_RESPONSE_AMPLITUDE", 0.05))
  if not (np.isfinite(drop) and np.isfinite(noise)):
    return None
  return bool(drop > max(min_amp, k * noise))


# -----------------------------------------------------------------------------
# --- SIZE WINDOW ---
# Radius range, in micrometres, that every vesicle must fall inside to enter
# the bulk table. None disables the filter and restores the whole population.
#
# The two preparations differ by 2.09 um in median radius (6.58 vs 4.49), and
# the induced transmembrane potential scales with radius, so a comparison at
# matched applied field is also a comparison at mismatched dose. There are two
# ways to answer that. Correct for it, by plotting against 1.5*E*R; or remove
# it, by comparing only vesicles of the same size. This constant does the
# second.
#
# The correction is the more powerful option and the restriction is the more
# conservative one. Mercadal et al. (2016) report that the inverse
# radius-field relationship Schwan's equation predicts is often not observed
# experimentally, or is much shallower than predicted, so dividing a dose axis
# by radius imports an assumption that may not hold in this regime. Matching
# on size assumes nothing about how dose scales.
#
# Inside 4.5-5.5 um the two preparations are indistinguishable in radius
# (medians 4.99 and 4.97 um, Mann-Whitney p = 0.27), so applied field is the
# dose and no model is needed.
#
# It is expensive. Roughly four vesicles in five fall outside, and the loss is
# not spread evenly: conditions where one preparation happens to be large or
# small lose almost everything. The report printed at aggregation time lists
# what each condition costs, and it should be read before the results are
# trusted -- a condition reduced to a handful of vesicles is not a condition.
SIZE_WINDOW_UM = (4.5, 5.5)


def _report_size_window(kept: dict, dropped: dict, n_no_radius: int) -> None:
  """What the size window cost, per phenotype and per condition.

  Built from tallies rather than from the finished table, because a vesicle
  outside the window is rejected inside the aggregation loop and never
  becomes a row. The condition list is the part worth reading: the loss is
  not spread evenly, and a condition reduced to one or two vesicles has
  stopped being a condition whatever the group totals say.
  """
  lo, hi = SIZE_WINDOW_UM
  n_keep, n_drop = sum(kept.values()), sum(dropped.values())
  total = n_keep + n_drop
  if total == 0:
    return
  print(f"\nSIZE WINDOW {lo}-{hi} um, applied before any endpoint is read.")
  print(f"  {n_keep} of {total} vesicle(s) retained ({100 * n_keep / total:.0f}%); "
        f"{n_no_radius} of the {n_drop} rejected had no measured radius.")

  groups = sorted({g for g, _ in kept} | {g for g, _ in dropped})
  print("  per group:")
  for g in groups:
    k = sum(v for (gg, _), v in kept.items() if gg == g)
    dd = sum(v for (gg, _), v in dropped.items() if gg == g)
    print(f"    {g:<24} {k + dd:4d} -> {k:4d}")

  thin = [(g, v, sum(x for (gg, vv), x in kept.items()
                     if (gg, vv) == (g, v)),
           sum(x for (gg, vv), x in dropped.items() if (gg, vv) == (g, v)))
          for g, v in sorted({*kept, *dropped})]
  thin = [t for t in thin if t[2] < 3]
  if thin:
    print(f"  {len(thin)} condition(s) left with fewer than 3 vesicles; these "
          "should not be read as conditions:")
    for g, v, k, dd in sorted(thin, key=lambda t: -(t[2] + t[3])):
      print(f"    {g:<24} {v:>5}  {k + dd:3d} -> {k:d}")
  print("  Set SIZE_WINDOW_UM = None to restore the whole population.")


def aggregate_pipeline_results(outputs_root: str) -> pd.DataFrame:
  root = Path(outputs_root)
  records = []
  # Tally of rows dropped by EXCLUDE_FATES, per fate and per experiment, so
  # the exclusion is visible in the run log rather than silent.
  dropped_fates = {}
  dropped_by_exp = {}
  # Experiments that cannot supply a matched-time endpoint, and the count of
  # response calls the matched endpoint changed. Both are reported at the end
  # rather than per row, so the cost of the setting is one visible number.
  endpoint_short = {}
  endpoint_no_time = set()
  n_response_flipped = 0
  n_endpoint_censored = 0
  # Which conditions the censoring falls on, not just how many vesicles it
  # takes. A total is not enough to write an n from: the loss is not spread
  # evenly, it lands on whichever sessions ran short, and one of those is a
  # deliberately truncated field whose vesicles are still present in the fate
  # and rupture analyses. Keyed (population, voltage, experiment).
  endpoint_censored_by = {}
  # Size-window tallies, accumulated in the loop because the rejected
  # vesicles never become rows and so cannot be counted from the table.
  size_kept, size_dropped = {}, {}
  n_size_no_radius = 0

  all_fate = [f for f in root.glob("**/*_guv_fate_classification.csv")
              if RESULTS_SUBFOLDER not in f.parts]
  fate_files = [f for f in all_fate
                if not is_excluded_experiment(f.parents[1].name)]
  for name in sorted({f.parents[1].name for f in all_fate}
                     - {f.parents[1].name for f in fate_files}):
    print(f"EXCLUDED experiment (EXCLUDE_EXPERIMENTS): {name}")

  for fate_file in fate_files:
    exp_dir = fate_file.parents[1]
    voltage = extract_voltage(exp_dir.name)
    population = extract_population(exp_dir.name)

    try:
      df_fate = pd.read_csv(fate_file)
    except Exception as e:
      print(f"Skipping {fate_file}: {e}")
      continue

    fit_files = list(exp_dir.glob("**/*_fit_parameters.csv"))
    df_fit = None
    if fit_files:
      try:
        df_fit = pd.read_csv(fit_files[0])
      except Exception as e:
        print(f"Skipping fit file {fit_files[0]}: {e}")

    norm_files = list(exp_dir.glob("**/*_normalized_curves.csv"))
    df_norm = None
    if norm_files:
      try:
        df_norm = pd.read_csv(norm_files[0])
      except Exception as e:
        print(f"Skipping normalized curve file {norm_files[0]}: {e}")

    df_cortex = load_cortex_status(exp_dir)
    pre_mask = post_mask = None
    t_final_s = np.nan
    
    if df_norm is not None:
      pre_mask = _pre_pulse_mask(df_norm)
      post_mask = _post_pulse_mask(df_norm)
      if "Time (s)" in df_norm.columns and post_mask.any():
        t_final_s = df_norm.loc[post_mask, "Time (s)"].max()

    # Endpoint reference for this experiment. t_final_s keeps meaning "when
    # this record ended" whatever mode is set, so report_endpoint_times still
    # shows the raw schedule; t_endpoint_used is where the value was actually
    # read, which is the two of them being different that matters.
    t_endpoint_used = t_final_s
    endpoint_censored = False
    if ENDPOINT_MATCHED_T_S is not None:
      if df_norm is None or "Time (s)" not in df_norm.columns:
        endpoint_no_time.add(exp_dir.name)
        endpoint_censored = True
        t_endpoint_used = np.nan
      elif (not np.isfinite(t_final_s)
            or t_final_s < ENDPOINT_MATCHED_T_S - ENDPOINT_MATCHED_TOLERANCE_S):
        endpoint_short[exp_dir.name] = float(t_final_s)
        endpoint_censored = True
        t_endpoint_used = np.nan
      else:
        t_endpoint_used = float(ENDPOINT_MATCHED_T_S)

    for _, row in df_fate.iterrows():
      gid = str(row["guv_id"])
      raw_fate = str(row.get("fate", "")).upper()

      # Vesicles without a usable post-pulse observation never reach the
      # table. This must come before the size_cat mapping, whose else-branch
      # would otherwise absorb them into "Stagnate".
      if raw_fate in EXCLUDE_FATES:
        dropped_fates[raw_fate] = dropped_fates.get(raw_fate, 0) + 1
        dropped_by_exp[exp_dir.name] = dropped_by_exp.get(exp_dir.name, 0) + 1
        continue

      if raw_fate == "GROWN":
        size_cat = "Grow"
      elif raw_fate == "SHRUNK":
        size_cat = "Reduce"
      elif raw_fate == "RUPTURED":
        size_cat = "Rupture/Collapse"
      else:
        size_cat = "Stagnate"

      # Terminal radius relative to the vesicle's own pre-pulse baseline.
      # Already computed by classify_guv_fates and sitting in the fate CSV;
      # carried through here because the area-loss onset needs a CONTINUOUS
      # measure, and size_category alone collapses it to four labels.
      terminal_norm_radius = np.nan
      try:
        _tnr = row.get("terminal_norm_radius", np.nan)
        if pd.notna(_tnr):
          terminal_norm_radius = float(_tnr)
      except Exception:
        pass

      radius_um = np.nan
      tau = np.nan
      tau_se = np.nan
      tau_identifiable = False
      is_responding = False
      response_class = None
      tau_over_record = np.nan
      
      if df_fit is not None:
        guv_fit = df_fit[df_fit["guv_id"].astype(str) == gid]
        if not guv_fit.empty:
          fr = guv_fit.iloc[0]
          radius_um = fr.get("radius_um", np.nan)
          tau = fr.get("tau", np.nan)
          tau_se = fr.get("tau_SE", np.nan)
          tau_identifiable = bool(fr.get("tau_identifiable", False))
          is_responding = bool(fr.get("is_responding", False))
          response_class = fr.get("response_class", None)
          tau_over_record = fr.get("tau_over_record", np.nan)

      cortex_status = "UNKNOWN"
      cortex_contrast = np.nan
      prepulse_lumen_abs = np.nan
      
      if df_cortex is not None:
        guv_cx = df_cortex[df_cortex["guv_id"].astype(str) == gid]
        if not guv_cx.empty:
          cortex_status = str(guv_cx.iloc[0].get("cortex_status", "UNKNOWN"))
          cortex_contrast = guv_cx.iloc[0].get("cortex_contrast", np.nan)
          prepulse_lumen_abs = guv_cx.iloc[0].get("prepulse_lumen_abs", np.nan)

      # SIZE WINDOW, applied here rather than to the finished table.
      # Everything below this point -- the endpoint read, the truncated-trace
      # response call, the per-frame column lookups -- is work done per
      # vesicle, and four vesicles in five are about to be discarded. Placing
      # the test after the cortex status lookup, which is a cheap row match,
      # lets the tally below break the loss down by phenotype rather than by
      # raw preparation.
      if SIZE_WINDOW_UM is not None:
        _lo, _hi = SIZE_WINDOW_UM
        if not (np.isfinite(radius_um) and _lo <= radius_um <= _hi):
          _g = assign_cortex_group({"population": population,
                                    "cortex_status": cortex_status,
                                    "cortex_contrast": cortex_contrast})
          _k = (_g, voltage)
          size_dropped[_k] = size_dropped.get(_k, 0) + 1
          if not np.isfinite(radius_um):
            n_size_no_radius += 1
          continue
        _g = assign_cortex_group({"population": population,
                                  "cortex_status": cortex_status,
                                  "cortex_contrast": cortex_contrast})
        _k = (_g, voltage)
        size_kept[_k] = size_kept.get(_k, 0) + 1

      radius_bin = assign_size_bin(radius_um)
      intensity_cat = None
      i_pre = np.nan
      i_final = np.nan
      diff = np.nan
      n_final_used = 0

      if size_cat == "Stagnate" and df_norm is not None and not endpoint_censored:
        target_col = f"GUV_{gid}_I_retained"
        if target_col in df_norm.columns:
          pre_vals = df_norm.loc[pre_mask, target_col].dropna().values

          if ENDPOINT_MATCHED_T_S is None:
            post_vals = df_norm.loc[post_mask, target_col].dropna().values
            post_t = None
          else:
            _p = df_norm.loc[post_mask, ["Time (s)", target_col]].dropna()
            _p = _p[_p["Time (s)"] <= ENDPOINT_MATCHED_T_S]
            post_vals = _p[target_col].to_numpy(float)
            post_t = _p["Time (s)"].to_numpy(float)

          if len(pre_vals) > 0 and len(post_vals) > 0:
            i_pre = float(np.mean(pre_vals))
            tail = post_vals[-ENDPOINT_N_FRAMES:]
            n_final_used = len(tail)
            i_final = float(np.mean(tail))
            diff = i_final - i_pre

            if diff > INTENSITY_DIFF_THRESHOLD:
              intensity_cat = "Increase Intensity"
            elif abs(diff) <= INTENSITY_DIFF_THRESHOLD:
              intensity_cat = "Flatline"
            else:
              intensity_cat = "Lose Intensity"

            # is_responding came off the FULL record, so leaving it alone
            # would give a matched-time release magnitude next to a
            # whole-record response call, and the dose-response figure plots
            # both. Recomputed here on the truncated trace using
            # run_analysis's own _observed_response and the same two config
            # thresholds -- imported, not restated, so there is one
            # definition of "responding" in the codebase and not two.
            if post_t is not None and len(post_t) >= 6:
              new_resp = _responding_at(post_t, post_vals)
              if new_resp is not None:
                if bool(new_resp) != bool(is_responding):
                  n_response_flipped += 1
                is_responding = bool(new_resp)
      elif size_cat == "Stagnate" and endpoint_censored:
        # The release value is unobtainable at the matched time, so the
        # response call taken over the whole record has to go with it. Left
        # in place it would put a whole-record verdict in the same column as
        # matched-time verdicts, and the responding fraction would then be a
        # mixture of two definitions.
        n_endpoint_censored += 1
        _key = (population, voltage, exp_dir.name)
        endpoint_censored_by[_key] = endpoint_censored_by.get(_key, 0) + 1
        is_responding = np.nan

      records.append({
          "population": population,
          "voltage": voltage,
          "experiment": exp_dir.name,
          "guv_id": gid,
          "size_category": size_cat,
          "radius_um": radius_um,
          "radius_bin": radius_bin,
          "terminal_norm_radius": terminal_norm_radius,
          "cortex_status": cortex_status,
          "cortex_contrast": cortex_contrast,
          "prepulse_lumen_abs": prepulse_lumen_abs,
          "tau": tau,
          "tau_SE": tau_se,
          "tau_identifiable": tau_identifiable,
          "is_responding": is_responding,
          "response_class": response_class,
          "tau_over_record": tau_over_record,
          "i_pre": i_pre,
          "i_final": i_final,
          "diff": diff,
          "intensity_category": intensity_cat,
          "t_final_s": t_final_s,
          "t_endpoint_used": t_endpoint_used,
          "n_final_used": n_final_used,
      })

  if ENDPOINT_MATCHED_T_S is not None:
    print(f"\nENDPOINT read at a matched {ENDPOINT_MATCHED_T_S:.0f} s after "
          "the pulse, not at each record's own end.")
    if endpoint_no_time:
      print("  No 'Time (s)' column, so no matched endpoint is possible: "
            + ", ".join(sorted(endpoint_no_time)))
    if endpoint_short:
      print(f"  {len(endpoint_short)} experiment(s) end before the target and "
            "are censored (i_final = NaN):")
      for name, t_end in sorted(endpoint_short.items(), key=lambda kv: kv[1]):
        print(f"    {t_end:7.1f} s  {name}")
    if n_endpoint_censored:
      print(f"  {n_endpoint_censored} stagnate GUV(s) lost to censoring. "
            "Lowering the target keeps them, at the cost of reading every "
            "condition earlier.")
      # Broken down, because the total hides where the hole is. These
      # vesicles are absent from the dye release and responding-fraction
      # figures but PRESENT in the size, fate and rupture ones, which take
      # nothing from the endpoint -- so the two sets of numbers have
      # different denominators and the difference is listed here rather than
      # left to be discovered from a figure.
      print("    condition                 experiment"
            "                                        n")
      for (pop, volt), _ in sorted(
          {(p, v): None for p, v, _ in endpoint_censored_by}.items(),
          key=lambda kv: (kv[0][0], _voltage_key(kv[0][1]))):
        for (p, v, exp), n in sorted(endpoint_censored_by.items()):
          if (p, v) == (pop, volt):
            print(f"    {POPULATION_LABELS.get(pop, pop):<12} {volt:<12} "
                  f"{exp:<48} {n:>3}")
    print(f"  {n_response_flipped} response call(s) changed against the "
          "whole-record version.")

  if dropped_fates:
    total = sum(dropped_fates.values())
    kept = len(records)
    print(f"\nEXCLUDED by fate (EXCLUDE_FATES): {total} GUVs dropped, "
          f"{kept} retained")
    for fate_name in sorted(dropped_fates):
      print(f"  {fate_name}: {dropped_fates[fate_name]}")
    worst = sorted(dropped_by_exp.items(), key=lambda kv: -kv[1])[:5]
    print("  worst-affected experiments:")
    for exp_name, n_dropped in worst:
      print(f"    {n_dropped:>3}  {exp_name}")

  if SIZE_WINDOW_UM is not None:
    _report_size_window(size_kept, size_dropped, n_size_no_radius)

  out = pd.DataFrame(records)

  # A censored vesicle has no response call, so this column carries missing
  # values and is no longer plain bool. NumPy has no missing bool: as object
  # it raises on being used as a mask, and as float it is silently taken for a
  # list of column labels, which is how a censoring change surfaced three
  # functions away as KeyError: 'radius_bin'. The nullable dtype is the only
  # one where NA both masks as False and stays out of a .sum().
  if "is_responding" in out.columns:
    out["is_responding"] = out["is_responding"].astype("boolean")
  return out


# -----------------------------------------------------------------------------
# --- ONSET FIELDS ---
# Perrier et al. (2019) read the field at which the actin shell breaks down,
# and the field at which the GUV starts to shrink, off sigmoids fitted "to
# guide the eye". The ordering of those two onsets is their mechanism: the
# cortex holds the membrane until it fails, and only then does the vesicle
# lose area. Testing that ordering needs the onsets as numbers with
# uncertainties, not as curves read by eye -- if the two confidence intervals
# overlap, there is no ordering to report.
#
# A single pulse per vesicle across a voltage series (this work) rather than a
# ramp on one vesicle (theirs) means these midpoints are not expected to match
# theirs numerically: they showed themselves that a ramp shifts the apparent
# onset relative to a single pulse of the same amplitude.
# -----------------------------------------------------------------------------

ONSET_N_BOOTSTRAP = 300       # resamples for the V50 confidence interval
# 300, not 2000. The interval is only ever read to answer a yes/no question --
# does the CI span more than half the sampled range -- and 300 resamples pin a
# percentile interval far more tightly than that decision needs. The cost is
# not small: six fits (three populations x two measures) over several hundred
# vesicles each, at roughly 30-100 ms per resample, is 10-25 minutes at 2000
# and about two at 300.
#
# It also buys nothing for the gate that is actually binding here. The
# full-sample fit, the residual scatter and `amplitude_ok` are all computed
# BEFORE the bootstrap loop; when a fit fails on amplitude, as every fit on
# this dataset currently does, the interval is never quoted at all and the
# resamples are spent on a number that is discarded.
#
# Raise it back to 2000 if a fit ever clears the amplitude gate and an
# interval has to be reported to three figures.
ONSET_MIN_N = 8               # GUVs required before a fit is attempted
ONSET_MIN_VOLTAGES = 4        # distinct voltages required
# Minimum fitted step, in units of the residual scatter about the fit. A
# logistic will happily fit flat, noisy data by placing an arbitrary midpoint
# under a step no larger than the noise; the CI can even come out narrow,
# because every bootstrap replicate agrees on the same meaningless value.
# Requiring the step to exceed the scatter is what separates "the onset is
# here" from "there is no onset in this range".
ONSET_MIN_AMPLITUDE_SD = 2.0


def _sigmoid(v, lo, hi, v50, k):
    """Logistic in field strength. lo/hi are the plateaus, v50 the midpoint."""
    return lo + (hi - lo) / (1.0 + np.exp(-k * (v - v50)))


def _fit_sigmoid_once(v, y):
    """One least-squares fit. Returns (lo, hi, v50, k) or None."""
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    if len(v) < 4 or len(np.unique(v)) < 3:
        return None
    lo0, hi0 = float(np.nanmin(y)), float(np.nanmax(y))
    v50_0 = float(np.median(v))
    span = max(float(np.ptp(v)), 1.0)
    try:
        p, _ = curve_fit(
            _sigmoid, v, y,
            p0=[lo0, hi0, v50_0, 4.0 / span],
            bounds=([-np.inf, -np.inf, float(np.min(v)) - span,
                     1e-4 / span],
                    [np.inf, np.inf, float(np.max(v)) + span,
                     100.0 / span]),
            maxfev=20000,
        )
    except Exception:
        return None
    return tuple(float(x) for x in p)


def fit_onset(df: pd.DataFrame, value_col: str, label: str,
              n_boot: int = None) -> dict:
    """Sigmoid midpoint in kV/cm, with a bootstrap CI, for one population.

    The CI comes from resampling GUVs with replacement rather than from the
    covariance matrix, because the residuals are neither independent (several
    GUVs share a field of view) nor homoscedastic (spread grows with voltage),
    which is exactly where the asymptotic standard error misleads.

    The returned `v50_ci_low`/`v50_ci_high` are the honest output. A midpoint
    whose CI spans most of the voltage range means the sigmoid is not
    identified by these data -- usually because no upper plateau was reached
    -- and it should be reported as such rather than quoted as an onset. This
    is the same failure the tau fits have, and it is not made better by
    reporting the point estimate alone.
    """
    sub = df[["voltage", value_col]].copy()
    # Resolved here rather than in the signature. A default bound at import
    # cannot be changed from the console afterwards, which is a trap when
    # cutting the count down for an exploratory run; _boot_median_ci already
    # does it this way and the two should match.
    n_boot = int(ONSET_N_BOOTSTRAP if n_boot is None else n_boot)
    # Field strength, not applied volts, so the fitted midpoint is an E50 in
    # kV/cm and comparable to work using a different electrode geometry. The
    # conversion is one constant, so the fit, the bootstrap and `identified`
    # are unchanged; only the units of v50 and its interval move.
    sub["v_num"] = sub["voltage"].map(field_kV_cm)
    sub = sub.dropna(subset=["v_num", value_col])
    out = {
        "measure": label, "n_guv": int(len(sub)),
        "n_voltages": int(sub["v_num"].nunique()),
        "v50": np.nan, "v50_ci_low": np.nan, "v50_ci_high": np.nan,
        "slope_k": np.nan, "plateau_low": np.nan, "plateau_high": np.nan,
        "amplitude": np.nan, "resid_sd": np.nan, "amplitude_over_sd": np.nan,
        "n_boot_ok": 0, "identified": False, "note": "",
    }
    if out["n_guv"] < ONSET_MIN_N or out["n_voltages"] < ONSET_MIN_VOLTAGES:
        out["note"] = (f"too few data: {out['n_guv']} GUV(s) across "
                       f"{out['n_voltages']} voltage(s)")
        return out

    v = sub["v_num"].to_numpy(float)
    y = sub[value_col].to_numpy(float)
    p = _fit_sigmoid_once(v, y)
    if p is None:
        out["note"] = "fit did not converge on the full sample"
        return out
    lo, hi, v50, k = p
    out.update(v50=v50, slope_k=k, plateau_low=lo, plateau_high=hi)

    resid = y - _sigmoid(v, lo, hi, v50, k)
    sd = float(np.std(resid))
    amp = abs(hi - lo)
    out["amplitude"] = amp
    out["resid_sd"] = sd
    out["amplitude_over_sd"] = (amp / sd) if sd > 0 else np.inf
    amplitude_ok = (sd <= 0) or (amp >= ONSET_MIN_AMPLITUDE_SD * sd)

    # Stop here when there is no step to locate. The bootstrap only sets the
    # width of the interval around v50, and a v50 under a step smaller than
    # the noise is not reported at all, so the resamples would be spent on a
    # number that is discarded either way.
    #
    # It is also the expensive branch, and gets more expensive as the data
    # get sparser: on a thin, noisy sample curve_fit runs to maxfev on most
    # resamples, which measured at roughly 0.4 s each against 0.03 s on the
    # full population. Six fits at 300 resamples is then minutes rather than
    # seconds, all of it to compute an interval that is thrown away.
    if not amplitude_ok:
        out["note"] = (f"no step to locate: fitted amplitude {amp:.3f} is "
                       f"only {out['amplitude_over_sd']:.1f}x the residual "
                       "scatter; bootstrap not attempted")
        return out

    rng = np.random.default_rng(0)          # fixed seed: reruns reproduce
    n = len(v)
    boot = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        pb = _fit_sigmoid_once(v[idx], y[idx])
        if pb is not None:
            boot.append(pb[2])
    out["n_boot_ok"] = len(boot)
    if len(boot) < 0.5 * n_boot:
        out["note"] = (f"only {len(boot)}/{n_boot} bootstrap fits converged; "
                       f"CI unreliable")
    if boot:
        out["v50_ci_low"] = float(np.percentile(boot, 2.5))
        out["v50_ci_high"] = float(np.percentile(boot, 97.5))
        width = out["v50_ci_high"] - out["v50_ci_low"]
        v_span = float(np.ptp(v))
        # A midpoint is only meaningful if the CI is narrow against the range
        # that was actually sampled, and if the midpoint sits inside it.
        inside = float(np.min(v)) <= v50 <= float(np.max(v))
        out["identified"] = bool(inside and v_span > 0
                                 and width < 0.5 * v_span and amplitude_ok)
        if not out["identified"] and not out["note"]:
            if not amplitude_ok:
                out["note"] = (f"no step to locate: fitted amplitude "
                               f"{amp:.3f} is only {out['amplitude_over_sd']:.1f}x "
                               f"the residual scatter")
            else:
                out["note"] = ("midpoint not constrained by these voltages "
                               "(CI spans >50% of the sampled range, or the "
                               "midpoint lies outside it)")
    return out


def report_onset_fields(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Cortex-breakdown and area-loss onsets per group, fitted and plotted.

    Area loss is expressed as fractional projected-area loss,
    1 - (terminal_norm_radius)^2, so it is directly comparable to the
    normalised area Perrier et al. plot rather than to a radius.

    Fitted on the cortex-split groups. The undivided BranchedCortex population
    mixes vesicles that assembled a cortex with vesicles that kept their actin
    in the lumen, and a cortex-breakdown onset taken over the second kind is
    an onset for a structure that was never there. The area-loss onset has the
    same problem in weaker form: the two phenotypes differ in radius, so they
    do not reach a given transmembrane potential at the same applied voltage,
    and pooling them smears the step this fit exists to locate.
    """
    out_path = sub_dir(output_dir, "cortex_breakdown")
    work = df.copy()
    work = work[work["size_category"] != "Rupture/Collapse"]

    # assign_cortex_group rather than apply_cortex_split: the split itself is
    # wanted, the several paragraphs of counts it prints are not, having
    # already been printed once for the figures that follow.
    work["population"] = work.apply(assign_cortex_group, axis=1)
    drop = work["population"].isin(EXCLUDE_GROUPS)
    if drop.any():
      print(f"\nOnset fields: {int(drop.sum())} unclassifiable GUV(s) "
            f"excluded, {int((~drop).sum())} fitted.")
      work = work[~drop]

    if "terminal_norm_radius" in work.columns:
      with np.errstate(invalid="ignore"):
        work["area_loss_frac"] = 1.0 - work["terminal_norm_radius"] ** 2

    measures = []
    if "area_loss_frac" in work.columns:
      measures.append(("area_loss_frac", "fractional area loss"))
    if "peak_drop_final" in work.columns:
      measures.append(("peak_drop_final", "cortex peak drop"))
    if not measures:
      print("Onset fields skipped: neither area nor cortex peak data present.")
      return pd.DataFrame()

    pops = _present_populations(work)
    rows = []
    fig, axes = plt.subplots(
        1, len(pops),
        figsize=fig_size("onset_fields", 6.0 * len(pops), 4.8, len(pops)),
        sharey=True, squeeze=False,
    )
    for ax, pop in zip(axes[0], pops):
      pop_df = work[work["population"] == pop]
      for value_col, label in measures:
        if value_col not in pop_df.columns:
          continue
        res = fit_onset(pop_df, value_col, label)
        res["population"] = pop
        rows.append(res)

        sub = pop_df[["voltage", value_col]].dropna()
        if sub.empty:
          continue
        vx = sub["voltage"].map(field_kV_cm).to_numpy(float)
        ax.plot(vx, sub[value_col].to_numpy(float), "o", ms=3, alpha=0.35,
                label=f"{label} (n = {len(sub)})")
        if np.isfinite(res["v50"]):
          grid = np.linspace(vx.min(), vx.max(), 200)
          ax.plot(grid, _sigmoid(grid, res["plateau_low"],
                                 res["plateau_high"], res["v50"],
                                 res["slope_k"]), "-", lw=1.6)
          if res["identified"]:
            ax.axvline(res["v50"], ls="--", lw=1.0, color="0.3")
            ax.axvspan(res["v50_ci_low"], res["v50_ci_high"],
                       color="0.6", alpha=0.15)
      ax.set_title(pop)
      ax.set_xlabel(FIELD_AXIS_LABEL)
      ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
      ax.legend(frameon=False, loc="upper left")
    axes[0][0].set_ylabel("fractional loss")

    style_figure("onset_fields")
    plt.tight_layout()
    pdf_path = out_path / "guv_onset_fields.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300)
    plt.close(fig)

    res_df = pd.DataFrame(rows)
    csv_path = out_path / "onset_fields.csv"
    res_df.to_csv(csv_path, index=False, float_format="%.4f")

    print("\nOnset fields (sigmoid midpoint in kV/cm, 95% bootstrap CI):")
    for _, r in res_df.iterrows():
      if r["identified"]:
        print(f"  {r['population']:<20} {r['measure']:<20} "
              f"E50 = {r['v50']:.2f} kV/cm  [{r['v50_ci_low']:.2f}, "
              f"{r['v50_ci_high']:.2f}]  (n = {r['n_guv']})")
      else:
        print(f"  {r['population']:<20} {r['measure']:<20} "
              f"not identified -- {r['note']}")
    print("  Two onsets are only ordered if their intervals do not overlap.")
    return res_df


def report_endpoint_times(df: pd.DataFrame):
  if df.empty:
    return
  tbl = (
      df.groupby(["voltage", "population", "experiment"])["t_final_s"]
      .max()
      .reset_index()
      .sort_values(["voltage", "population"])
  )
  print("\nEndpoint time per experiment (s post-pulse):")
  print(tbl.to_string(index=False))
  print()

def _filter_stagnate(df: pd.DataFrame) -> pd.DataFrame:
  return df[
      (df["size_category"] == "Stagnate")
      & (df["intensity_category"].isin(["Lose Intensity", "Flatline"]))
      & ((df["i_pre"] - 1.0).abs() <= PRE_PULSE_TOLERANCE)
  ].copy()

def _voltage_key(v) -> int:
  m = re.search(r"\d+", str(v))
  return int(m.group()) if m else 0


# -----------------------------------------------------------------------------
# --- FIELD STRENGTH ---
# Electrode separation in the 3D-printed chamber, in centimetres. Figures are
# labelled in kV/cm rather than in applied volts: the voltage is a property of
# this generator and this chamber and transfers to no other study, while the
# field does, and it is the field that appears in the Schwan expression the
# dose axis is built on.
#
# The `voltage` COLUMN is left alone. It stays the grouping key and the string
# written to every CSV, so nothing that joins, sorts or reads those files
# changes; only what is printed on an axis does. Sorting still runs through
# _voltage_key on the original label.
#
# The gap is not recorded per experiment anywhere in the pipeline outputs, so
# a session run with a different chamber would be mis-scaled here with no
# warning. If that ever happens this constant is not enough and the gap has to
# be carried per experiment.
# -----------------------------------------------------------------------------
ELECTRODE_GAP_CM = 0.3


def field_kV_cm(v) -> float:
  """Applied voltage label -> field strength in kV/cm."""
  return _voltage_key(v) / ELECTRODE_GAP_CM / 1000.0


def field_label(v) -> str:
  """Tick label in kV/cm, always with at least one decimal.

  Two places only where the amplitude needs them: 400, 650 and 775 V do not
  land on round fields (1.33, 2.17, 2.58 kV/cm), while the rest do. Trimming
  all the way to bare integers would print 300 V as "1", which reads as a
  count rather than a field, so the last zero is kept.
  """
  txt = f"{field_kV_cm(v):.2f}"
  return txt[:-1] if txt.endswith("0") else txt


def field_labels(vs) -> list:
  return [field_label(v) for v in vs]


FIELD_AXIS_LABEL = "Field strength (kV/cm)"

def _voltage_order(df: pd.DataFrame) -> list:
  return sorted(df["voltage"].unique(), key=_voltage_key)

def _present_populations(df: pd.DataFrame) -> list:
  present = [p for p in POPULATIONS if p in set(df["population"])]
  extra = sorted(set(df["population"]) - set(POPULATIONS))
  return present + extra

BIN_ORDER = ["< 5 \u00b5m", "5-8 \u00b5m", "9-12 \u00b5m", "> 13 \u00b5m"]

def _filter_tau(df: pd.DataFrame) -> pd.DataFrame:
  sub = df[
      (df["size_category"] == "Stagnate")
      & df["tau"].notna()
      & (df["tau"] > 0)
  ].copy()
  if TAU_REQUIRE_IDENTIFIABLE:
    sub = sub[sub["tau_identifiable"]]
  if TAU_REQUIRE_RESPONDING:
    sub = sub[sub["is_responding"]]
  return sub[sub["radius_bin"].isin(BIN_ORDER)]

def report_tau_yield(df: pd.DataFrame):
  if df.empty:
    return
  rows = []
  stag = df[df["size_category"] == "Stagnate"]
  for (pop, v), sub in stag.groupby(["population", "voltage"]):
    fitted = sub["tau"].notna().sum()
    # .sum() on the nullable dtype counts True only, so a censored vesicle is
    # not quietly scored as a non-responder. n_no_call carries them instead,
    # because "we could not tell" and "it did not respond" are different
    # denominators and only one of them belongs in a response rate.
    n_call = int(sub["is_responding"].notna().sum())
    rows.append({
        "population": pop,
        "voltage": v,
        "n_guv": len(sub),
        "n_fitted": int(fitted),
        "n_no_call": int(len(sub) - n_call),
        "n_responding": int(sub["is_responding"].sum()),
        "n_tau_identifiable": int(sub["tau_identifiable"].sum()),
        "n_plotted": len(_filter_tau(sub)),
    })
  tbl = pd.DataFrame(rows).sort_values(["population", "voltage"])
  print("\nTau yield per condition:")
  print(tbl.to_string(index=False))
  n_missing = int(tbl["n_no_call"].sum())
  if n_missing:
    print(f"  n_no_call: {n_missing} vesicle(s) whose record ends before "
          f"ENDPOINT_MATCHED_T_S = {ENDPOINT_MATCHED_T_S}, so they have no "
          "response call at the matched time. They are out of the numerator "
          "AND the denominator of n_responding.")
  print()

def plot_drop_by_cortex_status(df: pd.DataFrame, output_dir: str):
  out_path = sub_dir(output_dir, "cortex_class")
  df_st = _filter_stagnate(df)
  df_st = df_st[df_st["cortex_status"] != "UNKNOWN"]
  if df_st.empty:
    print("Cortex split skipped: no *_actin_cortex_status.csv found.")
    return

  df_st = df_st.assign(cortex_status=df_st.apply(assign_cortex_group, axis=1))
  pops = _present_populations(df)
  voltages = _voltage_order(df_st)
  statuses = [g for g in GROUP_ORDER
              if g.startswith("Branched") and g in set(df_st["cortex_status"])]

  fig, axes = plt.subplots(
      1, len(pops), figsize=fig_size("drop_by_cortex_status", 6.5 * len(pops), 5.5, len(pops)), sharey=True, squeeze=False
  )
  x = np.arange(len(voltages))
  slot_w = 0.8 / max(len(statuses), 1)
  rng = np.random.default_rng(0)

  for p_idx, pop in enumerate(pops):
    ax = axes[0][p_idx]
    pop_df = df_st[df_st["population"] == pop]

    for s_idx, status in enumerate(statuses):
      colour = GROUP_COLORS.get(status, ANNOTATION_TEXT)
      offset = (s_idx - (len(statuses) - 1) / 2) * slot_w
      med_x, med_y = [], []

      for v_idx, v in enumerate(voltages):
        vals = pop_df.loc[
            (pop_df["voltage"] == v) & (pop_df["cortex_status"] == status),
            "diff",
        ].dropna().values
        if len(vals) == 0:
          continue
        jitter = rng.uniform(-slot_w * 0.2, slot_w * 0.2, len(vals))
        ax.scatter(
            np.full(len(vals), x[v_idx] + offset) + jitter,
            vals,
            s=22, color=colour, alpha=0.55, edgecolor="none", zorder=3,
        )
        med = float(np.median(vals))
        ax.plot(
            [x[v_idx] + offset - slot_w * 0.3, x[v_idx] + offset + slot_w * 0.3],
            [med, med], color=colour, lw=2.0, solid_capstyle="butt", zorder=4,
        )
        med_x.append(x[v_idx] + offset)
        med_y.append(med)

      ax.plot(med_x, med_y, "-", color=colour, lw=1.2, alpha=0.7, zorder=2,
              label=status if p_idx == 0 else "")

    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(field_labels(voltages), rotation=45, fontsize=9)
    ax.set_xlabel(FIELD_AXIS_LABEL)
    ax.set_title(POPULATION_LABELS.get(pop, pop))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if p_idx == 0:
      ax.set_ylabel(r"$I_{final} - I_{pre}$")
      ax.legend(title="Pre-pulse cortex", fontsize=8, loc="best")

  fig.suptitle(
      "Dye retention change split by pre-pulse cortex presence"
      "\n(Stagnate GUVs; bars = median)"
  )
  style_figure("drop_by_cortex_status")
  plt.tight_layout()
  pdf_path = out_path / "guv_drop_by_cortex_status.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved Cortex Split Graph: {pdf_path}")

def report_cortex_status(df: pd.DataFrame):
  if df.empty or "cortex_status" not in df.columns:
    return
  stag = df[df["size_category"] == "Stagnate"]
  tbl = (
      stag.groupby(["population", "voltage", "cortex_status"])
      .size()
      .unstack(fill_value=0)
  )
  if tbl.empty:
    return
  print("\nPre-pulse cortex classification (Stagnate GUVs):")
  print(tbl.to_string())
  print()

def plot_tau_by_size_and_voltage(df: pd.DataFrame, output_dir: str):
  out_path = sub_dir(output_dir, "kinetics")
  pops = _present_populations(df)
  voltages = _voltage_order(df)
  df_tau = _filter_tau(df)

  if df_tau.empty:
    print("Tau plot skipped: no GUV passed the identifiability gate "
          "(see the tau yield table above, and tau_identifiability.pdf "
          "for which constraint failed).")
    return

  fig, axes = plt.subplots(
      1, len(pops), figsize=fig_size("tau_by_size_voltage", 6.5 * len(pops), 5.5, len(pops)), sharey=True, squeeze=False
  )
  x = np.arange(len(voltages))
  slot_w = 0.8 / len(BIN_ORDER)
  rng = np.random.default_rng(0)

  for p_idx, pop in enumerate(pops):
    ax = axes[0][p_idx]
    pop_df = df_tau[df_tau["population"] == pop]

    for b_idx, b_label in enumerate(BIN_ORDER):
      colour = SIZE_BIN_COLORS[b_label]
      offset = (b_idx - (len(BIN_ORDER) - 1) / 2) * slot_w
      med_x, med_y = [], []

      for v_idx, v in enumerate(voltages):
        vals = pop_df.loc[
            (pop_df["voltage"] == v) & (pop_df["radius_bin"] == b_label), "tau"
        ].values
        if len(vals) == 0:
          continue
        jitter = rng.uniform(-slot_w * 0.22, slot_w * 0.22, len(vals))
        ax.scatter(
            np.full(len(vals), x[v_idx] + offset) + jitter,
            vals,
            s=22,
            color=colour,
            alpha=0.55,
            edgecolor="none",
            zorder=3,
        )
        med = float(np.median(vals))
        ax.plot(
            [x[v_idx] + offset - slot_w * 0.32,
             x[v_idx] + offset + slot_w * 0.32],
            [med, med],
            color=colour,
            lw=2.0,
            solid_capstyle="butt",
            zorder=4,
        )
        med_x.append(x[v_idx] + offset)
        med_y.append(med)

      ax.plot(
          med_x,
          med_y,
          "-",
          color=colour,
          lw=1.2,
          alpha=0.7,
          zorder=2,
          label=b_label if p_idx == 0 else "",
      )

    ax.set_xticks(x)
    ax.set_xticklabels(field_labels(voltages), rotation=45, fontsize=9)
    ax.set_xlabel(FIELD_AXIS_LABEL)
    ax.set_yscale("log")
    ax.set_title(POPULATION_LABELS.get(pop, pop))
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.4)
    if len(pop_df) == 0:
      ax.text(
          0.5,
          0.5,
          "no GUV passed the\ntau identifiability gate",
          transform=ax.transAxes,
          ha="center",
          va="center",
          fontsize=10,
          color=SUMMARY_LINE,
      )
    if p_idx == 0:
      ax.set_ylabel(r"$\tau$ (s)")
      ax.legend(title="Size group", fontsize=8, loc="best")

  gates = []
  if TAU_REQUIRE_RESPONDING:
    gates.append("responding")
  if TAU_REQUIRE_IDENTIFIABLE:
    gates.append(r"$\tau$ identifiable")
  gate_txt = " + ".join(gates) if gates else "no gating"
  fig.suptitle(
      r"Dye efflux time constant $\tau$ vs. voltage, by size group"
      f"\n(bars = median; {gate_txt})"
  )
  style_figure("tau_by_size_voltage")
  plt.tight_layout()
  pdf_path = out_path / "guv_tau_by_size_voltage.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved Tau Graph: {pdf_path}")

def assign_cortex_group(row) -> str:
  if row["population"] != "BranchedCortex":
    return "Bare" if row["population"] in ("Empty", "Bare") else row["population"]

  status = row.get("cortex_status", "UNKNOWN")
  if status == "CORTEX":
    return "Branched, cortex"
  if status == "NO_CORTEX":
    return "Branched, lumenal only"
  if status == "AMBIGUOUS" and pd.notna(row.get("cortex_contrast", np.nan)):
    return "Branched, ambiguous"
  return "Branched, unclassified"

def apply_cortex_split(df: pd.DataFrame) -> pd.DataFrame:
  out = df.copy()
  out["population"] = out.apply(assign_cortex_group, axis=1)

  drop = out["population"].isin(EXCLUDE_GROUPS)
  if drop.any():
    print(f"\nExcluded {int(drop.sum())} of {len(out)} GUV(s) "
          f"({drop.mean():.1%}) as unclassifiable:")
    tbl = (out[drop].groupby(["population", "voltage"]).size()
           .unstack(fill_value=0))
    print(tbl.to_string())
    kept = out[~drop]
    print(f"  {len(kept)} GUV(s) retained. Per group:")
    for g, n in kept["population"].value_counts().items():
      print(f"    {g:<24} {n}")
    print()
    out = kept.copy()
  return out

PEAK_N_FRAMES = ENDPOINT_N_FRAMES
MIN_CONTROL_N = 5
CONTROL_VOLTAGE = "30V"
BLEACH_REFERENCE_PATTERN = rf"-{CONTROL_VOLTAGE}-"
EXCLUDE_EXPERIMENTS = [r"-0V-", r"260331.*400V"]

def is_excluded_experiment(name: str) -> bool:
  return any(re.search(pat, name) for pat in EXCLUDE_EXPERIMENTS)

APPLY_BLEACH_CORRECTION = True

def report_bleach_comparability(outputs_root: str) -> pd.DataFrame:
  root = Path(outputs_root)
  rows = []
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    files = [f for f in exp_dir.glob("**/*_normalized_curves.csv")
             if RESULTS_SUBFOLDER not in f.parts]
    if not files:
      continue
    try:
      cur = pd.read_csv(files[0])
    except Exception:
      continue
    tcol = next((c for c in ("Time (s)", "time_s", "t_aligned", "time")
                 if c in cur.columns), None)
    if tcol is None or cur.empty:
      continue
    t = cur[tcol].dropna().to_numpy(float)
    if t.size < 2:
      continue
    rows.append({
        "experiment": exp_dir.name,
        "is_control": bool(re.search(BLEACH_REFERENCE_PATTERN, exp_dir.name)),
        "n_frames": int(t.size),
        "record_s": float(t.max() - t.min()),
        "mean_interval_s": float(np.median(np.diff(t))),
    })
  if not rows:
    print("Bleach comparability skipped: no normalized curves found.")
    return pd.DataFrame()

  tbl = pd.DataFrame(rows)
  ctrl = tbl[tbl["is_control"]]
  if ctrl.empty:
    print(f"No bleach control matched {BLEACH_REFERENCE_PATTERN!r} "
          "(check EXCLUDE_EXPERIMENTS). With no no-pulse reference, cortex "
          "peak loss can only be read against 1.0, and photobleaching cannot "
          "be separated from breakdown in any decline reported here.")
    return tbl

  print("\nBleach control acquisition vs the rest:")
  for _, c in ctrl.iterrows():
    print(f"  control {c['experiment']}: {c['n_frames']} frames over "
          f"{c['record_s']:.0f} s (median interval {c['mean_interval_s']:.2f} s)")
  others = tbl[~tbl["is_control"]]
  print(f"  others : {others['n_frames'].median():.0f} frames (median) over "
        f"{others['record_s'].median():.0f} s "
        f"(median interval {others['mean_interval_s'].median():.2f} s)")

  c = ctrl.iloc[0]
  rate_ratio = (c["mean_interval_s"]
                / max(others["mean_interval_s"].median(), 1e-9))
  covers = c["record_s"] >= others["record_s"].median()

  if abs(rate_ratio - 1.0) > 0.2:
    print(f"  WARNING: the control was sampled at {rate_ratio:.2f}x the "
          "interval of a typical experiment, so it accumulated exposure at a "
          "different rate. Its decay per second is NOT transferable; rescale "
          "to a per-exposure basis or leave APPLY_BLEACH_CORRECTION off.")
  else:
    print("  Sampling interval matches, so decay per second is comparable.")
    print(f"  Control covers {c['record_s']:.0f} s vs a typical "
          f"{others['record_s'].median():.0f} s"
          + (" -- it spans the experiments, which is what is needed."
             if covers else
             " -- it is SHORTER than a typical experiment, so the tail of "
             "each curve has no control to compare against."))
    print("  Compare conditions at matched elapsed time, not at each record's "
          "own endpoint.")
  return tbl

def _peak_time_grid(t_max: float) -> np.ndarray:
  return np.unique(np.concatenate([
      np.arange(0.0, 2.01, 0.1),
      np.arange(2.0, 10.01, 0.5),
      np.arange(10.0, t_max + 5.0, 5.0),
  ]))

def collect_cortex_peak_traces(df: pd.DataFrame, outputs_root: str):
  root = Path(outputs_root)
  keep = df.assign(group=df.apply(assign_cortex_group, axis=1))
  keep = keep[(keep["group"] == "Branched, cortex")
              & (keep["size_category"] == "Stagnate")]
  if keep.empty:
    return None, None, None
  wanted = {(r["experiment"], str(r["guv_id"])): r["voltage"]
            for _, r in keep.iterrows()}

  raw, t_max = [], 0.0
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    files = [f for f in exp_dir.glob("**/*_actin_cortex_traces.csv")
             if RESULTS_SUBFOLDER not in f.parts]
    if not files:
      continue
    try:
      tr = pd.read_csv(files[0])
    except Exception:
      continue
    if "time_s" not in tr:
      continue
    t = tr["time_s"].to_numpy(float)
    for col in tr.columns:
      m = re.match(r"GUV_(.+)_cortex_peak$", col)
      if not m:
        continue
      key = (exp_dir.name, m.group(1))
      if key not in wanted:
        continue
      y = tr[col].to_numpy(float)
      ok = np.isfinite(t) & np.isfinite(y)
      if ok.sum() < 4:
        continue
      raw.append((wanted[key], t[ok], y[ok]))
      t_max = max(t_max, float(t[ok].max()))

  if not raw:
    return None, None, None

  grid = _peak_time_grid(t_max)
  by_voltage = {}
  for volt, t, y in raw:
    interp = np.interp(grid, t, y)
    interp[(grid < t.min()) | (grid > t.max())] = np.nan
    by_voltage.setdefault(volt, []).append(interp)
  return grid, {v: np.vstack(a) for v, a in by_voltage.items()}, len(raw)

def plot_cortex_peak_timecourse(df: pd.DataFrame, outputs_root: str,
                                output_dir: str, min_frac: float = 0.5):
  out_path = sub_dir(output_dir, "cortex_breakdown")
  out_path.mkdir(parents=True, exist_ok=True)

  grid, by_voltage, n_guv = collect_cortex_peak_traces(df, outputs_root)
  if grid is None:
    print("Cortex peak timecourse skipped: no cortex-bearing traces.")
    return

  voltages = sorted(by_voltage, key=_voltage_key)
  pulsed = [v for v in voltages if v != CONTROL_VOLTAGE]

  corrected = False
  if APPLY_BLEACH_CORRECTION and CONTROL_VOLTAGE in by_voltage:
    ref = np.nanmedian(by_voltage[CONTROL_VOLTAGE], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
      for volt in pulsed:
        arr = by_voltage[volt].copy()
        safe = np.isfinite(ref) & (ref > 0.05)
        arr[:, safe] = arr[:, safe] / ref[safe]
        arr[:, ~safe] = np.nan
        by_voltage[volt] = arr
    corrected = True
    print(f"  Bleach correction APPLIED: each curve divided by the "
          f"{CONTROL_VOLTAGE} reference median.")
  cmap = mcolors.LinearSegmentedColormap.from_list(
      "ep_blues", [PALETTE["pale_blue"], PALETTE["medium_blue"],
                   PALETTE["dark_blue"]])
  shades = {v: cmap(0.3 + 0.7 * i / max(len(pulsed) - 1, 1))
            for i, v in enumerate(pulsed)}

  fig, axes = plt.subplots(1, 2, figsize=fig_size("cortex_peak_timecourse", 13, 5.2, 2))
  rows, curves = [], {}

  for ax, (t_lo, t_hi, title) in zip(
      axes, [(0.0, 10.0, "Burst window"), (0.0, float(grid.max()),
                                           "Full record")]):
    win = (grid >= t_lo) & (grid <= t_hi)
    for volt in voltages:
      arr = by_voltage[volt]
      n_t = np.sum(np.isfinite(arr), axis=0)
      enough = n_t >= max(2, int(np.ceil(min_frac * arr.shape[0])))
      sel = win & enough
      if not sel.any():
        continue
      med = np.nanmedian(arr[:, sel], axis=0)
      q1 = np.nanpercentile(arr[:, sel], 25, axis=0)
      q3 = np.nanpercentile(arr[:, sel], 75, axis=0)
      is_ctrl = volt == CONTROL_VOLTAGE
      colour = PALETTE["dark_red"] if is_ctrl else shades[volt]
      ax.plot(grid[sel], med, color=colour, lw=2.0 if is_ctrl else 1.5,
              ls="--" if is_ctrl else "-", zorder=5 if is_ctrl else 3,
              label=f"{field_label(volt)} kV/cm (n={arr.shape[0]})")
      ax.fill_between(grid[sel], q1, q3, color=colour, alpha=0.13, lw=0,
                      zorder=2)
      if ax is axes[1]:
        rows.append({"voltage": volt, "n_guv": int(arr.shape[0]),
                     "t_last_s": float(grid[sel].max()),
                     "peak_at_t_last": float(med[-1])})
        curves[volt] = (grid[sel], med)

    ax.axhline(1.0, color=ANNOTATION_TEXT, lw=0.8, ls=":", zorder=1)
    ax.set_xlabel("time after pulse (s)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
  axes[0].set_ylabel("cortex peak height  (I$_{peak}$ / I$_{peak,pre}$)")
  axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")

  fig.suptitle("Radial cortex peak height over time, cortex-bearing GUVs\n"
               + (f"(median and IQR; divided by the {CONTROL_VOLTAGE} "
                  "bleach reference)" if corrected else
                  f"(median and IQR; dashed red = {CONTROL_VOLTAGE} "
                  "bleach reference, uncorrected)"),
               fontsize=11)
  style_figure("cortex_peak_timecourse")
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  pdf_path = out_path / ("guv_cortex_peak_timecourse"
                         + ("_bleach_corrected" if corrected else "")
                         + ".pdf")
  fig.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved cortex peak timecourse: {pdf_path}  ({n_guv} GUV traces)")

  if rows:
    tbl = pd.DataFrame(rows).sort_values("voltage", key=lambda c:
                                         c.map(_voltage_key))

    t_ref = 320.0
    matched = []
    for volt, (t, med) in curves.items():
        i = int(np.argmin(np.abs(t - t_ref)))
        matched.append({"voltage": volt, "peak_at_matched_t": float(med[i])})
    tbl = tbl.merge(pd.DataFrame(matched), on="voltage", how="left")
    ctrl_row = tbl[tbl["voltage"] == CONTROL_VOLTAGE]
    n_ctrl = int(ctrl_row["n_guv"].iloc[0]) if not ctrl_row.empty else 0
    ctrl_at_t = float(ctrl_row["peak_at_matched_t"].iloc[0]) if n_ctrl else np.nan

    if n_ctrl >= MIN_CONTROL_N:
        tbl["excess_loss_vs_control"] = ctrl_at_t - tbl["peak_at_matched_t"]
    elif n_ctrl:
        print(f"\n  {CONTROL_VOLTAGE} reference has only {n_ctrl} GUV(s) at t = {t_ref:.0f} s "
              f"(peak = {ctrl_at_t:.3f}); fewer than MIN_CONTROL_N="
              f"{MIN_CONTROL_N}, so no excess-over-control column is given.")
        if ctrl_at_t > 1.0:
            print("  The control also reads ABOVE its own pre-pulse level, "
                  "which a bleaching baseline cannot do -- it is sampling "
                  "noise, not a decay curve. Read peak_at_matched_t directly "
                  "and treat 1.0 as the no-change reference.")

    tbl.to_csv(out_path / "guv_cortex_peak_timecourse_summary.csv", index=False)
    if t_ref is not None:
      print(f"\nCortex peak, all conditions read at matched t = {t_ref:.0f} s.")
      print("peak_at_matched_t < 1 means peak height lost by that time; "
            "1.0 is no change.")
      if "excess_loss_vs_control" in tbl:
        print("excess_loss_vs_control > 0 means more peak lost than the "
              f"{CONTROL_VOLTAGE} reference at the same elapsed time.")
    print(tbl.to_string(index=False))
    print()

# -----------------------------------------------------------------------------
# --- POLE VS EQUATOR: IS CORTEX BREAKDOWN DIRECTIONAL? ---
#
# run_analysis.py writes one *_pole_vs_equator_traces.csv per experiment, with
# three columns per GUV: pole_mean and equator_mean (fold change against that
# GUV's own pre-pulse level, inside +/- POLE_EQUATOR_HALF_WIDTH_DEG of the
# electrode axis and of the equator respectively), and their contrast
#
#     polarization_index = (equator_mean - pole_mean) / (equator_mean + pole_mean)
#
# which is > 0 when the pole has lost more actin than the equator.
#
# Those files are only interpretable pooled, which is why this stage exists. A
# vesicle is free to rotate between frames, and a rotation carries signal
# between the pole and equator windows with no actin lost at all; on a single
# vesicle that is not distinguishable from directional breakdown. Rotation is
# not aligned to the electrode axis and averages out across a population,
# whereas a real field-driven asymmetry does not. Nothing read off one GUV's
# kymograph supports a directional claim.
#
# Two references have to be carried, because the index is a ratio of two noisy
# quantities and is not automatically centred on zero:
#
#   PRE-PULSE. Both windows are normalised to their own pre-pulse level, so
#   the index is 0 there by construction. Its scatter before the pulse is
#   therefore a direct measure of the noise floor, on the same vesicles and in
#   the same units as the post-pulse number it has to be judged against.
#
#   THE 30 V CONTROL. A field too small to porate, over a whole record, shows
#   what bleaching and rotation produce on their own.
#
# A post-pulse median that does not clear both is not a directional effect.
# -----------------------------------------------------------------------------

POLE_EQUATOR_MIN_GUV = 5          # GUVs required before a voltage is reported
POLE_EQUATOR_PRE_MIN_S = -30.0    # pre-pulse window, the systematic-offset check
POLE_EQUATOR_N_BOOTSTRAP = 2000   # resamples for the CI on each cell median

# Add a row per window pooling every pulsed condition, control excluded.
#
# n is the binding constraint per voltage: 8 GUVs at 900 V resolve to about
# +/-0.04, which cannot rule out anything. Cortex breakdown here is a late,
# slow process, and there is no evidence in these data that its DIRECTION is
# graded by voltage even though its extent is -- so pooling the pulsed
# conditions buys roughly an order of magnitude in n for a question that does
# not obviously need the voltage axis.
#
# What it costs: any voltage-dependence of the direction is averaged away, and
# a strong asymmetry confined to the highest fields would be diluted by the
# low ones. The per-voltage rows stay in the table for exactly that reason;
# the pooled row is the better-powered test, not the replacement.
POLE_EQUATOR_POOL_VOLTAGES = True
POOLED_LABEL = "pulsed (pooled)"

# Experiments required before a cell's interval is treated as trustworthy.
#
# A cluster bootstrap resamples whole chambers, so with only a handful of them
# there are very few distinct resamples and the interval comes out far too
# narrow. Measured on pure-null data (12 vesicles per chamber, between-chamber
# SD 0.010, within 0.030, 200 trials each), the rate at which the 95% interval
# wrongly excludes zero is:
#
#      3 chambers   27%        12 chambers  7.5%
#      5 chambers   15%        20 chambers  6.0%
#      8 chambers   11%        29 chambers  4.5%
#
# Per-voltage cells here have 2 to 6 experiments, so their intervals cannot
# support a verdict at all -- which is the quantitative reason the pooled row
# (29 experiments) is the test and the per-voltage rows are the check on it.
# Cells below this threshold still report their median and interval, since
# both are informative, but `resolved` is held at False and `ci_reliable`
# records why.
POLE_EQUATOR_MIN_CLUSTERS = 12

# Post-pulse windows the index is summarised over, as (label, t_lo, t_hi) in
# seconds. More than one because the answer depends on when you look, and a
# single window hides that.
#
# A 60 s window alone is not a test of this question on this dataset. The
# cortex peak breakdown table shows almost nothing lost immediately after the
# pulse at any voltage (medians from -0.020 to +0.025) and most of the loss
# arriving by the end of the record (0.25 at 400 V). Asking whether a loss
# that has not happened yet is directional returns "no asymmetry" for a reason
# that has nothing to do with direction. The burst window is kept because
# whether anything happens in the first minute is worth reporting; the late
# window is where the loss actually is.
#
# The late window ends at 431 s, the shortest full-length record here, so
# every experiment can supply it. Widening it past that would silently mean
# "the long sessions only".
POLE_EQUATOR_WINDOWS = (
    ("burst", 0.0, 60.0),
    ("late", 300.0, 431.0),
)

# When set, the index is reported a second time over only those GUVs whose
# cortex peak fell by at least this fraction by the end of the record.
#
# This is the sharper form of the question. Pooled over every cortex-bearing
# vesicle, most of which lost no cortex at all, a real asymmetry among the
# ones that did break down is diluted toward zero by vesicles that had nothing
# to be asymmetric about. Conditioning asks: of the cortices that failed, did
# they fail directionally?
#
# It is also a selection, and selections invite their own artefacts, so both
# strata are always reported side by side rather than the conditioned one
# replacing the pooled one. Set it from the observed peak_drop_final
# distribution, which is printed below whenever this is None. Requires
# peak_drop_final on the input frame (pass the output of
# add_cortex_peak_metrics, not the raw summary).
POLE_EQUATOR_MIN_PEAK_DROP = None


def _time_mean(t, y, lo: float, hi: float):
  """Time-weighted mean of y over lo < t <= hi, plus the sample count.

  Trapezoidal, and taken on the RAW samples rather than on the interpolated
  grid, because both are non-uniform in time and a plain mean or median would
  then report the sampling schedule as much as the signal. The acquisition
  runs a fast segment of about a second immediately after the pulse and a slow
  one for the rest of the record, so an unweighted average over the first
  minute is dominated by that first second.

  Returns (value, n_samples); (NaN, 0) if nothing finite falls in the window.
  """
  t = np.asarray(t, dtype=float)
  y = np.asarray(y, dtype=float)
  ok = np.isfinite(t) & np.isfinite(y) & (t > lo) & (t <= hi)
  t, y = t[ok], y[ok]
  if t.size == 0:
    return np.nan, 0
  if t.size == 1:
    return float(y[0]), 1
  order = np.argsort(t)
  t, y = t[order], y[order]
  span = float(t[-1] - t[0])
  if span <= 0:
    return float(np.mean(y)), int(t.size)
  area = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))
  return area / span, int(t.size)


def _pole_equator_time_grid(t_min: float, t_max: float) -> np.ndarray:
  """_peak_time_grid extended backwards, since these traces span the pulse.

  The actin cortex traces start at the pulse; the pole/equator traces carry
  negative times as well, and the pre-pulse segment is the noise floor rather
  than padding, so it is kept.
  """
  pre = np.arange(np.floor(min(t_min, 0.0)), 0.0, 0.5) if t_min < 0 else \
      np.array([], dtype=float)
  return np.unique(np.concatenate([pre, _peak_time_grid(t_max)]))


def collect_pole_equator_traces(df: pd.DataFrame, outputs_root: str):
  """Pool every cortex-bearing GUV's pole/equator traces onto one time axis.

  Mirrors collect_cortex_peak_traces: same population filter (cortex-bearing
  and Stagnate), same walk over the per-experiment folders, same
  interpolation onto a shared grid with samples outside a GUV's own record
  left NaN rather than extrapolated.

  Returns (grid, by_voltage, per_guv):
    grid       shared time axis in seconds, negative before the pulse
    by_voltage voltage -> dict of (n_guv, n_times) arrays keyed 'pole',
               'equator', 'polarization'
    per_guv    LONG format, one row per (vesicle, window). Long rather than a
               column per window so that adding a window does not change the
               shape of the table or of anything reading it.
  Returns (None, None, None) if nothing usable was found.
  """
  root = Path(outputs_root)
  keep = df.assign(group=df.apply(assign_cortex_group, axis=1))
  keep = keep[(keep["group"] == "Branched, cortex")
              & (keep["size_category"] == "Stagnate")]
  if keep.empty:
    print("Pole/equator pooling skipped: no cortex-bearing stagnate GUVs.")
    return None, None, None
  has_drop = "peak_drop_final" in keep.columns
  wanted = {(r["experiment"], str(r["guv_id"])):
            (r["voltage"],
             float(r["peak_drop_final"]) if has_drop else np.nan)
            for _, r in keep.iterrows()}

  raw, n_files = [], 0
  t_min, t_max = 0.0, 0.0
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    files = [f for f in exp_dir.glob("**/*_pole_vs_equator_traces.csv")
             if RESULTS_SUBFOLDER not in f.parts]
    if not files:
      continue
    n_files += 1
    try:
      tr = pd.read_csv(files[0])
    except Exception as e:
      print(f"Skipping pole/equator file {files[0]}: {e}")
      continue
    if "time_s" not in tr:
      continue
    t = tr["time_s"].to_numpy(float)
    for col in tr.columns:
      m = re.match(r"GUV_(.+)_polarization_index$", col)
      if not m:
        continue
      gid = m.group(1)
      key = (exp_dir.name, gid)
      if key not in wanted:
        continue
      cols = {"polarization": col,
              "pole": f"GUV_{gid}_pole_mean",
              "equator": f"GUV_{gid}_equator_mean"}
      if not all(c in tr.columns for c in cols.values()):
        continue
      vals = {k: tr[c].to_numpy(float) for k, c in cols.items()}
      ok = np.isfinite(t) & np.isfinite(vals["polarization"])
      if ok.sum() < 4:
        continue
      volt, drop = wanted[key]
      raw.append((volt, drop, exp_dir.name, gid, t, vals))
      t_min = min(t_min, float(t[ok].min()))
      t_max = max(t_max, float(t[ok].max()))

  if not raw:
    if n_files:
      print(f"Pole/equator pooling: {n_files} file(s) found, but no GUV in "
            "them is a cortex-bearing stagnate vesicle in the summary table.")
    else:
      print("Pole/equator pooling skipped: no *_pole_vs_equator_traces.csv "
            "found. Re-run the per-experiment stage with "
            "EXPORT_ACTIN_KYMOGRAPH = True.")
    return None, None, None

  grid = _pole_equator_time_grid(t_min, t_max)

  stacks, rows = {}, []
  for volt, drop, exp, gid, t, vals in raw:
    # The interpolated grid is for the pooled median curve only. Every
    # per-GUV number below is taken from that GUV's own raw samples, so the
    # summary does not inherit the grid's density.
    for k, y in vals.items():
      ok = np.isfinite(t) & np.isfinite(y)
      if ok.sum() < 4:
        z = np.full(grid.shape, np.nan)
      else:
        z = np.interp(grid, t[ok], y[ok])
        z[(grid < t[ok].min()) | (grid > t[ok].max())] = np.nan
      stacks.setdefault(volt, {kk: [] for kk in vals})[k].append(z)

    pol_pre, n_pre = _time_mean(t, vals["polarization"],
                                POLE_EQUATOR_PRE_MIN_S, 0.0)
    for label, lo, hi in POLE_EQUATOR_WINDOWS:
      pol, n_w = _time_mean(t, vals["polarization"], lo, hi)
      pole, _ = _time_mean(t, vals["pole"], lo, hi)
      eq, _ = _time_mean(t, vals["equator"], lo, hi)
      rows.append({
          "experiment": exp, "guv_id": gid, "voltage": volt,
          "window": label, "t_lo": lo, "t_hi": hi,
          "peak_drop_final": drop,
          "pol_pre": pol_pre, "n_pre": n_pre,
          "pol_post": pol, "pole_post": pole, "equator_post": eq,
          "n_post": n_w,
      })

  by_voltage = {v: {k: np.vstack(a) for k, a in d.items()}
                for v, d in stacks.items()}
  return grid, by_voltage, pd.DataFrame(rows)


def _pole_equator_strata(per_guv: pd.DataFrame):
  """(label, boolean mask) for each population the index is reported over.

  Always includes every cortex-bearing vesicle. Adds the cortex-loss subset
  only when POLE_EQUATOR_MIN_PEAK_DROP is set AND peak_drop_final is actually
  present, so a frame passed without the actin merge degrades to the pooled
  test with a warning rather than silently reporting an unconditioned result
  under a conditioned label.
  """
  strata = [("all cortex-bearing", pd.Series(True, index=per_guv.index))]
  if POLE_EQUATOR_MIN_PEAK_DROP is None:
    return strata
  if per_guv["peak_drop_final"].notna().sum() == 0:
    print("  POLE_EQUATOR_MIN_PEAK_DROP is set but no peak_drop_final values "
          "reached this function. Pass the output of add_cortex_peak_metrics "
          "rather than the raw summary; reporting the pooled test only.")
    return strata
  thr = float(POLE_EQUATOR_MIN_PEAK_DROP)
  strata.append((f"peak drop >= {thr:.2f}",
                 per_guv["peak_drop_final"] >= thr))
  return strata


def holm(pvals):
  """Holm-Bonferroni adjusted p-values, NaN-preserving.

  Lives here rather than in each script so the two places that correct for
  multiplicity cannot end up doing it differently.
  """
  p = np.asarray(pvals, dtype=float)
  out = np.full(p.shape, np.nan)
  idx = np.where(np.isfinite(p))[0]
  if idx.size == 0:
    return out
  order = idx[np.argsort(p[idx])]
  m = len(order)
  running = 0.0
  for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * float(p[i])))
    out[i] = running
  return out


def _boot_median_ci(v, clusters=None, n_boot: int = None, seed: int = 0):
  """Percentile bootstrap CI for a median, and the half-width it implies.

  The CI is what makes a null statement mean anything. Comparing a population
  median against the spread of INDIVIDUAL pre-pulse values, which is what this
  function replaces, asks whether one vesicle's asymmetry would have been
  visible -- a far coarser question than whether the median over thirty of
  them has moved, and one that reports "not resolved" for effects the data
  could easily have seen. The per-vesicle scatter here is about 0.033, so a
  cell of 30 pins its median to roughly 0.008; judging that median against
  0.043 discards a factor of five.

  Bootstrap rather than a normal approximation because these values are
  bounded on [-1, 1], skewed when only some vesicles have broken down, and
  not independent (several GUVs share a field of view). None of those is fatal
  to a resampled median; all of them are to a standard error.

  When *clusters* is given (one label per value, here the experiment), whole
  clusters are resampled rather than individual vesicles. Vesicles in one
  chamber share a preparation, a field and a day; treating them as independent
  is what makes a naive interval too narrow, and the error grows with pooling
  -- across 250 vesicles from 29 experiments the difference is roughly the
  square root of the mean cluster size, so about 3x. Falls back to resampling
  vesicles when fewer than 3 clusters are present, since a cluster bootstrap
  over 2 groups estimates nothing.

  Returns (lo, hi, half_width, p). half_width is the resolution: the smallest
  median this cell could have distinguished from zero. p is the two-sided
  percentile bootstrap p-value against a median of zero, taken from the SAME
  resampling as the interval.

  That last point is the reason this replaced the Wilcoxon signed-rank test
  that used to sit alongside it. Wilcoxon has no notion of clustering, so it
  treated every vesicle as independent while the interval did not; on the
  pooled row that produced a p of 0.034 next to a CI spanning zero, two
  statistics disagreeing for a reason belonging to the code rather than the
  data. A p and an interval from one resampling cannot contradict each other.

  Resolution floor: with n_boot resamples the smallest reportable p is
  1/n_boot, so a p at that floor means "below this", not an exact value.
  """
  n_boot = int(POLE_EQUATOR_N_BOOTSTRAP if n_boot is None else n_boot)
  v = np.asarray(v, dtype=float)
  keep = np.isfinite(v)
  if clusters is not None:
    clusters = np.asarray(clusters)[keep]
  v = v[keep]
  if v.size < 3:
    # Four values, not three. Both callers unpack four, so the short return
    # this used to make raised ValueError on any cell with one or two
    # measurable vesicles -- which killed report_pole_equator before it
    # printed anything, and with it every stage after it in __main__.
    return np.nan, np.nan, np.nan, np.nan
  rng = np.random.default_rng(seed)          # fixed: reruns reproduce

  if clusters is not None:
    uniq = list(dict.fromkeys(clusters.tolist()))
    if len(uniq) >= 3:
      idx_by_cluster = [np.where(clusters == u)[0] for u in uniq]
      n_c = len(uniq)
      meds = np.empty(n_boot, dtype=float)
      picks = rng.integers(0, n_c, (n_boot, n_c))
      for b in range(n_boot):
        take = np.concatenate([idx_by_cluster[j] for j in picks[b]])
        meds[b] = np.median(v[take])
    else:
      meds = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
  else:
    meds = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)

  lo = float(np.percentile(meds, 2.5))
  hi = float(np.percentile(meds, 97.5))
  # Two-sided: twice the smaller tail, floored at one resample so it is never
  # reported as exactly zero.
  frac_le = float(np.mean(meds <= 0.0))
  frac_ge = float(np.mean(meds >= 0.0))
  p = min(1.0, 2.0 * min(frac_le, frac_ge))
  p = max(p, 1.0 / len(meds))
  return lo, hi, float(0.5 * (hi - lo)), float(p)


def report_pole_equator(per_guv: pd.DataFrame, out_path: Path) -> pd.DataFrame:
  """Per-window, per-stratum, per-voltage summary of the polarization index.

  The test is a Wilcoxon signed-rank of each GUV's window value against zero,
  not a comparison of two group means: every vesicle carries its own
  pre-pulse normalisation, so zero is a fixed reference rather than an
  estimated one, and the paired form is what matches that design.
  """
  if per_guv is None or per_guv.empty:
    return pd.DataFrame()

  # One row per GUV for the checks that are about vesicles, not windows.
  first = per_guv.drop_duplicates(subset=["experiment", "guv_id"])

  pre_all = first["pol_pre"].dropna()
  if len(pre_all) >= 3:
    # The pre-pulse index is 0 by construction, so its MEDIAN over hundreds of
    # vesicles should sit within a standard error or two of zero. Where it does
    # not, the residue is a systematic offset in the measurement -- unequal
    # background between the pole and equator windows, or an electrode angle
    # slightly off the one assumed -- and unlike random scatter it does not
    # shrink with n. It is therefore a hard floor on accuracy that no sample
    # size defeats, and it is a different quantity from the bootstrap CI below,
    # which only bounds precision.
    sys_offset = float(pre_all.median())
    pre_sd = float(pre_all.std())
    se_pre = pre_sd / np.sqrt(len(pre_all)) if len(pre_all) else np.nan
    print(f"\n  Pre-pulse check: median {sys_offset:+.4f}, "
          f"IQR [{pre_all.quantile(.25):+.4f}, {pre_all.quantile(.75):+.4f}], "
          f"per-vesicle SD {pre_sd:.4f}, across {len(pre_all)} GUV(s).")
    if np.isfinite(se_pre) and se_pre > 0 and abs(sys_offset) > 2 * se_pre:
      print(f"    This should be 0 by construction but sits {abs(sys_offset) / se_pre:.1f} "
            f"standard errors off it. Treat |{abs(sys_offset):.4f}| as a "
            "systematic floor on accuracy: a post-pulse median smaller than "
            "that is not distinguishable from the offset, however many "
            "vesicles are pooled.")
    else:
      print("    Consistent with zero, so there is no systematic offset to "
            "subtract; precision alone limits what can be resolved.")
  else:
    sys_offset = np.nan
    print("\n  Pre-pulse check unavailable: fewer than 3 GUVs have pre-pulse "
          "samples, so there is no way to tell a systematic offset in the "
          "index from a real post-pulse asymmetry.")

  drops = first["peak_drop_final"].dropna()
  if len(drops) >= 4:
    print(f"  Cortex peak lost by end of record, these {len(drops)} GUV(s): "
          f"quartiles {drops.quantile(.25):+.3f} / {drops.median():+.3f} / "
          f"{drops.quantile(.75):+.3f}")
    if POLE_EQUATOR_MIN_PEAK_DROP is None:
      print("  POLE_EQUATOR_MIN_PEAK_DROP is None, so the index below pools "
            "every cortex-bearing vesicle, including those that lost no "
            "cortex and so have no asymmetry to show. Set it from the "
            "quartiles above to add the conditioned test.")

  strata = _pole_equator_strata(per_guv)
  rows = []
  for s_label, s_mask in strata:
    sub_s = per_guv[s_mask]
    for w_label, lo, hi in POLE_EQUATOR_WINDOWS:
      sub_w = sub_s[sub_s["window"] == w_label]
      volts = sorted(sub_w["voltage"].unique(), key=_voltage_key)
      # The pooled row is built from the same cell code as a single voltage,
      # so it cannot drift from the rows it summarises.
      if POLE_EQUATOR_POOL_VOLTAGES:
        volts = volts + [POOLED_LABEL]
      for volt in volts:
        # cell_all keeps the vesicles with no measurable index. They are not
        # missing data: the index is undefined exactly when no cortex peak is
        # detected, which is what complete breakdown looks like. Dropping them
        # silently would condition the late window on vesicles that STILL HAD
        # a cortex at the end of it -- selecting against the very population
        # in which a directional failure would show, and reporting the
        # survivors as though they were the sample.
        is_pooled = volt == POOLED_LABEL
        if is_pooled:
          cell_all = sub_w[sub_w["voltage"] != CONTROL_VOLTAGE]
        else:
          cell_all = sub_w[sub_w["voltage"] == volt]
        cell = cell_all[cell_all["n_post"] > 0]
        v = cell["pol_post"].dropna()
        n_gone = int(len(cell_all) - len(v))
        base = {
            "stratum": s_label, "window": w_label,
            "t_lo": lo, "t_hi": hi, "voltage": volt, "pooled": is_pooled,
            "n_no_cortex": n_gone,
            "n_exp": int(cell["experiment"].nunique()) if len(cell) else 0,
        }
        if v.empty:
          rows.append({**base, "n_guv": 0,
                       "frac_no_cortex": 1.0 if len(cell_all) else np.nan,
                       "median_pol": np.nan, "q1": np.nan, "q3": np.nan,
                       "ci_lo": np.nan, "ci_hi": np.nan,
                       "resolution": np.nan, "median_pole": np.nan,
                       "median_equator": np.nan,
                       "boot_p_vs_zero": np.nan, "p_holm": np.nan,
                       "ci_reliable": False, "resolved": False})
          continue
        # Clustered on experiment: vesicles in one chamber are not
        # independent, and pooling voltages pools chambers as well, so an
        # unclustered interval here would be badly overconfident. The p comes
        # out of the same resampling, so it carries the same clustering.
        ci_lo, ci_hi, half, p = _boot_median_ci(
            v.to_numpy(float), clusters=cell.loc[v.index, "experiment"])
        med = float(v.median())
        excl_zero = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi)
                         and (ci_lo > 0 or ci_hi < 0))
        over_sys = bool(not np.isfinite(sys_offset)
                        or abs(med) > abs(sys_offset))
        rows.append({**base, "n_guv": int(len(v)),
                     "frac_no_cortex": float(n_gone / len(cell_all))
                                       if len(cell_all) else np.nan,
                     "median_pol": med,
                     "q1": float(v.quantile(.25)),
                     "q3": float(v.quantile(.75)),
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "resolution": half,
                     "median_pole": float(cell["pole_post"].median()),
                     "median_equator": float(cell["equator_post"].median()),
                     "boot_p_vs_zero": p, "p_holm": np.nan,
                     "ci_reliable": bool(base["n_exp"]
                                         >= POLE_EQUATOR_MIN_CLUSTERS),
                     "_excl_zero": excl_zero, "_over_sys": over_sys,
                     "resolved": False})

  tbl = pd.DataFrame(rows)
  if tbl.empty:
    print("  No window produced a usable value. Check that the record "
          "reaches the window bounds in POLE_EQUATOR_WINDOWS.")
    return tbl

  # Multiplicity. Without this, `resolved` is a per-cell test run twenty-odd
  # times per report and will flag roughly one cell per run by chance -- which
  # is not a bug in the data but in the flag. Holm is applied within each
  # (stratum, window) family across the per-voltage pulsed cells. The pooled
  # row is excluded from that family and corrected separately across windows:
  # it is one prespecified test per window, not one of many.
  if "p_holm" in tbl.columns:
    per_volt = (~tbl["pooled"].astype(bool)) & (tbl["voltage"] != CONTROL_VOLTAGE)
    for (s_label, w_label), grp in tbl[per_volt].groupby(["stratum", "window"]):
      tbl.loc[grp.index, "p_holm"] = holm(grp["boot_p_vs_zero"].to_numpy())
    pooled_rows = tbl[tbl["pooled"].astype(bool)]
    if not pooled_rows.empty:
      for s_label, grp in pooled_rows.groupby("stratum"):
        tbl.loc[grp.index, "p_holm"] = holm(
            grp["boot_p_vs_zero"].to_numpy())
    ctrl_rows = tbl[tbl["voltage"] == CONTROL_VOLTAGE]
    tbl.loc[ctrl_rows.index, "p_holm"] = ctrl_rows["boot_p_vs_zero"]

  if "_excl_zero" in tbl.columns:
    ph = tbl["p_holm"]
    tbl["resolved"] = (tbl["_excl_zero"].fillna(False).astype(bool)
                       & tbl["_over_sys"].fillna(False).astype(bool)
                       & tbl["ci_reliable"].fillna(False).astype(bool)
                       & (ph.isna() | (ph < 0.05)))
    tbl = tbl.drop(columns=["_excl_zero", "_over_sys"])
  tbl.to_csv(out_path / "pole_equator_summary.csv", index=False,
             float_format="%.4f")
  per_guv.to_csv(out_path / "pole_equator_per_guv.csv", index=False,
                 float_format="%.4f")

  for s_label, _ in strata:
    for w_label, lo, hi in POLE_EQUATOR_WINDOWS:
      part = tbl[(tbl["stratum"] == s_label) & (tbl["window"] == w_label)]
      print(f"\n  {s_label}, {w_label} window ({lo:.0f} to {hi:.0f} s)")
      if part.empty:
        print("    nothing measurable in this window")
        continue
      show = ["voltage", "n_guv", "n_exp", "n_no_cortex", "median_pol",
              "ci_lo", "ci_hi", "resolution", "median_pole",
              "median_equator", "boot_p_vs_zero", "p_holm", "ci_reliable",
              "resolved"]
      print(part[[c for c in show if c in part.columns]].to_string(index=False))
      # A window in which neither pole nor equator has moved off 1.0 cannot
      # answer the question either way, and reads as a null if not said.
      moved = ((part[~part["pooled"].astype(bool)][["median_pole",
                                                     "median_equator"]] - 1.0)
               .abs().max().max())
      if np.isfinite(moved) and moved < 0.05:
        print(f"    Neither window has moved more than {moved:.3f} from its "
              "pre-pulse level here, so there is no cortex loss yet to be "
              "directional. Read this as 'nothing has happened', not as "
              "'the loss is symmetric'.")
      # A near-zero median with a long upper tail is a mixture, not a null:
      # some vesicles are strongly polarised and the rest, having lost no
      # cortex, sit at zero and outvote them. The median is the wrong summary
      # for that and would be reported as "no effect" if only it were read.
      mix_ref = max(abs(sys_offset) if np.isfinite(sys_offset) else 0.0, 0.01)
      split = part[(~part["resolved"].astype(bool))
                   & (~part["pooled"].astype(bool))
                   & (part["q3"] > 4.0 * mix_ref)]
      for _, r in split.iterrows():
        print(f"    {r['voltage']}: median {r['median_pol']:+.3f} is not "
              f"resolved but the upper quartile is {r['q3']:+.3f}. That is a "
              "subset of vesicles moving while the rest sit at zero, not an "
              "absent effect. Set POLE_EQUATOR_MIN_PEAK_DROP to test that "
              "subset directly.")

      # Vesicles whose cortex has gone entirely by this window. Reported per
      # window because it grows with time and is the selection that would
      # otherwise be invisible.
      per_v = part[~part["pooled"].astype(bool)]
      gone = per_v[per_v["n_no_cortex"] > 0]
      if not gone.empty:
        worst = gone.loc[gone["frac_no_cortex"].idxmax()]
        total_gone = int(per_v["n_no_cortex"].sum())
        print(f"    {total_gone} vesicle(s) have no detectable cortex peak "
              f"anywhere in this window and so carry no index; worst at "
              f"{worst['voltage']} ({int(worst['n_no_cortex'])} of "
              f"{int(worst['n_no_cortex'] + worst['n_guv'])}, "
              f"{worst['frac_no_cortex']:.0%}). The index exists only where a "
              "cortex still does, so this window describes the vesicles that "
              "had not finished breaking down -- which is a selection against "
              "the ones most likely to be asymmetric.")

      # The sensitivity statement, led by the pooled row where there is one.
      # A null is only a result if it comes with the size of effect that
      # would have been seen.
      pooled_row = part[part["pooled"].astype(bool)]
      if not pooled_row.empty:
        r = pooled_row.iloc[0]
        ctrl_row = part[part["voltage"] == CONTROL_VOLTAGE]
        bound = float(np.nanmax([r["resolution"],
                                 abs(sys_offset) if np.isfinite(sys_offset)
                                 else 0.0]))
        verdict = ("RESOLVED" if bool(r["resolved"]) else "not resolved")
        print(f"    POOLED over pulsed conditions: median "
              f"{r['median_pol']:+.4f}, CI [{r['ci_lo']:+.4f}, "
              f"{r['ci_hi']:+.4f}], {int(r['n_guv'])} GUV(s) from "
              f"{int(r['n_exp'])} experiment(s) -- {verdict}.")
        # How much the clustering cost, since it varies with how correlated
        # vesicles in a chamber turn out to be and is otherwise invisible. A
        # ratio near 1 means the vesicles behaved independently and the
        # pooled n is close to its face value; a large one means the
        # effective n is much smaller than the vesicle count suggests.
        pool_v = sub_w[(sub_w["voltage"] != CONTROL_VOLTAGE)
                       & (sub_w["n_post"] > 0)]["pol_post"].dropna()
        if len(pool_v) >= 3 and np.isfinite(r["resolution"]):
          naive = _boot_median_ci(pool_v.to_numpy(float))[2]
          if np.isfinite(naive) and naive > 0:
            print(f"    Clustering on experiment widened that interval "
                  f"{r['resolution'] / naive:.1f}x against treating vesicles "
                  f"as independent. The vesicle count is "
                  f"{int(r['n_guv'])}; the effective one is nearer "
                  f"{int(r['n_guv']) / max((r['resolution'] / naive) ** 2, 1):.0f}.")
        if not bool(r["resolved"]):
          print(f"    Bounding that: the pooled median resolves to "
                f"+/-{r['resolution']:.4f}"
                + (f" and the systematic offset is {abs(sys_offset):.4f}"
                   if np.isfinite(sys_offset) else "")
                + f", so any pole-equator asymmetry common to the pulsed "
                  f"conditions is below about {bound:.3f} on this index. "
                  "Report the bound, not just the absence.")
        if not ctrl_row.empty and np.isfinite(ctrl_row["median_pol"].iloc[0]):
          cm = float(ctrl_row["median_pol"].iloc[0])
          if abs(cm) >= abs(r["median_pol"]):
            print(f"    The {CONTROL_VOLTAGE} control median is {cm:+.4f}, at "
                  f"least as large as the pooled pulsed value "
                  f"{r['median_pol']:+.4f}. A field-driven asymmetry cannot "
                  "be largest where there is no field, so the spread across "
                  "conditions is noise -- which supports the bound "
                  "independently of the interval.")
        print("    The per-voltage rows above stay the check on this: pooling "
              "would hide an asymmetry confined to the highest fields.")

      pulsed = part[(part["voltage"] != CONTROL_VOLTAGE)
                    & (~part["pooled"].astype(bool))
                    & part["n_guv"].ge(POLE_EQUATOR_MIN_GUV)]
      unreliable = pulsed[~pulsed["ci_reliable"].astype(bool)]
      if not unreliable.empty:
        print(f"    {len(unreliable)} of {len(pulsed)} per-voltage cells come "
              f"from fewer than {POLE_EQUATOR_MIN_CLUSTERS} experiments "
              f"(here {int(pulsed['n_exp'].min())}-{int(pulsed['n_exp'].max())}"
              "), so their intervals are too narrow to carry a verdict and "
              "`resolved` is held False for them. Their medians are still "
              "worth reading as a check on whether the pooled value hides a "
              "pattern across voltage.")
      hits = pulsed[pulsed["resolved"].astype(bool)]
      if not hits.empty:
        print(f"    Individually resolved after Holm: "
              f"{', '.join(hits['voltage'])}.")
      # Name the gate that actually blocked each near-miss. Printing only the
      # adjusted p implies multiplicity was the reason even when it was not,
      # and the three gates fail for genuinely different reasons.
      near = pulsed[(~pulsed["resolved"].astype(bool))
                    & pulsed["boot_p_vs_zero"].lt(0.05)]
      for _, r in near.iterrows():
        if not bool(r["ci_reliable"]):
          why = (f"only {int(r['n_exp'])} experiment(s), below "
                 f"{POLE_EQUATOR_MIN_CLUSTERS}, so the interval is not "
                 "trustworthy")
        elif np.isfinite(r["p_holm"]) and r["p_holm"] >= 0.05:
          why = (f"Holm-adjusted p = {r['p_holm']:.3f} across the "
                 f"{len(pulsed)} pulsed cells in this window")
        elif np.isfinite(sys_offset) and abs(r["median_pol"]) <= abs(sys_offset):
          why = (f"median {r['median_pol']:+.4f} does not exceed the "
                 f"systematic offset {abs(sys_offset):.4f}, so it is a "
                 "precise measurement of the offset rather than of an effect")
        else:
          why = "the interval includes zero"
        print(f"    {r['voltage']}: raw p = {r['boot_p_vs_zero']:.3f}, but "
              f"{why}. Not a finding.")

      thin = part[(~part["pooled"].astype(bool))
                  & (part["n_guv"] < POLE_EQUATOR_MIN_GUV)]["voltage"].tolist()
      if thin:
        print(f"    Fewer than {POLE_EQUATOR_MIN_GUV} GUVs at: "
              f"{', '.join(thin)} -- indicative only.")

  per_v = tbl[(tbl["voltage"] != CONTROL_VOLTAGE)
              & (~tbl["pooled"].astype(bool))]
  n_tests = int(len(per_v))
  if n_tests > 1:
    n_thin = int((~per_v["ci_reliable"].astype(bool)).sum())
    print(f"\n  {n_tests} per-voltage tests are reported above. `resolved` "
          "requires three things of a cell: an interval excluding zero, a "
          "median above the systematic offset, and a Holm-adjusted p. Read "
          "boot_p_vs_zero against p_holm and ci_reliable, not against 0.05.")
    if n_thin:
      print(f"  For {n_thin} of them the binding constraint is neither p nor "
            f"the offset but the chamber count: below "
            f"{POLE_EQUATOR_MIN_CLUSTERS} experiments the interval itself is "
            "not trustworthy, so no p could rescue them.")
  return tbl


def plot_pole_equator_pooled(df: pd.DataFrame, outputs_root: str,
                             output_dir: str, min_frac: float = 0.5):
  """Pooled time course, then one per-GUV panel per post-pulse window.

  Left panel mirrors plot_cortex_peak_timecourse (median and IQR per voltage,
  drawn only where at least min_frac of the contributing GUVs are still
  measurable). The window panels mirror plot_cortex_peak_breakdown (per-GUV
  points with median and IQR against the 30 V control band).
  """
  out_path = sub_dir(output_dir, "cortex_breakdown")
  out_path.mkdir(parents=True, exist_ok=True)

  grid, by_voltage, per_guv = collect_pole_equator_traces(df, outputs_root)
  if grid is None:
    return

  n_guv_total = per_guv.drop_duplicates(["experiment", "guv_id"]).shape[0]
  print(f"\nPooled pole-vs-equator traces from {n_guv_total} cortex-bearing "
        f"stagnate GUV(s) across {per_guv['experiment'].nunique()} "
        "experiment(s).")

  tbl = report_pole_equator(per_guv, out_path)

  voltages = sorted(by_voltage, key=_voltage_key)
  pulsed = [v for v in voltages if v != CONTROL_VOLTAGE]
  cmap = mcolors.LinearSegmentedColormap.from_list(
      "ep_blues", [PALETTE["pale_blue"], PALETTE["medium_blue"],
                   PALETTE["dark_blue"]])
  shades = {v: cmap(0.3 + 0.7 * i / max(len(pulsed) - 1, 1))
            for i, v in enumerate(pulsed)}

  t_hi_max = max(hi for _, _, hi in POLE_EQUATOR_WINDOWS)
  n_panels = 1 + len(POLE_EQUATOR_WINDOWS)
  fig, axes = plt.subplots(
      1, n_panels,
      figsize=fig_size("pole_equator_pooled", 6.0 * n_panels, 5.2, n_panels))
  axes = np.atleast_1d(axes)

  ax = axes[0]
  win = grid <= t_hi_max
  for volt in voltages:
    arr = by_voltage[volt]["polarization"]
    n_t = np.sum(np.isfinite(arr), axis=0)
    enough = n_t >= max(2, int(np.ceil(min_frac * arr.shape[0])))
    sel = win & enough
    if not sel.any():
      continue
    med = np.nanmedian(arr[:, sel], axis=0)
    q1 = np.nanpercentile(arr[:, sel], 25, axis=0)
    q3 = np.nanpercentile(arr[:, sel], 75, axis=0)
    is_ctrl = volt == CONTROL_VOLTAGE
    colour = PALETTE["dark_red"] if is_ctrl else shades[volt]
    ax.plot(grid[sel], med, color=colour, lw=2.0 if is_ctrl else 1.5,
            ls="--" if is_ctrl else "-", zorder=5 if is_ctrl else 3,
            label=f"{field_label(volt)} kV/cm (n={arr.shape[0]})")
    ax.fill_between(grid[sel], q1, q3, color=colour, alpha=0.13, lw=0,
                    zorder=2)
  for _, lo, hi in POLE_EQUATOR_WINDOWS:
    ax.axvspan(lo, hi, color=PALETTE["light_grey"], alpha=0.35, zorder=0)
  ax.axhline(0.0, color=ANNOTATION_TEXT, lw=0.8, ls=":", zorder=1)
  ax.axvline(0.0, color=ANNOTATION_TEXT, lw=0.8, ls="-", alpha=0.6, zorder=1)
  ax.set_xlabel("time relative to pulse (s)")
  ax.set_ylabel("polarization index  (equator $-$ pole) / (equator $+$ pole)")
  ax.set_title("Pooled time course (shaded = summary windows)", fontsize=10)
  ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  usable = per_guv[per_guv["n_post"] > 0]
  pre_first = per_guv.drop_duplicates(["experiment", "guv_id"])["pol_pre"]
  pre_first = pre_first.dropna()
  x_volts = [v for v in voltages if v != CONTROL_VOLTAGE]
  rng = np.random.default_rng(0)

  for ax, (w_label, lo, hi) in zip(axes[1:], POLE_EQUATOR_WINDOWS):
    sub_w = usable[usable["window"] == w_label]
    ctrl = sub_w[sub_w["voltage"] == CONTROL_VOLTAGE]["pol_post"].dropna()
    if len(ctrl) >= 2:
      ax.axhspan(float(ctrl.quantile(.25)), float(ctrl.quantile(.75)),
                 color=PALETTE["pale_red"], alpha=0.55, zorder=0)
      ax.axhline(float(ctrl.median()), color=PALETTE["dark_red"], lw=1.2,
                 ls="--", zorder=1,
                 label=f"{CONTROL_VOLTAGE} control (n={len(ctrl)})")
    elif len(ctrl) == 1:
      ax.axhline(float(ctrl.iloc[0]), color=PALETTE["dark_red"], lw=1.2,
                 ls="--", zorder=1,
                 label=f"{CONTROL_VOLTAGE} control (n=1)")
    if len(ctrl) and len(ctrl) < MIN_CONTROL_N:
      print(f"  NOTE: the {CONTROL_VOLTAGE} band in the {w_label} panel is "
            f"drawn from {len(ctrl)} GUV(s), below MIN_CONTROL_N="
            f"{MIN_CONTROL_N}. Indicative only.")
    if len(pre_first) >= 3:
      so = abs(float(pre_first.median()))
      if so > 0:
        ax.axhspan(-so, so, color=PALETTE["light_grey"], alpha=0.5, zorder=0,
                   label="systematic offset (pre-pulse median)")

    x = np.arange(len(x_volts))
    for idx, volt in enumerate(x_volts):
      cell = sub_w[sub_w["voltage"] == volt]
      hi_drop = cell
      if POLE_EQUATOR_MIN_PEAK_DROP is not None:
        hi_drop = cell[cell["peak_drop_final"]
                       >= float(POLE_EQUATOR_MIN_PEAK_DROP)]
      vals = cell["pol_post"].dropna()
      if vals.empty:
        continue
      jitter = rng.uniform(-0.16, 0.16, len(vals))
      ax.scatter(x[idx] + jitter, vals.values, s=16, alpha=0.35,
                 color=PALETTE["grey"], edgecolors="none", zorder=2)
      hv = hi_drop["pol_post"].dropna()
      if not hv.empty and POLE_EQUATOR_MIN_PEAK_DROP is not None:
        jl = rng.uniform(-0.16, 0.16, len(hv))
        ax.scatter(x[idx] + jl, hv.values, s=18, alpha=0.75,
                   color=PALETTE["dark_blue"], edgecolors="none", zorder=3,
                   label=("lost cortex" if idx == 0 else ""))
        med2 = float(hv.median())
        ax.plot([x[idx] - 0.26, x[idx] + 0.26], [med2, med2],
                color=PALETTE["dark_blue"], lw=2.2, zorder=5)
      med = float(vals.median())
      ax.plot([x[idx] - 0.26, x[idx] + 0.26], [med, med],
              color=PALETTE["grey"], lw=2.0, zorder=4)
      ax.plot([x[idx], x[idx]],
              [float(vals.quantile(.25)), float(vals.quantile(.75))],
              color=PALETTE["grey"], lw=1.0, alpha=0.8, zorder=3)
      # 95% CI on the median, drawn heavier than the IQR: the IQR is how much
      # the vesicles differ from each other, the CI is how well the median is
      # pinned down, and only the second says what the data can resolve.
      c_lo, c_hi, _, _ = _boot_median_ci(vals.to_numpy(float))
      if np.isfinite(c_lo):
        ax.plot([x[idx], x[idx]], [c_lo, c_hi], color=PALETTE["dark_red"],
                lw=2.6, alpha=0.9, solid_capstyle="butt", zorder=6,
                label=("95% CI on median" if idx == 0 else ""))
      ax.text(x[idx], 0.015, f"n={len(vals)}", ha="center", va="bottom",
              fontsize=7, color=ANNOTATION_TEXT,
              transform=ax.get_xaxis_transform())
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_xticks(np.arange(len(x_volts)))
    ax.set_xticklabels(field_labels(x_volts), rotation=45)
    ax.set_xlabel(FIELD_AXIS_LABEL)
    ax.set_title(f"{w_label}: per-GUV mean, {lo:.0f} to {hi:.0f} s",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

  fig.suptitle("Directionality of cortex breakdown: poles vs equator\n"
               "(> 0 = electrode-facing poles lose more actin; pooled "
               "because a single vesicle is free to rotate)", fontsize=11)
  style_figure("pole_equator_pooled")
  fig.tight_layout(rect=(0, 0, 1, 0.90))
  pdf_path = out_path / "guv_pole_equator_pooled.pdf"
  fig.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"\nSaved pole/equator figure: {pdf_path}")
  return tbl


# Experiments whose actin trace cannot reach ENDPOINT_MATCHED_T_S. Module
# level so add_cortex_peak_metrics can report them in one place: dropping a
# whole experiment's cortex metrics is not something that should happen
# without a line in the log, and on a 5 s frame grid an experiment can miss
# the target by less than one frame.
_ACTIN_SHORT = {}


def load_actin_peak_metrics(exp_dir: Path) -> pd.DataFrame:
  files = [f for f in exp_dir.glob("**/*_actin_cortex_traces.csv")
           if RESULTS_SUBFOLDER not in f.parts]
  if not files:
    return None
  try:
    tr = pd.read_csv(files[0])
  except Exception as e:
    print(f"Skipping actin traces {files[0]}: {e}")
    return None

  # Same matched-time rule as the dye endpoint. Without it peak_drop_final is
  # still read at each record's own end, so the actin and dye halves of the
  # same figure would be measured on different clocks.
  t_col = tr["time_s"].to_numpy(float) if "time_s" in tr.columns else None
  if ENDPOINT_MATCHED_T_S is not None:
    if t_col is None:
      print(f"  {exp_dir.name}: actin traces have no time_s column, so no "
            "matched-time cortex endpoint is possible; skipped.")
      return None
    t_max = float(np.nanmax(t_col)) if np.isfinite(np.nanmax(t_col)) else np.nan
    # Same tolerance as the dye endpoint, and it has to be the same number:
    # an experiment admitted on the dye side and refused here would put a
    # matched-time release magnitude and a missing cortex metric on the same
    # vesicle, which is what the merge in add_cortex_peak_metrics then drops
    # without saying so.
    if (not np.isfinite(t_max)
        or t_max < ENDPOINT_MATCHED_T_S - ENDPOINT_MATCHED_TOLERANCE_S):
      _ACTIN_SHORT[exp_dir.name] = t_max
      return None

  rows = []
  for col in tr.columns:
    m = re.match(r"GUV_(.+)_cortex_peak$", col)
    if not m:
      continue
    if ENDPOINT_MATCHED_T_S is None:
      vals = tr[col].dropna()
    else:
      keep = pd.DataFrame({"t": t_col, "y": tr[col]}).dropna()
      vals = keep.loc[keep["t"] <= ENDPOINT_MATCHED_T_S, "y"]
    if len(vals) < 2 * PEAK_N_FRAMES:
      continue
    burst = float(vals.iloc[:PEAK_N_FRAMES].median())
    final = float(vals.iloc[-PEAK_N_FRAMES:].median())
    rows.append({
        "guv_id": m.group(1),
        "peak_norm_burst": burst,
        "peak_norm_final": final,
        "peak_drop_burst": 1.0 - burst,
        "peak_drop_final": 1.0 - final,
        "n_post_frames": int(len(vals)),
    })
  return pd.DataFrame(rows) if rows else None

def add_cortex_peak_metrics(df: pd.DataFrame, outputs_root: str) -> pd.DataFrame:
  root = Path(outputs_root)
  frames = []
  _ACTIN_SHORT.clear()
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    pk = load_actin_peak_metrics(exp_dir)
    if pk is None:
      continue
    pk["experiment"] = exp_dir.name
    frames.append(pk)
  if _ACTIN_SHORT:
    print(f"\nCortex peak metrics: {len(_ACTIN_SHORT)} experiment(s) have an "
          f"actin trace ending before ENDPOINT_MATCHED_T_S = "
          f"{ENDPOINT_MATCHED_T_S} and contribute no cortex metrics at all:")
    for name, t_max in sorted(_ACTIN_SHORT.items(), key=lambda kv: kv[1]):
      print(f"    ends {t_max:7.1f} s  {name}")
    print("  On a 5 s frame grid an experiment can miss the target by less "
          "than one frame, so check these against the target before reading "
          "the cortex figures as a complete set.")
  if not frames:
    print("No actin traces found; cortex peak metrics skipped.")
    return df.iloc[0:0]

  pk_all = pd.concat(frames, ignore_index=True)
  out = df.copy()
  out["guv_id"] = out["guv_id"].astype(str)
  pk_all["guv_id"] = pk_all["guv_id"].astype(str)
  return out.merge(pk_all, on=["experiment", "guv_id"], how="inner")

def plot_cortex_peak_breakdown(df: pd.DataFrame, output_dir: str):
  out_path = sub_dir(output_dir, "cortex_breakdown")
  out_path.mkdir(parents=True, exist_ok=True)
  if df.empty or "peak_drop_final" not in df:
    print("Cortex peak breakdown skipped: no peak metrics.")
    return

  sub = df.assign(group=df.apply(assign_cortex_group, axis=1))
  sub = sub[sub["group"] == "Branched, cortex"]
  sub = sub[sub["size_category"] == "Stagnate"]
  if sub.empty:
    print("Cortex peak breakdown skipped: no cortex-bearing stagnate GUVs.")
    return

  voltages = _voltage_order(sub)
  ctrl = sub[sub["voltage"] == CONTROL_VOLTAGE]["peak_drop_final"].dropna()
  voltages = [v for v in voltages if v != CONTROL_VOLTAGE]
  if not voltages:
    print("Cortex peak breakdown skipped: only the 30V control is present.")
    return

  x = np.arange(len(voltages))
  series = [("peak_drop_burst", "immediately after pulse", PALETTE["light_blue"]),
            ("peak_drop_final", "end of record", PALETTE["dark_blue"])]
  offset_w = 0.16
  rng = np.random.default_rng(0)

  fig, ax = plt.subplots(figsize=fig_size("cortex_peak_breakdown", max(8, 1.1 * len(voltages) + 3), 5.5))

  if len(ctrl) and len(ctrl) < MIN_CONTROL_N:
    print(f"  NOTE: the {CONTROL_VOLTAGE} band is drawn from {len(ctrl)} GUV(s), below "
          f"MIN_CONTROL_N={MIN_CONTROL_N}. Treat it as indicative only.")
  if len(ctrl) >= 2:
    lo, hi = float(ctrl.quantile(.25)), float(ctrl.quantile(.75))
    ax.axhspan(lo, hi, color=PALETTE["pale_red"], alpha=0.55, zorder=0)
    ax.axhline(float(ctrl.median()), color=PALETTE["dark_red"], lw=1.2,
               ls="--", zorder=1,
               label=f"{CONTROL_VOLTAGE} bleach reference (n={len(ctrl)})")
  elif len(ctrl) == 1:
    ax.axhline(float(ctrl.iloc[0]), color=PALETTE["dark_red"], lw=1.2, ls="--",
               zorder=1, label=f"{CONTROL_VOLTAGE} bleach reference (n=1)")

  rows = []
  for s_idx, (col, label, colour) in enumerate(series):
    offset = (s_idx - (len(series) - 1) / 2) * 2 * offset_w
    labelled = False
    for idx, v in enumerate(voltages):
      cell = sub[sub["voltage"] == v][col].dropna()
      if cell.empty:
        continue
      jitter = rng.uniform(-offset_w * 0.45, offset_w * 0.45, len(cell))
      ax.scatter(x[idx] + offset + jitter, cell.values, s=14, alpha=0.45,
                 color=colour, edgecolors="none", zorder=2)
      med = float(cell.median())
      q1, q3 = float(cell.quantile(.25)), float(cell.quantile(.75))
      ax.plot([x[idx] + offset - offset_w * 0.7,
               x[idx] + offset + offset_w * 0.7], [med, med],
              color=colour, lw=2.2, zorder=4,
              label="" if labelled else label)
      labelled = True
      ax.plot([x[idx] + offset, x[idx] + offset], [q1, q3],
              color=colour, lw=1.0, alpha=0.8, zorder=3)
      rows.append({"voltage": v, "when": label, "n": len(cell),
                   "median_peak_drop": med, "q1": q1, "q3": q3})

  ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.7)
  ax.set_xticks(x)
  ax.set_xticklabels(field_labels(voltages), rotation=45)
  ax.set_xlabel(FIELD_AXIS_LABEL)
  ax.set_ylabel("Cortex peak height lost  (1 - I$_{peak}$ / I$_{peak,pre}$)")
  ax.set_title("Radial cortex peak breakdown, cortex-bearing GUVs\n"
               f"(median and IQR; band = {CONTROL_VOLTAGE} bleach "
               "reference, no measurable response)")
  ax.legend(frameon=False, loc="upper left", fontsize=8)
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  style_figure("cortex_peak_breakdown")
  plt.tight_layout()

  pdf_path = out_path / "guv_cortex_peak_breakdown.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved cortex peak breakdown: {pdf_path}")

  tbl = pd.DataFrame(rows)
  csv_path = out_path / "guv_cortex_peak_summary.csv"
  sub.to_csv(out_path / "guv_cortex_peak_per_guv.csv", index=False)
  tbl.to_csv(csv_path, index=False)
  print(f"Saved cortex peak summary: {csv_path}")
  if len(ctrl):
    print(f"  {CONTROL_VOLTAGE} bleach reference: median peak lost "
          f"{ctrl.median():.3f} (n={len(ctrl)}). Conditions at or below "
          "this show no breakdown beyond bleaching.")
  print("\nCortex peak height lost per condition:")
  print(tbl.to_string(index=False))
  print()

def plot_lumen_abs_histogram(df: pd.DataFrame, output_dir: str):
  out_path = sub_dir(output_dir, "cortex_class")
  out_path.mkdir(parents=True, exist_ok=True)

  sub = df[df["population"] == "BranchedCortex"]
  vals = sub["prepulse_lumen_abs"].dropna() if "prepulse_lumen_abs" in sub \
      else pd.Series(dtype=float)
  if vals.empty:
    print("Lumen histogram skipped: no prepulse_lumen_abs values.")
    return

  fig, ax = plt.subplots(figsize=fig_size("lumen_abs_histogram", 7.5, 4.8))
  ax.hist(vals, bins=40, color=PALETTE["light_blue"],
          edgecolor=PALETTE["white"], linewidth=0.4)
  ax.set_xlabel("pre-pulse lumen above background (camera counts)")
  ax.set_ylabel("GUV count")
  ax.set_title("Lumenal actin level, BranchedCortex population")
  ax.grid(axis="y", linestyle="--", alpha=0.4)
  style_figure("lumen_abs_histogram")
  plt.tight_layout()

  pdf_path = out_path / "guv_lumen_abs_histogram.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved lumen level histogram: {pdf_path}")
  print(f"  lumen quartiles: {vals.quantile(.25):.1f} / {vals.median():.1f} / "
        f"{vals.quantile(.75):.1f}   range {vals.min():.1f}-{vals.max():.1f}")

def plot_cortex_contrast_histogram(df: pd.DataFrame, output_dir: str):
  out_path = sub_dir(output_dir, "cortex_class")
  out_path.mkdir(parents=True, exist_ok=True)

  sub = df[df["population"] == "BranchedCortex"]
  if sub.empty:
    print("Contrast histogram skipped: no BranchedCortex experiments.")
    return

  vals = sub["cortex_contrast"]
  n_total, n_nan = len(vals), int(vals.isna().sum())
  vals = vals.dropna()
  if vals.empty:
    print(f"Contrast histogram skipped: all {n_total} cortex_contrast values "
          "are NaN -- check the 'Mean of empty slice' warnings from the batch.")
    return

  lo = getattr(cfg, "CORTEX_CONTRAST_LOW", None)
  hi = getattr(cfg, "CORTEX_CONTRAST_HIGH", None)

  fig, ax = plt.subplots(figsize=fig_size("cortex_contrast_histogram", 7.5, 4.8))
  ax.hist(vals, bins=40, color=PALETTE["medium_blue"],
          edgecolor=PALETTE["white"], linewidth=0.4)
  for thr, name in ((lo, "CORTEX_CONTRAST_LOW"), (hi, "CORTEX_CONTRAST_HIGH")):
    if thr is not None:
      ax.axvline(float(thr), color=PALETTE["dark_red"], lw=1.2, ls="--")
      ax.text(float(thr), ax.get_ylim()[1] * 0.97, f" {name}={thr}",
              rotation=90, va="top", ha="left", fontsize=7,
              color=PALETTE["dark_red"])

  ax.set_xlabel(r"pre-pulse cortex contrast  $(I_{peak}-I_{lumen})/(I_{lumen}-I_{bg})$")
  ax.set_ylabel("GUV count")
  ax.set_title("Cortex contrast, BranchedCortex population\n"
               f"n={len(vals)} classified, {n_nan} NaN of {n_total}")
  ax.grid(axis="y", linestyle="--", alpha=0.4)
  style_figure("cortex_contrast_histogram")
  plt.tight_layout()

  pdf_path = out_path / "guv_cortex_contrast_histogram.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved cortex contrast histogram: {pdf_path}")
  if n_nan:
    print(f"  WARNING: {n_nan}/{n_total} BranchedCortex GUVs have no contrast "
          "value and are unclassifiable; they fall into 'ambiguous'.")
  print(f"  contrast quartiles: {vals.quantile(.25):.3f} / "
        f"{vals.median():.3f} / {vals.quantile(.75):.3f}"
        f"   range {vals.min():.3f}-{vals.max():.3f}")

def plot_released_fraction_by_voltage(df: pd.DataFrame, output_dir: str,
                                      by: str = "population"):
  if df.empty:
    print("No valid GUV records found.")
    return

  out_path = sub_dir(output_dir, "dye")
  out_path.mkdir(parents=True, exist_ok=True)

  sub_all = _filter_stagnate(df)
  if sub_all.empty:
    print("Released-fraction plot skipped: no GUV passed the stagnate/QC gate.")
    return

  sub_all = sub_all.assign(released=-sub_all["diff"])

  if by == "cortex_group":
    sub_all = sub_all.assign(
        group=sub_all.apply(assign_cortex_group, axis=1))
    pops = [g for g in GROUP_ORDER if g in set(sub_all["group"])]
    colours = GROUP_COLORS
    group_col = "group"
    suffix, what = "_by_cortex", "cortex presence"
  else:
    pops = _present_populations(sub_all)
    colours = POPULATION_COLORS
    group_col = "population"
    suffix, what = "", "population"

  voltages = _voltage_order(sub_all)
  x = np.arange(len(voltages))
  offset_w = 0.18 if len(pops) <= 2 else 0.11

  fig, ax = plt.subplots(figsize=fig_size("released_fraction_by_voltage", max(8, 1.1 * len(voltages) + 3), 5.5))
  rng = np.random.default_rng(0)

  rows = []
  for p_idx, pop in enumerate(pops):
    pop_df = sub_all[sub_all[group_col] == pop]
    offset = (p_idx - (len(pops) - 1) / 2) * 2 * offset_w
    colour = colours.get(pop, ANNOTATION_TEXT)
    labelled = False

    for idx, v in enumerate(voltages):
      cell = pop_df[pop_df["voltage"] == v]["released"].dropna()
      if cell.empty:
        continue
      jitter = rng.uniform(-offset_w * 0.45, offset_w * 0.45, len(cell))
      ax.scatter(x[idx] + offset + jitter, cell.values, s=14, alpha=0.45,
                 color=colour, edgecolors="none", zorder=2)

      med = float(cell.median())
      q1, q3 = float(cell.quantile(0.25)), float(cell.quantile(0.75))
      ax.plot([x[idx] + offset - offset_w * 0.7,
               x[idx] + offset + offset_w * 0.7], [med, med],
              color=colour, lw=2.2, zorder=4,
              label="" if labelled else POPULATION_LABELS.get(pop, pop))
      labelled = True
      ax.plot([x[idx] + offset, x[idx] + offset], [q1, q3],
              color=colour, lw=1.0, alpha=0.8, zorder=3)

      rows.append({"group": pop, "voltage": v, "n": len(cell),
                   "median_released": med, "q1": q1, "q3": q3})

  ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.7)
  ax.set_xticks(x)
  ax.set_xticklabels(field_labels(voltages), rotation=45)
  ax.set_xlabel(FIELD_AXIS_LABEL)
  ax.set_ylabel("Released fraction  (I$_{pre}$ - I$_{final}$)")
  ax.set_title(
      f"Dye released per vesicle, by {what}\n"
      f"(pre-pulse vs final {ENDPOINT_N_FRAMES}-frame average; median and IQR; "
      "all stagnate GUVs, responding or not)"
  )
  ax.legend(frameon=False, loc="upper left")
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  style_figure("released_fraction_by_voltage")
  plt.tight_layout()

  pdf_path = out_path / f"guv_released_fraction_by_voltage{suffix}.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved released-fraction plot: {pdf_path}")

  if rows:
    tbl = pd.DataFrame(rows).sort_values(["group", "voltage"])
    csv_path = out_path / f"guv_released_fraction_summary{suffix}.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"Saved released-fraction summary: {csv_path}")
    print(f"\nReleased fraction per condition (by {what}):")
    print(tbl.to_string(index=False))
    print()

def generate_separate_pdf_plots(df: pd.DataFrame, output_dir: str):
  if df.empty:
    print("No valid GUV records found. Please check output directory path.")
    return

  report_endpoint_times(df)

  voltages = _voltage_order(df)
  pops = _present_populations(df)
  df_stagnate = _filter_stagnate(df)

  size_categories = ["Grow", "Stagnate", "Reduce", "Rupture/Collapse"]
  size_colors = {c: SIZE_CATEGORY_COLORS[c] for c in size_categories}
  intensity_categories = ["Lose Intensity", "Flatline"]

  fig1, ax1 = plt.subplots(figsize=fig_size("size_category_distribution", max(8, 1.1 * len(voltages) + 3), 5.5))
  x = np.arange(len(voltages))
  bar_w = 0.7 / max(len(pops), 1)

  for p_idx, pop in enumerate(pops):
    sub = df[df["population"] == pop]
    counts = (
        sub.groupby(["voltage", "size_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=voltages, columns=size_categories, fill_value=0)
    )
    offset = (p_idx - (len(pops) - 1) / 2) * bar_w
    bottom = np.zeros(len(voltages))
    for cat in size_categories:
      vals = counts[cat].values.astype(float)
      ax1.bar(
          x + offset,
          vals,
          width=bar_w * 0.9,
          bottom=bottom,
          color=size_colors[cat],
          edgecolor="black",
          linewidth=0.5,
          label=cat if p_idx == 0 else "",
      )
      bottom += vals
    for xi, tot in zip(x + offset, bottom):
      if tot > 0:
        ax1.text(
            xi,
            tot + 0.5,
            "Ba" if pop in ("Empty", "Bare") else "Br",
            ha="center",
            va="bottom",
            fontsize=7,
            color=POPULATION_COLORS.get(pop, ANNOTATION_TEXT),
            fontweight="bold",
        )

  ax1.set_xticks(x)
  ax1.set_xticklabels(field_labels(voltages))
  ax1.set_title(
      "GUV Size Category Distribution\n(Ba = Bare, Br = Branched)"
  )
  ax1.set_xlabel(FIELD_AXIS_LABEL)
  ax1.set_ylabel("GUV Count")
  ax1.legend(title="Size Category", bbox_to_anchor=(1.02, 1), loc="upper left")
  ax1.grid(axis="y", linestyle="--", alpha=0.5)
  style_figure("size_category_distribution")
  plt.tight_layout()

  pdf_path1 = sub_dir(output_dir, "fate") / "guv_size_category_distribution.pdf"
  plt.savefig(pdf_path1, format="pdf", dpi=300)
  plt.close(fig1)
  print(f"Saved Graph 1: {pdf_path1}")

  fig2, ax2 = plt.subplots(figsize=fig_size("stagnate_intensity_counts", max(8, 1.1 * len(voltages) + 3), 5.5))
  n_series = len(pops) * len(intensity_categories)
  bar_w = 0.8 / max(n_series, 1)
  hatches = {"Lose Intensity": "", "Flatline": "///"}

  s_idx = 0
  for pop in pops:
    sub = df_stagnate[df_stagnate["population"] == pop]
    if sub.empty:
      counts = pd.DataFrame(0, index=voltages, columns=intensity_categories)
    else:
      counts = (
          sub.groupby(["voltage", "intensity_category"])
          .size()
          .unstack(fill_value=0)
          .reindex(index=voltages, columns=intensity_categories, fill_value=0)
      )
    for cat in intensity_categories:
      offset = (s_idx - (n_series - 1) / 2) * bar_w
      ax2.bar(
          x + offset,
          counts[cat].values.astype(float),
          width=bar_w * 0.9,
          color=POPULATION_COLORS.get(pop, ANNOTATION_TEXT),
          edgecolor="black",
          linewidth=0.5,
          hatch=hatches[cat],
          label=f"{POPULATION_LABELS.get(pop, pop)} — {cat}",
      )
      s_idx += 1

  ax2.set_xticks(x)
  ax2.set_xticklabels(field_labels(voltages))
  ax2.set_title(
      "Stagnate GUV Intensity Breakdown\n(Pre-Pulse vs. Final"
      f" {ENDPOINT_N_FRAMES} Frames)"
  )
  ax2.set_xlabel(FIELD_AXIS_LABEL)
  ax2.set_ylabel("GUV Count")
  ax2.legend(
      title="Population / Behaviour", bbox_to_anchor=(1.02, 1), loc="upper left"
  )
  ax2.grid(axis="y", linestyle="--", alpha=0.5)
  style_figure("stagnate_intensity_counts")
  plt.tight_layout()

  pdf_path2 = sub_dir(output_dir, "dye") / "guv_stagnate_intensity_counts.pdf"
  plt.savefig(pdf_path2, format="pdf", dpi=300)
  plt.close(fig2)
  print(f"Saved Graph 2: {pdf_path2}")

  fig3, axes3 = plt.subplots(
      1, len(pops), figsize=fig_size("intensity_drop_trajectories", 7 * len(pops), 5.5, len(pops)), sharey=True, squeeze=False
  )
  width = 0.22

  for p_idx, pop in enumerate(pops):
    ax3 = axes3[0][p_idx]
    pop_df = df_stagnate[df_stagnate["population"] == pop]
    mean_labelled = False

    for idx, v in enumerate(voltages):
      sub = pop_df[pop_df["voltage"] == v]
      x_pre = idx - width
      x_post = idx + width

      for _, r in sub.iterrows():
        if pd.notna(r["i_pre"]) and pd.notna(r["i_final"]):
          ax3.plot(
              [x_pre, x_post],
              [r["i_pre"], r["i_final"]],
              "-o",
              color=INDIVIDUAL_TRACE,
              alpha=0.35,
              ms=4,
          )

      if not sub.empty:
        ax3.plot(
            [x_pre, x_post],
            [sub["i_pre"].mean(), sub["i_final"].mean()],
            "-s",
            color=SUMMARY_LINE,
            lw=1.8,
            ms=5,
            alpha=0.85,
            zorder=5,
            label="Mean Drop" if not mean_labelled else "",
        )
        mean_labelled = True
        ax3.text(
            idx,
            0.02,
            f"n={len(sub)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=ANNOTATION_TEXT,
        )

    ax3.set_xticks(np.arange(len(voltages)))
    ax3.set_xticklabels(field_labels(voltages), fontsize=9, rotation=45)
    ax3.set_xlabel(FIELD_AXIS_LABEL)
    ax3.set_title(POPULATION_LABELS.get(pop, pop))
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    if p_idx == 0:
      ax3.set_ylabel("Normalized Intensity")
      if mean_labelled:
        ax3.legend(loc="lower left", fontsize=9)

  fig3.suptitle(
      "Intensity Drop Trajectories per Field Strength"
      f"\n(Pre-Pulse → Final {ENDPOINT_N_FRAMES}-Frame Average)"
  )
  style_figure("intensity_drop_trajectories")
  plt.tight_layout()

  pdf_path3 = sub_dir(output_dir, "dye") / "guv_intensity_drop_trajectories.pdf"
  plt.savefig(pdf_path3, format="pdf", dpi=300)
  plt.close(fig3)
  print(f"Saved Graph 3: {pdf_path3}")

  bin_order = BIN_ORDER
  df_binned = df_stagnate[df_stagnate["radius_bin"].isin(bin_order)].copy()

  for v in voltages:
    fig_v, axes_v = plt.subplots(
        1, len(pops), figsize=fig_size("drop_by_size_per_voltage", 5.5 * len(pops), 5, len(pops)), sharey=True, squeeze=False
    )

    for p_idx, pop in enumerate(pops):
      ax_v = axes_v[0][p_idx]
      sub = df_binned[
          (df_binned["voltage"] == v) & (df_binned["population"] == pop)
      ]

      for b_idx, b_label in enumerate(bin_order):
        bin_sub = sub[sub["radius_bin"] == b_label]
        x_pre = b_idx - width
        x_post = b_idx + width

        for _, r in bin_sub.iterrows():
          ax_v.plot(
              [x_pre, x_post],
              [r["i_pre"], r["i_final"]],
              "-o",
              color=INDIVIDUAL_TRACE,
              alpha=0.4,
              ms=4,
          )

        if not bin_sub.empty:
          ax_v.plot(
              [x_pre, x_post],
              [bin_sub["i_pre"].mean(), bin_sub["i_final"].mean()],
              "-s",
              color=SUMMARY_LINE,
              lw=1.8,
              ms=5,
              alpha=0.85,
              zorder=5,
          )
          ax_v.text(
              b_idx,
              0.02,
              f"n={len(bin_sub)}",
              ha="center",
              va="bottom",
              fontsize=7,
              color=ANNOTATION_TEXT,
          )

      ax_v.set_xticks(np.arange(len(bin_order)))
      ax_v.set_xticklabels(bin_order, fontsize=9)
      ax_v.set_xlabel("Size Group")
      ax_v.set_title(POPULATION_LABELS.get(pop, pop))
      ax_v.set_ylim(bottom=0, top=1.15)
      ax_v.grid(axis="y", linestyle="--", alpha=0.5)
      if p_idx == 0:
        ax_v.set_ylabel("Normalized Intensity")

    fig_v.suptitle(f"{field_label(v)} kV/cm — Size Group vs. Intensity Drop")
    style_figure("drop_by_size_per_voltage")
    plt.tight_layout()

    pdf_v_path = (sub_dir(output_dir, "dye_by_voltage")
                  / f"guv_binned_size_vs_intensity_{v}.pdf")
    plt.savefig(pdf_v_path, format="pdf", dpi=300)
    plt.close(fig_v)
    print(f"Saved Voltage Graph ({v}): {pdf_v_path}")

if __name__ == "__main__":
  outputs_root = getattr(cfg, "PARENT_OUTPUT_FOLDER", r"D:\EP\Outputs")
  results_dir = Path(outputs_root) / RESULTS_SUBFOLDER
  results_dir.mkdir(parents=True, exist_ok=True)

  df_summary = aggregate_pipeline_results(outputs_root)
  df_summary.to_csv(results_dir / "guv_bulk_summary.csv", index=False)
  write_layout_readme(results_dir)
  plot_lumen_abs_histogram(df_summary, str(results_dir))
  df_peak = add_cortex_peak_metrics(df_summary, outputs_root)
  plot_cortex_peak_breakdown(df_peak, str(results_dir))
  # Onsets are fitted on df_peak, not df_summary: the cortex-loss curve needs
  # peak_drop_final, which only exists after the merge above. Bare GUVs carry
  # no actin columns, so their onset is fitted on area loss alone.
  report_onset_fields(
      df_summary.merge(
          df_peak[["experiment", "guv_id", "peak_drop_final"]],
          on=["experiment", "guv_id"], how="left")
      if not df_peak.empty else df_summary,
      str(results_dir))
  report_bleach_comparability(outputs_root)
  plot_cortex_peak_timecourse(df_summary, outputs_root, str(results_dir))
  plot_pole_equator_pooled(df_summary, outputs_root, str(results_dir))
  plot_cortex_contrast_histogram(df_summary, str(results_dir))
  report_cortex_status(df_summary)

  df_split = apply_cortex_split(df_summary)
  generate_separate_pdf_plots(df_split, str(results_dir))
  plot_released_fraction_by_voltage(df_split, str(results_dir))
  plot_drop_by_cortex_status(df_summary, str(results_dir))
  report_tau_yield(df_split)
  plot_tau_by_size_and_voltage(df_split, str(results_dir))

  pooled_dir = sub_dir(results_dir, "pooled")
  generate_separate_pdf_plots(df_summary, str(pooled_dir))
  plot_released_fraction_by_voltage(df_summary, str(pooled_dir))
  report_tau_yield(df_summary)
  plot_tau_by_size_and_voltage(df_summary, str(pooled_dir))