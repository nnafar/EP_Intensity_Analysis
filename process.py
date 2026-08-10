import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load pipeline configuration
import config as cfg

# -----------------------------------------------------------------------------
# --- POPULATION + ENDPOINT SETTINGS ---
# -----------------------------------------------------------------------------

# Population labels as they appear in the output folder / experiment names.
# Order controls left-to-right placement in every side-by-side plot.
POPULATIONS = ["Empty", "BranchedCortex"]
POPULATION_LABELS = {"Empty": "Empty", "BranchedCortex": "Branched cortex"}
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

# Number of terminal post-pulse frames averaged into the endpoint. The
# endpoint is the last N post-pulse frames of each acquisition: the metric is
# the magnitude of the decline, independent of how long the acquisition ran.
# Everything this script writes goes here, as a subfolder of
# PARENT_OUTPUT_FOLDER, so bulk results never mix with the
# per-experiment folders they were aggregated from.
# tau_identifiability_report.py repeats this literal -- keep them equal.
RESULTS_SUBFOLDER = "Bulk_Analysis_Results"

ENDPOINT_N_FRAMES = 3

# Pre-pulse QC: keep a GUV only if its mean pre-pulse retention sits within
# this tolerance of 1.0. Replaces the old `i_pre <= 1.0` cut, which discarded
# every GUV whose pre-pulse mean landed above 1.0 by noise (~50-90% of them).
PRE_PULSE_TOLERANCE = 0.05

# Classification threshold on (i_final - i_pre).
INTENSITY_DIFF_THRESHOLD = 0.05

# --- cortex presence -------------------------------------------------------
# Read from *_actin_cortex_status.csv (see actin_cortex_status_patch.py).
# When the file is absent every GUV is labelled UNKNOWN and the cortex-split
# figure is skipped, so the rest of the script still runs on Empty data.
CORTEX_STATUS_ORDER = ["CORTEX", "AMBIGUOUS", "NO_CORTEX", "UNKNOWN"]
# CORTEX_STATUS_COLORS now comes from palette.py (imported above).

# --- tau plot gating -------------------------------------------------------
# Only GUVs the pipeline judged to have a genuine, well-constrained decay are
# plotted. Both gates come straight from *_fit_parameters.csv.
TAU_REQUIRE_IDENTIFIABLE = True   # tau_identifiable == True
TAU_REQUIRE_RESPONDING = True     # is_responding == True

# Colours for the size groups in the tau plot.
# SIZE_BIN_COLORS now comes from palette.py (imported above).


def extract_voltage(path_str: str) -> str:
  """Extract voltage label (e.g., '400V') from directory name."""
  match = re.search(r"(\d+)\s*V", path_str, re.IGNORECASE)
  return f"{match.group(1)}V" if match else "Unknown"


def extract_population(path_str: str) -> str:
  """Extract population label (e.g., 'BranchedCortex') from directory name."""
  for pop in POPULATIONS:
    if re.search(pop, path_str, re.IGNORECASE):
      return pop
  return "Unknown"


def assign_size_bin(r_um: float) -> str:
  """Categorize GUV radius into specified size groups."""
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
  """Boolean mask selecting post-pulse rows."""
  if "phase" in df_norm.columns:
    return df_norm["phase"] == "post_pulse"
  elif "Time (s)" in df_norm.columns:
    return df_norm["Time (s)"] > 0
  return df_norm.index >= 5


def _pre_pulse_mask(df_norm: pd.DataFrame) -> pd.Series:
  """Boolean mask selecting pre-pulse rows."""
  if "phase" in df_norm.columns:
    return df_norm["phase"] == "pre_pulse"
  elif "Time (s)" in df_norm.columns:
    return df_norm["Time (s)"] < 0
  return df_norm.index < 5


def load_cortex_status(exp_dir: Path) -> pd.DataFrame:
  """Per-GUV cortex classification for one experiment, or None if absent."""
  files = list(exp_dir.glob("**/*_actin_cortex_status.csv"))
  if not files:
    return None
  try:
    return pd.read_csv(files[0])
  except Exception as e:
    print(f"Skipping cortex status file {files[0]}: {e}")
    return None


def aggregate_pipeline_results(outputs_root: str) -> pd.DataFrame:
  """Parse tracking fates, unnormalized radius, and pre-pulse vs. endpoint
  intensity traces across output folders, for every population."""
  root = Path(outputs_root)
  records = []

  fate_files = [f for f in root.glob("**/*_guv_fate_classification.csv")
                if RESULTS_SUBFOLDER not in f.parts]

  for fate_file in fate_files:
    exp_dir = fate_file.parents[1]
    voltage = extract_voltage(exp_dir.name)
    population = extract_population(exp_dir.name)

    # 1. Load Fate Classification CSV
    try:
      df_fate = pd.read_csv(fate_file)
    except Exception as e:
      print(f"Skipping {fate_file}: {e}")
      continue

    # 2. Load Fit Parameters CSV (for raw unnormalized radius_um)
    fit_files = list(exp_dir.glob("**/*_fit_parameters.csv"))
    df_fit = None
    if fit_files:
      try:
        df_fit = pd.read_csv(fit_files[0])
      except Exception as e:
        print(f"Skipping fit file {fit_files[0]}: {e}")

    # 3. Load Normalized Curves CSV (for frame-level intensity traces)
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

    for _, row in df_fate.iterrows():
      gid = str(row["guv_id"])
      raw_fate = str(row.get("fate", "")).upper()

      # Map size dynamics
      if raw_fate == "GROWN":
        size_cat = "Grow"
      elif raw_fate == "SHRUNK":
        size_cat = "Reduce"
      elif raw_fate == "RUPTURED":
        size_cat = "Rupture/Collapse"
      else:
        size_cat = "Stagnate"

      # Get raw unnormalized radius_um (and kinetic fit fields) and bin it
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
      if df_cortex is not None:
        guv_cx = df_cortex[df_cortex["guv_id"].astype(str) == gid]
        if not guv_cx.empty:
          cortex_status = str(guv_cx.iloc[0].get("cortex_status", "UNKNOWN"))
          cortex_contrast = guv_cx.iloc[0].get("cortex_contrast", np.nan)

      radius_bin = assign_size_bin(radius_um)

      intensity_cat = None
      i_pre = np.nan
      i_final = np.nan
      diff = np.nan
      n_final_used = 0

      if size_cat == "Stagnate" and df_norm is not None:
        target_col = f"GUV_{gid}_I_retained"
        if target_col in df_norm.columns:
          pre_vals = df_norm.loc[pre_mask, target_col].dropna().values
          post_vals = df_norm.loc[post_mask, target_col].dropna().values

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

      records.append({
          "population": population,
          "voltage": voltage,
          "experiment": exp_dir.name,
          "guv_id": gid,
          "size_category": size_cat,
          "radius_um": radius_um,
          "radius_bin": radius_bin,
          "cortex_status": cortex_status,
          "cortex_contrast": cortex_contrast,
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
          "n_final_used": n_final_used,
      })

  return pd.DataFrame(records)


def report_endpoint_times(df: pd.DataFrame):
  """Print the endpoint time of each experiment, for the record."""
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
  """Stagnate GUVs passing pre-pulse QC, excluding intensity increases."""
  return df[
      (df["size_category"] == "Stagnate")
      & (df["intensity_category"].isin(["Lose Intensity", "Flatline"]))
      & ((df["i_pre"] - 1.0).abs() <= PRE_PULSE_TOLERANCE)
  ].copy()


def _voltage_order(df: pd.DataFrame) -> list:
  return sorted(
      df["voltage"].unique(),
      key=lambda v: int(re.search(r"\d+", v).group())
      if re.search(r"\d+", v)
      else 0,
  )


def _present_populations(df: pd.DataFrame) -> list:
  present = [p for p in POPULATIONS if p in set(df["population"])]
  extra = sorted(set(df["population"]) - set(POPULATIONS))
  return present + extra



BIN_ORDER = ["< 5 \u00b5m", "5-8 \u00b5m", "9-12 \u00b5m", "> 13 \u00b5m"]


def _filter_tau(df: pd.DataFrame) -> pd.DataFrame:
  """Stagnate GUVs with a usable decay constant, per the pipeline's own gates.

  Ruptured / grown / shrunk vesicles are excluded: their intensity trace is
  dominated by the geometry change, so a fitted efflux tau is meaningless.
  """
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
  """Print how many GUVs survive each tau gate, per population and voltage.

  A panel that comes out empty is not a plotting bug: it means no GUV in that
  condition had a decay constant the fit could constrain.
  """
  if df.empty:
    return
  rows = []
  stag = df[df["size_category"] == "Stagnate"]
  for (pop, v), sub in stag.groupby(["population", "voltage"]):
    fitted = sub["tau"].notna().sum()
    rows.append({
        "population": pop,
        "voltage": v,
        "n_guv": len(sub),
        "n_fitted": int(fitted),
        "n_responding": int(sub["is_responding"].sum()),
        "n_tau_identifiable": int(sub["tau_identifiable"].sum()),
        "n_plotted": len(_filter_tau(sub)),
    })
  tbl = pd.DataFrame(rows).sort_values(["population", "voltage"])
  print("\nTau yield per condition:")
  print(tbl.to_string(index=False))
  print()


def plot_drop_by_cortex_status(df: pd.DataFrame, output_dir: str):
  """Pre-pulse -> final drop, split by whether the GUV had a cortex.

  Skipped when no experiment exported a cortex status file.
  """
  out_path = Path(output_dir)
  df_st = _filter_stagnate(df)
  df_st = df_st[df_st["cortex_status"] != "UNKNOWN"]
  if df_st.empty:
    print("Cortex split skipped: no *_actin_cortex_status.csv found.")
    return

  pops = _present_populations(df_st)
  voltages = _voltage_order(df_st)
  statuses = [s for s in CORTEX_STATUS_ORDER
              if s != "UNKNOWN" and s in set(df_st["cortex_status"])]

  fig, axes = plt.subplots(
      1, len(pops), figsize=(6.5 * len(pops), 5.5), sharey=True, squeeze=False
  )
  x = np.arange(len(voltages))
  slot_w = 0.8 / max(len(statuses), 1)
  rng = np.random.default_rng(0)

  for p_idx, pop in enumerate(pops):
    ax = axes[0][p_idx]
    pop_df = df_st[df_st["population"] == pop]

    for s_idx, status in enumerate(statuses):
      colour = CORTEX_STATUS_COLORS[status]
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
    ax.set_xticklabels(voltages, rotation=45, fontsize=9)
    ax.set_xlabel("Voltage")
    ax.set_title(POPULATION_LABELS.get(pop, pop))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if p_idx == 0:
      ax.set_ylabel(r"$I_{final} - I_{pre}$")
      ax.legend(title="Pre-pulse cortex", fontsize=8, loc="best")

  fig.suptitle(
      "Dye retention change split by pre-pulse cortex presence"
      "\n(Stagnate GUVs; bars = median)"
  )
  plt.tight_layout()
  pdf_path = out_path / "guv_drop_by_cortex_status.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved Cortex Split Graph: {pdf_path}")


def report_cortex_status(df: pd.DataFrame):
  """Counts per population and voltage, so an empty group is never a mystery."""
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
  """Tau vs. voltage, split by size group, populations side by side."""
  out_path = Path(output_dir)
  pops = _present_populations(df)
  voltages = _voltage_order(df)
  df_tau = _filter_tau(df)

  # With no GUV past the identifiability gate there is nothing to draw, and
  # the log y-axis raises rather than producing an empty panel. Report the
  # yield and return: an empty tau plot is a result, not a crash.
  if df_tau.empty:
    print("Tau plot skipped: no GUV passed the identifiability gate "
          "(see the tau yield table above, and tau_identifiability.pdf "
          "for which constraint failed).")
    return

  fig, axes = plt.subplots(
      1, len(pops), figsize=(6.5 * len(pops), 5.5), sharey=True, squeeze=False
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
    ax.set_xticklabels(voltages, rotation=45, fontsize=9)
    ax.set_xlabel("Voltage")
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
  plt.tight_layout()

  pdf_path = out_path / "guv_tau_by_size_voltage.pdf"
  plt.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved Tau Graph: {pdf_path}")


def assign_cortex_group(row) -> str:
  """Population, with BranchedCortex split on pre-pulse cortex presence.

  Deliberately computed at plot time rather than added as a column to
  guv_bulk_summary.csv: the split depends on CORTEX_CONTRAST_HIGH/LOW, which
  are still provisional, so it must not be frozen into an exported table that
  later gets read as though it were measured.

  "Lumenal only" means actin was present but never polymerised onto the
  membrane -- a different object from a corticated vesicle, and from an Empty
  one, which is why it is its own group rather than being pooled either way.
  """
  if row["population"] != "BranchedCortex":
    return "Empty" if row["population"] == "Empty" else row["population"]
  status = row.get("cortex_status", "UNKNOWN")
  if status == "CORTEX":
    return "Branched, cortex"
  if status == "NO_CORTEX":
    return "Branched, lumenal only"
  return "Branched, ambiguous"      # AMBIGUOUS and UNKNOWN both land here


def plot_cortex_contrast_histogram(df: pd.DataFrame, output_dir: str):
  """Distribution of pre-pulse cortex contrast, with the current thresholds.

  This figure has to be read before any cortex-split result is trusted. The
  thresholds were set to reproduce a visual sort of one montage, not from the
  data. If the distribution here is bimodal, the cut is defensible and the
  lines should sit in the trough. If it is unimodal, there is no natural
  boundary and the split is imposed rather than found -- which does not make
  it useless, but it must be reported as a chosen cut, not a discovered one.
  """
  out_path = Path(output_dir)
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

  fig, ax = plt.subplots(figsize=(7.5, 4.8))
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
  """Released dye fraction against voltage, populations side by side.

  This is the magnitude counterpart to plot_tau_by_size_and_voltage. It is its
  own figure because the two quantities have different yields: the timescale is
  reportable only for GUVs whose decay resolved inside the record, while the
  magnitude is defined for every GUV with a valid pre-pulse baseline and
  endpoint. Sharing one axis would silently impose the stricter gate on both.

  Gated on _filter_stagnate only -- NOT on tau_identifiable or is_responding.
  A non-responding vesicle contributes a released fraction near zero, which is
  a measurement, not a missing value: dropping it would bias the population
  median upward and erase the voltage threshold the experiment exists to find.

  by : "population"    Empty vs BranchedCortex.
       "cortex_group"  BranchedCortex further split into vesicles with a rim
                       and vesicles carrying lumenal actin only. Read
                       guv_cortex_contrast_histogram.pdf before using this
                       version: the split depends on provisional thresholds.
  """
  if df.empty:
    print("No valid GUV records found.")
    return

  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  sub_all = _filter_stagnate(df)
  if sub_all.empty:
    print("Released-fraction plot skipped: no GUV passed the stagnate/QC gate.")
    return

  # diff = i_final - i_pre, so a loss is negative. Flip the sign once here so
  # the axis reads as "released", increasing upward.
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

  fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(voltages) + 3), 5.5))
  rng = np.random.default_rng(0)   # fixed seed: jitter must not move between runs

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
  ax.set_xticklabels(voltages, rotation=45)
  ax.set_xlabel("Voltage")
  ax.set_ylabel("Released fraction  (I$_{pre}$ - I$_{final}$)")
  ax.set_title(
      f"Dye released per vesicle, by {what}\n"
      f"(pre-pulse vs final {ENDPOINT_N_FRAMES}-frame average; median and IQR; "
      "all stagnate GUVs, responding or not)"
  )
  ax.legend(frameon=False, loc="upper left")
  ax.grid(axis="y", linestyle="--", alpha=0.5)
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
  """Generate side-by-side population comparison graphs as separate PDFs."""
  if df.empty:
    print("No valid GUV records found. Please check output directory path.")
    return

  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  report_endpoint_times(df)

  voltages = _voltage_order(df)
  pops = _present_populations(df)
  df_stagnate = _filter_stagnate(df)

  size_categories = ["Grow", "Stagnate", "Reduce", "Rupture/Collapse"]
  size_colors = {c: SIZE_CATEGORY_COLORS[c] for c in size_categories}
  intensity_categories = ["Lose Intensity", "Flatline"]

  # -------------------------------------------------------------------------
  # GRAPH 1: Size Category Distribution — populations side by side
  # -------------------------------------------------------------------------
  fig1, ax1 = plt.subplots(figsize=(max(8, 1.1 * len(voltages) + 3), 5.5))
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
    # Population tag under each bar group
    for xi, tot in zip(x + offset, bottom):
      if tot > 0:
        ax1.text(
            xi,
            tot + 0.5,
            "E" if pop == "Empty" else "B",
            ha="center",
            va="bottom",
            fontsize=7,
            color=POPULATION_COLORS.get(pop, ANNOTATION_TEXT),
            fontweight="bold",
        )

  ax1.set_xticks(x)
  ax1.set_xticklabels(voltages)
  ax1.set_title(
      "GUV Size Category Distribution\n(E = Empty, B = Branched cortex)"
  )
  ax1.set_xlabel("Voltage")
  ax1.set_ylabel("GUV Count")
  ax1.legend(title="Size Category", bbox_to_anchor=(1.02, 1), loc="upper left")
  ax1.grid(axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()

  pdf_path1 = out_path / "guv_size_category_distribution.pdf"
  plt.savefig(pdf_path1, format="pdf", dpi=300)
  plt.close(fig1)
  print(f"Saved Graph 1: {pdf_path1}")

  # -------------------------------------------------------------------------
  # GRAPH 2: Stagnate Intensity Breakdown — populations side by side
  # -------------------------------------------------------------------------
  fig2, ax2 = plt.subplots(figsize=(max(8, 1.1 * len(voltages) + 3), 5.5))
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
  ax2.set_xticklabels(voltages)
  ax2.set_title(
      "Stagnate GUV Intensity Breakdown\n(Pre-Pulse vs. Final"
      f" {ENDPOINT_N_FRAMES} Frames)"
  )
  ax2.set_xlabel("Voltage")
  ax2.set_ylabel("GUV Count")
  ax2.legend(
      title="Population / Behaviour", bbox_to_anchor=(1.02, 1), loc="upper left"
  )
  ax2.grid(axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()

  pdf_path2 = out_path / "guv_stagnate_intensity_counts.pdf"
  plt.savefig(pdf_path2, format="pdf", dpi=300)
  plt.close(fig2)
  print(f"Saved Graph 2: {pdf_path2}")

  # -------------------------------------------------------------------------
  # GRAPH 3: Voltage Trajectory Plot — one panel per population, shared y
  # -------------------------------------------------------------------------
  fig3, axes3 = plt.subplots(
      1, len(pops), figsize=(7 * len(pops), 5.5), sharey=True, squeeze=False
  )
  width = 0.22

  for p_idx, pop in enumerate(pops):
    ax3 = axes3[0][p_idx]
    pop_df = df_stagnate[df_stagnate["population"] == pop]

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
            label="Mean Drop" if idx == 0 else "",
        )
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
    ax3.set_xticklabels(voltages, fontsize=9, rotation=45)
    ax3.set_xlabel("Voltage")
    ax3.set_title(POPULATION_LABELS.get(pop, pop))
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    if p_idx == 0:
      ax3.set_ylabel("Normalized Intensity")
      ax3.legend(loc="lower left", fontsize=9)

  fig3.suptitle(
      "Intensity Drop Trajectories per Voltage"
      f"\n(Pre-Pulse → Final {ENDPOINT_N_FRAMES}-Frame Average)"
  )
  plt.tight_layout()

  pdf_path3 = out_path / "guv_intensity_drop_trajectories.pdf"
  plt.savefig(pdf_path3, format="pdf", dpi=300)
  plt.close(fig3)
  print(f"Saved Graph 3: {pdf_path3}")

  # -------------------------------------------------------------------------
  # GRAPH 4+: Binned Size vs. Intensity Drop — one PDF per voltage,
  #           populations side by side
  # -------------------------------------------------------------------------
  bin_order = BIN_ORDER
  df_binned = df_stagnate[df_stagnate["radius_bin"].isin(bin_order)].copy()

  for v in voltages:
    fig_v, axes_v = plt.subplots(
        1, len(pops), figsize=(5.5 * len(pops), 5), sharey=True, squeeze=False
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

    fig_v.suptitle(f"{v} — Size Group vs. Intensity Drop")
    plt.tight_layout()

    pdf_v_path = out_path / f"guv_binned_size_vs_intensity_{v}.pdf"
    plt.savefig(pdf_v_path, format="pdf", dpi=300)
    plt.close(fig_v)
    print(f"Saved Voltage Graph ({v}): {pdf_v_path}")


if __name__ == "__main__":
  outputs_root = getattr(cfg, "PARENT_OUTPUT_FOLDER", r"D:\EP\Outputs")

  # Read from the per-experiment tree, write into its results subfolder.
  results_dir = Path(outputs_root) / RESULTS_SUBFOLDER
  results_dir.mkdir(parents=True, exist_ok=True)

  df_summary = aggregate_pipeline_results(outputs_root)
  df_summary.to_csv(results_dir / "guv_bulk_summary.csv", index=False)
  generate_separate_pdf_plots(df_summary, str(results_dir))
  plot_released_fraction_by_voltage(df_summary, str(results_dir))
  plot_cortex_contrast_histogram(df_summary, str(results_dir))
  plot_released_fraction_by_voltage(df_summary, str(results_dir),
                                    by="cortex_group")
  report_cortex_status(df_summary)
  plot_drop_by_cortex_status(df_summary, str(results_dir))
  report_tau_yield(df_summary)
  plot_tau_by_size_and_voltage(df_summary, str(results_dir))