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
                          height lost, and its full time course. Both carry
                          the 30 V control as a reference.

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
CORTEX_STATUS_ORDER = ["CORTEX", "AMBIGUOUS", "NO_CORTEX", "UNKNOWN"]
TAU_REQUIRE_IDENTIFIABLE = True
TAU_REQUIRE_RESPONDING = True

def extract_voltage(path_str: str) -> str:
  match = re.search(r"(\d+)\s*V", path_str, re.IGNORECASE)
  return f"{match.group(1)}V" if match else "Unknown"

def extract_population(path_str: str) -> str:
  for pop in POPULATIONS:
    if re.search(pop, path_str, re.IGNORECASE):
      return pop
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

def aggregate_pipeline_results(outputs_root: str) -> pd.DataFrame:
  root = Path(outputs_root)
  records = []

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

    for _, row in df_fate.iterrows():
      gid = str(row["guv_id"])
      raw_fate = str(row.get("fate", "")).upper()

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
          "n_final_used": n_final_used,
      })

  return pd.DataFrame(records)


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

ONSET_N_BOOTSTRAP = 2000      # resamples for the V50 confidence interval
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
    """Logistic in voltage. lo/hi are the plateaus, v50 the midpoint."""
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
              n_boot: int = ONSET_N_BOOTSTRAP) -> dict:
    """Sigmoid midpoint in volts, with a bootstrap CI, for one population.

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
    sub["v_num"] = sub["voltage"].map(_voltage_key)
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
    """Cortex-breakdown and area-loss onsets per population, fitted and plotted.

    Area loss is expressed as fractional projected-area loss,
    1 - (terminal_norm_radius)^2, so it is directly comparable to the
    normalised area Perrier et al. plot rather than to a radius.
    """
    out_path = sub_dir(output_dir, "cortex_breakdown")
    work = df.copy()
    work = work[work["size_category"] != "Rupture/Collapse"]

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
        vx = sub["voltage"].map(_voltage_key).to_numpy(float)
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
      ax.set_xlabel("Voltage (V)")
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

    print("\nOnset fields (sigmoid midpoint, 95% bootstrap CI):")
    for _, r in res_df.iterrows():
      if r["identified"]:
        print(f"  {r['population']:<20} {r['measure']:<20} "
              f"V50 = {r['v50']:.0f} V  [{r['v50_ci_low']:.0f}, "
              f"{r['v50_ci_high']:.0f}]  (n = {r['n_guv']})")
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
              label=f"{volt} (n={arr.shape[0]})")
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

  rows = []
  for col in tr.columns:
    m = re.match(r"GUV_(.+)_cortex_peak$", col)
    if not m:
      continue
    vals = tr[col].dropna()
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
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    pk = load_actin_peak_metrics(exp_dir)
    if pk is None:
      continue
    pk["experiment"] = exp_dir.name
    frames.append(pk)
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
  ax.set_xticklabels(voltages, rotation=45)
  ax.set_xlabel("Voltage")
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
  ax1.set_xticklabels(voltages)
  ax1.set_title(
      "GUV Size Category Distribution\n(Ba = Bare, Br = Branched)"
  )
  ax1.set_xlabel("Voltage")
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
    ax3.set_xticklabels(voltages, fontsize=9, rotation=45)
    ax3.set_xlabel("Voltage")
    ax3.set_title(POPULATION_LABELS.get(pop, pop))
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    if p_idx == 0:
      ax3.set_ylabel("Normalized Intensity")
      if mean_labelled:
        ax3.legend(loc="lower left", fontsize=9)

  fig3.suptitle(
      "Intensity Drop Trajectories per Voltage"
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

    fig_v.suptitle(f"{v} — Size Group vs. Intensity Drop")
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