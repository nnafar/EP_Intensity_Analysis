"""Bare vs BranchedCortex, compared on transmembrane potential rather than voltage.

Four questions, in the order they have to be answered:

  1. Do the two populations have the same size distribution? The induced
     transmembrane potential is dV_m = 1.5 E R cos(theta) -- linear in radius.
     A 3 um vesicle at 400 V sees roughly what an 8 um vesicle sees at 150 V.
     If the populations differ in size, a difference in release at matched
     VOLTAGE is a size effect, not a cortex effect.

  2. Does release still differ once compared at matched dV_m?

  3. Does the FRACTION of vesicles that respond differ? A median over
     responders and non-responders together answers "what does a typical
     vesicle do"; susceptibility is a dose-response question and needs the
     responding fraction against dose.

  4. Does the difference survive treating the EXPERIMENT as the unit of
     replication? Vesicles in one chamber share a preparation, a field and a
     day; they are not independent, so the effective n is the number of
     experiments, not the number of GUVs.

Vesicles are compared as THREE groups, not two. A BranchedCortex
preparation contains vesicles that assembled a shell and vesicles that
merely hold unpolymerised actin in the lumen; pooling them dilutes any
cortex effect toward the bare result, and the lumenal-only vesicles are
the better control for the bare ones anyway, since they carry the same
protein load without the shell. The split is process.apply_cortex_split,
imported rather than reimplemented so both stages cannot drift apart, and
it drops the ambiguous and unclassified vesicles (process.EXCLUDE_GROUPS).
It rests on the provisional cortex_contrast thresholds in config.py; if
those move, every figure here moves with them.

The dose axis is the induced transmembrane potential at the pole, in volts:

    dV_m = 1.5 * E * R,   E = U / ELECTRODE_GAP_CM

Reported this way rather than as an applied voltage because the voltage is a
property of this generator and this chamber and transfers to no other study,
while the field and the potential it induces do. The gap is fixed by the
3D-printed chamber and is NOT recorded per experiment in these outputs, so a
session run with a different chamber would be silently mis-scaled. Set
ELECTRODE_GAP_CM to whatever the chamber actually was.

The conversion is one multiplicative constant, so quantile bin membership,
every median and every test below are identical to the old V.um axis. Only
the units change.

The 1.5 factor is the steady-state Schwan result and assumes the pulse
outlasts the membrane charging time. That holds here by a wide margin: with
Cm = 1 uF/cm^2 and the measured outer conductivity of 416.1 uS/cm,
tau_c = R*Cm*(1/sigma_in + 1/(2*sigma_out)) is 1-5 us across the observed
radius range, against a 500 us pulse.

In Spyder: open and press F5.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palette import PALETTE, POPULATION_COLORS

try:
  import process
  RESULTS_SUBFOLDER = process.RESULTS_SUBFOLDER
except Exception:
  process = None
  RESULTS_SUBFOLDER = "Bulk_Analysis_Results"

try:
  from scipy.optimize import curve_fit
except Exception:
  curve_fit = None

# -----------------------------------------------------------------------------
# --- SETTINGS ---
# -----------------------------------------------------------------------------

OUTPUTS_ROOT = None        # None = config.PARENT_OUTPUT_FOLDER
N_DOSE_BINS = 6            # quantile bins on the pooled dose axis
MIN_BIN_N = 5              # a bin below this is dropped, not drawn thin
SUBFOLDER = "07_susceptibility"

# Keys are the group names process.assign_cortex_group emits; the order
# here sets plotting and print order everywhere in this module.
POP_LABEL = {"Bare": "Bare",
             "Branched, cortex": "Branched cortex",
             "Branched, lumenal only": "Branched lumenal only"}


def outputs_root() -> Path:
  if OUTPUTS_ROOT:
    return Path(OUTPUTS_ROOT)
  import config as cfg
  return Path(cfg.PARENT_OUTPUT_FOLDER)


def _colour(pop: str):
  return POPULATION_COLORS.get(POP_LABEL.get(pop, pop),
                               POPULATION_COLORS.get(pop, PALETTE["grey"]))


# Electrode separation in the 3D-printed chamber, in centimetres. E = U / gap,
# so this one number sets the whole dose axis. It is not recorded per
# experiment in the pipeline outputs and has to be stated here.
ELECTRODE_GAP_CM = 0.3


def load_summary(results_dir: Path) -> pd.DataFrame:
  path = results_dir / "guv_bulk_summary.csv"
  if not path.exists():
    raise SystemExit(f"{path} not found -- run the intensity stage first.")
  df = pd.read_csv(path)

  # Split BranchedCortex into cortex-bearing and lumenal-only before anything
  # is measured, so no figure in this module can silently pool the two. This
  # is a hard failure rather than a fallback to the old two-group behaviour:
  # a two-group figure and a three-group figure are not distinguishable once
  # written to disk, and a run that quietly produced the wrong one would be
  # worse than a run that did not finish.
  if process is None:
    raise SystemExit(
        "process.py could not be imported, so the cortex split cannot be "
        "applied. Every figure here would pool cortex-bearing and "
        "lumenal-only vesicles under one label. Fix the import first.")
  df = process.apply_cortex_split(df)

  df["voltage_V"] = df["voltage"].astype(str).str.extract(r"(\d+)").astype(float)
  # Field strength, and the potential it induces at the pole. Both are kept:
  # the field is what a reader of another study can match their own protocol
  # against, the potential is what the membrane responds to.
  df["field_kV_cm"] = df["voltage_V"] / ELECTRODE_GAP_CM / 1000.0
  df["dvm_proxy"] = (1.5 * (df["voltage_V"] / (ELECTRODE_GAP_CM * 1e-2))
                     * df["radius_um"] * 1e-6)
  df["released"] = -df["diff"]

  present = [p for p in POP_LABEL if p in set(df["population"])]
  missing = [p for p in POP_LABEL if p not in present]
  print("Groups after the split: "
        + ", ".join(f"{POP_LABEL[p]} n={int((df['population'] == p).sum())}"
                    for p in present))
  if missing:
    print("  Not present at all: "
          + ", ".join(POP_LABEL[p] for p in missing))
  return df


def wilson(k: int, n: int, z: float = 1.96):
  """Wilson score interval. Normal-approximation intervals run off the end of
  [0, 1] at the small n and extreme proportions this data has."""
  if n == 0:
    return np.nan, np.nan
  p = k / n
  d = 1 + z * z / n
  centre = (p + z * z / (2 * n)) / d
  half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return max(0.0, centre - half), min(1.0, centre + half)


# --- 1. size -----------------------------------------------------------------

def report_size_distributions(df: pd.DataFrame, out: Path):
  sub = df[df["radius_um"].notna()]
  if sub.empty:
    print("Size comparison skipped: no radius_um values.")
    return

  tbl = (sub.groupby(["population", "voltage"])["radius_um"]
         .agg(n="count", q25=lambda s: s.quantile(.25), median="median",
              q75=lambda s: s.quantile(.75)).reset_index())
  print("\nRadius per condition (um):")
  print(tbl.to_string(index=False))
  tbl.to_csv(out / "size_by_condition.csv", index=False)

  pops = [p for p in POP_LABEL if p in set(sub["population"])]
  for p in pops:
    v = sub[sub["population"] == p]["radius_um"]
    print(f"\n  {POP_LABEL[p]}: median {v.median():.2f} um (n={len(v)})")

  # Every pair, not one comparison: with three groups the question is not
  # only bare vs cortex but also whether lumenal-only sits with the bare
  # vesicles or with the corticated ones, and that is the pair that decides
  # whether it is usable as a same-preparation control. Holm-adjusted because
  # three tests are run on one table; both p values are printed so the
  # correction can be seen rather than taken on trust.
  pairs = [(a, b) for i, a in enumerate(pops) for b in pops[i + 1:]]
  if pairs:
    try:
      from scipy.stats import mannwhitneyu
    except Exception:
      mannwhitneyu = None
    if mannwhitneyu is not None:
      raw = []
      for a, b in pairs:
        va = sub[sub["population"] == a]["radius_um"]
        vb = sub[sub["population"] == b]["radius_um"]
        if len(va) < 3 or len(vb) < 3:
          continue
        p = float(mannwhitneyu(va, vb, alternative="two-sided").pvalue)
        raw.append({"group_a": POP_LABEL[a], "group_b": POP_LABEL[b],
                    "n_a": len(va), "n_b": len(vb),
                    "median_a": float(va.median()),
                    "median_b": float(vb.median()),
                    "median_diff_um": float(va.median() - vb.median()),
                    "p_raw": p})
      if raw:
        # process.holm, not a second copy: two implementations of the same
        # correction in one codebase is how they end up disagreeing.
        adj = process.holm([r["p_raw"] for r in raw])
        for r, a in zip(raw, adj):
          r["p_holm"] = float(a)
        pw = pd.DataFrame(raw)
        print("\n  Pairwise size comparison (Mann-Whitney, Holm-adjusted):")
        print(pw.to_string(index=False))
        pw.to_csv(out / "size_pairwise_tests.csv", index=False)
        big = pw[pw["median_diff_um"].abs() > 0.5]
        if not big.empty:
          print("  Groups differing by more than 0.5 um in median radius: "
                + "; ".join(f"{r.group_a} vs {r.group_b}"
                            for r in big.itertuples()))
          print("  Any comparison at matched VOLTAGE therefore confounds "
                "cortex with transmembrane potential; use the dose axis "
                "below.")

  fig, ax = plt.subplots(figsize=(7.5, 4.8))
  for pop in pops:
    vals = np.sort(sub[sub["population"] == pop]["radius_um"].to_numpy())
    ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post",
            color=_colour(pop), lw=2,
            label=f"{POP_LABEL[pop]} (n={len(vals)})")
  ax.set_xlabel("radius (um)")
  ax.set_ylabel("cumulative fraction")
  ax.set_title("Vesicle size distribution by population\n"
               "(ECDF; separation here confounds any voltage-matched "
               "comparison)")
  ax.legend(frameon=False)
  ax.grid(alpha=0.35, linestyle="--")
  fig.tight_layout()
  fig.savefig(out / "size_distribution_ecdf.pdf", dpi=300)
  plt.close(fig)


# --- 2 and 3. dose response --------------------------------------------------

def _dose_bins(sub: pd.DataFrame):
  """Quantile edges on the POOLED dose, so both populations share them.

  Binning each population separately would compare different dose ranges
  under the same bin label, which is the error this whole script exists to
  avoid.
  """
  q = np.linspace(0, 1, N_DOSE_BINS + 1)
  edges = np.unique(np.nanquantile(sub["dvm_proxy"], q))
  return edges if len(edges) >= 3 else None


def plot_dose_response(df: pd.DataFrame, out: Path):
  sub = df[df["dvm_proxy"].notna()]
  stag = sub[sub["size_category"] == "Stagnate"]
  if stag.empty:
    print("Dose response skipped: no stagnate GUVs with a radius.")
    return
  edges = _dose_bins(stag)
  if edges is None:
    print("Dose response skipped: dose range too narrow to bin.")
    return
  centres = 0.5 * (edges[:-1] + edges[1:])
  pops = [p for p in POP_LABEL if p in set(stag["population"])]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
  rows = []
  for pop in pops:
    p_df = stag[stag["population"] == pop].copy()
    p_df["bin"] = pd.cut(p_df["dvm_proxy"], edges, include_lowest=True,
                         labels=False)
    xs, med, lo, hi, frac, f_lo, f_hi = [], [], [], [], [], [], []
    for i in range(len(centres)):
      cell = p_df[p_df["bin"] == i]
      if len(cell) < MIN_BIN_N:
        continue
      rel = cell["released"].dropna()
      # dropna, not fillna(False): under process.ENDPOINT_MATCHED_T_S a
      # censored vesicle has no response call at all, and scoring it as a
      # non-responder would put every censored GUV in the denominator as
      # evidence of not responding.
      resp = cell["is_responding"].dropna().astype(bool)
      if resp.empty:
        continue
      xs.append(float(cell["dvm_proxy"].median()))
      med.append(rel.median())
      lo.append(rel.quantile(.25))
      hi.append(rel.quantile(.75))
      k, n = int(resp.sum()), len(resp)
      frac.append(k / n)
      a, b = wilson(k, n)
      f_lo.append(a)
      f_hi.append(b)
      rows.append({"population": POP_LABEL[pop],
                   "dose_lo": edges[i], "dose_hi": edges[i + 1],
                   "dose_median": float(cell["dvm_proxy"].median()),
                   "n": n, "median_released": rel.median(),
                   "frac_responding": k / n, "ci_lo": a, "ci_hi": b})
    if not xs:
      continue
    c = _colour(pop)
    ax1.scatter(p_df["dvm_proxy"], p_df["released"], s=8, alpha=0.25,
                color=c, edgecolors="none")
    ax1.plot(xs, med, "-o", color=c, lw=2, ms=5, label=POP_LABEL[pop])
    ax1.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
    ax2.plot(xs, frac, "-o", color=c, lw=2, ms=5, label=POP_LABEL[pop])
    ax2.fill_between(xs, f_lo, f_hi, color=c, alpha=0.18, lw=0)

  ax1.axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
  ax1.set_ylabel("released fraction")
  ax1.set_title("Release magnitude vs dose", fontsize=10)
  ax2.set_ylabel("fraction responding")
  ax2.set_ylim(-0.02, 1.02)
  ax2.set_title("Responding fraction vs dose (Wilson 95% CI)", fontsize=10)
  for ax in (ax1, ax2):
    ax.set_xlabel(r"induced transmembrane potential  $\Delta V_m = 1.5\,E\,R$  (V)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.35, linestyle="--")
  fig.suptitle("Vesicle groups at matched transmembrane potential\n"
               "(shared dose bins; stagnate GUVs)", fontsize=11)
  fig.tight_layout(rect=(0, 0, 1, 0.91))
  fig.savefig(out / "dose_response_by_population.pdf", dpi=300)
  plt.close(fig)

  if rows:
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "dose_response_summary.csv", index=False)
    print("\nDose response (shared bins on voltage x radius):")
    print(tbl.to_string(index=False))
    widths = tbl["dose_hi"] - tbl["dose_lo"]
    if len(widths) and widths.max() > 3 * widths.median():
      w = tbl.loc[widths.idxmax()]
      print(f"\n  NOTE: the bin {w['dose_lo']:.2f}-{w['dose_hi']:.2f} V is "
            f"{widths.max() / widths.median():.1f}x wider than the median bin. "
            "Quantile bins spread out where the data thins, so this one pools "
            "doses that are not really comparable -- treat any reversal in it "
            "as unresolved rather than as a turning point.")


# --- 4. experiment as the unit of replication --------------------------------

def plot_experiment_level(df: pd.DataFrame, out: Path):
  """One point per EXPERIMENT, not per GUV.

  GUVs in a chamber share a preparation, a field and a day, so pooling them
  overstates n by an order of magnitude. Whatever survives here is what the
  design actually supports.
  """
  stag = df[(df["size_category"] == "Stagnate") & df["released"].notna()]
  if stag.empty:
    print("Experiment-level plot skipped: no stagnate GUVs.")
    return

  per_exp = (stag.groupby(["population", "voltage", "voltage_V", "experiment"])
             .agg(median_released=("released", "median"),
                  frac_responding=("is_responding",
                                   lambda s: s.dropna().mean()),
                  n_guv=("released", "size"),
                  median_radius=("radius_um", "median"))
             .reset_index())
  per_exp.to_csv(out / "per_experiment_summary.csv", index=False)

  volts = sorted(per_exp["voltage_V"].unique())
  x = {v: i for i, v in enumerate(volts)}
  pops = [p for p in POP_LABEL if p in set(per_exp["population"])]
  rng = np.random.default_rng(0)

  fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(volts) + 3), 5.0))
  for k, pop in enumerate(pops):
    p_df = per_exp[per_exp["population"] == pop]
    off = (k - (len(pops) - 1) / 2) * 0.3
    xs = [x[v] + off for v in p_df["voltage_V"]]
    xs = xs + rng.uniform(-0.05, 0.05, len(xs))
    ax.scatter(xs, p_df["median_released"], s=44, color=_colour(pop),
               edgecolors="white", linewidth=0.6, zorder=3,
               label=f"{POP_LABEL[pop]} ({len(p_df)} experiments)")
  ax.axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
  ax.set_xticks(range(len(volts)))
  ax.set_xticklabels([f"{int(v)}V" for v in volts], rotation=45)
  ax.set_xlabel("Voltage")
  ax.set_ylabel("experiment median released fraction")
  ax.set_title("One point per experiment, not per vesicle\n"
               "(the unit of replication the design supports)")
  ax.legend(frameon=False, fontsize=9)
  ax.grid(axis="y", alpha=0.35, linestyle="--")
  fig.tight_layout()
  fig.savefig(out / "per_experiment_release.pdf", dpi=300)
  plt.close(fig)

  print("\nExperiments per condition (the effective n):")
  counts = (per_exp.groupby(["population", "voltage"]).size()
            .unstack(fill_value=0))
  print(counts.to_string())


def load_guv_qc(outputs_root) -> pd.DataFrame:
  """Per-GUV detectability measures, from the per-experiment CSVs.

  response_drop / response_noise come from *_fit_parameters.csv;
  sep0_over_sigma (pre-pulse dye contrast against background noise) from
  *_dye_qc.csv. Neither is in guv_bulk_summary.csv, and both are needed to
  ask whether one population is simply easier to score as responding.
  """
  root = Path(outputs_root)
  frames = []
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER:
      continue
    if process is not None and process.is_excluded_experiment(exp_dir.name):
      continue
    fits = [f for f in exp_dir.glob("**/*_fit_parameters.csv")
            if RESULTS_SUBFOLDER not in f.parts]
    if not fits:
      continue
    try:
      fit = pd.read_csv(fits[0])
    except Exception:
      continue
    keep = [c for c in ("guv_id", "response_drop", "response_noise")
            if c in fit.columns]
    if "guv_id" not in keep:
      continue
    d = fit[keep].copy()
    d["experiment"] = exp_dir.name

    qc = [f for f in exp_dir.glob("**/*_dye_qc.csv")
          if RESULTS_SUBFOLDER not in f.parts]
    if qc:
      try:
        q = pd.read_csv(qc[0])
        if "sep0_over_sigma" in q.columns:
          q = q[["guv_id", "sep0_over_sigma"]].copy()
          q["guv_id"] = q["guv_id"].astype(str)
          d["guv_id"] = d["guv_id"].astype(str)
          d = d.merge(q, on="guv_id", how="left")
      except Exception:
        pass
    frames.append(d)
  if not frames:
    return pd.DataFrame()
  out = pd.concat(frames, ignore_index=True)
  out["guv_id"] = out["guv_id"].astype(str)
  return out


def check_detectability(df: pd.DataFrame, outputs_root, out: Path):
  """Is one population simply easier to score as responding?

  `is_responding` fires when a vesicle's observed decline clears a noise
  floor derived from its own trace. A population with quieter traces, or with
  better pre-pulse dye contrast, clears that floor on a smaller real change --
  which would produce a susceptibility difference with no difference in
  poration at all. This asks whether all three groups can be scored
  equally well, before any response is interpreted.
  """
  qc = load_guv_qc(outputs_root)
  if qc.empty:
    print("Detectability check skipped: no fit parameters found.")
    return

  merged = df.assign(guv_id=df["guv_id"].astype(str)).merge(
      qc, on=["experiment", "guv_id"], how="inner")
  if merged.empty:
    print("Detectability check skipped: no GUVs matched between tables.")
    return

  pops = [p for p in POP_LABEL if p in set(merged["population"])]
  cols = [c for c in ("response_noise", "sep0_over_sigma") if c in merged]
  if not cols or len(pops) < 2:
    print("Detectability check skipped: nothing to compare.")
    return

  print("\nDetectability by population "
        "(is one easier to score as responding?):")
  rows = []
  for col in cols:
    for pop in pops:
      v = merged[merged["population"] == pop][col].dropna()
      if v.empty:
        continue
      rows.append({"measure": col, "population": POP_LABEL[pop], "n": len(v),
                   "q25": v.quantile(.25), "median": v.median(),
                   "q75": v.quantile(.75)})
  tbl = pd.DataFrame(rows)
  print(tbl.to_string(index=False))
  tbl.to_csv(out / "detectability_by_population.csv", index=False)

  # A group with LOWER noise or HIGHER contrast is easier to score. But a
  # detectability difference only explains away the result if it favours the
  # group that responds MORE. If the easier-to-score group is the one
  # responding LESS, the bias runs against the observed effect and the result
  # is conservative -- the opposite conclusion from the same numbers, so the
  # direction has to be checked, not just the magnitude.
  #
  # With three groups the comparison is between the extremes of the spread:
  # the easiest group to score against the hardest. A ratio taken over a
  # middle group would understate how far apart the ends are.
  resp_rate = {POP_LABEL[p]:
               merged[merged["population"] == p]["is_responding"]
               .dropna().mean() for p in pops}
  more_responsive = max(resp_rate, key=resp_rate.get)
  print(f"  responding overall: "
        + ", ".join(f"{k} {v:.1%}" for k, v in resp_rate.items()))

  for col in cols:
    s = tbl[tbl["measure"] == col]
    if len(s) < 2:
      continue
    med = s["median"].to_numpy()
    if min(med) <= 0:
      continue
    ratio = max(med) / min(med)
    # response_noise: lower is easier to score. sep0_over_sigma: higher is.
    easiest = (s.iloc[int(np.argmin(med))]["population"]
               if col == "response_noise"
               else s.iloc[int(np.argmax(med))]["population"])
    hardest = (s.iloc[int(np.argmax(med))]["population"]
               if col == "response_noise"
               else s.iloc[int(np.argmin(med))]["population"])
    why = ("quieter traces" if col == "response_noise"
           else "better pre-pulse contrast")
    if ratio <= 1.3:
      print(f"  {col}: medians within {ratio:.2f}x across all groups -- "
            "comparable, so this does not explain a susceptibility "
            "difference.")
    elif easiest == more_responsive:
      print(f"  {col}: {easiest} to {hardest} spans {ratio:.2f}x. {easiest} "
            f"has {why} AND responds most, so being easier to score could "
            "produce the difference on its own -- CONFOUNDED.")
    else:
      print(f"  {col}: {easiest} to {hardest} spans {ratio:.2f}x. {easiest} "
            f"has {why} but does not respond most, while {more_responsive} "
            "responds most despite being harder to score. The bias runs "
            "against the observed effect, so the difference is CONSERVATIVE, "
            "not explained away.")

  fig, axes = plt.subplots(1, len(cols) + 1,
                           figsize=(4.8 * (len(cols) + 1), 4.6))
  axes = np.atleast_1d(axes)
  for ax, col in zip(axes, cols):
    for pop in pops:
      v = np.sort(merged[merged["population"] == pop][col].dropna().to_numpy())
      if len(v):
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color=_colour(pop), lw=2, label=POP_LABEL[pop])
    ax.set_xlabel(col)
    ax.set_ylabel("cumulative fraction")
    ax.set_title(col, fontsize=10)
    ax.grid(alpha=0.35, linestyle="--")
    ax.legend(frameon=False, fontsize=8)

  ax = axes[-1]
  if {"response_noise", "response_drop"} <= set(merged.columns):
    for pop in pops:
      m = merged[merged["population"] == pop]
      ax.scatter(m["response_noise"], m["response_drop"], s=9, alpha=0.35,
                 color=_colour(pop), edgecolors="none", label=POP_LABEL[pop])
    finite = merged["response_noise"].dropna()
    lim = float(np.nanpercentile(finite, 99)) if len(finite) else 1.0
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, 3 * xs, ls="--", lw=1.0, color=PALETTE["dark_red"],
            label="3 sigma floor")
    ax.set_xlim(0, lim)
    ax.set_xlabel("response_noise")
    ax.set_ylabel("response_drop")
    ax.set_title("Signal against its own noise floor", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.35, linestyle="--")
  fig.suptitle("Detectability: can every group be scored equally well?",
               fontsize=11)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  fig.savefig(out / "detectability_by_population.pdf", dpi=300)
  plt.close(fig)


def check_per_experiment_within_bins(df: pd.DataFrame, out: Path):
  """Responding fraction per EXPERIMENT, inside each shared dose bin.

  The pooled dose response treats every vesicle as independent. If the
  difference between populations is carried by one or two sessions, it is a
  session effect wearing a population label. Each point here is one
  experiment; the populations separate only if their experiments do.
  """
  sub = df[df["dvm_proxy"].notna() & (df["size_category"] == "Stagnate")]
  edges = _dose_bins(sub) if not sub.empty else None
  if edges is None:
    print("Per-experiment bin check skipped: dose range too narrow.")
    return

  sub = sub.copy()
  sub["bin"] = pd.cut(sub["dvm_proxy"], edges, include_lowest=True,
                      labels=False)
  per = (sub.groupby(["bin", "population", "experiment"])
         .agg(n_guv=("is_responding", "count"),
              frac_responding=("is_responding",
                               lambda s: s.dropna().mean()))
         .reset_index())
  # An experiment contributing one or two vesicles to a bin can only report
  # 0, 0.5 or 1 by construction; that is not an estimate of a fraction.
  per = per[per["n_guv"] >= 3]
  if per.empty:
    print("Per-experiment bin check skipped: no experiment has >=3 GUVs "
          "in any bin.")
    return
  per.to_csv(out / "per_experiment_within_bin.csv", index=False)

  counts = per.groupby(["bin", "population"]).size().unstack(fill_value=0)
  pops = [p for p in POP_LABEL if p in counts.columns]

  print("\nResponding fraction per EXPERIMENT, within shared dose bins:")
  print("  (a bin can only separate two groups if BOTH have >=2 experiments "
        "in it; with three groups a bin may compare some pairs and not "
        "others)")
  for b in sorted(per["bin"].unique()):
    lo, hi = edges[int(b)], edges[int(b) + 1]
    have = [p for p in pops if counts.loc[b, p] >= 2]
    ok = len(have) >= 2
    mark = ("" if len(have) == len(pops) else
            f"   (comparable here: {', '.join(POP_LABEL[p] for p in have)})"
            if ok else "   (too few experiments to compare)")
    print(f"  bin dV_m {lo:.2f}-{hi:.2f} V{mark}")
    cell = per[per["bin"] == b]
    for pop in pops:
      vals = cell[cell["population"] == pop]
      if vals.empty:
        continue
      fr = ", ".join(f"{v:.2f}(n={int(n)})" for v, n
                     in zip(vals["frac_responding"], vals["n_guv"]))
      n_exp = len(vals)
      n_nonzero = int((vals["frac_responding"] > 0).sum())
      print(f"    {POP_LABEL[pop]:<16} median {vals['frac_responding'].median():.2f}"
            f"  ({n_nonzero}/{n_exp} experiments with any responder)")
      print(f"      {fr}")

  fig, ax = plt.subplots(figsize=(9.5, 5))
  rng = np.random.default_rng(0)
  for k, pop in enumerate(pops):
    m = per[per["population"] == pop]
    off = (k - (len(pops) - 1) / 2) * 0.28
    xs = m["bin"].to_numpy(float) + off + rng.uniform(-0.05, 0.05, len(m))
    ax.scatter(xs, m["frac_responding"], s=30 + 3 * m["n_guv"],
               color=_colour(pop), alpha=0.85, edgecolors="white",
               linewidth=0.6, label=f"{POP_LABEL[pop]} (1 point = 1 experiment)")
  ax.set_xticks(range(len(edges) - 1))
  ax.set_xticklabels([f"{edges[i]:.0f}-\n{edges[i+1]:.0f}"
                      for i in range(len(edges) - 1)], fontsize=8)
  ax.set_xlabel(r"dose bin  ($\Delta V_m$, V)")
  ax.set_ylabel("fraction responding")
  ax.set_ylim(-0.05, 1.05)
  ax.set_title("Responding fraction per experiment, within dose bin\n"
               "(marker size = GUVs contributed; <3 GUVs per bin dropped)")
  ax.legend(frameon=False, fontsize=9)
  ax.grid(axis="y", alpha=0.35, linestyle="--")
  fig.tight_layout()
  fig.savefig(out / "per_experiment_within_bin.pdf", dpi=300)
  plt.close(fig)


def main(results_dir=None, root=None):
  base = Path(results_dir) if results_dir else outputs_root() / RESULTS_SUBFOLDER
  out = base / SUBFOLDER
  out.mkdir(parents=True, exist_ok=True)
  # The per-experiment CSVs live one level up from the results folder.
  tree = Path(root) if root else base.parent

  df = load_summary(base)
  print(f"Loaded {len(df)} GUV rows from {base / 'guv_bulk_summary.csv'}")
  report_size_distributions(df, out)
  plot_dose_response(df, out)
  check_per_experiment_within_bins(df, out)
  check_detectability(df, tree, out)
  plot_experiment_level(df, out)
  print(f"\nSaved susceptibility figures to {out}")
  return df


if __name__ == "__main__":
  main()