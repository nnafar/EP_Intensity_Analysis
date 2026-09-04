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
     vesicle do"; susceptibility asks how many respond at a given field and
     needs the responding fraction against field strength.

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

The x axis is the applied field strength in kV/cm, E = U / ELECTRODE_GAP_CM.

Nothing here is corrected for vesicle size. It does not need to be: process.py
restricts the whole analysis to a radius window inside which the preparations
are indistinguishable (SIZE_WINDOW_UM), so two vesicles at the same field are
also at the same induced transmembrane potential, and the field is the whole
of the treatment.

That is a deliberate choice of the restriction over the correction. Dividing
an axis by radius assumes the inverse radius-field relationship Schwan's
equation predicts, and Mercadal et al. (2016) report that this is often not
observed experimentally, or is far shallower than predicted. Matching on size
assumes nothing about how the response scales, at the cost of most of the
vesicles.

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
# Vesicles a population must contribute at one field before that field is
# plotted or tabulated. The applied fields are 11 discrete values, so there is
# nothing to bin: each is its own condition. A condition of one or two
# vesicles reports 0, 0.5 or 1 by construction and is not an estimate.
MIN_N_PER_FIELD = 4
MIN_BIN_N = 5              # a bin below this is dropped, not drawn thin
SUBFOLDER = "07_susceptibility"

# Highest field the dose series can be read across. Fields above this are
# still measured and tabulated, but never drawn into a main-text dose-response
# figure: they are session-confounded (see report_session_confound), and as of
# the 2026-08 pipeline update they are excluded from every main plot outright
# rather than drawn broken alongside the interpreted range. They still appear,
# in full, in the dedicated high-field supplementary figures
# (plot_field_response_supplementary).
#
# The value now lives in config.py, since process.py's raw-trace figure needs
# it too and cannot import this module (susceptibility already imports
# process). report_session_confound still recomputes the cut from the session
# table on every run and prints a warning if the data no longer agree with the
# config value. Set config.INTERPRETED_MAX_FIELD to None to interpret the
# whole series.
try:
  import config as cfg
  INTERPRETED_MAX_FIELD = cfg.INTERPRETED_MAX_FIELD
except Exception:
  INTERPRETED_MAX_FIELD = 1.34

# Keys are the group names process.assign_cortex_group emits; the order
# here sets plotting and print order everywhere in this module.
POP_LABEL = {"Bare": "Bare",
             "Branched, cortex": "Branched cortex",
             "Branched, lumenal only": "Branched lumenal only"}
LUMEN_GROUP = "Branched, lumenal only"


def outputs_root() -> Path:
  if OUTPUTS_ROOT:
    return Path(OUTPUTS_ROOT)
  import config as cfg
  return Path(cfg.PARENT_OUTPUT_FOLDER)


def _colour(pop: str):
  return POPULATION_COLORS.get(POP_LABEL.get(pop, pop),
                               POPULATION_COLORS.get(pop, PALETTE["grey"]))


# Electrode separation in the 3D-printed chamber, in centimetres. E = U / gap,
# so this one number sets the whole x axis. It is not recorded per
# experiment in the pipeline outputs and has to be stated here.
ELECTRODE_GAP_CM = 0.3


def efflux_bool(s: pd.Series) -> pd.Series:
  """The efflux column as a real bool, with missing read as no efflux.

  fillna() on an object column is what raises the downcasting FutureWarning,
  and infer_objects() does not silence it because the warning fires inside
  fillna itself. Mapping element by element avoids the question: a missing
  call is not efflux, a string survives a round trip through CSV, and the
  result is bool either way.
  """
  if pd.api.types.is_bool_dtype(s):
    return s

  def one(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
      return False
    if isinstance(v, str):
      return v.strip().lower() in ("true", "1", "yes")
    return bool(v)

  return s.map(one).astype(bool)


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Columns every stage in this module expects, added once.

  Kept out of load_summary so the radius sensitivity, which reads a different
  file, derives its field axis and its release values the same way rather
  than from a second copy of these three lines.
  """
  df = df.copy()
  df["voltage_V"] = (df["voltage"].astype(str).str.extract(r"(\d+)")
                     .astype(float))
  # Applied field strength. This is the treatment, and inside the size window
  # it is also the whole of it -- no correction for radius is applied or
  # needed. See the module docstring.
  df["field_kV_cm"] = df["voltage_V"] / ELECTRODE_GAP_CM / 1000.0
  df["released"] = -df["diff"]
  # Acquisition session, from the yymmdd prefix the output folders carry.
  # Chambers recorded on one day share a lipid film, a protein prep and a
  # microscope alignment, so this is the coarsest unit that can confound the
  # field axis, and the one that does.
  df["session"] = df["experiment"].astype(str).str.extract(r"^(\d{6})")[0]
  return df


def load_summary(results_dir: Path, attrition_dir=None) -> pd.DataFrame:
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
  # The analysis population, defined once in process.py and applied here so
  # that this module cannot report a denominator the kinetics stage does not
  # share. See process.analysis_population for what it removes and why.
  df = process.analysis_population(df, attrition_dir=attrition_dir)

  df = derive_columns(df)

  present = [p for p in POP_LABEL if p in set(df["population"])]
  missing = [p for p in POP_LABEL if p not in present]
  print("Groups after the split: "
        + ", ".join(f"{POP_LABEL[p]} n={int((df['population'] == p).sum())}"
                    for p in present))
  if missing:
    print("  Not present at all: "
          + ", ".join(POP_LABEL[p] for p in missing))
  return df


# Chambers a cell must have on BOTH sides before an interval is attempted.
# A bootstrap over one or two clusters resamples the same chamber almost every
# draw, so the interval it returns is a statement about that chamber's spread
# and can exclude zero on three vesicles. Below this the difference is printed
# without an interval rather than with a misleading one.
MIN_CLUSTERS_FOR_CI = 3


def cluster_diff(a: pd.DataFrame, b: pd.DataFrame, col: str = "efflux",
                 n_boot: int = 5000, seed: int = 0):
  """Difference in the fraction with efflux, b minus a, with a percentile
  interval from resampling whole EXPERIMENTS.

  Vesicles in one chamber share a preparation, a field and a day. Resampling
  vesicles treats them as independent and returns an interval narrower than
  the design supports; the unit of replication is the chamber. This replaced
  a vesicle-level Fisher exact test, which had the same defect and reported it
  as a p-value, which is harder to discount than a visibly wide interval.

  With 5 to 8 chambers per cell the interval is wide. That is the honest
  width, not a failure of the method.
  """
  if col not in a.columns or col not in b.columns:
    return np.nan, np.nan, np.nan
  ga = [x for _, x in a.groupby("experiment")]
  gb = [x for _, x in b.groupby("experiment")]
  obs = float(b[col].mean() - a[col].mean()) if len(a) and len(b) else np.nan
  if min(len(ga), len(gb)) < MIN_CLUSTERS_FOR_CI:
    return obs, np.nan, np.nan
  rng = np.random.default_rng(seed)
  draws = np.empty(n_boot)
  for i in range(n_boot):
    ra = pd.concat([ga[j] for j in rng.integers(0, len(ga), len(ga))])
    rb = pd.concat([gb[j] for j in rng.integers(0, len(gb), len(gb))])
    draws[i] = rb[col].mean() - ra[col].mean()
  lo, hi = np.percentile(draws, [2.5, 97.5])
  return obs, float(lo), float(hi)


def clustered_field_ci(cell: pd.DataFrame, col: str, n_boot: int = 4000,
                       seed: int = 0):
  """Percentile CI for the mean of `col`, resampling whole chambers.

  Same logic as cluster_diff, applied to one population-field cell instead
  of a two-group difference: vesicles from the same chamber share a
  preparation, a field and a day, so the unit of replication is the
  experiment, not the vesicle. Used for both panels of plot_field_response
  -- `col="released"` for the mean-release CI, `col="efflux"` for the
  responding-fraction CI -- so that neither panel quietly reverts to a
  vesicle-level interval while the other stays chamber-clustered.

  Returns (mean, lo, hi, n_chambers). Below MIN_CLUSTERS_FOR_CI chambers,
  lo/hi are NaN rather than an interval built from resampling one or two
  chambers over and over, matching the gate cluster_diff already applies.
  """
  chambers = [g[col].dropna().to_numpy(float)
              for _, g in cell.groupby("experiment")]
  chambers = [g for g in chambers if len(g)]
  n_chambers = len(chambers)
  pooled = cell[col].dropna().to_numpy(float)
  mean = float(pooled.mean()) if len(pooled) else np.nan
  if n_chambers < MIN_CLUSTERS_FOR_CI:
    return mean, np.nan, np.nan, n_chambers
  rng = np.random.default_rng(seed)
  draws = np.empty(n_boot)
  for i in range(n_boot):
    picks = rng.integers(0, n_chambers, n_chambers)
    draws[i] = np.concatenate([chambers[j] for j in picks]).mean()
  lo, hi = np.percentile(draws, [2.5, 97.5])
  return mean, float(lo), float(hi), n_chambers


# --- Section 3: flatline vs efflux -------------------------------------------

def split_field(d: pd.DataFrame) -> float:
  """Lowest interpreted field at which any Bare vesicle shows efflux.

  One definition, used by Section 3 and by the size-matched block. The two
  used to derive it separately -- Section 3 from the data, the size-matched
  block from a hard-coded 1.2 -- and agreed only because the data happened to
  put the first Bare responder there. A change of threshold or of population
  would have moved one and not the other, silently.
  """
  if "efflux" not in d.columns:
    return np.inf
  bare = d[(d["population"] == "Bare") & d["field_kV_cm"].map(interpreted)]
  first = bare.loc[bare["efflux"] == True, "field_kV_cm"]
  return float(first.min()) if len(first) else np.inf


def report_split_support(d: pd.DataFrame, cut: float, out: Path):
  """What the split field itself is built on.

  The two-regime description is defined by the field where Bare first
  responds, so the framing inherits the support of that one cell. Printing the
  chambers behind it, and the vesicles the matched endpoint censored there,
  keeps the concession in the chapter attached to numbers rather than left for
  a reader to reconstruct from the attrition file.
  """
  if not np.isfinite(cut):
    return
  cell = d[np.isclose(d["field_kV_cm"], cut)]
  if cell.empty:
    return
  print(f"\n  What the {cut:.2f} kV/cm split field rests on:")
  for pop in POP_LABEL:
    v = cell[cell["population"] == pop]
    if v.empty:
      continue
    per = ", ".join(f"{int((x['efflux'] == True).sum())}/{len(x)}"
                    for _, x in v.groupby("experiment"))
    print(f"    {POP_LABEL[pop]:<22} "
          f"{int((v['efflux'] == True).sum())}/{len(v)} in "
          f"{v['experiment'].nunique()} chamber(s): {per}")

  n_ch = cell[cell["population"] == "Bare"]["experiment"].nunique()
  if n_ch < 2:
    print("    The Bare side of this field is ONE chamber, so the field at "
          "which Bare first responds -- and the two-regime split built on it "
          "-- cannot be separated from that chamber. Describe the split as "
          "chosen from the data, not as an estimated threshold.")

  path = Path(out) / process.POPULATION_ATTRITION_CSV
  try:
    gone = pd.read_csv(path)
  except Exception:
    return
  volt = f"{int(round(cut * ELECTRODE_GAP_CM * 1000))}V"
  lost = gone[(gone.get("voltage", pd.Series(dtype=object)).astype(str) == volt)
              & gone.get("reason", pd.Series(dtype=object)).astype(str)
              .str.contains("no endpoint call", na=False)]
  if len(lost):
    print(f"    Censored at this field: {len(lost)} vesicle(s) with no call "
          f"at t = {process.ENDPOINT_MATCHED_T_S:.0f} s, from "
          f"{lost['experiment'].nunique()} chamber(s):")
    for exp, g in lost.groupby("experiment"):
      print(f"      {len(g):>3}  {exp}")
    print("    A chamber that lost all of its vesicles here is missing from "
          "the counts above, not scored as non-responding.")


def report_cut_sensitivity(ok: pd.DataFrame, cut: float):
  """The below-cut comparison against every field the cut could sit at.

  The cut is chosen from the data, so the below-cut cell is chosen too. If the
  separation only appears at one placement it is a property of that placement.
  """
  fields = sorted(ok["field_kV_cm"].unique())
  rows = []
  for c in fields[1:]:
    lo = ok[ok["field_kV_cm"] < c]
    a = lo[lo["population"] == "Bare"]
    b = lo[lo["population"] == "Branched, cortex"]
    if len(a) < 5 or len(b) < 5:
      continue
    ka = int((a["efflux"] == True).sum())
    kb = int((b["efflux"] == True).sum())
    diff, l, h = cluster_diff(a, b)
    ci = f"[{l:+.3f}, {h:+.3f}]" if np.isfinite(l) else "no interval"
    rows.append((c, ka, len(a), kb, len(b), diff, ci,
                 "  <- the cut used" if abs(c - cut) < 1e-9 else ""))
  if not rows:
    return
  print("\n  Below-cut comparison against where the cut is placed:")
  print(f"    {'cut':>6} {'Bare':>9} {'cortex':>9}  cortex minus Bare")
  for c, ka, na, kb, nb, diff, ci, mark in rows:
    print(f"    {c:6.2f} {ka:>3}/{na:<5} {kb:>3}/{nb:<5} "
          f"{diff:+.3f} {ci}{mark}")
  print("    Read the threshold claim from this column, not from the one row "
        "the chosen cut selects.")


def report_section3(df: pd.DataFrame, out: Path) -> pd.DataFrame:
  """Flatline against efflux, per field, for the three cortex groups.

  Two exhaustive classes on the analysis population: a vesicle either holds
  its intensity to within INTENSITY_DIFF_THRESHOLD or loses more than that.
  Gainers and endpoint-less records are already gone, so the counts sum to n
  and the efflux fraction is a proportion of the whole group rather than of
  whatever survived an unstated filter.

  Intensity is summarised as a MEAN. The median over a population that is
  mostly flatlines is a statement about the flatlines: it sits near zero at
  every field regardless of how much dye the responders lost.
  """
  d = df.copy()
  # The call is made once, in process.aggregate_pipeline_results, against each
  # vesicle's own pre-pulse mean and its own noise. Recomputing it here from a
  # flat threshold would drop the noise term and put a second definition in
  # the file that reports the headline numbers. The fallback covers an old
  # summary CSV written before the column existed, and says so.
  if "efflux" in d.columns:
    d["efflux"] = efflux_bool(d["efflux"])
  else:
    print("  WARNING: no 'efflux' column in this summary -- falling back to a "
          f"flat {process.INTENSITY_DIFF_THRESHOLD:.0%} cut with no noise "
          "term. Re-run the bulk stage; these counts are not the reported "
          "ones.")
    d["efflux"] = d["released"] > process.INTENSITY_DIFF_THRESHOLD

  rows = []
  for (pop, E), cell in d.groupby(["population", "field_kV_cm"]):
    n = len(cell)
    k = int(cell["efflux"].sum())
    rows.append({
        "population": POP_LABEL.get(pop, pop),
        "field_kV_cm": E,
        "n": n,
        "n_flatline": n - k,
        "n_efflux": k,
        "frac_efflux": k / n if n else np.nan,
        "mean_released": cell["released"].mean(),
        "mean_released_efflux": (cell.loc[cell["efflux"], "released"].mean()
                                 if k else np.nan),
        "chambers": cell["experiment"].nunique(),
        "interpreted": interpreted(E),
    })
  tbl = pd.DataFrame(rows).sort_values(["population", "field_kV_cm"])
  tbl.to_csv(out / "section3_flatline_vs_efflux.csv", index=False)

  print("\n" + "=" * 70)
  print("Section 3: flatline vs efflux on the analysis population")
  print("=" * 70)
  print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
  print("  Fields marked interpreted=False share no session with any field "
        "below them and are reported, not read.")

  # Pooled cells. The series is split at the lowest field where the bare
  # population first shows any efflux: below it one group is at zero and the
  # other is not, which is the threshold claim, and pooling the two halves
  # averages that separation away against the wider high-field cells.
  ok = d[d["field_kV_cm"].map(interpreted)]
  cut = split_field(d)
  print(f"\n  Bare first shows efflux at {cut:.2f} kV/cm; the series is split "
        f"there.")
  report_split_support(ok, cut, out)

  for label, part in (("all interpreted fields", ok),
                      (f"E < {cut:.2f} kV/cm", ok[ok["field_kV_cm"] < cut]),
                      (f"E >= {cut:.2f} kV/cm", ok[ok["field_kV_cm"] >= cut])):
    print(f"\n  {label}:")
    ref = part[part["population"] == "Bare"]
    for pop in POP_LABEL:
      cell = part[part["population"] == pop]
      if cell.empty:
        continue
      k, n = int(cell["efflux"].sum()), len(cell)
      print(f"    {POP_LABEL[pop]:<22} {k:>3}/{n:<4} "
            f"({k / n:5.1%})  {cell['experiment'].nunique():>2} chamber(s)  "
            f"mean released {cell['released'].mean():+.4f}")
    for pop in POP_LABEL:
      if pop == "Bare" or ref.empty:
        continue
      cell = part[part["population"] == pop]
      if cell.empty:
        continue
      diff, lo, hi = cluster_diff(ref, cell)
      if not np.isfinite(lo):
        print(f"    {POP_LABEL[pop]} minus Bare: {diff:+.3f} -- no interval, "
              f"{min(ref['experiment'].nunique(), cell['experiment'].nunique())}"
              f" chamber(s) on one side (need {MIN_CLUSTERS_FOR_CI})")
        continue
      verdict = "resolved" if (lo > 0 or hi < 0) else "not resolved"
      print(f"    {POP_LABEL[pop]} minus Bare: {diff:+.3f} "
            f"95% CI [{lo:+.3f}, {hi:+.3f}] -- {verdict}")
  report_cut_sensitivity(ok, cut)
  return tbl


# Written by run_bulk.py before the size window is applied.
ALL_RADII_CSV = "guv_bulk_all_radii.csv"

# Windows the headline cells are recomputed under. None is no restriction at
# all; the configured window sits in the middle of the sweep so its row can be
# read against both looser and tighter ones.
RADIUS_SENSITIVITY_WINDOWS = [None, (3.0, 7.0), (3.5, 6.5), (4.0, 6.0),
                              (4.5, 5.5)]


def _win_label(win) -> str:
  return "none" if win is None else f"{win[0]:.1f}-{win[1]:.1f}"


def _cell_line(part: pd.DataFrame):
  """Bare and cortex counts and the clustered difference for one cell."""
  a = part[part["population"] == "Bare"]
  b = part[part["population"] == "Branched, cortex"]
  ka, kb = int(a["efflux"].sum()), int(b["efflux"].sum())
  diff, lo, hi = cluster_diff(a, b)
  ci = f"[{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else "no interval"
  r = (a["radius_um"].median() - b["radius_um"].median()
       if len(a) and len(b) else np.nan)
  return (f"{ka:>3}/{len(a):<5} {kb:>3}/{len(b):<5} {diff:+.3f} {ci:<20} "
          f"{r:+.2f}")


def report_radius_sensitivity(base: Path, out: Path):
  """Does the Bare-vs-cortex result depend on the 4-6 um window?

  The window discards most of the tracked population and does not discard it
  evenly: at 1.33 kV/cm it removes more Bare vesicles than cortex ones, and
  some of the Bare vesicles it removes had released dye. That is the direction
  that would manufacture the reported difference rather than guard against it,
  so it is checked here instead of argued about.

  Each row re-runs process.analysis_population under a different window, so
  the other five conditions of the population are applied identically and
  only the radius restriction moves. The split field is held at the value the
  configured window gives, so the cells being compared stay the same cells.
  """
  path = Path(base) / ALL_RADII_CSV
  print("\n" + "=" * 70)
  print("Radius window sensitivity")
  print("=" * 70)
  if not path.exists():
    print(f"  Skipped: {path} not found. run_bulk.py writes it beside "
          "guv_bulk_summary.csv; re-run the intensity stage to produce it.")
    return
  raw = process.apply_cortex_split(pd.read_csv(path))
  raw = raw[raw["radius_um"].notna()]
  print(f"  {len(raw)} vesicle(s) with a measured radius, before any window.")

  ref = derive_columns(process.analysis_population(
      raw, verbose=False, size_window=process.SIZE_WINDOW_UM))
  ref["efflux"] = efflux_bool(ref["efflux"])
  cut = split_field(ref)
  if not np.isfinite(cut):
    print("  No Bare responder inside the configured window, so there is no "
          "split field to hold fixed. Sweep not run.")
    return

  # --- what the window removes, field by field ------------------------------
  full = derive_columns(process.analysis_population(
      raw, verbose=False, size_window=None))
  full["efflux"] = efflux_bool(full["efflux"])
  lo, hi = process.SIZE_WINDOW_UM
  full["in_window"] = full["radius_um"].between(lo, hi)
  ok = full[full["field_kV_cm"].map(interpreted)]

  print(f"\n  What the {lo}-{hi} um window removes, per interpreted field:")
  print(f"    {'kV/cm':>6} {'population':<22} "
        f"{'kept efflux/n':>14} {'removed efflux/n':>17}")
  for E in sorted(ok["field_kV_cm"].unique()):
    for pop in ("Bare", "Branched, cortex"):
      cell = ok[(ok["field_kV_cm"] == E) & (ok["population"] == pop)]
      if cell.empty:
        continue
      kept = cell[cell["in_window"]]
      gone = cell[~cell["in_window"]]
      print(f"    {E:6.2f} {POP_LABEL[pop]:<22} "
            f"{int(kept['efflux'].sum()):>7}/{len(kept):<6} "
            f"{int(gone['efflux'].sum()):>9}/{len(gone):<6}")

  # --- the responders the window discards, named ----------------------------
  lost = ok[(~ok["in_window"]) & ok["efflux"]]
  if len(lost):
    print(f"\n  Responding vesicles the window discards ({len(lost)}):")
    for _, v in lost.sort_values(["field_kV_cm", "population"]).iterrows():
      print(f"    {v['field_kV_cm']:.2f} kV/cm  "
            f"{POP_LABEL.get(v['population'], v['population']):<22} "
            f"r = {v['radius_um']:.2f} um  released {v['released']:+.3f}  "
            f"{v['experiment']}")
    print("    A responder discarded from one group and not the other moves "
          "the difference. Read the sweep below with these in mind.")

  # --- the headline cells, window by window ---------------------------------
  parts = [("all interpreted", lambda d: d),
           (f"E < {cut:.2f} kV/cm", lambda d: d[d["field_kV_cm"] < cut]),
           (f"E >= {cut:.2f} kV/cm", lambda d: d[d["field_kV_cm"] >= cut])]
  runs = []
  for win in RADIUS_SENSITIVITY_WINDOWS:
    d = derive_columns(process.analysis_population(
        raw, verbose=False, size_window=win))
    d["efflux"] = efflux_bool(d["efflux"])
    runs.append((win, d[d["field_kV_cm"].map(interpreted)],
                 split_field(d)))

  for label, sel in parts:
    print(f"\n  {label}:")
    print(f"    {'window':>9} {'Bare':>9} {'cortex':>9} "
          f"{'cortex minus Bare':<28} {'median r, Bare-cortex':>10}")
    for win, d, _ in runs:
      mark = "  <- configured" if win == process.SIZE_WINDOW_UM else ""
      print(f"    {_win_label(win):>9} {_cell_line(sel(d))}{mark}")

  print("\n  Where Bare first responds, under each window:")
  for win, _, c in runs:
    print(f"    {_win_label(win):>9}  "
          + (f"{c:.2f} kV/cm" if np.isfinite(c) else "never"))
  print("    A window that moves this has changed which fields the split "
        "describes, not just how many vesicles are in them.")


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
                "cortex with transmembrane potential; check the size window "
                "below.")

  fig, ax = plt.subplots(
      figsize=process.fig_size("size_distribution_ecdf", 7.5, 4.8))
  for pop in pops:
    vals = np.sort(sub[sub["population"] == pop]["radius_um"].to_numpy())
    ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post",
            color=_colour(pop), lw=2,
            label=f"{POP_LABEL[pop]} (n={len(vals)})")
  ax.set_xlabel("radius (um)")
  ax.set_ylabel("cumulative fraction")
  ax.set_xlim(left=0)
  ax.set_ylim(0, 1.02)
  ax.legend(frameon=False)
  ax.grid(alpha=0.35, linestyle="--")
  process.style_figure("size_distribution_ecdf")
  fig.tight_layout()
  fig.savefig(out / "size_distribution_ecdf.pdf", dpi=300)
  plt.close(fig)


# --- 2 and 3. response against field strength --------------------------------

def _field_cells(df: pd.DataFrame):
  """Stagnate vesicles with a field and a release value, per population."""
  return df[(df["size_category"] == "Stagnate")
            & df["released"].notna()
            & df["field_kV_cm"].notna()]


def _field_cells_interpreted(df: pd.DataFrame):
  """_field_cells, restricted to the interpreted range (<= INTERPRETED_MAX_FIELD).

  This is what every main-text field-strength figure should plot from, as of
  the 2026-08 update: fields above the cut are excluded outright here, not
  drawn broken alongside the interpreted range. Use
  _field_cells_high_field for the complementary, supplementary-only set.
  """
  sub = _field_cells(df)
  if INTERPRETED_MAX_FIELD is None:
    return sub
  return sub[sub["field_kV_cm"].map(interpreted)]


def _field_cells_high_field(df: pd.DataFrame):
  """_field_cells, restricted to ABOVE the interpreted range.

  The complement of _field_cells_interpreted. Feeds the supplementary
  high-field figures only -- never a main-text plot -- and is not read as a
  dose response for the reasons in report_session_confound.
  """
  sub = _field_cells(df)
  if INTERPRETED_MAX_FIELD is None:
    return sub.iloc[0:0]
  return sub[~sub["field_kV_cm"].map(interpreted)]


def interpreted(E) -> bool:
  """Is this field inside the range the dose series can be read across?"""
  if INTERPRETED_MAX_FIELD is None:
    return True
  return float(E) <= INTERPRETED_MAX_FIELD


def _mark_uninterpreted(ax, fields):
  """Shade the field range that is not read as a dose response.

  Drawn rather than dropped. A silent gap at the top of a dose series invites
  the reader to assume the missing conditions were unfavourable, and here
  they were measured and are simply not separable from their session.
  """
  if INTERPRETED_MAX_FIELD is None:
    return
  hi = [E for E in fields if not interpreted(E)]
  if not hi:
    return
  lo_edge = INTERPRETED_MAX_FIELD + 0.05
  ax.axvspan(lo_edge, max(hi) * 1.05, color=PALETTE["pale_red"], alpha=0.55,
             lw=0, zorder=0)
  ax.annotate("one session only;\nnot read as a dose response",
              xy=((lo_edge + max(hi)) / 2, 0.72),
              xycoords=("data", "axes fraction"), ha="center", fontsize=7.5,
              color=PALETTE["grey"], zorder=1)


def _runs(mask: np.ndarray):
  """Start/stop index pairs of each contiguous True run in a boolean array.

  Used to draw a filled CI band only across points that actually have one,
  leaving a real gap at points without a chamber-clustered interval instead
  of bridging over them or dropping the band for the whole series.
  """
  idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int),
                                              [0]))))
  return list(zip(idx[0::2], idx[1::2]))


def plot_field_response(
    df: pd.DataFrame, 
    out: Path,
    font_size: int = 15,
    tick_size: int = 15,
    legend_size: int = 15
):
  """Release magnitude and responding fraction against applied field.

  One point per field per population, not per bin. The eleven amplitudes are
  discrete conditions and binning them would only blur conditions together
  that were never mixed in the first place.

  Both panels are reported because they answer different questions and can
  disagree. The mean release is what left the vesicles and needs no
  threshold; the responding fraction counts how many crossed one. A
  population where a few vesicles empty and the rest do nothing shows a high
  mean and a low fraction, and the two together say which is happening.

  As of the 2026-08 pipeline update, this figure is restricted outright to
  the interpreted range (<= INTERPRETED_MAX_FIELD, i.e. 0.1-1.33 kV/cm /
  30-400 V): fields above that are session-confounded (see
  report_session_confound) and are no longer drawn broken alongside the
  interpreted range. They are drawn in full in their own figure by
  plot_field_response_supplementary. field_response_summary.csv still
  tabulates every field, interpreted or not, with an "interpreted" column, so
  nothing measured is silently dropped -- only kept off the main plot.
  """
  stag_all = _field_cells(df)
  stag = _field_cells_interpreted(df)
  if stag.empty:
    print("Field response skipped: no stagnate GUVs with a release value "
          "in the interpreted range.")
    return
  pops = [p for p in POP_LABEL if p in set(stag["population"])]
  rng = np.random.default_rng(0)

  fig, (ax1, ax2) = plt.subplots(
      1, 2, figsize=process.fig_size("field_response_by_population",
                                     12.5, 5.0, 2))
  rows, thin = [], []
  for pop in pops:
    # Rows (and therefore the CSV) still cover every field for this
    # population, interpreted or not; only what gets plotted is restricted.
    p_df_all = stag_all[stag_all["population"] == pop]
    p_df = stag[stag["population"] == pop]
    c = _colour(pop)
    xs, mean_r, lo_r, hi_r, frac, f_lo, f_hi = [], [], [], [], [], [], []
    for E, cell in p_df_all.groupby("field_kV_cm"):
      v = cell["released"].dropna().to_numpy(float)
      if len(v) < MIN_N_PER_FIELD:
        thin.append((POP_LABEL[pop], E, len(v)))
        continue
      # Chamber-clustered, not vesicle-level: matches the Methods text
      # ("all confidence intervals in this chapter resample whole
      # chambers"), which previously described cluster_diff's behaviour but
      # not this figure's. A cell with fewer than MIN_CLUSTERS_FOR_CI
      # chambers gets its point plotted from every pooled vesicle but no
      # interval -- see n_chambers in the CSV rather than a falsely narrow
      # band built from resampling one or two chambers repeatedly.
      mean_v, lo_v, hi_v, n_ch = clustered_field_ci(cell, "released")
      resp = cell["efflux"].dropna().astype(bool)
      k, n = int(resp.sum()), int(len(resp))
      cell_num = cell.assign(efflux=cell["efflux"].astype(float))
      mean_f, lo_f, hi_f, n_ch_f = clustered_field_ci(cell_num, "efflux")
      rows.append({"population": POP_LABEL[pop], "field_kV_cm": E,
                   "n": len(v), "n_chambers": n_ch,
                   "median_released": float(np.median(v)),
                   "mean_released": float(v.mean()),
                   "mean_ci_lo": lo_v, "mean_ci_hi": hi_v,
                   "n_with_call": n, "n_efflux": k,
                   "frac_efflux": (k / n if n else np.nan),
                   "frac_ci_lo": lo_f, "frac_ci_hi": hi_f,
                   "interpreted": interpreted(E),
                   "sessions": ",".join(sorted(cell["session"].dropna()
                                               .unique()))})
      if not interpreted(E):
        continue
      xs.append(E)
      mean_r.append(v.mean())
      lo_r.append(lo_v)
      hi_r.append(hi_v)
      frac.append(k / n if n else np.nan)
      f_lo.append(lo_f)
      f_hi.append(hi_f)
      
    # Plot larger, darker points on ax1, with a bit of x-jitter
    valid_rel = p_df.dropna(subset=["released"])
    if not valid_rel.empty:
      x_jit1 = rng.uniform(-0.015, 0.015, len(valid_rel))
      ax1.scatter(valid_rel["field_kV_cm"] + x_jit1, valid_rel["released"], 
                  s=35, alpha=0.6, color=c, edgecolors="none")

    # Add raw points to ax2 with matching style and xy-jitter for 0/1 separation
    valid_eff = p_df.dropna(subset=["efflux"])
    if not valid_eff.empty:
      x_jit2 = rng.uniform(-0.015, 0.015, len(valid_eff))
      y_jit2 = rng.uniform(-0.02, 0.02, len(valid_eff))
      ax2.scatter(valid_eff["field_kV_cm"] + x_jit2, valid_eff["efflux"].astype(float) + y_jit2, 
                  s=35, alpha=0.6, color=c, edgecolors="none")

    xs = np.asarray(xs, float)
    if xs.size:
      for ax, mid, lo_v, hi_v, alpha in (
          (ax1, mean_r, lo_r, hi_r, 0.15), (ax2, frac, f_lo, f_hi, 0.18)):
        y = np.asarray(mid)
        lo_a, hi_a = np.asarray(lo_v), np.asarray(hi_v)
        # zorder=5 elevates the summary lines above the new larger scatter dots
        ax.plot(xs, y, "o-", color=c, lw=2, ms=5, mfc=c,
                label=POP_LABEL[pop], zorder=5)
        # NaN-safe: points below MIN_CLUSTERS_FOR_CI chambers have no
        # interval, and fill_between would otherwise raise or silently drop
        # the whole band at the first NaN. Drawn per contiguous run of
        # resolvable points instead, so a gap in the middle of the series
        # (e.g. one thin field) leaves a real gap rather than a false bridge
        # or an empty plot.
        ok = np.isfinite(lo_a) & np.isfinite(hi_a)
        for start, stop in _runs(ok):
          ax.fill_between(xs[start:stop], lo_a[start:stop], hi_a[start:stop],
                          color=c, alpha=alpha, lw=0, zorder=1)

  # 1D axes start at 0: the release fraction and the responding fraction are
  # both non-negative quantities by construction, so the axis floor should
  # say so rather than implying values below zero are possible. If a
  # bootstrap CI band is ever clipped visibly at the floor, that is real
  # information (the estimate's lower bound crosses zero), not a plotting
  # artefact -- leave it clipped rather than padding below 0 to show it.
  ax1.set_ylim(bottom=0)
  ax1.set_ylabel("released fraction of lumenal signal", fontsize=font_size)
  ax2.set_ylabel("fraction with efflux", fontsize=font_size)
  # Expanded the limit slightly so jittered points near 0 and 1 don't get clipped
  ax2.set_ylim(-0.05, 1.05) 
  for ax in (ax1, ax2):
    ax.set_xlim(left=0)
    ax.set_xlabel("field strength (kV/cm)", fontsize=font_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(alpha=0.35, linestyle="--")
    
  # Extract handles and labels once from the first axis to create a single legend below
  handles, labels = ax1.get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), 
             ncol=len(labels), frameon=False, fontsize=legend_size)

  fig.tight_layout()
  # Adjust bottom margin to make space for the unified legend
  fig.subplots_adjust(bottom=0.2)
  fig.savefig(out / "field_response_by_population.pdf", dpi=300, bbox_inches="tight")
  plt.close(fig)

  if rows:
    tbl = pd.DataFrame(rows).sort_values(["population", "field_kV_cm"])
    tbl.to_csv(out / "field_response_summary.csv", index=False)
    print("\nResponse against applied field strength (no size correction; "
          "every field, interpreted or not -- see the 'interpreted' "
          "column):")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
  if thin:
    print(f"\n  {len(thin)} population-field cell(s) below "
          f"{MIN_N_PER_FIELD} vesicles and omitted:")
    for lab, E, n in sorted(thin, key=lambda t: (t[0], t[1])):
      print(f"    {lab:<24} {E:.2f} kV/cm  n={n}")


def plot_field_response_supplementary(df: pd.DataFrame, out: Path):
  """Same two panels as plot_field_response, for fields ABOVE the interpreted
  range only.

  Not a dose-response figure: see report_session_confound for why the fields
  drawn here cannot be separated from the sessions that supplied them. Exists
  so the higher-field vesicles remain visible somewhere -- a silent gap at
  the top of the series invites the reader to assume the missing conditions
  were unfavourable, and here they were measured, just not interpretable as
  part of the dose series.
  """
  stag = _field_cells_high_field(df)
  if stag.empty:
    print("High-field supplementary figure skipped: no vesicles above "
          f"{INTERPRETED_MAX_FIELD} kV/cm.")
    return
  pops = [p for p in POP_LABEL if p in set(stag["population"])]
  rng = np.random.default_rng(0)

  fig, (ax1, ax2) = plt.subplots(
      1, 2, figsize=process.fig_size(
          "field_response_by_population_supplementary_highfield", 12.5, 5.0, 2))
  rows, thin = [], []
  for pop in pops:
    p_df = stag[stag["population"] == pop]
    c = _colour(pop)
    xs, mean_r, lo_r, hi_r, frac, f_lo, f_hi = [], [], [], [], [], [], []
    for E, cell in p_df.groupby("field_kV_cm"):
      v = cell["released"].dropna().to_numpy(float)
      if len(v) < MIN_N_PER_FIELD:
        thin.append((POP_LABEL[pop], E, len(v)))
        continue
      boot = np.mean(v[rng.integers(0, len(v), (4000, len(v)))], axis=1)
      resp = cell["efflux"].dropna().astype(bool)
      k, n = int(resp.sum()), int(len(resp))
      wl, wh = wilson(k, n) if n else (np.nan, np.nan)
      xs.append(E)
      mean_r.append(v.mean())
      lo_r.append(np.percentile(boot, 2.5))
      hi_r.append(np.percentile(boot, 97.5))
      frac.append(k / n if n else np.nan)
      f_lo.append(wl)
      f_hi.append(wh)
      rows.append({"population": POP_LABEL[pop], "field_kV_cm": E,
                   "n": len(v), "median_released": float(np.median(v)),
                   "mean_released": float(v.mean()),
                   "n_with_call": n, "n_efflux": k,
                   "frac_efflux": (k / n if n else np.nan),
                   "sessions": ",".join(sorted(cell["session"].dropna()
                                               .unique()))})
    ax1.scatter(p_df["field_kV_cm"], p_df["released"], s=8, alpha=0.22,
                color=c, edgecolors="none")
    xs = np.asarray(xs, float)
    if xs.size:
      order = np.argsort(xs)
      xs_o = xs[order]
      for ax, mid, lo_v, hi_v, alpha in (
          (ax1, mean_r, lo_r, hi_r, 0.15), (ax2, frac, f_lo, f_hi, 0.18)):
        y = np.asarray(mid)[order]
        lo_a, hi_a = np.asarray(lo_v)[order], np.asarray(hi_v)[order]
        # Open markers, no connecting dose-response line: these fields are
        # each their own isolated condition (one session apiece), not a
        # series, so the same drawing choice as the old broken-line segment
        # is kept here without needing the interpreted/uninterpreted split.
        ax.plot(xs_o, y, "o", ls="none", color=c, mfc="white", mec=c,
                ms=6, mew=1.5, label=POP_LABEL[pop])
        ax.errorbar(xs_o, y, yerr=[np.clip(y - lo_a, 0, None),
                                   np.clip(hi_a - y, 0, None)],
                    fmt="none", ecolor=c, elinewidth=0.9, capsize=2,
                    alpha=0.8)

  ax1.set_ylim(bottom=0)
  ax1.set_ylabel("released fraction of lumenal signal")
  ax2.set_ylabel("fraction with efflux")
  ax2.set_ylim(0, 1.02)
  for ax in (ax1, ax2):
    ax.set_xlim(left=0)
    ax.set_xlabel("field strength (kV/cm)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.35, linestyle="--")
  ax1.annotate("supplementary: fields above the interpreted range,\n"
               "each one session only -- not a dose response",
               xy=(0.02, 0.96), xycoords="axes fraction", ha="left",
               va="top", fontsize=8, color=PALETTE["grey"])
  process.style_figure("field_response_by_population_supplementary_highfield")
  fig.tight_layout()
  fig.savefig(out / "field_response_by_population_supplementary_highfield.pdf",
              dpi=300)
  plt.close(fig)

  if rows:
    tbl = pd.DataFrame(rows).sort_values(["population", "field_kV_cm"])
    tbl.to_csv(out / "field_response_summary_supplementary_highfield.csv",
              index=False)
    print("\nHigh-field supplementary response (session-confounded, not a "
          "dose response):")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --- 3b. is the field axis also the session axis? ----------------------------

def _suggested_field_cut(sess_by_field: dict):
  """Highest field below the largest top block that shares no session.

  Walks down from the top of the series and stops at the first field whose
  sessions also appear lower down. Everything above that point was recorded
  in sessions that contributed nothing else, so within it a change in
  response cannot be attributed to the field. Returns None when no such block
  exists, which is the outcome to want.
  """
  fields = sorted(sess_by_field)
  cut_idx = len(fields)
  # Every split is tested rather than scanning until the first overlap.
  # Disjointness is not monotonic in the split point: the top field usually
  # shares its session with the field just below it, which would stop the
  # scan before it reached the block boundary that matters.
  for i in range(1, len(fields)):
    upper = set().union(*[sess_by_field[f] for f in fields[i:]])
    lower = set().union(*[sess_by_field[f] for f in fields[:i]])
    if not (upper & lower):
      cut_idx = i
      break
  return fields[cut_idx - 1] if cut_idx < len(fields) else None


def report_session_confound(df: pd.DataFrame, out: Path):
  """Which sessions supplied which fields, and how much that matters.

  Every other check in this module asks whether the two preparations differ.
  This one asks whether the x axis means what it says. If a field is supplied
  by sessions that supply no other field, then its response rate carries the
  session as well as the field, and the two cannot be told apart by any
  amount of resampling within it.
  """
  sub = _field_cells(df)
  if sub.empty or sub["session"].isna().all():
    print("Session check skipped: no session could be parsed from the "
          "experiment names.")
    return

  counts = (sub.groupby(["field_kV_cm", "session"]).size()
            .unstack(fill_value=0).sort_index())
  counts.to_csv(out / "field_session_counts.csv")
  print(f"\n{'=' * 70}")
  print("Vesicles per field and session (is the dose axis also a date axis?)")
  print("=" * 70)
  print(counts.to_string())

  rate = (sub.groupby("session")["efflux"]
          .agg(n="count", responding="sum"))
  rate["frac"] = rate["responding"] / rate["n"]
  print("\nResponding fraction per session, pooled over fields and groups:")
  print(rate.to_string(float_format=lambda x: f"{x:.3f}"))
  if len(rate) > 1 and rate["frac"].min() > 0:
    print(f"  Between-session spread: {rate['frac'].min():.1%} to "
          f"{rate['frac'].max():.1%}, a factor of "
          f"{rate['frac'].max() / rate['frac'].min():.1f}. Any field carried "
          "by one session inherits that spread.")

  sess_by_field = {E: set(g["session"].dropna())
                   for E, g in sub.groupby("field_kV_cm")}
  # Within a field, the groups have to come from the same sessions or the
  # comparison at that field carries the session too. Most fields here are
  # supplied by one session, which is harmless as long as both groups sit in
  # it; what is not harmless is the two groups sitting in different ones.
  for E, g in sub.groupby("field_kV_cm"):
    per_pop = {p: set(x["session"].dropna())
               for p, x in g.groupby("population") if len(x) >= MIN_N_PER_FIELD}
    if len(per_pop) < 2:
      continue
    sets = list(per_pop.values())
    if all(s == sets[0] for s in sets):
      continue
    detail = "; ".join(f"{POP_LABEL.get(p, p)} {','.join(sorted(s))}"
                       for p, s in per_pop.items())
    disjoint = all(not (x & y) for i, x in enumerate(sets)
                   for y in sets[i + 1:])
    print(f"\n{'  WARNING: ' if disjoint else '  Note: '}"
          f"at {E:.2f} kV/cm the groups do not come from the same sessions "
          f"({detail})."
          + (" They share none, so this field cannot separate the groups "
             "from their sessions." if disjoint else ""))

  cut = _suggested_field_cut(sess_by_field)
  if cut is None:
    print("No block of top fields is session-isolated; the whole series is "
          "readable as a dose response.")
  else:
    print(f"\nFields above {cut:.2f} kV/cm share no session with any field "
          "below them, so across that break the dose axis and the session "
          "axis cannot be separated.")
  if INTERPRETED_MAX_FIELD is not None and cut is not None \
      and abs(cut - INTERPRETED_MAX_FIELD) > 0.05:
    print(f"  WARNING: INTERPRETED_MAX_FIELD is {INTERPRETED_MAX_FIELD} but "
          f"the session table implies {cut:.2f}. The figures are drawn "
          "against the setting, not against the data. Update it or explain "
          "the difference.")
  elif INTERPRETED_MAX_FIELD is None and cut is not None:
    print("  WARNING: INTERPRETED_MAX_FIELD is None, so the figures draw an "
          "unbroken dose curve across that break.")

  # --- figure ---------------------------------------------------------------
  pops = [p for p in POP_LABEL if p in set(sub["population"])]
  sessions = sorted(sub["session"].dropna().unique())
  marks = ["o", "s", "^", "D", "v", "P"]
  fig, (ax1, ax2) = plt.subplots(
      1, 2, figsize=process.fig_size("field_response_by_session", 11.5, 4.4, 2),
      gridspec_kw={"width_ratios": [1.6, 1]})
  for pop in pops:
    for j, s in enumerate(sessions):
      cell = sub[(sub["population"] == pop) & (sub["session"] == s)]
      g = cell.groupby("field_kV_cm")["efflux"].agg(["count", "sum"])
      g = g[g["count"] >= 3]
      if g.empty:
        continue
      ax1.scatter(g.index, g["sum"] / g["count"],
                  s=12 + 2.2 * g["count"], marker=marks[j % len(marks)],
                  color=_colour(pop), edgecolors="white", linewidth=0.5,
                  zorder=3)
  _mark_uninterpreted(ax1, sorted(sess_by_field))
  ax1.set_xlabel("field strength (kV/cm)")
  ax1.set_ylabel("fraction with efflux")
  ax1.set_ylim(-0.05, 1.05)
  handles = ([plt.Line2D([], [], marker=marks[j % len(marks)], ls="",
                         mfc="none", mec=PALETTE["grey"],
                         color=PALETTE["grey"], ms=5, label=s)
              for j, s in enumerate(sessions)]
             + [plt.Line2D([], [], marker="o", ls="", color=_colour(p), ms=5,
                           label=POP_LABEL[p]) for p in pops])
  ax1.legend(handles=handles, frameon=False, fontsize=7.5, ncol=2,
             loc="upper left", handletextpad=0.4, columnspacing=1.0)

  ax2.bar(range(len(rate)), rate["frac"], color=PALETTE["light_blue"],
          edgecolor=PALETTE["dark_blue"], lw=0.8)
  for i, (n, f) in enumerate(zip(rate["n"], rate["frac"])):
    ax2.text(i, f + 0.004, f"n={int(n)}", ha="center", fontsize=7.5,
             color=PALETTE["grey"])
  ax2.set_xticks(range(len(rate)))
  ax2.set_xticklabels(rate.index, rotation=45, ha="right", fontsize=8)
  ax2.set_ylabel("fraction with efflux")
  process.style_figure("field_response_by_session")
  fig.tight_layout()
  fig.savefig(out / "field_response_by_session.pdf", dpi=300)
  plt.close(fig)


# --- 3c. endpoint distribution and the block that is not interpreted ---------

try:
  from process import INTENSITY_DIFF_THRESHOLD as INTACT_BAND
except Exception:
  INTACT_BAND = 0.05

# Emit \input-able tables alongside the figures. The alternative is copying
# counts out of the console into the thesis by hand, which is where the
# transcription errors come from.
WRITE_LATEX_TABLES = True
LATEX_SUBFOLDER = "latex"


def endpoint_classes(df: pd.DataFrame) -> pd.DataFrame:
  """Per population and field: how each vesicle ended, and whether it scored.

  Four mutually exclusive outcomes plus the response call, which is a
  separate measurement of the same trace and is therefore reported beside
  them rather than derived from them.
  """
  sub = _field_cells(df).copy()
  sub["intact"] = sub["diff"].abs() <= INTACT_BAND
  sub["lost"] = sub["diff"] < -INTACT_BAND
  sub["gained"] = sub["diff"] > INTACT_BAND
  rows = []
  for (pop, E), cell in sub.groupby(["population", "field_kV_cm"]):
    resp = cell["efflux"].dropna().astype(bool)
    rows.append({"population": pop, "field_kV_cm": E, "n": len(cell),
                 "n_intact": int(cell["intact"].sum()),
                 "n_lost": int(cell["lost"].sum()),
                 "n_gained": int(cell["gained"].sum()),
                 "n_efflux": int(resp.sum()),
                 "frac_intact": float(cell["intact"].mean()),
                 "median_released": float(cell["released"].median()),
                 "interpreted": interpreted(E)})
  out = pd.DataFrame(rows)
  if out.empty:
    return out
  out["population"] = pd.Categorical(out["population"],
                                     [p for p in POP_LABEL], ordered=True)
  return out.sort_values(["population", "field_kV_cm"]).reset_index(drop=True)


def plot_endpoint_distribution(df: pd.DataFrame, out: Path):
  """Where the endpoint change actually sits, and the intact fraction.

  The trajectory figure draws one line per vesicle, so a mode of two hundred
  vesicles at zero renders as a solid block while a dozen extreme traces
  dominate the eye. That reads as two groups whether or not there are two.
  A count axis settles it.
  """
  # Restricted to the interpreted range, as of the 2026-08 update: this is a
  # main-text figure and should agree with plot_field_response about which
  # vesicles are shown, not pool in the session-confounded high-field block
  # unmarked. The right-hand panel already filtered on cls["interpreted"];
  # this brings the left-hand histogram into line with it.
  sub = _field_cells_interpreted(df)
  if sub.empty:
    print("Endpoint distribution skipped: nothing to plot in the "
          "interpreted range.")
    return
  cls = endpoint_classes(df)
  pops = [p for p in POP_LABEL if p in set(sub["population"])]

  fig, (ax1, ax2) = plt.subplots(
      1, 2, figsize=process.fig_size("endpoint_distribution", 11.0, 4.2, 2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
  # Full range, not a percentile: clipping the axis at the 99th percentile
  # would fold the largest releases into the last bin, and those are the
  # vesicles the tail is being drawn to characterise.
  hi = float(np.nanmax(sub["released"]))
  edges = np.arange(-0.10, max(0.15, hi + 0.05), 0.025)
  bottom = np.zeros(len(edges) - 1)
  for pop in pops:
    v = sub[sub["population"] == pop]["released"].clip(edges[0],
                                                       edges[-1] - 1e-9)
    h, _ = np.histogram(v, bins=edges)
    ax1.bar(edges[:-1], h, width=np.diff(edges), bottom=bottom, align="edge",
            color=_colour(pop), lw=0.3, edgecolor="white",
            label=POP_LABEL[pop])
    bottom += h
  ax1.axvspan(-INTACT_BAND, INTACT_BAND, color=PALETTE["pale_blue"],
              alpha=0.55, lw=0, zorder=0)
  # Symlog, not log: the mode is two orders of magnitude above the tail and a
  # linear axis hides the tail entirely, but a plain log axis cannot show the
  # empty bins between the two.
  ax1.set_yscale("symlog", linthresh=10)
  ax1.set_ylim(bottom=0)
  ax1.set_xlabel("released fraction at the matched endpoint")
  ax1.set_ylabel("vesicles")
  ax1.legend(frameon=False, fontsize=8, loc="upper right")

  # Counts are annotated only for the two populations the section compares.
  # The lumenal-only group has three usable fields and its labels land on top
  # of the other two where all three sit at 100%.
  for pop, dy in zip(pops, (9, -14, -14)):
    m = cls[(cls["population"] == pop) & (cls["n"] >= MIN_N_PER_FIELD)
            & cls["interpreted"]]
    if m.empty:
      continue
    ax2.plot(m["field_kV_cm"], 100 * m["frac_intact"], "-o",
             color=_colour(pop), lw=1.8, ms=5, label=POP_LABEL[pop])
    if pop == LUMEN_GROUP:
      # Its labels land on top of the other two where all three sit at 100%,
      # and its three usable fields are already in the table.
      continue
    for _, r in m.iterrows():
      ax2.annotate(f"{int(r['n'])}", (r["field_kV_cm"],
                                      100 * r["frac_intact"]),
                   textcoords="offset points", xytext=(0, dy), ha="center",
                   fontsize=6.5, color=_colour(pop))
  ax2.set_xlabel("field strength (kV/cm)")
  ax2.set_xlim(left=0)
  ax2.set_ylabel(f"vesicles ending within {INTACT_BAND:.0%} (%)")
  ax2.set_ylim(0, 105)
  ax2.legend(frameon=False, fontsize=8, loc="lower left")

  process.style_figure("endpoint_distribution")
  fig.tight_layout()
  fig.savefig(out / "endpoint_distribution.pdf", dpi=300)
  plt.close(fig)
  cls.to_csv(out / "endpoint_classes.csv", index=False)

  tot = _field_cells(df)
  n_in = int((tot["diff"].abs() <= INTACT_BAND).sum())
  n_lo = int((tot["diff"] < -INTACT_BAND).sum())
  n_hi = int((tot["diff"] > INTACT_BAND).sum())
  print(f"\nEndpoint outcome, all {len(tot)} vesicles: {n_in} within "
        f"{INTACT_BAND:.0%}, {n_lo} lost more, {n_hi} gained more.")
  losses = -tot.loc[tot["diff"] < -INTACT_BAND, "diff"]
  for lo_e, hi_e in ((INTACT_BAND, 0.10), (0.10, 0.20), (0.20, 0.40),
                     (0.40, np.inf)):
    n = int(((losses > lo_e) & (losses <= hi_e)).sum())
    print(f"    {lo_e:.2f} to {hi_e:.2f}: {n}")
  print("  A tail that thins away from the band is a graded response; two "
        "separated humps would be a binary one. Read this before describing "
        "the trajectory figure.")


def report_uninterpreted_block(df: pd.DataFrame, out: Path):
  """Why the fields above the cut are excluded, tested rather than asserted.

  Three things could make a block of high fields respond less than the
  fields below it without the field being the cause: too few vesicles to
  see anything, loss of the vesicles that did respond before they were
  scored, or traces those vesicles could not be scored on. Each is checkable
  from the frame already loaded, and each is checked here so the exclusion
  rests on a measurement rather than on the fact that it looks odd.
  """
  if INTERPRETED_MAX_FIELD is None:
    return
  sub = _field_cells(df)
  hi = sub[~sub["field_kV_cm"].apply(interpreted)]
  if hi.empty:
    return
  ref_E = max(E for E in sub["field_kV_cm"].unique() if interpreted(E))
  ref = sub[sub["field_kV_cm"] == ref_E]
  k_ref, n_ref = int(ref["efflux"].sum()), len(ref)
  k_hi, n_hi = int(hi["efflux"].sum()), len(hi)
  p_ref = k_ref / n_ref if n_ref else np.nan

  print(f"\n{'=' * 70}")
  print(f"Fields above {INTERPRETED_MAX_FIELD} kV/cm: why they are excluded")
  print("=" * 70)
  print(f"  responding {k_hi}/{n_hi} against {k_ref}/{n_ref} at {ref_E:.2f} "
        "kV/cm, the highest interpreted field")

  # 1. sampling
  try:
    from scipy.stats import binom
    p_bin = float(binom.cdf(k_hi, n_hi, p_ref))
    print(f"  1. not sampling: at the {ref_E:.2f} kV/cm rate "
          f"({p_ref:.1%}), {n_hi} vesicles would be expected to return "
          f"{p_ref * n_hi:.0f} responders. P(X <= {k_hi}) = {p_bin:.2g}")
  except Exception:
    pass

  # 2. attrition -- vesicles that never reached the response call. df still
  # holds them; _field_cells is what drops them, so the comparison is
  # between the two.
  all_hi = df[~df["field_kV_cm"].apply(interpreted) & df["field_kV_cm"].notna()]
  all_lo = df[df["field_kV_cm"].apply(interpreted) & df["field_kV_cm"].notna()]
  rej_hi = len(all_hi) - len(hi)
  rej_lo = len(all_lo) - len(sub[sub["field_kV_cm"].apply(interpreted)])
  print(f"  2. not attrition among scored vesicles: {rej_hi} of {len(all_hi)} "
        f"tracked vesicles above the cut were dropped before the response "
        f"call ({rej_hi / max(len(all_hi), 1):.1%}) against {rej_lo} of "
        f"{len(all_lo)} below it ({rej_lo / max(len(all_lo), 1):.1%}); "
        f"counting every one as a responder gives "
        f"{k_hi + rej_hi}/{n_hi + rej_hi}.")
  print("     This counts only vesicles that reached tracking. A vesicle "
        "that lysed at the pulse and was never tracked is not in either "
        "denominator; compare the per-chamber yield below.")

  # 3. detectability and geometry, per session
  cols = [c for c in ("radius_um", "prepulse_lumen_abs") if c in df.columns]
  if cols and "session" in df.columns:
    g = df.groupby("session").agg(
        vesicles=("guv_id", "size"), chambers=("experiment", "nunique"),
        **{c: (c, "median") for c in cols})
    g["per_chamber"] = (g["vesicles"] / g["chambers"]).round(1)
    hi_sess = sorted(all_hi["session"].dropna().unique())
    print("  3. session properties (the block's sessions: "
          f"{', '.join(map(str, hi_sess))}):")
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    print("     Higher pre-pulse contrast or a larger radius in the excluded "
          "block would make a response EASIER to see and to induce, so they "
          "cannot explain a lower rate; a lower per-chamber yield would "
          "point at loss before tracking.")

  pd.DataFrame({"metric": ["k_hi", "n_hi", "k_ref", "n_ref", "rejected_hi",
                           "rejected_lo"],
                "value": [k_hi, n_hi, k_ref, n_ref, rej_hi, rej_lo]}).to_csv(
      out / "uninterpreted_block_checks.csv", index=False)


def plot_experiment_level(df: pd.DataFrame, out: Path):
  """One point per EXPERIMENT, not per GUV.

  GUVs in a chamber share a preparation, a field and a day, so pooling them
  overstates n by an order of magnitude. Whatever survives here is what the
  design actually supports.
  """
  # Restricted to the interpreted range, as of the 2026-08 update, matching
  # every other main-text field-strength figure.
  stag = _field_cells_interpreted(df)
  if stag.empty:
    print("Experiment-level plot skipped: no stagnate GUVs in the "
          "interpreted range.")
    return

  per_exp = (stag.groupby(["population", "voltage", "voltage_V", "experiment"])
             .agg(median_released=("released", "median"),
                  frac_efflux=("efflux",
                                   lambda s: s.dropna().mean()),
                  n_guv=("released", "size"),
                  median_radius=("radius_um", "median"))
             .reset_index())
  per_exp.to_csv(out / "per_experiment_summary.csv", index=False)

  volts = sorted(per_exp["voltage_V"].unique())
  x = {v: i for i, v in enumerate(volts)}
  pops = [p for p in POP_LABEL if p in set(per_exp["population"])]
  rng = np.random.default_rng(0)

  fig, ax = plt.subplots(figsize=process.fig_size(
      "per_experiment_release", max(8, 1.0 * len(volts) + 3), 5.0))
  for k, pop in enumerate(pops):
    p_df = per_exp[per_exp["population"] == pop]
    off = (k - (len(pops) - 1) / 2) * 0.3
    xs = [x[v] + off for v in p_df["voltage_V"]]
    xs = xs + rng.uniform(-0.05, 0.05, len(xs))
    ax.scatter(xs, p_df["median_released"], s=44, color=_colour(pop),
               edgecolors="white", linewidth=0.6, zorder=3,
               label=f"{POP_LABEL[pop]} ({len(p_df)} experiments)")
  ax.axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
  ax.set_ylim(bottom=0)
  ax.set_xticks(range(len(volts)))
  # Field strength, matching every other axis in the chapter. The grouping
  # stays on voltage_V because the conversion is one constant and the
  # ordering is identical; only the label changes.
  ax.set_xticklabels([f"{v / ELECTRODE_GAP_CM / 1000:.2f}" for v in volts],
                     rotation=45)
  ax.set_xlabel("Field strength (kV/cm)")
  ax.set_ylabel("experiment median released fraction")
  ax.legend(frameon=False, fontsize=9)
  ax.grid(axis="y", alpha=0.35, linestyle="--")
  process.style_figure("per_experiment_release")
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


# Fate whose vesicles carry a matched-endpoint response call. process.py
# the efflux call is made at ENDPOINT_MATCHED_T_S for this fate only, so it
# is the only one whose responding rate is on a single clock.
FATE_FOR_RESPONSE = "Stagnate"


def check_detectability(df: pd.DataFrame, outputs_root, out: Path):
  """Is one population simply easier to score as responding?

  `efflux` fires when a vesicle's fall from its PRE-pulse mean clears a noise
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
  # Restricted to the interpreted range, as of the 2026-08 update: this
  # check exists to rule out a confound in the SAME comparison the
  # field-response figures make, so it has to describe the same population,
  # not a wider one that includes the session-confounded high-field block.
  n_all_field = len(merged)
  merged = merged[merged["field_kV_cm"].map(interpreted)]
  if n_all_field != len(merged):
    print(f"  Detectability restricted to the interpreted field range: "
          f"{n_all_field - len(merged)} of {n_all_field} dropped.")
  if merged.empty:
    print("Detectability check skipped: nothing left in the interpreted "
          "range.")
    return

  # Stagnate only, matching every other response figure in this module.
  #
  # Two things go wrong without this. The responding RATE printed below is
  # taken from the same `efflux` column the field response uses, but
  # process.py only recomputes that column at the matched endpoint for
  # Stagnate vesicles; Grow, Reduce and Rupture rows keep the whole-record
  # call read straight from _fit_parameters.csv. Pooling the two puts three
  # definitions in one percentage -- matched-endpoint, whole-record, and NaN
  # for the censored -- and the resulting figure disagrees with the
  # field-response panels drawn from the same file.
  #
  # The noise and contrast distributions move with it deliberately. This
  # function asks whether one population is easier to SCORE as responding, so
  # it has to describe the vesicles whose responding rate is being explained,
  # not a wider set that includes vesicles no rate was computed for.
  n_before = len(merged)
  merged = merged[merged["size_category"] == FATE_FOR_RESPONSE]
  if merged.empty:
    print(f"Detectability check skipped: no {FATE_FOR_RESPONSE} GUVs after "
          "the merge.")
    return
  if n_before != len(merged):
    print(f"\n  Detectability restricted to {FATE_FOR_RESPONSE} vesicles: "
          f"{n_before - len(merged)} of {n_before} dropped, {len(merged)} "
          "retained. The responding rate below is on the same vesicles as "
          "the field-response figures.")

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
               merged[merged["population"] == p]["efflux"]
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

  n_panels = len(cols) + 1
  fig, axes = plt.subplots(1, n_panels, figsize=process.fig_size(
      "detectability_by_population", 4.8 * n_panels, 4.6, n_panels))
  axes = np.atleast_1d(axes)
  for ax, col in zip(axes, cols):
    for pop in pops:
      v = np.sort(merged[merged["population"] == pop][col].dropna().to_numpy())
      if len(v):
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                color=_colour(pop), lw=2, label=POP_LABEL[pop])
    ax.set_xlabel(col)
    ax.set_ylabel("cumulative fraction")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
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
    ax.set_ylim(bottom=0)
    ax.set_xlabel("response_noise")
    ax.set_ylabel("response_drop")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.35, linestyle="--")
  process.style_figure("detectability_by_population")
  fig.tight_layout()
  fig.savefig(out / "detectability_by_population.pdf", dpi=300)
  plt.close(fig)


def check_per_experiment_within_fields(df: pd.DataFrame, out: Path):
  """Responding fraction per EXPERIMENT, at each applied field.

  The pooled curve treats every vesicle as independent. Vesicles in one
  chamber share a preparation, a field of view and a day, so a difference
  carried by one chamber is a statement about that chamber. This is the
  check that the difference is not.
  """
  # Restricted to the interpreted range, as of the 2026-08 update, matching
  # every other main-text field-strength figure.
  sub = _field_cells_interpreted(df)
  if sub.empty:
    print("Per-experiment field check skipped: nothing to compare in the "
          "interpreted range.")
    return
  per = (sub.groupby(["field_kV_cm", "population", "experiment"])
         .agg(n_guv=("efflux", "count"),
              frac_efflux=("efflux", lambda x: x.dropna().mean()))
         .reset_index())
  # An experiment contributing one or two vesicles to a field can only report
  # 0, 0.5 or 1 by construction; that is not an estimate of a fraction.
  per = per[per["n_guv"] >= 3]
  if per.empty:
    print("Per-experiment field check skipped: no experiment has >=3 GUVs "
          "at any single field.")
    return
  per.to_csv(out / "per_experiment_within_field.csv", index=False)

  counts = per.groupby(["field_kV_cm", "population"]).size().unstack(fill_value=0)
  pops = [p for p in POP_LABEL if p in counts.columns]
  print("\nResponding fraction per EXPERIMENT, at each field:")
  print("  (a field separates two groups only if BOTH have >=2 experiments "
        "there)")
  for E in sorted(per["field_kV_cm"].unique()):
    have = [p for p in pops if counts.loc[E, p] >= 2]
    mark = ("" if len(have) == len(pops) else
            f"   (comparable here: {', '.join(POP_LABEL[p] for p in have)})"
            if len(have) >= 2 else "   (too few experiments to compare)")
    print(f"  {E:.2f} kV/cm{mark}")
    cell = per[per["field_kV_cm"] == E]
    for pop in pops:
      vals = cell[cell["population"] == pop]
      if vals.empty:
        continue
      fr = ", ".join(f"{v:.2f}(n={int(n)})" for v, n
                     in zip(vals["frac_efflux"], vals["n_guv"]))
      print(f"    {POP_LABEL[pop]:<16} median "
            f"{vals['frac_efflux'].median():.2f}  "
            f"({int((vals['frac_efflux'] > 0).sum())}/{len(vals)} "
            "experiments with any responder)")
      print(f"      {fr}")

  fig, ax = plt.subplots(
      figsize=process.fig_size("per_experiment_within_field", 9.5, 5))
  rng = np.random.default_rng(0)
  fields = sorted(per["field_kV_cm"].unique())
  xpos = {E: i for i, E in enumerate(fields)}
  for k, pop in enumerate(pops):
    m = per[per["population"] == pop]
    off = (k - (len(pops) - 1) / 2) * 0.28
    xs = np.array([xpos[E] for E in m["field_kV_cm"]], float) + off \
        + rng.uniform(-0.05, 0.05, len(m))
    ax.scatter(xs, m["frac_efflux"], s=30 + 3 * m["n_guv"],
               color=_colour(pop), alpha=0.85, edgecolors="white",
               linewidth=0.6,
               label=f"{POP_LABEL[pop]} (1 point = 1 experiment)")
  ax.set_xticks(range(len(fields)))
  ax.set_xticklabels([f"{E:.2f}" for E in fields], fontsize=8)
  ax.set_xlabel("field strength (kV/cm)")
  ax.set_ylabel("fraction with efflux")
  ax.set_ylim(0, 1.05)
  ax.legend(frameon=False, fontsize=9)
  ax.grid(axis="y", alpha=0.35, linestyle="--")
  process.style_figure("per_experiment_within_field")
  fig.tight_layout()
  fig.savefig(out / "per_experiment_within_field.pdf", dpi=300)
  plt.close(fig)


# Radius window the analysis already runs inside, taken from process.py so
# there is one definition. This module used to carry its own copy, which
# silently applied a SECOND, narrower restriction on top of the first once the
# two numbers diverged.
try:
  from process import SIZE_WINDOW_UM as _PROC_WINDOW
except Exception:
  _PROC_WINDOW = None
SIZE_MATCHED_WINDOW = _PROC_WINDOW

# Field at or above which any vesicle responds. Below it both groups sit at
# the assay floor and contribute only zeros, so pooling the whole series
# dilutes the comparison with conditions that cannot separate anything. Read
# the per-field table before trusting the pooled row: this cut is chosen from
# the data, not prespecified.
# Used only if no Bare vesicle responds anywhere in the interpreted range, in
# which case split_field() has nothing to return. Normally the cut comes from
# the data through split_field(), so this file and Section 3 cannot disagree
# about where the series divides.
SIZE_MATCHED_MIN_FIELD = 1.2


def check_size_matched(df: pd.DataFrame, out: Path):
  """Bare vs cortex at matched radius, pooled across fields.

  The dV_m analysis answers the confound by modelling it. This answers it by
  removing it: restrict to vesicles of the same size and the applied field is
  the treatment. The two should agree, and if they do the result does not depend
  on the Schwan expression being right.
  """
  m = df[(df["size_category"] == FATE_FOR_RESPONSE)
         & df["released"].notna() & df["radius_um"].notna()].copy()
  if m.empty:
    print("Size-matched check skipped: no vesicles with a release value.")
    return
  win = ("radius %.2f-%.2f um" % SIZE_MATCHED_WINDOW
         if SIZE_MATCHED_WINDOW else "no size window set")

  print(f"\n{'=' * 70}")
  print(f"Bare vs cortex pooled across fields ({win})")
  print("=" * 70)
  # Uninterpreted fields are dropped here as they are in Section 3. Without
  # this the two blocks described the same vesicles over different field
  # ranges and disagreed above the cut -- Bare 8/58 in this block against 7/35
  # in Section 3 -- while agreeing below it, which is the worst arrangement to
  # read a chapter from.
  m_int = m[m["field_kV_cm"].map(interpreted)]
  print(f"  Interpreted fields only (<= {INTERPRETED_MAX_FIELD} kV/cm): "
        f"{len(m_int)} of {len(m)} vesicles; the CSV holds all of them.")
  pops = [p for p in POP_LABEL if p in set(m_int["population"])]
  for p in pops:
    v = m_int[m_int["population"] == p]
    print(f"  {POP_LABEL[p]:<24} n={len(v):3d}  "
          f"median r={v['radius_um'].median():.2f} um"
          f"  chambers {v['experiment'].nunique()}")

  a = m_int[m_int["population"] == "Bare"]
  b = m_int[m_int["population"] == "Branched, cortex"]
  if len(a) < 5 or len(b) < 5:
    print("  Too few vesicles of both kinds in the window to compare.")
    return
  try:
    from scipy.stats import mannwhitneyu
    p_r = float(mannwhitneyu(a["radius_um"], b["radius_um"]).pvalue)
    d_r = float(a["radius_um"].median() - b["radius_um"].median())
    print(f"\n  Residual radius difference: {d_r:+.2f} um "
          f"(Bare minus cortex), p = {p_r:.3g}.")
    if p_r < 0.05:
      print("    Detectable, so these groups are NOT matched -- only closer "
            "than the unrestricted populations. Report the residual and its "
            "direction rather than claiming a match.")
    if d_r > 0:
      print("    It runs conservatively: Bare vesicles remain the larger, so "
            "they reach the higher induced potential and should porate more "
            "readily. An excess measured in the cortex-bearing group is "
            "obtained against this bias, not because of it.")
  except Exception:
    pass

  # Radius per field, not efflux per field: the response counts are in the
  # Section 3 table, and printing them twice from two frames is what let the
  # two copies drift apart in the first place.
  print(f"\n  Per field strength, radius (cells with >=5 of both):")
  print(f"    {'kV/cm':>6} {'n Bare':>7} {'n cortex':>9}   median radius")
  for E, cell in m_int.groupby(m_int["field_kV_cm"].round(2)):
    va = cell[cell["population"] == "Bare"]
    vb = cell[cell["population"] == "Branched, cortex"]
    if len(va) < 5 or len(vb) < 5:
      continue
    print(f"    {E:6.2f} {len(va):7d} {len(vb):9d}   "
          f"{va['radius_um'].median():.2f} vs "
          f"{vb['radius_um'].median():.2f} um")

  # The responding fractions for this window are reported ONCE, by Section 3
  # below: same vesicles, same interpreted fields, same clustered interval.
  # This block used to print them again from an unfiltered frame, so a reader
  # met two sets of numbers for one comparison and had no way to tell which
  # the chapter quoted. What is left here is what only this block does -- the
  # residual size check above, and the single-session sensitivity below.
  print("\n  Responding fractions for this window: see Section 3 below.")

  cut = split_field(m_int)
  if not np.isfinite(cut):
    cut = SIZE_MATCHED_MIN_FIELD
  al = a[a["field_kV_cm"] < cut]
  bl = b[b["field_kV_cm"] < cut]
  if len(al) >= 5 and len(bl) >= 5:
    # Same cell inside one session. The pooled interval treats vesicles as
    # independent when they are clustered in chambers and days; restricting
    # to the session that supplies most of this range removes the day as an
    # explanation entirely, at the cost of the vesicles it discards. If the
    # difference survives here it does not rest on a between-day change.
    if "session" in al.columns and al["session"].notna().any():
      s = pd.concat([al["session"], bl["session"]]).value_counts().idxmax()
      a_s, b_s = al[al["session"] == s], bl[bl["session"] == s]
      if len(a_s) >= 5 and len(b_s) >= 5:
        _report_matched_cell(
            a_s, b_s,
            f"E < {cut:.2f} kV/cm, session {s} only "
            f"({len(a_s) + len(b_s)} of {len(al) + len(bl)} vesicles)")

  m.to_csv(out / "size_matched_window.csv", index=False)


def _report_matched_cell(a: pd.DataFrame, b: pd.DataFrame, label: str):
  """Responding fraction, mean release and chamber counts for one cell."""
  ka = int((a["efflux"] == True).sum())
  kb = int((b["efflux"] == True).sum())
  print(f"\n  {label}:")
  print(f"    responding   Bare {ka}/{len(a)} ({100 * ka / len(a):.1f}%)"
        f"   cortex {kb}/{len(b)} ({100 * kb / len(b):.1f}%)")
  if "efflux" not in a.columns:
    a = a.assign(efflux=a["efflux"] == True)
    b = b.assign(efflux=b["efflux"] == True)
  d, lo, hi = cluster_diff(a, b)
  if np.isfinite(lo):
    print(f"    difference   {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"  (bootstrap clustered on experiment)")
  else:
    print(f"    difference   {d:+.3f}  -- no interval, too few chambers")

  rng = np.random.default_rng(0)
  for name, v in (("Bare", a["released"].dropna().to_numpy(float)),
                  ("cortex", b["released"].dropna().to_numpy(float))):
    if len(v) < 3:
      continue
    boot = np.mean(v[rng.integers(0, len(v), (4000, len(v)))], axis=1)
    print(f"    mean released {name:<7} {v.mean():.4f} "
          f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")

  # The chamber count is the check that matters most and the one this window
  # is least able to supply. A vesicle-level p built on a handful of chambers
  # is a statement about those chambers.
  for name, v in (("Bare", a), ("cortex", b)):
    per = [(x["efflux"] == True).mean()
           for _, x in v.groupby("experiment") if len(x) >= 3]
    if not per:
      print(f"    {name:<7} no chamber contributes 3+ vesicles here")
      continue
    print(f"    {name:<7} {len(per)} chamber(s) with 3+ vesicles, "
          f"{sum(f > 0 for f in per)} with any responder, "
          f"median {np.median(per):.2f}")


def report_size_distributions_unrestricted(results_dir: Path, out: Path):
  """ECDF of pre-pulse radius for the full tracked population, before
  SIZE_WINDOW_UM is applied.

  report_size_distributions, above, cannot answer this on its own: by the
  time load_summary() hands it a df, process.analysis_population() has
  already restricted to SIZE_WINDOW_UM. guv_bulk_summary.csv is not the fix
  either -- it is not guaranteed to be the pre-restriction population, and
  reading it directly is what the previous version of this function did
  wrong. ALL_RADII_CSV (guv_bulk_all_radii.csv) is what run_bulk.py writes
  specifically for this comparison, before the size window; it is the same
  file report_radius_sensitivity reads, so this function reads it the same
  way rather than opening a second path to the same population.

  Sees the comparison Chapter 2 Results Sec 2 opens with -- Bare vs the full
  protein-loaded population, before any phenotype split. Compares on the raw
  "population" label (Bare / BranchedCortex) precisely so that
  process.EXCLUDE_GROUPS is NOT applied here: those ~126 ambiguous/
  unclassified vesicles are still part of "protein-loaded" for this
  comparison, matching the 497 Bare / 513 protein-loaded counts Results Sec 1
  reports. Check the printed n against those two numbers when this runs --
  they should match exactly, modulo any vesicles with a missing radius_um.
  """
  path = Path(results_dir) / ALL_RADII_CSV
  if not path.exists():
    print(f"  Skipped: {path} not found. run_bulk.py writes it beside "
          "guv_bulk_summary.csv; re-run the intensity stage to produce it.")
    return
  df = pd.read_csv(path)
  df = df[df["radius_um"].notna()]
  if df.empty:
    print("Unrestricted size comparison skipped: no radius_um values.")
    return

  # Bare vs every protein-loaded vesicle, on the RAW population label
  # ("Bare" / "BranchedCortex"), before process.apply_cortex_split. This is
  # deliberately NOT the split table: apply_cortex_split drops
  # process.EXCLUDE_GROUPS (the ambiguous/unclassified ~126 vesicles) before
  # anything downstream sees it, which is correct for a cortex-vs-lumenal
  # comparison but wrong here. Results Sec 2 reports the 6.58 vs ~4.4 um
  # split for the FULL protein-loaded population, all phenotypes and
  # unclassified vesicles pooled (497 Bare / 513 protein-loaded, matching
  # Results Sec 1's tracked counts) -- calling apply_cortex_split first
  # silently shrank the Branched group to 387 and shifted its median, which
  # is why this figure could not be reproduced from the split table.

  bare = df[df["population"] == "Bare"]["radius_um"]
  branched = df[df["population"] != "Bare"]["radius_um"]
  print("\nUnrestricted radius comparison (before SIZE_WINDOW_UM):")
  print(f"  Bare:     n={len(bare)}  median={bare.median():.2f} um")
  print(f"  Branched: n={len(branched)}  median={branched.median():.2f} um")
  if len(bare) < 3 or len(branched) < 3:
    print("  Too few vesicles in one group for a comparison.")
    return

  try:
    from scipy.stats import mannwhitneyu
    p = float(mannwhitneyu(bare, branched, alternative="two-sided").pvalue)
    print(f"  Mann-Whitney U, p = {p:.3g}")
  except Exception:
    print("  scipy unavailable; p-value not computed.")

  fig, ax = plt.subplots(
      figsize=process.fig_size("guv_size_full_population_ecdf", 7.5, 4.8))
  for label, vals, color in (
      ("Bare", bare, PALETTE["grey"]),
      ("Branched (all phenotypes)", branched, PALETTE["dark_blue"])):
    v = np.sort(vals.to_numpy())
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
            color=color, lw=2, label=f"{label} (n={len(v)})")
  window = process.SIZE_WINDOW_UM if process is not None else None
  if window is not None:
    lo, hi = window
    ax.axvspan(lo, hi, color=PALETTE["grey"], alpha=0.12,
               label=f"SIZE_WINDOW_UM ({lo:.1f}-{hi:.1f} um)")
  ax.set_xlabel("radius (um)")
  ax.set_ylabel("cumulative fraction")
  ax.set_xlim(left=0)
  ax.set_ylim(0, 1.02)
  ax.legend(frameon=False)
  ax.grid(alpha=0.35, linestyle="--")
  process.style_figure("guv_size_full_population_ecdf")
  fig.tight_layout()
  fig.savefig(out / "guv_size_full_population_ecdf.pdf", dpi=300)
  plt.close(fig)
  print(f"Saved {out / 'guv_size_full_population_ecdf.pdf'}")


def main(results_dir=None, root=None):
  base = Path(results_dir) if results_dir else outputs_root() / RESULTS_SUBFOLDER
  out = base / SUBFOLDER
  out.mkdir(parents=True, exist_ok=True)
  # The per-experiment CSVs live one level up from the results folder.
  tree = Path(root) if root else base.parent

  df = load_summary(base, attrition_dir=out)
  print(f"Loaded {len(df)} GUV rows from {base / 'guv_bulk_summary.csv'}")
  # Must run on the population BEFORE the restriction load_summary() just
  # applied, so it re-reads guv_bulk_summary.csv itself rather than reusing df.
  report_size_distributions_unrestricted(base, out)
  report_size_distributions(df, out)
  # Runs before the dose figures so that a disagreement between
  # INTERPRETED_MAX_FIELD and the session table is printed above them rather
  # than after they have already been written.
  report_session_confound(df, out)
  report_uninterpreted_block(df, out)
  plot_field_response(df, out)
  plot_field_response_supplementary(df, out)
  plot_endpoint_distribution(df, out)
  check_per_experiment_within_fields(df, out)
  check_detectability(df, tree, out)
  check_size_matched(df, out)
  report_section3(df, out)
  report_radius_sensitivity(base, out)
  plot_experiment_level(df, out)
  print(f"\nSaved susceptibility figures to {out}")
  return df


if __name__ == "__main__":
  main()