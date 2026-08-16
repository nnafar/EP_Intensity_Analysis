import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palette import PALETTE, POPULATION_COLORS

try:
    import config as cfg
except Exception:            
    cfg = None

OUTPUTS_ROOT = Path(getattr(cfg, 'PARENT_OUTPUT_FOLDER', r"D:\Data\EP\Outputs"))

RESULTS_SUBFOLDER = 'Bulk_Analysis_Results'

try:
  from process import (EXCLUDE_EXPERIMENTS, is_excluded_experiment,
                       SIZE_WINDOW_UM)
except Exception as _e:
  print(f"WARNING: could not import EXCLUDE_EXPERIMENTS from process.py "
        f"({type(_e).__name__}: {_e}). No experiments will be excluded from "
        "this report; its counts may not match the figures.")
  EXCLUDE_EXPERIMENTS = []
  SIZE_WINDOW_UM = None

  def is_excluded_experiment(name: str) -> bool:
    return any(re.search(pat, name) for pat in EXCLUDE_EXPERIMENTS)

def _match_analysis_population(out):
  """Keep only rows whose (experiment, guv_id) survives process.analysis_population."""
  path = OUTPUTS_ROOT / RESULTS_SUBFOLDER / 'guv_bulk_summary.csv'
  if not path.exists():
    print(f"WARNING: {path} not found, so this report cannot be matched to "
          "the analysis population. Its counts will not agree with the "
          "other stages.")
    return out
  try:
    import process
    bulk = process.analysis_population(process.apply_cortex_split(
        pd.read_csv(path)), verbose=False)
  except Exception as _e:
    print(f"WARNING: could not apply process.analysis_population "
          f"({type(_e).__name__}: {_e}). Counts here may not match the "
          "other stages.")
    return out
  keys = set(zip(bulk['experiment'].astype(str),
                 bulk['guv_id'].astype('Int64')))
  keep = [(str(e), pd.NA if pd.isna(g) else int(g)) in keys
          for e, g in zip(out['experiment'], out['guv_id'])]
  keep = pd.Series(keep, index=out.index)
  print(f"Analysis population: {int(keep.sum())} of {len(out)} GUV(s) "
        f"retained (matched to guv_bulk_summary.csv).")
  out = out[keep].copy()

  # Take the CALLS from the bulk table too, not just the membership. This
  # report is built from the per-experiment fit files, which carry their own
  # copy of every flag. Filtering rows to the analysis population while
  # leaving those copies in place is how this file came to print a responder
  # count that disagreed with the one in the results table: same vesicles,
  # two columns, two answers. The bulk table is the authority because that is
  # where the matched endpoint is applied.
  cols = [c for c in ('efflux', 'tau_identifiable', 'diff', 'released')
          if c in bulk.columns]
  if cols:
    key_out = list(zip(out['experiment'].astype(str),
                       out['guv_id'].astype('Int64')))
    idx = bulk.set_index([bulk['experiment'].astype(str),
                          bulk['guv_id'].astype('Int64')])
    for c in cols:
      out[c] = [idx[c].get(k, pd.NA) for k in key_out]
  else:
    print("WARNING: guv_bulk_summary.csv carries no efflux column, so this "
          "report is using the fit-file flags. Those are computed on the "
          "whole record and will not agree with the results tables.")
  return out


SE_RATIO_MAX = float(getattr(cfg, 'TAU_SE_RATIO_MAX', 0.5))
TAU_OVER_RECORD_MAX = float(getattr(cfg, 'TAU_MAX_FRACTION_OF_RECORD', 1 / 3))

def population_of(name: str) -> str:
  return 'Empty' if 'empty' in name.lower() else 'BranchedCortex'

EXCLUDE_GROUPS = ['Branched, ambiguous', 'Branched, unclassified']

def cortex_group_of(population: str, status) -> str:
  if population != 'BranchedCortex':
    return 'Bare' if population in ('Empty', 'Bare') else population
  status = str(status)
  if status == 'CORTEX':
    return 'Branched, cortex'
  if status == 'NO_CORTEX':
    return 'Branched, lumenal only'
  if status == 'AMBIGUOUS':
    return 'Branched, ambiguous'
  return 'Branched, unclassified'

def cortex_status_for(exp_dir: Path):
  files = [f for f in exp_dir.glob("**/*_actin_cortex_status.csv")
           if RESULTS_SUBFOLDER not in f.parts]
  if not files:
    return None
  try:
    cx = pd.read_csv(files[0])
    ids = cx['guv_id'].astype(str)
    return {'status': dict(zip(ids, cx.get('cortex_status',
                                           pd.Series(dtype=str))))}
  except Exception:
    return None

def voltage_of(name: str) -> float:
  m = re.search(r'-(\d+)V-', name)
  return float(m.group(1)) if m else np.nan

def load() -> pd.DataFrame:
  frames = []
  n_excluded = 0
  for f in sorted(OUTPUTS_ROOT.glob("**/*_fit_parameters.csv")):
    if f.parent == OUTPUTS_ROOT or RESULTS_SUBFOLDER in f.parts:
      continue
    exp = f.parents[2].name if f.parent.name == 'fitting' else f.parent.name
    if is_excluded_experiment(exp):
      n_excluded += 1
      continue
    df = pd.read_csv(f)
    df['experiment'] = exp
    df['population'] = population_of(exp)
    df['voltage_V'] = voltage_of(exp)

    exp_dir = f.parents[2] if f.parent.name == 'fitting' else f.parents[1]
    cx = cortex_status_for(exp_dir)
    gid = df['guv_id'].astype(str)
    df['cortex_status'] = gid.map(cx['status']) if cx else None
    df['cortex_group'] = [cortex_group_of(p, s_)
                          for p, s_ in zip(df['population'],
                                               df['cortex_status'])]
    frames.append(df)
  if not frames:
    raise SystemExit(f"No *_fit_parameters.csv found under {OUTPUTS_ROOT}")
  if n_excluded:
    print(f"Excluded {n_excluded} experiment(s) via "
          f"process.EXCLUDE_EXPERIMENTS: {EXCLUDE_EXPERIMENTS}")
  out = pd.concat(frames, ignore_index=True)

  # Same size window process.py applies to the bulk table. This report reads
  # the per-experiment fit files directly rather than guv_bulk_summary.csv, so
  # without this it would describe a different population from every other
  # stage and its identifiability percentages would not be comparable to them.
  if SIZE_WINDOW_UM is not None and 'radius_um' in out.columns:
    lo, hi = SIZE_WINDOW_UM
    keep = out['radius_um'].between(lo, hi)
    print(f"Size window {lo}-{hi} um: {int(keep.sum())} of {len(out)} GUV(s) "
          f"retained ({int((~keep).sum())} outside or unmeasured).")
    out = out[keep].copy()

  # Restrict to the vesicles every other stage reports on.
  #
  # This report reads the per-experiment fit files, which carry no fate,
  # endpoint or radius-stability column, so process.analysis_population cannot
  # be applied to them directly. Matching on (experiment, guv_id) against the
  # already-filtered bulk table gets the same population without restating the
  # filter here, where it could drift.
  #
  # Without this the report ran on 363 GUVs while the susceptibility stage ran
  # on fewer, and the identifiability percentages quoted next to a results
  # table did not share its denominator.
  out = _match_analysis_population(out)

  if EXCLUDE_GROUPS:
    drop = out['cortex_group'].isin(EXCLUDE_GROUPS)
    if drop.any():
      print(f"Excluding {int(drop.sum())} unclassifiable GUV(s) "
            f"({drop.mean():.1%}):")
      for g, n in out.loc[drop, 'cortex_group'].value_counts().items():
        print(f"    {g:<24} {n}")
      n_ident = int(out.loc[drop, 'tau_identifiable'].fillna(False).sum())
      if n_ident:
        print(f"    of which {n_ident} had an identifiable tau -- excluding "
              "them removes measurements, not just noise")
      out = out[~drop].copy()
  return out

def summarise(df: pd.DataFrame) -> None:
  n = len(df)
  ident = df['tau_identifiable'].fillna(False).astype(bool)
  # One release call, taken from the bulk table (fall from the PRE-pulse mean
  # at the matched endpoint, against the vesicle's own noise).
  resp = df['efflux'].fillna(False).astype(bool)
  slow = df['tau_over_record'] > TAU_OVER_RECORD_MAX
  loose = df['tau_se_ratio'] > SE_RATIO_MAX
  step = df.get('has_step', pd.Series(False, index=df.index)).fillna(False).astype(bool)

  print(f"\nPooled over {df['experiment'].nunique()} experiment(s), {n} GUV(s)\n")
  print(f"  tau identifiable                     {ident.sum():5d}  ({ident.mean():.1%})")
  print(f"  released dye                         {resp.sum():5d}  ({resp.mean():.1%})")
  print()
  print("  Of the GUVs that failed the gate:")
  failed = ~ident
  if failed.any():
    print(f"    tau > {TAU_OVER_RECORD_MAX:.2f} x record (too slow)   "
          f"{(failed & slow).sum():5d}  ({(failed & slow).sum() / failed.sum():.1%})")
    print(f"    tau_SE/tau > {SE_RATIO_MAX:.2f} (unconstrained) "
          f"{(failed & loose).sum():5d}  ({(failed & loose).sum() / failed.sum():.1%})")
    print(f"    single-frame step (too fast)      "
          f"{(failed & step).sum():5d}  ({(failed & step).sum() / failed.sum():.1%})")
    print(f"    no dye released at all            "
          f"{(failed & ~resp).sum():5d}  ({(failed & ~resp).sum() / failed.sum():.1%})")

  with np.errstate(divide='ignore', invalid='ignore'):
    df = df.assign(record_s=df['tau'] / df['tau_over_record'])
  print("\n  Record length and tau, per population"
        "  (record medians taken over experiments, not GUVs, since a")
  print("  long experiment with many vesicles would otherwise dominate):")
  for pop, sub in df.groupby('cortex_group'
                             if 'cortex_group' in df.columns
                             else 'population'):
    rec = sub.groupby('experiment')['record_s'].median()
    print(f"    {pop:<24} record {rec.median():8.1f} s "
          f"({rec.min():.1f}-{rec.max():.1f})   "
          f"median tau {sub['tau'].median():9.1f} s   "
          f"tau/record {sub['tau_over_record'].median():6.2f}")

  # Magnitude taken from the endpoint difference, not from response_drop.
  # response_drop measures the fall from the first few POST-pulse frames, and
  # onset falls inside the first frame, so it understates the release by
  # whatever left before its baseline was taken.
  print("\n  Release magnitude (fall from pre-pulse mean), releasing GUVs:")
  grp = 'cortex_group' if 'cortex_group' in df.columns else 'population'
  for pop, sub in df[resp].groupby(grp):
    mag = -pd.to_numeric(sub['diff'], errors='coerce')
    print(f"    {pop:<24} median {mag.median():.3f}  "
          f"IQR {mag.quantile(.25):.3f}-{mag.quantile(.75):.3f}  (n={len(sub)})")

def figure(df: pd.DataFrame, out_path: Path) -> None:
  colours = POPULATION_COLORS
  key = 'cortex_group' if 'cortex_group' in df.columns else 'population'
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

  x = df['tau_over_record'].to_numpy(float)
  y = df['tau_se_ratio'].to_numpy(float)
  ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

  ax1.axvspan(TAU_OVER_RECORD_MAX, 1e6, color=PALETTE['light_grey'], alpha=0.45, zorder=0)
  ax1.axhspan(SE_RATIO_MAX, 1e6, color=PALETTE['light_grey'], alpha=0.45, zorder=0)
  for pop, sub in df[ok].groupby(key):
    ax1.scatter(sub['tau_over_record'], sub['tau_se_ratio'], s=14, alpha=0.6,
                edgecolors='none', color=colours.get(pop, PALETTE['grey']), label=pop)
  ax1.axvline(TAU_OVER_RECORD_MAX, color=PALETTE['grey'], lw=0.8, ls='--')
  ax1.axhline(SE_RATIO_MAX, color=PALETTE['grey'], lw=0.8, ls='--')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_xlabel(r'$\tau$ / record length')
  ax1.set_ylabel(r'$\tau_{SE}$ / $\tau$')
  ax1.set_title('Neither constraint is satisfied', fontsize=10)
  ax1.legend(frameon=False, fontsize=8, loc='lower right')
  ax1.text(0.03, 0.95, 'identifiable\nregion', transform=ax1.transAxes,
           fontsize=8, va='top', color=PALETTE['grey'])

  d = df['response_drop'].to_numpy(float)
  nz = df['response_noise'].to_numpy(float)
  ok2 = np.isfinite(d) & np.isfinite(nz) & (nz > 0)
  for pop, sub in df[ok2].groupby(key):
    ax2.scatter(sub['response_noise'], sub['response_drop'], s=14, alpha=0.6,
                edgecolors='none', color=colours.get(pop, PALETTE['grey']), label=pop)
  lim = np.nanpercentile(nz[ok2], 99) if ok2.any() else 1.0
  xs = np.linspace(0, lim * 1.05, 50)
  ax2.plot(xs, 3 * xs, ls='--', lw=0.8, color=PALETTE['dark_red'],
         label=r'3$\sigma$ detection floor')
  ax2.set_xlabel('trace noise (MAD of successive differences)')
  ax2.set_ylabel('observed decline (normalised units)')
  ax2.set_title('Response magnitude is measurable', fontsize=10)
  ax2.legend(frameon=False, fontsize=8)

  fig.tight_layout()
  fig.savefig(out_path, bbox_inches='tight')
  fig.savefig(out_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
  print(f"\nSaved {out_path}")

def main(outputs_root=None, results_dir=None) -> pd.DataFrame:
  global OUTPUTS_ROOT
  if outputs_root is not None:
    OUTPUTS_ROOT = Path(outputs_root)
  base = Path(results_dir) if results_dir else OUTPUTS_ROOT / RESULTS_SUBFOLDER
  results = base / "05_kinetics"
  results.mkdir(parents=True, exist_ok=True)

  data = load()
  summarise(data)
  figure(data, results / 'tau_identifiability.pdf')
  pooled = results / 'tau_identifiability_pooled.csv'
  data.to_csv(pooled, index=False)
  print(f"Saved {pooled}")
  return data

if __name__ == '__main__':
  main()