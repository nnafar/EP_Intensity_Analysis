"""Why tau is not reported: a pooled identifiability diagnostic.

Reads every *_fit_parameters.csv under PARENT_OUTPUT_FOLDER and answers one
question — when the identifiability gate rejects a GUV, WHICH constraint did
it fail?

    tau_over_record > 0.33   the decay never turned over inside the record
                             (efflux slower than the observation window)
    tau_se_ratio    > 0.5    tau is unconstrained by the data
    has_step                 the drop completed between two frames
                             (efflux faster than the sampling interval)

Those first and third causes are opposite, and the distinction matters: the
first says "record too short", the second "sampling too coarse". They imply
different follow-up experiments, so the figure separates them rather than
lumping both under "tau unidentifiable".

Run in Spyder with F5. Writes tau_identifiability.pdf next to the CSVs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palette import PALETTE, POPULATION_COLORS

try:
    import config as cfg
except Exception:            # runnable outside the repo folder too
    cfg = None

OUTPUTS_ROOT = Path(getattr(cfg, 'PARENT_OUTPUT_FOLDER', r"D:\Data\EP\Outputs"))

# Must match process.RESULTS_SUBFOLDER. Not imported from process.py on
# purpose: this report reads the raw fit parameters and applies none of
# process.py's gates, and importing it would invite that coupling.
RESULTS_SUBFOLDER = 'Bulk_Analysis_Results'

# Gate values read from config, not hardcoded, so this figure cannot drift
# away from the gate it is describing.
SE_RATIO_MAX = float(getattr(cfg, 'TAU_SE_RATIO_MAX', 0.5))
TAU_OVER_RECORD_MAX = float(getattr(cfg, 'TAU_MAX_FRACTION_OF_RECORD', 1 / 3))


def population_of(name: str) -> str:
  return 'Empty' if 'empty' in name.lower() else 'BranchedCortex'


def voltage_of(name: str) -> float:
  import re
  m = re.search(r'-(\d+)V-', name)
  return float(m.group(1)) if m else np.nan


def load() -> pd.DataFrame:
  frames = []
  for f in sorted(OUTPUTS_ROOT.glob("**/*_fit_parameters.csv")):
    # Never re-ingest anything this script itself wrote.
    if f.parent == OUTPUTS_ROOT or RESULTS_SUBFOLDER in f.parts:
      continue
    exp = f.parents[2].name if f.parent.name == 'fitting' else f.parent.name
    df = pd.read_csv(f)
    df['experiment'] = exp
    df['population'] = population_of(exp)
    df['voltage_V'] = voltage_of(exp)
    frames.append(df)
  if not frames:
    raise SystemExit(f"No *_fit_parameters.csv found under {OUTPUTS_ROOT}")
  return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> None:
  n = len(df)
  ident = df['tau_identifiable'].fillna(False).astype(bool)
  resp = df['is_responding'].fillna(False).astype(bool)
  slow = df['tau_over_record'] > TAU_OVER_RECORD_MAX
  loose = df['tau_se_ratio'] > SE_RATIO_MAX
  step = df.get('has_step', pd.Series(False, index=df.index)).fillna(False).astype(bool)

  print(f"\nPooled over {df['experiment'].nunique()} experiment(s), {n} GUV(s)\n")
  print(f"  tau identifiable                     {ident.sum():5d}  ({ident.mean():.1%})")
  print(f"  responding (model-free)              {resp.sum():5d}  ({resp.mean():.1%})")
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
    print(f"    not responding at all             "
          f"{(failed & ~resp).sum():5d}  ({(failed & ~resp).sum() / failed.sum():.1%})")

  # Record length is not exported directly; tau_over_record = tau / t_span
  # recovers it, which lets the median tau be quoted in units of the record.
  with np.errstate(divide='ignore', invalid='ignore'):
    df = df.assign(record_s=df['tau'] / df['tau_over_record'])
  print("\n  Record length and tau, per population"
        "  (record medians taken over experiments, not GUVs, since a")
  print("  long experiment with many vesicles would otherwise dominate):")
  for pop, sub in df.groupby('population'):
    rec = sub.groupby('experiment')['record_s'].median()
    print(f"    {pop:<16} record {rec.median():8.1f} s "
          f"({rec.min():.1f}-{rec.max():.1f})   "
          f"median tau {sub['tau'].median():9.1f} s   "
          f"tau/record {sub['tau_over_record'].median():6.2f}")

  print("\n  Response class:")
  for cls, cnt in df['response_class'].value_counts().items():
    print(f"    {cls:<16} {cnt:5d}  ({cnt / n:.1%})")

  print("\n  Model-free response amplitude (response_drop), responding GUVs:")
  for pop, sub in df[resp].groupby('population'):
    print(f"    {pop:<16} median {sub['response_drop'].median():.3f}  "
          f"IQR {sub['response_drop'].quantile(.25):.3f}-"
          f"{sub['response_drop'].quantile(.75):.3f}  (n={len(sub)})")


def figure(df: pd.DataFrame, out_path: Path) -> None:
  colours = POPULATION_COLORS
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

  # --- Panel A: where each GUV sits relative to the two gate constraints ---
  x = df['tau_over_record'].to_numpy(float)
  y = df['tau_se_ratio'].to_numpy(float)
  ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

  ax1.axvspan(TAU_OVER_RECORD_MAX, 1e6, color=PALETTE['light_grey'], alpha=0.45, zorder=0)
  ax1.axhspan(SE_RATIO_MAX, 1e6, color=PALETTE['light_grey'], alpha=0.45, zorder=0)
  for pop, sub in df[ok].groupby('population'):
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

  # --- Panel B: the amplitude IS resolved where the timescale is not -------
  d = df['response_drop'].to_numpy(float)
  nz = df['response_noise'].to_numpy(float)
  ok2 = np.isfinite(d) & np.isfinite(nz) & (nz > 0)
  for pop, sub in df[ok2].groupby('population'):
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
  """Run the whole report. Importable so run_bulk.py can drive it.

  outputs_root : tree of per-experiment folders to READ from.
  results_dir  : where to WRITE. Defaults to outputs_root/RESULTS_SUBFOLDER.
  """
  global OUTPUTS_ROOT
  if outputs_root is not None:
    OUTPUTS_ROOT = Path(outputs_root)
  results = Path(results_dir) if results_dir else OUTPUTS_ROOT / RESULTS_SUBFOLDER
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