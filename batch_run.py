"""Run run_analysis.py across many experiments without editing config.py.

Each experiment runs in its own subprocess with EP_DATA_FOLDER,
EP_EXPERIMENT_BASE_NAME and (optionally) EP_ANALYZE_ACTIN set in the
environment. config.py reads those; when they are absent it falls back to its
hardcoded literals, so single runs are unaffected.

A subprocess per experiment, rather than a loop that mutates cfg, because the
worker pool uses the 'spawn' start method: workers re-import config from disk,
so in-process attribute edits would reach the parent and nothing else. A
subprocess also contains crashes and releases the Windows memmap file locks
between experiments.

Usage (from the repo folder):

    python batch_run.py --dry-run          # list what would run, change nothing
    python batch_run.py                    # run everything not yet analysed
    python batch_run.py --force            # re-run even if outputs exist
    python batch_run.py --include Branched # only matching experiments
    python batch_run.py --exclude 900V

In Spyder:

    %runfile C:/Github/EP_Intensity_Analysis/batch_run.py --wdir --args "--dry-run"
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import config as cfg

# -----------------------------------------------------------------------------
# --- SPYDER SETTINGS ---
# Used when the script is run with no command-line arguments, which is what
# %runfile does by default. Edit these four, press F5, done. Passing --args
# on the command line overrides them.
# -----------------------------------------------------------------------------

DRY_RUN = False   # True = print the plan and stop. Set False to actually run.
INCLUDE = None    # e.g. "BranchedCortex" or "400V"; None = everything
EXCLUDE = None    # e.g. "900V"; None = exclude nothing
FORCE   = True    # True = re-run experiments that already have outputs.
                  # True is required for the first pass after the cortex
                  # exporter was added: previously analysed experiments
                  # have fate CSVs, so already_analysed() would skip the
                  # very runs that need to produce *_actin_cortex_status.csv.

# Root scanned for .nd2 files.
DATA_ROOT = r"D:\Data\EP"

# Directories under DATA_ROOT that are never experiment data.
SKIP_DIRS = {"Outputs", "outputs", "Analysis", "temp", "tmp"}


def discover_experiments(data_root: str):
  """Every .nd2 under data_root, as (data_folder, base_name) pairs."""
  root = Path(data_root)
  if not root.is_dir():
    raise SystemExit(f"DATA_ROOT does not exist: {root}")

  found = []
  for nd2 in sorted(root.glob("**/*.nd2")):
    if any(part in SKIP_DIRS for part in nd2.relative_to(root).parts[:-1]):
      continue
    found.append((str(nd2.parent), nd2.stem))
  return found


def output_folder_for(data_folder: str, base_name: str) -> Path:
  """Mirror config.py's OUTPUT_IMAGE_FOLDER construction."""
  dir_name = os.path.basename(os.path.normpath(data_folder))
  match = re.search(r"^(\d{6})", dir_name)
  yymmdd = match.group(1) if match else "000000"
  return Path(cfg.PARENT_OUTPUT_FOLDER) / f"{yymmdd}_{base_name}"


def circles_json_for(data_folder: str, base_name: str) -> Path:
  return output_folder_for(data_folder, base_name) / f"{base_name}_guv_circles.json"


def already_analysed(data_folder: str, base_name: str) -> bool:
  """True when a fate classification CSV already exists for this experiment."""
  out = output_folder_for(data_folder, base_name)
  return any(out.glob("**/*_guv_fate_classification.csv"))


def actin_expected(base_name: str) -> bool:
  """Same rule config.py applies, reproduced here for the dry-run table."""
  return "empty" not in base_name.lower()


def run_one(data_folder: str, base_name: str, repo_dir: Path) -> int:
  env = os.environ.copy()
  env["EP_DATA_FOLDER"] = data_folder
  env["EP_EXPERIMENT_BASE_NAME"] = base_name
  env.pop("EP_ANALYZE_ACTIN", None)  # let config derive it from the name

  # Guard against a stale variable left in a long-lived Spyder kernel:
  # os.environ.copy() would carry it into every child.

  # Piped and re-printed line by line rather than inherited. Spyder's
  # console is not a real terminal, so a child writing straight to the
  # inherited stdout produces nothing visible until the process exits —
  # hours of apparent silence. flush=True is required for the same reason.
  proc = subprocess.Popen(
      [sys.executable, "-u", str(repo_dir / "run_analysis.py")],
      cwd=str(repo_dir),
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      bufsize=1,
  )
  for line in proc.stdout:
      print(line.rstrip(), flush=True)
  proc.wait()
  return proc.returncode


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--data-root", default=DATA_ROOT)
  ap.add_argument("--include", default=None,
                  help="regex; only experiments whose name matches are run")
  ap.add_argument("--exclude", default=None,
                  help="regex; experiments whose name matches are skipped")
  ap.add_argument("--force", action="store_true",
                  help="re-run experiments that already have outputs")
  ap.add_argument("--dry-run", action="store_true",
                  help="print the plan and exit without running anything")
  # Spyder's %runfile passes no arguments, so fall back to the settings
  # block at the top of this file. A command line still wins.
  argv = sys.argv[1:]
  if not argv:
    argv = []
    if DRY_RUN:
      argv.append("--dry-run")
    if FORCE:
      argv.append("--force")
    if INCLUDE:
      argv += ["--include", INCLUDE]
    if EXCLUDE:
      argv += ["--exclude", EXCLUDE]

  args = ap.parse_args(argv)

  repo_dir = Path(__file__).resolve().parent
  experiments = discover_experiments(args.data_root)

  if args.include:
    pat = re.compile(args.include, re.IGNORECASE)
    experiments = [e for e in experiments if pat.search(e[1])]
  if args.exclude:
    pat = re.compile(args.exclude, re.IGNORECASE)
    experiments = [e for e in experiments if not pat.search(e[1])]

  if not experiments:
    raise SystemExit(f"No .nd2 files matched under {args.data_root}")

  # --- Plan ----------------------------------------------------------------
  plan, skipped_done, missing_circles = [], [], []
  for data_folder, base_name in experiments:
    if not circles_json_for(data_folder, base_name).exists():
      missing_circles.append(base_name)
      continue
    if already_analysed(data_folder, base_name) and not args.force:
      skipped_done.append(base_name)
      continue
    plan.append((data_folder, base_name))

  print(f"\nFound {len(experiments)} ND2 file(s) under {args.data_root}")
  if skipped_done:
    print(f"  {len(skipped_done)} already analysed (use --force to redo)")
  if missing_circles:
    print(f"  {len(missing_circles)} have no GUV circles JSON and CANNOT be")
    print("  batched — run interactive_select.py for each of these first:")
    for name in missing_circles:
      print(f"      {name}")
  print(f"\n{len(plan)} experiment(s) to run:")
  print(f"  {'actin':<7} {'experiment'}")
  for _, base_name in plan:
    print(f"  {str(actin_expected(base_name)):<7} {base_name}")
  print()

  if args.dry_run or not plan:
    return

  # --- Execute -------------------------------------------------------------
  results = []
  for idx, (data_folder, base_name) in enumerate(plan, start=1):
    print(f"\n{'=' * 70}")
    print(f"[{idx}/{len(plan)}] {base_name}")
    print(f"{'=' * 70}\n")
    started = time.time()
    try:
      code = run_one(data_folder, base_name, repo_dir)
    except KeyboardInterrupt:
      print("\nInterrupted.")
      break
    except Exception as e:  # a crash here must not kill the remaining queue
      print(f"  launcher error: {e}")
      code = -1
    results.append((base_name, code, time.time() - started))

  print(f"\n{'=' * 70}\nBatch summary\n{'=' * 70}")
  for base_name, code, elapsed in results:
    status = "ok  " if code == 0 else f"FAIL({code})"
    print(f"  {status}  {elapsed / 60:6.1f} min  {base_name}")
  n_failed = sum(1 for _, code, _ in results if code != 0)
  print(f"\n{len(results) - n_failed} succeeded, {n_failed} failed.")
  if n_failed:
    print("Failed experiments left their logs in their own output folders.")


if __name__ == "__main__":
  main()