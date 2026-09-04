"""Everything to do with the actin cortex after it has been classified.

Two jobs, kept in one file because they answer to each other.

MONTAGES. One panel per GUV classified CORTEX, one per GUV classified
NO_CORTEX (actin present but never polymerised onto the membrane). This is the
visual check on the classifier -- guv_cortex_contrast_histogram.pdf says
whether the threshold sits in a trough, and these say whether the objects
either side of it actually look different. Panels are sorted by
cortex_contrast, ascending; read left to right the montage should show a
progression, and if the two figures look interchangeable at their shared
boundary the threshold is not separating anything real. Requires the raw
ND2s, so DATA_ROOT below must point at them.

BREAKDOWN. What happened to the cortex of those GUVs after the pulse: peak
height lost by the matched endpoint, its time course, the mean radial profile,
and the onset fields fitted to it. These moved here from process.py, which now
keeps only what more than one stage needs -- the population filter, the time
grid, the control voltage and the classification figures.

I_peak is measured from *_actin_evolution.csv, the radial profiles
themselves: one profile per GUV per frame, from which the peak is located,
the background subtracted, and the whole trace divided by that vesicle's own
pre-pulse mean.

It used to be read instead from the cortex_peak columns of
*_actin_cortex_traces.csv, which run_analysis.py has already reduced and
normalised the same way -- (peak - bg) / pre-pulse mean of (peak - bg), on the
same smoothed radial profile. Those stages are gone, and the reason is not
that the trace was wrong. It is that two summaries of one quantity, differing
only in how the peak is located, invite being read as two measurements. Where
they part company is peak DETECTION: the trace requires a prominence of 0.10
and blanks the frame when no distinct peak is found, which censors exactly the
frames where the cortex has broken down enough to stop being a peak. Measuring
the profile directly keeps those frames, at the cost of reporting a lumenal
shoulder where there is no cortex left -- a bias toward finding LESS loss,
which is the safe direction for a breakdown claim.

The profiles are also what the per-experiment *_actin_evolution.png panels
show, so every number here can be checked by eye against them.

In Spyder: open and press F5.
"""

import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palette import ANNOTATION_TEXT, PALETTE

try:
  import config as cfg
except Exception:
  cfg = None

# process.py is a hard dependency now, not an optional one: the breakdown
# stages below run on its population filter, its endpoint constants and its
# figure styling. There is no meaningful fallback -- a montage drawn without
# EXCLUDE_EXPERIMENTS reports a different denominator than every figure it is
# meant to be checked against, and the peak stages cannot run at all.
import process
from process import (CONTROL_VOLTAGE, ENDPOINT_MATCHED_T_S,
                     ENDPOINT_MATCHED_TOLERANCE_S, ENDPOINT_N_FRAMES,
                     EXCLUDE_EXPERIMENTS, FIELD_AXIS_LABEL, MIN_CONTROL_N,
                     RESULTS_SUBFOLDER, SIZE_WINDOW_UM,
                     _peak_time_grid, _voltage_key, _voltage_order,
                     cortex_stage_population, field_label, field_labels,
                     fig_size, is_excluded_experiment, style_figure, sub_dir)

# -----------------------------------------------------------------------------
# --- SPYDER SETTINGS ---
# -----------------------------------------------------------------------------

DATA_ROOT = r"D:\Data\EP"          # tree containing the .nd2 files
OUTPUTS_ROOT = None                # None = config.PARENT_OUTPUT_FOLDER

# Which halves run when this file is executed on its own. run_bulk.py has its
# own toggles and ignores these.
RUN_CORTEX_FIGURES = True          # peak breakdown, time course, evolution
RUN_CROP_MONTAGES = True           # the ND2 crops; slow, needs DATA_ROOT

# --- montages ---
MAX_PANELS = 36                    # per figure; excess GUVs are dropped evenly
N_COLS = 6
CROP_RADII = 1.8                   # half-width of the crop, in GUV radii

# Display scaling. "shared" puts every panel on one intensity scale computed
# from the pooled crops, so a dim vesicle looks dim. "per_panel" rescales each
# crop to its own range, which makes faint structure visible but destroys any
# comparison of brightness between panels -- including the very comparison
# these figures exist to support. Shared is the honest default; use per_panel
# only to inspect morphology, and say so in any caption.
SCALING = "shared"                 # "shared" | "per_panel"
CLIP_PERCENTILES = (1.0, 99.5)

# Frame to show. None = the frame before the pulse index in the filename tag
# (…-frame5 → frame 4), which is always inside the pre-pulse window even when
# the auto-detected pulse frame differs from the tag by a frame or two.
FRAME_OVERRIDE = None

# --- cortex peak ---
#
# Frames averaged at each end of the record for the burst and final values.
PEAK_N_FRAMES = ENDPOINT_N_FRAMES
BLEACH_REFERENCE_PATTERN = rf"-{CONTROL_VOLTAGE}-"

# --- the radial profiles themselves ---
#
# Each row of that file is one radial sample of one GUV in one frame:
# guv_id, frame, time_s, radius_px, radius_um, intensity_raw,
# intensity_smoothed. The profile runs from the tracked centre out to
# PROFILE_EXTENT_RADII times the mean ellipse axis, so the membrane sits at a
# FIXED fraction of each profile's extent (1 / 1.6 = 0.625) whatever the
# vesicle's size, and every window below can be given as a fraction rather
# than in micrometres.
#
# This must match utils.plot_actin_radial_evolution, which sets
# r_max = int(avg_r * 1.6). If that constant changes there, the membrane moves
# in every window here and nothing raises.
PROFILE_EXTENT_RADII = 1.6

# Where I_peak is looked for, as a fraction of the profile extent. The lower
# bound keeps the search off the centre spike visible in several GUVs (a
# handful of pixels averaged over a circumference of almost nothing); the
# upper bound keeps it off the background tail. A cortex peak outside 0.40 to
# 0.85 is not a cortex peak.
PEAK_SEARCH_FRAC = (0.40, 0.85)

# Background: the flat tail outside the vesicle, taken as the median over this
# fraction of the extent outward. 0.90 to 1.0 of the extent is 1.44 to 1.6
# vesicle radii, clear of the membrane by half a radius.
BG_FRAC = 0.90

# Subtract that background from I_peak before normalising.
#
# It matters more than it looks. The camera offset here is around 105 counts
# against a peak of 140, so a raw ratio I_peak/I_peak,pre cannot fall below
# about 0.75 even if the cortex disappears completely -- the floor is set by
# the offset, and it differs between vesicles. Subtracting removes that floor
# (what is left is the lumenal actin level, which is a real signal rather than
# an instrumental one). Set False only to reproduce the trace-based numbers.
PEAK_SUBTRACT_BACKGROUND = True

# Time bins for the mean profile figure, over 0 to the matched endpoint. The
# pre-pulse frames are always drawn as their own curve on top of these.
PROFILE_N_TIME_BINS = 10
PROFILE_N_X = 161                  # radial grid points, 0 to PROFILE_EXTENT_RADII

# Fraction of a condition's GUVs that must still be contributing at a time
# point before the mean is drawn there, so the curves do not thin into
# whichever vesicles were tracked longest.
EVOLUTION_MIN_FRAC = 0.5

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- MONTAGES: WHAT THE CLASSIFIER PUT EITHER SIDE OF THE THRESHOLD ---
# -----------------------------------------------------------------------------


def outputs_root() -> Path:
  if OUTPUTS_ROOT:
    return Path(OUTPUTS_ROOT)
  return Path(cfg.PARENT_OUTPUT_FOLDER)


def index_nd2(data_root: Path) -> dict:
  """Map '{yymmdd}_{stem}' -> path, matching config.py's output folder key.

  Keying on the stem alone is WRONG: several base names occur twice under
  different date folders (e.g. BranchedCortex_Experiment3-400V-500us-frame5),
  so a stem index silently points both output folders at whichever movie was
  found first, and one of them gets crops cut from the wrong field of view.
  The date prefix is what distinguishes them, exactly as it does for the
  output folder name, so the key includes it.
  """
  out, collisions = {}, {}
  for p in data_root.glob("**/*.nd2"):
    if "Outputs" in p.parts:
      continue
    m = re.search(r"^(\d{6})", p.parent.name)
    yymmdd = m.group(1) if m else "000000"
    key = f"{yymmdd}_{p.stem}"
    if key in out:
      collisions.setdefault(key, [out[key]]).append(p)
      continue
    out[key] = p
  for key, paths in collisions.items():
    print(f"  AMBIGUOUS: {len(paths)} movies share the key {key}; "
          "skipping crops for it. Rename one of:")
    for q in paths:
      print(f"      {q}")
    out.pop(key, None)
  return out


def prepulse_frame(base_name: str) -> int:
  if FRAME_OVERRIDE is not None:
    return int(FRAME_OVERRIDE)
  m = re.search(r"frame(\d+)", base_name)
  tag = int(m.group(1)) if m else 5
  return max(0, tag - 1)


def load_actin_frame(nd2_path: Path, frame: int, ch: int) -> np.ndarray:
  """One 2-D actin frame. Single-frame read where the reader supports it."""
  import nd2
  with nd2.ND2File(str(nd2_path)) as f:
    try:
      arr = f.read_frame(frame)            # (C, Y, X) for this timepoint
      return np.asarray(arr[ch] if arr.ndim == 3 else arr, dtype=float)
    except Exception:
      data = f.asarray()                   # fallback: whole stack
      if data.ndim == 4:
        return np.asarray(data[frame, ch], dtype=float)
      return np.asarray(data[frame], dtype=float)


def collect(root: Path, nd2_index: dict) -> pd.DataFrame:
  """One row per classified GUV, with everything needed to cut its crop."""
  rows, n_excluded = [], 0
  for status_csv in sorted(root.glob("**/*_actin_cortex_status.csv")):
    if RESULTS_SUBFOLDER in status_csv.parts:
      continue
    exp_dir = status_csv.parents[1]
    if is_excluded_experiment(exp_dir.name):
      n_excluded += 1
      continue
    base = status_csv.name.replace("_actin_cortex_status.csv", "")

    # exp_dir.name is already '{yymmdd}_{base}', the same key index_nd2 builds.
    nd2_path = nd2_index.get(exp_dir.name)
    if nd2_path is None:
      print(f"  no unambiguous ND2 for {exp_dir.name}; skipped")
      continue

    track_files = list(exp_dir.glob("**/*_tracking_data.csv"))
    if not track_files:
      print(f"  no tracking data for {base}; skipped")
      continue

    status = pd.read_csv(status_csv)
    track = pd.read_csv(track_files[0])
    frame = prepulse_frame(base)

    # Nearest available frame, in case tracking dropped the exact one.
    avail = track["frame"].unique()
    if len(avail) == 0:
      continue
    use_frame = int(avail[np.argmin(np.abs(avail - frame))])
    tf = track[track["frame"] == use_frame].set_index("guv_id")

    for _, r in status.iterrows():
      gid = r["guv_id"]
      if gid not in tf.index:
        continue
      t = tf.loc[gid]
      t = t.iloc[0] if isinstance(t, pd.DataFrame) else t
      voltage = process.extract_voltage(exp_dir.name)
      rows.append({
          "experiment": exp_dir.name,
          "base": base,
          "nd2": nd2_path,
          "frame": use_frame,
          "guv_id": gid,
          "status": str(r.get("cortex_status", "UNKNOWN")),
          "contrast": r.get("cortex_contrast", np.nan),
          "x": float(t["x"]), "y": float(t["y"]), "radius": float(t["radius"]),
          "voltage": voltage,
          "field_kV_cm": process.field_kV_cm(voltage),
      })
  if n_excluded:
    print(f"  excluded {n_excluded} experiment(s) via "
          f"process.EXCLUDE_EXPERIMENTS: {EXCLUDE_EXPERIMENTS}")
  return pd.DataFrame(rows)


def cut_crops(sel: pd.DataFrame, ch: int) -> list:
  """Read each GUV's crop, opening every ND2 once."""
  from collections import Counter
  crops, skipped = [], Counter()
  for nd2_path, grp in sel.groupby("nd2", sort=False):
    for frame, sub in grp.groupby("frame", sort=False):
      try:
        img = load_actin_frame(Path(nd2_path), int(frame), ch)
      except Exception as e:
        print(f"  could not read {Path(nd2_path).name} frame {frame}: {e}")
        continue
      h, w = img.shape[:2]
      for _, r in sub.iterrows():
        if not np.isfinite([r["x"], r["y"], r["radius"]]).all():
          skipped["no tracking position at this frame"] += 1
          continue
        half = int(round(r["radius"] * CROP_RADII))
        if half < 3:
          skipped["tracked radius too small"] += 1
          continue
        x0, x1 = int(round(r["x"])) - half, int(round(r["x"])) + half
        y0, y1 = int(round(r["y"])) - half, int(round(r["y"])) + half
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
          skipped["crop extends outside the frame"] += 1
          continue
        crops.append({**r.to_dict(), "img": img[y0:y1, x0:x1]})

  if skipped:
    print("Crops skipped:")
    for reason, n in sorted(skipped.items()):
      print(f"  {n:4d}  {reason}")
  return crops


def short_tag(experiment: str) -> str:
  """Voltage plus experiment number, e.g. '400V E2'. Falls back to the name."""
  v = re.search(r"-(\d+V)-", experiment)
  e = re.search(r"Experiment(\d+)", experiment)
  if v and e:
    return f"{v.group(1)} E{e.group(1)}"
  return experiment[:14]


def montage(crops: list, title: str, out_path: Path, vmin=None, vmax=None):
  # title is accepted for call-site compatibility but no longer drawn: the
  # figure suptitle it used to fill was removed with every other plot title
  # (2026-08 update). The phenotype and scaling info it carried lives in the
  # output filename and the thesis caption instead.
  if not crops:
    print(f"  nothing to draw for {title}")
    return
  crops = sorted(crops, key=lambda c: (np.inf if pd.isna(c["contrast"])
                                       else c["contrast"]))
  if len(crops) > MAX_PANELS:                 # even thinning, keeps the range
    idx = np.linspace(0, len(crops) - 1, MAX_PANELS).round().astype(int)
    crops = [crops[i] for i in idx]

  n_rows = int(np.ceil(len(crops) / N_COLS))
  # A small top margin only -- the header used to reserve a full inch for a
  # figure-level suptitle; that title is gone (see below), and the per-panel
  # identifier is drawn inside its own axes now, not as an axes title, so it
  # needs no reserved strip either.
  header_in = 0.15
  fig_h = 2.1 * n_rows + header_in
  style_key = out_path.stem
  fig, axes = plt.subplots(
      n_rows, N_COLS,
      figsize=fig_size(style_key, 1.9 * N_COLS, fig_h), squeeze=False)

  for ax in axes.ravel():
    ax.axis("off")

  for i, c in enumerate(crops):
    ax = axes[i // N_COLS][i % N_COLS]
    kw = {} if SCALING == "per_panel" else {"vmin": vmin, "vmax": vmax}
    ax.imshow(c["img"], cmap="gray", interpolation="nearest",
              aspect="equal", **kw)
    # GUV ids restart at 1 in every experiment, so the id alone is ambiguous
    # across a pooled montage; the voltage tag disambiguates it. Drawn as an
    # axes annotation rather than ax.set_title -- panels still need this
    # identifier to be legible as one, it just is not a plot title.
    label = f"{short_tag(c['experiment'])} G{c['guv_id']}"
    if pd.notna(c["contrast"]):
      label += f"  c={c['contrast']:.2f}"
    ax.text(0.5, 1.0, label, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.5, color=PALETTE["grey"])
    ax.axis("off")

  style_figure(style_key)
  # tight_layout packs the rows until the panel labels collide with the row
  # above, so the spacing is set explicitly instead.
  fig.subplots_adjust(top=1 - header_in / fig_h, hspace=0.32, wspace=0.06)
  fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
  fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved {out_path}  ({len(crops)} panels)")


def main(data_root=None, results_dir=None):
  """Build both montages.

  data_root   : tree containing the .nd2 files. None = the DATA_ROOT setting.
  results_dir : where to write. None = outputs_root()/RESULTS_SUBFOLDER.
  """
  base = (Path(results_dir) if results_dir
          else outputs_root() / RESULTS_SUBFOLDER)
  # Must match process.LAYOUT["cortex_class"]; the montages are the visual
  # check on the classification, so they sit with it.
  results = base / "03_cortex_classification" / "montages"
  results.mkdir(parents=True, exist_ok=True)
  root = outputs_root()

  ch = int(getattr(cfg, "ND2_CHANNEL_IDX_ACTIN", 0))
  data_root = Path(data_root or DATA_ROOT)
  if not data_root.is_dir():
    raise SystemExit(f"DATA_ROOT does not exist: {data_root}")

  print(f"Indexing ND2 files under {data_root} ...")
  nd2_index = index_nd2(data_root)
  print(f"  {len(nd2_index)} movie(s) found")

  df = collect(root, nd2_index)
  if df.empty:
    raise SystemExit("No classified GUVs with tracking data found.")
  # Same size window as the rest of the pipeline. The montage reads the
  # per-experiment status and tracking files directly rather than
  # guv_bulk_summary.csv, so without this it would show panels drawn from the
  # whole population while every number in the analysis comes from the
  # window, and the counts printed here would match nothing else.
  #
  # The tracked radius is in PIXELS here, not micrometres -- cut_crops uses it
  # as a pixel offset -- so it has to be converted before it can be compared
  # with a window given in um.
  if SIZE_WINDOW_UM is not None and not df.empty:
    lo, hi = SIZE_WINDOW_UM
    um_per_px = float(getattr(cfg, "MICRONS_PER_PIXEL", 0.11))
    r_um = df["radius"] * um_per_px
    keep = r_um.between(lo, hi)
    print(f"Size window {lo}-{hi} um "
          f"(MICRONS_PER_PIXEL = {um_per_px}): {int(keep.sum())} of "
          f"{len(df)} classified GUV(s) retained.")
    df = df[keep].copy()
    if df.empty:
      raise SystemExit("No classified GUVs left inside the size window. "
                       "Set process.SIZE_WINDOW_UM = None to restore them.")

  print(f"Classified GUVs: {len(df)}")
  print(df["status"].value_counts().to_string())

  # Interpreted field range only, as of the 2026-08 update: fields above
  # config.INTERPRETED_MAX_FIELD are session-confounded (see
  # susceptibility.report_session_confound) and are excluded from every
  # other main-text figure, so the montage -- which is read alongside them
  # as the visual check on the classifier -- should not quietly include
  # vesicles from a field range the rest of the chapter treats as
  # unreadable as a dose response.
  max_field = getattr(cfg, "INTERPRETED_MAX_FIELD", None)
  if max_field is not None:
    n_before = len(df)
    df = df[df["field_kV_cm"] <= max_field].copy()
    print(f"Interpreted field range (<= {max_field} kV/cm): {len(df)} of "
          f"{n_before} classified GUV(s) retained.")
    if df.empty:
      raise SystemExit("No classified GUVs left inside the interpreted "
                       "field range. Set config.INTERPRETED_MAX_FIELD = "
                       "None to restore them.")

  wanted = df[df["status"].isin(["CORTEX", "NO_CORTEX"])]
  crops = cut_crops(wanted, ch)
  if not crops:
    raise SystemExit("No crops could be cut -- check DATA_ROOT and the "
                     "actin channel index.")

  # One display scale for BOTH figures, so the cortex and no-cortex montages
  # can be compared with each other and not just internally.
  vmin = vmax = None
  if SCALING == "shared":
    pooled = np.concatenate([c["img"].ravel() for c in crops])
    vmin, vmax = np.percentile(pooled, CLIP_PERCENTILES)
    print(f"Shared display range: {vmin:.0f}-{vmax:.0f} "
          f"(percentiles {CLIP_PERCENTILES})")

  for status, name, label in (
      ("CORTEX", "guv_actin_montage_cortex.pdf", "with a cortex"),
      ("NO_CORTEX", "guv_actin_montage_lumenal_only.pdf",
       "lumenal actin only, no cortex")):
    sel = [c for c in crops if c["status"] == status]
    montage(sel, f"GUVs {label}", results / name, vmin, vmax)


# -----------------------------------------------------------------------------
# --- CORTEX PEAK FROM THE RADIAL PROFILES ---
#
# I_peak is measured from *_actin_evolution.csv, the radial profiles
# themselves, rather than read from the reduced cortex_peak columns of
# *_actin_cortex_traces.csv.
#
# Three things that buys:
#
#   The background is removed. A raw peak carries the camera offset, and a
#   ratio built on it cannot fall past that offset however much cortex is
#   lost -- so "complete loss" and "half the loss" sit closer together than
#   they should, by an amount that differs per vesicle.
#
#   The pre-pulse baseline is formed from the frames before the pulse, rather
#   than inherited from whatever normalisation the trace export applied.
#
#   Vesicles with tracking gaps are kept. Traces are aligned on time_s, so a
#   GUV that lost two frames mid-record contributes the frames it has instead
#   of being dropped for having the wrong length.
#
# What it does NOT do is apply a bleach correction. The 30 V reference is
# drawn as its own curve and left to be read against the others; dividing by
# it is a decision the reference cannot currently support.
# -----------------------------------------------------------------------------

_EVOLUTION_ATTRITION = {}


def _profile_peak(r: np.ndarray, y: np.ndarray):
  """I_peak, its position, and the background, from one radial profile.

  Returns (i_peak, r_peak_frac, i_bg), all NaN if the profile is unusable.
  Positions are fractions of the profile extent, so 0.625 is the membrane.
  """
  ok = np.isfinite(r) & np.isfinite(y)
  if ok.sum() < 8:
    return np.nan, np.nan, np.nan
  r, y = r[ok], y[ok]
  order = np.argsort(r)
  r, y = r[order], y[order]
  extent = float(r[-1])
  if not np.isfinite(extent) or extent <= 0:
    return np.nan, np.nan, np.nan

  frac = r / extent
  lo, hi = PEAK_SEARCH_FRAC
  win = (frac >= lo) & (frac <= hi)
  bg_win = frac >= BG_FRAC
  if win.sum() < 3 or bg_win.sum() < 2:
    return np.nan, np.nan, np.nan

  i = int(np.argmax(y[win]))
  return float(y[win][i]), float(frac[win][i]), float(np.median(y[bg_win]))


def _evolution_csv(exp_dir: Path):
  files = [f for f in exp_dir.glob("**/*_actin_evolution.csv")
           if RESULTS_SUBFOLDER not in f.parts]
  return files[0] if files else None


def collect_evolution_peaks(df: pd.DataFrame, outputs_root: str):
  """I_peak traces and mean radial profiles, from *_actin_evolution.csv.

  Population is cortex_stage_population: cortex-bearing, Stagnate and
  radius-stable, the same vesicles the trace-based stages use. Each GUV is
  normalised to its own pre-pulse mean peak, so what comes back is fold change
  and vesicles of different brightness can be averaged.

  Returns (traces, profiles, grid_x, edges), where

      traces    list of dicts, one per GUV, with t and y arrays
      profiles  {(voltage, bin_index): [sum, count, {guv keys}]} over grid_x,
                bin -1 being the pre-pulse frames. The GUV set is what the
                coverage gate in plot_mean_actin_evolution is applied to:
                counting profiles would let one long-tracked vesicle stand in
                for a whole condition.
      grid_x    radial grid in vesicle radii, 0 to PROFILE_EXTENT_RADII
      edges     time bin edges, 0 to the matched endpoint
  """
  root = Path(outputs_root)
  keep = cortex_stage_population(df)
  _EVOLUTION_ATTRITION.clear()
  if keep.empty:
    return [], {}, None, None
  # Interpreted field range only, as of the 2026-08 update: fields above
  # config.INTERPRETED_MAX_FIELD are session-confounded (see
  # susceptibility.report_session_confound) and are excluded from every
  # other main-text figure. Filtered here rather than in each consumer
  # (plot_mean_actin_evolution, plot_evolution_peak_timecourse,
  # evolution_peak_metrics -> plot_cortex_peak_breakdown /
  # report_cortex_size_dependence) since all four read traces/profiles built
  # from this same wanted set -- restricting once here is what keeps them
  # from drifting apart.
  keep = keep.copy()
  keep["field_kV_cm"] = keep["voltage"].map(process.field_kV_cm)
  max_field = getattr(cfg, "INTERPRETED_MAX_FIELD", None)
  if max_field is not None:
    n_before = len(keep)
    keep = keep[keep["field_kV_cm"] <= max_field]
    print(f"Cortex-stage population restricted to the interpreted field "
          f"range (<= {max_field} kV/cm): {len(keep)} of {n_before} "
          "retained.")
    if keep.empty:
      return [], {}, None, None
  wanted = {(r["experiment"], str(r["guv_id"])): r["voltage"]
            for _, r in keep.iterrows()}

  t_end = (float(ENDPOINT_MATCHED_T_S) if ENDPOINT_MATCHED_T_S is not None
           else np.inf)
  grid_x = np.linspace(0.0, PROFILE_EXTENT_RADII, PROFILE_N_X)
  edges = None if not np.isfinite(t_end) else np.linspace(
      0.0, t_end, PROFILE_N_TIME_BINS + 1)

  traces, profiles, seen_t_max = [], {}, 0.0
  n_files = 0
  for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    if exp_dir.name == RESULTS_SUBFOLDER or is_excluded_experiment(exp_dir.name):
      continue
    path = _evolution_csv(exp_dir)
    if path is None:
      if any(k[0] == exp_dir.name for k in wanted):
        _EVOLUTION_ATTRITION.setdefault("no radial profile export", 0)
        _EVOLUTION_ATTRITION["no radial profile export"] += sum(
            1 for k in wanted if k[0] == exp_dir.name)
      continue
    try:
      ev = pd.read_csv(path)
    except Exception as e:
      print(f"  could not read {path.name}: {e}")
      continue
    need = {"guv_id", "time_s", "radius_um", "intensity_smoothed"}
    if not need.issubset(ev.columns):
      print(f"  {path.name} is missing {sorted(need - set(ev.columns))}; "
            "skipped")
      continue
    n_files += 1
    ev["guv_id"] = ev["guv_id"].astype(str)

    for gid, g in ev.groupby("guv_id", sort=False):
      key = (exp_dir.name, gid)
      if key not in wanted:
        continue

      # One profile per frame. Grouping on time_s rather than frame keeps the
      # two in step even where the frame column has gaps.
      frames = []
      for t_s, fr in g.groupby("time_s", sort=True):
        r = fr["radius_um"].to_numpy(float)
        y = fr["intensity_smoothed"].to_numpy(float)
        i_peak, r_frac, i_bg = _profile_peak(r, y)
        if not np.isfinite(i_peak):
          continue
        frames.append((float(t_s), i_peak, r_frac, i_bg, r, y))
      if not frames:
        _EVOLUTION_ATTRITION["no usable profile"] = (
            _EVOLUTION_ATTRITION.get("no usable profile", 0) + 1)
        continue

      def corrected(i_peak, i_bg):
        return i_peak - i_bg if PEAK_SUBTRACT_BACKGROUND else i_peak

      pre = [corrected(f[1], f[3]) for f in frames if f[0] < 0]
      if not pre:
        _EVOLUTION_ATTRITION["no pre-pulse frame"] = (
            _EVOLUTION_ATTRITION.get("no pre-pulse frame", 0) + 1)
        continue
      i_pre = float(np.mean(pre))
      if not np.isfinite(i_pre) or i_pre <= 0:
        _EVOLUTION_ATTRITION["pre-pulse peak not positive"] = (
            _EVOLUTION_ATTRITION.get("pre-pulse peak not positive", 0) + 1)
        continue

      volt = wanted[key]
      t_post = np.array([f[0] for f in frames if 0.0 <= f[0] <= t_end])
      y_post = np.array([corrected(f[1], f[3]) / i_pre for f in frames
                         if 0.0 <= f[0] <= t_end])
      if t_post.size < 4:
        _EVOLUTION_ATTRITION["fewer than 4 post-pulse frames"] = (
            _EVOLUTION_ATTRITION.get("fewer than 4 post-pulse frames", 0) + 1)
        continue
      seen_t_max = max(seen_t_max, float(t_post.max()))
      traces.append({
          "experiment": exp_dir.name, "guv_id": gid, "voltage": volt,
          "i_peak_pre": i_pre,
          "r_peak_frac_pre": float(np.median([f[2] for f in frames
                                              if f[0] < 0])),
          "n_frames": int(t_post.size),
          "t": t_post, "y": y_post,
      })

      # Profiles, on a common radius axis. Dividing by each profile's own
      # extent puts the membrane at 1.0 for every vesicle; multiplying by
      # PROFILE_EXTENT_RADII restores the axis to vesicle radii.
      for t_s, i_peak, _r_frac, i_bg, r, y in frames:
        if t_s > t_end:
          continue
        b = -1 if t_s < 0 else int(np.clip(np.searchsorted(
            edges, t_s, side="right") - 1, 0, PROFILE_N_TIME_BINS - 1))
        ok = np.isfinite(r) & np.isfinite(y)
        if ok.sum() < 8:
          continue
        r_ok, y_ok = r[ok], y[ok]
        order = np.argsort(r_ok)
        r_ok, y_ok = r_ok[order], y_ok[order]
        extent = float(r_ok[-1])
        if extent <= 0:
          continue
        x = r_ok / extent * PROFILE_EXTENT_RADII
        v = (y_ok - i_bg if PEAK_SUBTRACT_BACKGROUND else y_ok) / i_pre
        prof = np.interp(grid_x, x, v, left=np.nan, right=np.nan)
        acc = profiles.setdefault((volt, b),
                                  [np.zeros_like(grid_x),
                                   np.zeros_like(grid_x), set()])
        good = np.isfinite(prof)
        acc[0][good] += prof[good]
        acc[1][good] += 1.0
        acc[2].add(key)

  if _EVOLUTION_ATTRITION:
    print("\nRadial profile peaks, vesicles not measured:")
    for reason, n in sorted(_EVOLUTION_ATTRITION.items()):
      print(f"  {n:4d}  {reason}")
  print(f"Radial profiles read from {n_files} experiment(s); "
        f"{len(traces)} of {len(wanted)} cortex-stage GUV(s) measured, "
        f"records reaching {seen_t_max:.0f} s.")
  if traces:
    fr = np.array([tr["r_peak_frac_pre"] for tr in traces])
    fr = fr[np.isfinite(fr)]
    lo, hi = PEAK_SEARCH_FRAC
    n_edge = int(((fr <= lo + 0.02) | (fr >= hi - 0.02)).sum())
    print(f"  pre-pulse peak position, fraction of profile extent: "
          f"{np.percentile(fr, 25):.3f} / {np.median(fr):.3f} / "
          f"{np.percentile(fr, 75):.3f} (membrane sits at "
          f"{1.0 / PROFILE_EXTENT_RADII:.3f})")
    if n_edge:
      print(f"  WARNING: {n_edge} GUV(s) peak within 0.02 of the "
            f"PEAK_SEARCH_FRAC bounds {PEAK_SEARCH_FRAC}, so the window may "
            "be clipping the peak rather than containing it.")
  return traces, profiles, grid_x, edges


def plot_representative_radial_profiles(df_all: pd.DataFrame, outputs_root,
                                         output_dir):
  """One representative pre-pulse radial actin profile per phenotype class.

  Answers what the classifier in guv_analysis_utils.classify_cortex_presence
  actually saw: CORTEX, NO_CORTEX and AMBIGUOUS vesicles, each represented by
  one pre-pulse radial profile, on a common radius axis in VESICLE RADII
  (matching collect_evolution_peaks' grid_x, not the raw 0-1 fraction
  _profile_peak uses internally for PEAK_SEARCH_FRAC) -- so the membrane
  sits at x = 1.0 regardless of vesicle size, the same convention
  plot_mean_actin_evolution's reference line uses.

  Deliberately NOT built from collect_evolution_peaks / cortex_stage_
  population: those restrict to cortex-bearing GUVs only, and this figure
  needs all three phenotypes. The population here is Stagnate and phenotyped
  (CORTEX, NO_CORTEX or AMBIGUOUS), full stop -- not size-windowed and not
  restricted to cortex-bearing, matching the rest of this module's stance
  that the actin channel is analysed on its own population.

  "Representative" means the vesicle whose pre-pulse cortex_contrast sits
  closest to its class's own median. Each profile is the MEAN of that
  vesicle's own pre-pulse frames only (classification is pre-pulse-only,
  deliberately -- see export_actin_cortex_status_csv in
  guv_analysis_utils.py), interpolated onto a common radial grid, and
  normalised to its own innermost (centre-most) sample as a simple proxy for
  the lumenal baseline so the three curves are visually comparable
  regardless of absolute brightness. This is NOT the same quantity as
  prepulse_lumen_abs (which is measured from a masked inner region, not the
  radial profile) -- close enough to plot, not a substitute for the
  classifier's own number in cortex_contrast.

  Restricted to the interpreted field range (<= config.INTERPRETED_MAX_FIELD),
  as of the 2026-08 update, matching every other main-text figure -- the
  chosen "representative" vesicle should not come from the session-confounded
  high-field block. df_all does not carry field_kV_cm (see
  process.plot_representative_intensity_traces for why); derived locally the
  same way.
  """
  root = Path(outputs_root)
  keep = df_all[(df_all["size_category"] == "Stagnate")
               & (df_all["cortex_status"].isin(
                   ["CORTEX", "NO_CORTEX", "AMBIGUOUS"]))].copy()
  keep["field_kV_cm"] = keep["voltage"].map(process.field_kV_cm)
  max_field = getattr(cfg, "INTERPRETED_MAX_FIELD", None)
  if max_field is not None:
    keep = keep[keep["field_kV_cm"] <= max_field]
  if keep.empty:
    print("Representative radial profiles skipped: no phenotyped vesicles "
          "in the interpreted range.")
    return

  chosen = {}
  for status in ("CORTEX", "NO_CORTEX", "AMBIGUOUS"):
    cell = keep[keep["cortex_status"] == status]
    if cell.empty:
      print(f"  Representative radial profiles: no {status} vesicle "
            "available.")
      continue
    med = cell["cortex_contrast"].median()
    row = cell.loc[(cell["cortex_contrast"] - med).abs().idxmin()]
    chosen[status] = (str(row["experiment"]), str(row["guv_id"]))

  if not chosen:
    print("Representative radial profiles skipped: no class had a "
          "usable example.")
    return

  grid_x = np.linspace(0.0, PROFILE_EXTENT_RADII, PROFILE_N_X)
  labels = {"CORTEX": "cortex", "NO_CORTEX": "lumenal only",
           "AMBIGUOUS": "ambiguous"}
  # Matched to POPULATION_COLORS / GROUP_COLORS (palette.py) -- the scheme
  # every dye-release and susceptibility figure in the thesis already uses
  # for these same three groups ('Branched, cortex' / 'Branched, lumenal
  # only' / 'Branched, ambiguous'), not CORTEX_STATUS_COLORS, which is a
  # separate palette for the classification-QC histograms and uses
  # medium_red for NO_CORTEX where the rest of the thesis uses light_blue.
  colours = {"CORTEX": PALETTE["dark_blue"], "NO_CORTEX": PALETTE["light_blue"],
            "AMBIGUOUS": PALETTE["light_grey"]}

  fig, ax = plt.subplots(
      figsize=fig_size("guv_representative_radial_profiles", 5.6, 5.2))
  for status, (exp_name, gid) in chosen.items():
    path = _evolution_csv(root / exp_name)
    if path is None:
      print(f"  {exp_name}: no *_actin_evolution.csv found; {status} "
            "example skipped")
      continue
    try:
      ev = pd.read_csv(path)
    except Exception as e:
      print(f"  could not read {path.name}: {e}")
      continue
    need = {"guv_id", "time_s", "radius_um", "intensity_smoothed"}
    if not need.issubset(ev.columns):
      print(f"  {path.name} is missing {sorted(need - set(ev.columns))}; "
            f"{status} example skipped")
      continue
    ev["guv_id"] = ev["guv_id"].astype(str)
    g = ev[(ev["guv_id"] == gid) & (ev["time_s"] < 0)]
    if g.empty:
      print(f"  {exp_name} GUV {gid}: no pre-pulse frames; {status} "
            "example skipped")
      continue

    curves = []
    for t_s, fr in g.groupby("time_s"):
      r = fr["radius_um"].to_numpy(float)
      y = fr["intensity_smoothed"].to_numpy(float)
      ok = np.isfinite(r) & np.isfinite(y)
      if ok.sum() < 8:
        continue
      r_ok, y_ok = r[ok], y[ok]
      order = np.argsort(r_ok)
      r_ok, y_ok = r_ok[order], y_ok[order]
      extent = float(r_ok[-1])
      if extent <= 0:
        continue
      x = r_ok / extent * PROFILE_EXTENT_RADII
      curves.append(np.interp(grid_x, x, y_ok, left=np.nan, right=np.nan))
    if not curves:
      print(f"  {exp_name} GUV {gid}: no usable pre-pulse profile; "
            f"{status} example skipped")
      continue
    mean_curve = np.nanmean(np.vstack(curves), axis=0)

    finite = np.where(np.isfinite(mean_curve))[0]
    if finite.size == 0:
      print(f"  {exp_name} GUV {gid}: profile is all-NaN; {status} "
            "example skipped")
      continue
    baseline = mean_curve[finite[0]]
    if not np.isfinite(baseline) or baseline == 0:
      baseline = 1.0
    ax.plot(grid_x, mean_curve / baseline, color=colours[status], lw=2.2,
            label=labels[status], zorder=3, solid_capstyle="round")

  ax.axvline(1.0, color=PALETTE["grey"], lw=0.8, ls=":", zorder=1)
  ax.text(1.0, 0.97, "membrane", fontsize=7.5,
          color=PALETTE["grey"], ha="center", va="top",
          transform=ax.get_xaxis_transform())
  # Not forced to 0: this is intensity normalised to each vesicle's own
  # centre-most sample, not a fraction or a count, so 0 is not the floor the
  # data actually lives near -- forcing it would waste most of the plot on
  # empty space below the real range. Autoscaled with a little headroom
  # instead.
  ax.margins(y=0.10)
  ax.set_xlim(0, PROFILE_EXTENT_RADII)
  ax.set_xlabel("radius / vesicle radius", fontsize=10)
  ax.set_ylabel("actin intensity, normalized to centre-most sample",
                fontsize=10)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(alpha=0.25, linestyle="--", lw=0.6)
  ax.tick_params(labelsize=8.5)
  ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1.6,
            handletextpad=0.6, borderaxespad=0.3)
  style_figure("guv_representative_radial_profiles")
  fig.tight_layout()
  out_path = (sub_dir(output_dir, "representative_traces")
             / "guv_representative_radial_profiles.pdf")
  fig.savefig(out_path, dpi=300)
  plt.close(fig)
  print(f"Saved {out_path}")

def plot_mean_actin_evolution(
    traces, 
    profiles, 
    grid_x, 
    edges, 
    output_dir,
    min_frac: float = EVOLUTION_MIN_FRAC,
    font_size: int = 15,
    tick_size: int = 15,
    annot_size: int = 15,
    legend_size: int = 15
):
  """Mean radial profile evolution, one panel per field strength.

  The pooled version of the per-experiment *_actin_evolution.png panels: every
  cortex-bearing GUV in a condition, normalised to its own pre-pulse peak and
  averaged within time bins. Light grey is early, black is late.

  A bin is drawn only where at least min_frac of the condition's vesicles are
  still contributing to it. Records here run either ~431 s or ~721 s and the
  short ones are not censored, so without the gate a late bin would be an
  average over whichever chambers happened to record longer -- the
  record-length confound, moved out of the endpoint and into the bins. Gated
  bins are dropped from the figure and written to the CSV as NaN with their
  vesicle count, so a missing curve is distinguishable from a flat one.
  """
  out_path = sub_dir(output_dir, "cortex_breakdown")
  if not profiles:
    print("Mean actin evolution skipped: no radial profiles.")
    return

  voltages = sorted({v for v, _ in profiles}, key=_voltage_key)
  n_by_volt = pd.Series([tr["voltage"] for tr in traces]).value_counts()
  n_cols = min(3, len(voltages))
  n_rows = int(np.ceil(len(voltages) / n_cols))
  
  fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True,
                           sharey=True, layout="constrained",
                           figsize=fig_size("actin_evolution_mean",
                                            4.4 * n_cols, 3.9 * n_rows,
                                            len(voltages)))
  greys = mcolors.LinearSegmentedColormap.from_list(
      "evo_greys", plt.cm.Greys(np.linspace(0.25, 1.0, 256)))
  t_mid = 0.5 * (edges[:-1] + edges[1:])
  norm = mcolors.Normalize(vmin=float(edges[0]), vmax=float(edges[-1]))

  def enough(volt, acc):
    """Vesicle coverage gate for one bin."""
    need = max(2, int(np.ceil(min_frac * int(n_by_volt.get(volt, 0)))))
    return len(acc[2]) >= need, need

  for ax in axes.ravel():
    ax.set_visible(False)
    
  dropped = []
  for k, volt in enumerate(voltages):
    ax = axes[k // n_cols][k % n_cols]
    ax.set_visible(True)
    for b in range(PROFILE_N_TIME_BINS):
      acc = profiles.get((volt, b))
      if acc is None:
        continue
      ok, need = enough(volt, acc)
      if not ok:
        dropped.append((volt, float(t_mid[b]), len(acc[2]), need))
        continue
      s, n = acc[0], acc[1]
      mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
      ax.plot(grid_x, mean, color=greys(norm(t_mid[b])), lw=1.1)
      
    pre = profiles.get((volt, -1))
    if pre is not None:
      s, n = pre[0], pre[1]
      mean_pre = np.where(n > 0, s / np.maximum(n, 1), np.nan)
      ax.plot(grid_x, mean_pre, color=PALETTE["dark_red"], lw=1.6, ls="--",
              zorder=6, label="pre-pulse")
              
    ax.axvline(1.0, color=ANNOTATION_TEXT, lw=0.8, ls=":", zorder=1)
    # Moved the text inside the plot to the bottom left corner
    ax.text(0.05, 0.05, f"{field_label(volt)} kV/cm "
            f"(n={int(n_by_volt.get(volt, 0))})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=annot_size)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="both", labelsize=tick_size)
    
    if k // n_cols == n_rows - 1:
      ax.set_xlabel("radius / vesicle radius", fontsize=font_size)
      
  axes[0][0].legend(frameon=False, fontsize=legend_size, loc="upper left")
  
  # Set a single, global y-axis label for the entire figure
  fig.supylabel("actin intensity / pre-pulse peak", fontsize=font_size)

  sm = plt.cm.ScalarMappable(norm=norm, cmap=greys)
  
  n_filled = len(voltages)
  empty_axes = axes.ravel()[n_filled:]
  
  if len(empty_axes) > 0:
    gs = empty_axes[0].get_subplotspec().get_gridspec()
    for ax in empty_axes:
      ax.remove()
    row = n_filled // n_cols
    col = n_filled % n_cols
    
    anchor_ax = fig.add_subplot(gs[row, col:])
    anchor_ax.axis("off")
    
    cax = anchor_ax.inset_axes([0.05, 0.4, 0.9, 0.15])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
  else:
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.05)

  cbar.set_label("time after pulse (s)", fontsize=font_size)
  cbar.ax.tick_params(labelsize=tick_size)

  pdf_path = out_path / "guv_actin_evolution_mean.pdf"
  fig.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved mean actin evolution: {pdf_path}")
  
  if dropped:
    print(f"  Time bins blanked for coverage below {min_frac:.0%} of the "
          "condition's vesicles:")
    for volt, t_c, n_have, n_need in dropped:
      print(f"    {volt:>5}  bin centred {t_c:6.0f} s: {n_have} GUV(s), "
            f"{n_need} needed")

  prof_rows = []
  for (volt, b), acc in sorted(profiles.items(),
                               key=lambda kv: (_voltage_key(kv[0][0]),
                                               kv[0][1])):
    s_arr, n = acc[0], acc[1]
    mean = np.where(n > 0, s_arr / np.maximum(n, 1), np.nan)
    if b >= 0 and not enough(volt, acc)[0]:
      mean = np.full_like(grid_x, np.nan)
    label = "pre-pulse" if b < 0 else f"{t_mid[b]:.0f}"
    prof_rows.append(pd.DataFrame({
        "voltage": volt, "t_bin_mid_s": label, "r_over_R": grid_x,
        "mean_intensity": mean, "n_profiles": n.astype(int),
        "n_guv": len(acc[2])}))
  pd.concat(prof_rows, ignore_index=True).to_csv(
      out_path / "guv_actin_evolution_mean_profiles.csv", index=False)

def plot_evolution_peak_timecourse(traces, per_guv, output_dir,
                                   min_frac: float = EVOLUTION_MIN_FRAC):
  """Mean I_peak +/- SD against time, one curve per field strength."""
  out_path = sub_dir(output_dir, "cortex_breakdown")
  if not traces:
    print("Evolution peak timecourse skipped: no traces.")
    return

  t_max = max(float(tr["t"].max()) for tr in traces)
  t_grid_max = t_max if ENDPOINT_MATCHED_T_S is None else min(
      t_max, float(ENDPOINT_MATCHED_T_S))
  grid = _peak_time_grid(t_grid_max)

  by_voltage = {}
  for tr in traces:
    interp = np.interp(grid, tr["t"], tr["y"])
    interp[(grid < tr["t"].min()) | (grid > tr["t"].max())] = np.nan
    by_voltage.setdefault(tr["voltage"], []).append(interp)
  by_voltage = {v: np.vstack(a) for v, a in by_voltage.items()}

  voltages = sorted(by_voltage, key=_voltage_key)
  pulsed = [v for v in voltages if v != CONTROL_VOLTAGE]
  cmap = mcolors.LinearSegmentedColormap.from_list(
      "ep_blues", [PALETTE["pale_blue"], PALETTE["medium_blue"],
                   PALETTE["dark_blue"]])
  shades = {v: cmap(0.3 + 0.7 * i / max(len(pulsed) - 1, 1))
            for i, v in enumerate(pulsed)}

  fig, ax = plt.subplots(figsize=fig_size("actin_evolution_peak", 8.5, 5.2))
  rows = []
  for volt in voltages:
    arr = by_voltage[volt]
    n_t = np.sum(np.isfinite(arr), axis=0)
    sel = n_t >= max(2, int(np.ceil(min_frac * arr.shape[0])))
    if not sel.any():
      continue
    with np.errstate(invalid="ignore"):
      mean = np.nanmean(arr[:, sel], axis=0)
      sd = np.nanstd(arr[:, sel], axis=0, ddof=1)
    is_ctrl = volt == CONTROL_VOLTAGE
    colour = PALETTE["dark_red"] if is_ctrl else shades[volt]
    ax.plot(grid[sel], mean, color=colour, lw=2.0 if is_ctrl else 1.5,
            ls="--" if is_ctrl else "-", zorder=5 if is_ctrl else 3,
            label=f"{field_label(volt)} kV/cm (n={arr.shape[0]})")
    ax.fill_between(grid[sel], mean - sd, mean + sd, color=colour, alpha=0.13,
                    lw=0, zorder=2)
    rows.append(pd.DataFrame({
        "voltage": volt, "time_s": grid[sel], "mean_peak": mean, "sd_peak": sd,
        "n_guv": n_t[sel]}))

  ax.axhline(1.0, color=ANNOTATION_TEXT, lw=0.8, ls=":", zorder=1)
  ax.set_xlabel("time after pulse (s)")
  ax.set_ylabel("cortex peak height  (I$_{peak}$ / I$_{peak,pre}$)")
  ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
  ax.grid(axis="y", linestyle="--", alpha=0.4)
  style_figure("actin_evolution_peak")
  fig.tight_layout()
  pdf_path = out_path / "guv_actin_evolution_peak_timecourse.pdf"
  fig.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)
  print(f"Saved evolution peak timecourse: {pdf_path}")

  if rows:
    pd.concat(rows, ignore_index=True).to_csv(
        out_path / "guv_actin_evolution_peak_timecourse.csv", index=False)

  per_guv.to_csv(out_path / "guv_actin_evolution_peak_per_guv.csv",
                 index=False)

  summary = (per_guv.groupby("voltage")["peak_norm_final"]
             .agg(n_guv="count", mean_peak="mean", sd_peak="std")
             .reset_index()
             .sort_values("voltage", key=lambda c: c.map(_voltage_key)))
  print(f"\nCortex peak from the radial profiles, read at t = "
        f"{ENDPOINT_MATCHED_T_S:.0f} s (1.0 = no change). n_guv counts only "
        "vesicles whose record reaches it:")
  print(summary.to_string(index=False))
  print()


def evolution_peak_metrics(traces) -> pd.DataFrame:
  """One row per GUV: burst and final peak height, and how much was lost.

  The final value is the median of the last PEAK_N_FRAMES samples at or before
  ENDPOINT_MATCHED_T_S, and it is NaN for any vesicle whose record stops more
  than ENDPOINT_MATCHED_TOLERANCE_S short of it. That is the same tolerance
  the dye endpoint uses -- one frame on a 5 s grid -- so a cortex number and a
  release number in the same sentence are read at the same instant. A shorter
  record is censored rather than read at its own end, which is the bias the
  matched endpoint exists to remove.
  """
  rows = []
  for tr in traces:
    t, y = tr["t"], tr["y"]
    if y.size < 2 * PEAK_N_FRAMES:
      continue
    burst = float(np.median(y[:PEAK_N_FRAMES]))
    final = np.nan
    if ENDPOINT_MATCHED_T_S is None:
      final = float(np.median(y[-PEAK_N_FRAMES:]))
    elif float(t.max()) >= ENDPOINT_MATCHED_T_S - ENDPOINT_MATCHED_TOLERANCE_S:
      final = float(np.median(y[-PEAK_N_FRAMES:]))
    rows.append({
        "experiment": tr["experiment"], "guv_id": tr["guv_id"],
        "voltage": tr["voltage"], "i_peak_pre": tr["i_peak_pre"],
        "r_peak_frac_pre": tr["r_peak_frac_pre"],
        "n_post_frames": tr["n_frames"], "t_last_s": float(t.max()),
        "peak_norm_burst": burst, "peak_norm_final": final,
        "peak_drop_burst": 1.0 - burst, "peak_drop_final": 1.0 - final,
    })
  out = pd.DataFrame(rows)
  if not out.empty:
    n_short = int(out["peak_norm_final"].isna().sum())
    if n_short:
      print(f"  {n_short} of {len(out)} GUV(s) have no endpoint value: their "
            f"record stops more than {ENDPOINT_MATCHED_TOLERANCE_S:.0f} s "
            f"short of {ENDPOINT_MATCHED_T_S:.0f} s.")
  return out


def add_cortex_peak_metrics(df: pd.DataFrame, per_guv) -> pd.DataFrame:
  """Attach the per-GUV peak metrics to the bulk table.

  Inner merge, so what comes back is only the vesicles with a measured cortex
  peak. Bare GUVs have no actin profile and are absent by construction; that
  is why report_onset_fields is given a LEFT merge onto the full table
  instead, and not this frame.
  """
  pk = per_guv.copy()
  if pk.empty:
    print("No radial profile peaks; cortex peak metrics skipped.")
    return df.iloc[0:0]
  out = df.copy()
  out["guv_id"] = out["guv_id"].astype(str)
  pk["guv_id"] = pk["guv_id"].astype(str)
  return out.merge(pk.drop(columns=["voltage"]), on=["experiment", "guv_id"],
                   how="inner")


def plot_cortex_peak_breakdown(df: pd.DataFrame, output_dir: str):
  """Cortex peak lost per field, as median and IQR over vesicles.

  Median rather than mean, because several conditions rest on five to nine
  vesicles and are skewed in both directions: 1.33 kV/cm to the right (median
  0.149, mean 0.196, largest single loss 0.586), 775 and 900 V to the left,
  where one low vesicle in nine pulls the mean below the median. A median and
  an IQR do not move with one vesicle.

  Both statistics go to the CSV, together with the skew, so the tail at
  1.33 kV/cm can be read without re-running.
  """
  out_path = sub_dir(output_dir, "cortex_breakdown")
  out_path.mkdir(parents=True, exist_ok=True)
  if df.empty or "peak_drop_final" not in df:
    print("Cortex peak breakdown skipped: no peak metrics.")
    return

  sub = cortex_stage_population(df, verbose=False)
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
      sd = float(cell.std(ddof=1)) if len(cell) > 1 else np.nan
      ax.plot([x[idx] + offset - offset_w * 0.7,
               x[idx] + offset + offset_w * 0.7], [med, med],
              color=colour, lw=2.2, zorder=4,
              label="" if labelled else label)
      labelled = True
      ax.plot([x[idx] + offset, x[idx] + offset], [q1, q3],
              color=colour, lw=1.0, alpha=0.8, zorder=3)
      rows.append({"voltage": v, "when": label, "n": len(cell),
                   "median_peak_drop": med, "q1": q1, "q3": q3,
                   "mean_peak_drop": float(cell.mean()), "sd_peak_drop": sd,
                   "sem_peak_drop": (sd / np.sqrt(len(cell))
                                     if np.isfinite(sd) else np.nan),
                   "skew": float(cell.skew()) if len(cell) > 2 else np.nan})

  ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.7)
  ax.set_xticks(x)
  ax.set_xticklabels(field_labels(voltages), rotation=45)
  ax.set_xlabel(FIELD_AXIS_LABEL)
  ax.set_ylabel("Cortex peak height lost  (1 - I$_{peak}$ / I$_{peak,pre}$)")
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
          f"{ctrl.median():.3f} (IQR {ctrl.quantile(.25):.3f} to "
          f"{ctrl.quantile(.75):.3f}, n={len(ctrl)}). Conditions at or below "
          "this show no breakdown beyond bleaching.")
  skewed = tbl[tbl["skew"].abs() > 1.0]
  if not skewed.empty:
    print("  Conditions where the mean and median disagree because the "
          "distribution is skewed (|skew| > 1); both are in the CSV:")
    for _, r in skewed.iterrows():
      print(f"    {r['voltage']:>5}  {r['when']:<22} mean "
            f"{r['mean_peak_drop']:+.3f}  median {r['median_peak_drop']:+.3f}"
            f"  skew {r['skew']:+.2f}  n={int(r['n'])}")
  print("\nCortex peak height lost per condition:")
  print(tbl.to_string(index=False))
  print()


# -----------------------------------------------------------------------------
# --- DOES CORTEX LOSS DEPEND ON VESICLE SIZE? ---
#
# Read this section knowing what it cannot separate.
#
# The induced transmembrane potential is proportional to the radius
# (dV_m = 1.5 E R cos(theta) at steady state), so at one field strength a
# 10 um vesicle sits at twice the potential of a 5 um one. A positive
# correlation between radius and cortex loss is therefore the EXPECTED result
# and is not evidence that size matters over and above dose -- it is the dose
# axis, measured in micrometres instead of volts. The interesting outcomes are
# the other two: no correlation, which would say the cortex fails on something
# other than membrane potential, or a negative one, which would need a
# mechanism of its own.
#
# What this cannot do on these data is separate radius from field, because
# neither was randomised against the other: voltage is confounded with session
# and the radius distribution is not matched across conditions. So the
# correlation is computed WITHIN each field strength, where every vesicle
# received the same E and radius is the only thing varying, and then pooled
# across fields on within-field ranks. The pooled row is the better-powered
# test; the per-field rows are the check that it is not being driven by one
# condition.
#
# The size window is deliberately absent here, as in every cortex stage. It
# exists to match Bare against Branched on radius; this section is a
# comparison of cortex-bearing vesicles against each other, and applying it
# would remove most of the range the question is about.
# -----------------------------------------------------------------------------

# Experiments required before an interval is reported at all, and before the
# resolved flag is allowed. Same thresholds and the same reason as the
# pole/equator stage: a cluster bootstrap over 3 chambers wrongly excludes
# zero about 27% of the time on pure-null data, against 7.5% at 12.
SIZE_MIN_CLUSTERS_CI = 3
SIZE_MIN_CLUSTERS_RESOLVED = 12
SIZE_N_BOOTSTRAP = 2000


def _stratified_spearman(radius, drop, strata) -> float:
  """Spearman correlation on ranks taken WITHIN each stratum, then pooled.

  Ranking inside the field strength and only then pooling is what removes the
  field effect. Pooling the raw values instead would let a condition that
  happened to contain the larger vesicles AND the stronger field manufacture
  a correlation out of the confound.
  """
  radius = np.asarray(radius, float)
  drop = np.asarray(drop, float)
  strata = np.asarray(strata)
  rx, ry = [], []
  for s in np.unique(strata):
    m = strata == s
    if m.sum() < 3:
      continue
    # Average ranks scaled to (0, 1) so strata of different size contribute
    # on one scale.
    n = int(m.sum())
    rx.append(pd.Series(radius[m]).rank().to_numpy() / (n + 1))
    ry.append(pd.Series(drop[m]).rank().to_numpy() / (n + 1))
  if not rx:
    return np.nan
  rx, ry = np.concatenate(rx), np.concatenate(ry)
  if rx.size < 3 or np.std(rx) == 0 or np.std(ry) == 0:
    return np.nan
  return float(np.corrcoef(rx, ry)[0, 1])


def _boot_corr_ci(radius, drop, clusters, strata=None, n_boot: int = None,
                  seed: int = 0):
  """Clustered bootstrap CI and p for a rank correlation.

  Whole chambers are resampled, not vesicles: vesicles in one chamber share a
  preparation, a field and a day, and a vesicle-level resample would report an
  interval several times narrower than the data support. When *strata* is
  given the correlation is the within-field one above, recomputed on each
  resample so the ranking follows the resampled data rather than being fixed
  from the full sample.

  Returns (rho, lo, hi, p, n_clusters). p is two-sided against rho = 0, taken
  from the same resampling as the interval so the two cannot disagree.
  """
  n_boot = int(SIZE_N_BOOTSTRAP if n_boot is None else n_boot)
  radius = np.asarray(radius, float)
  drop = np.asarray(drop, float)
  clusters = np.asarray(clusters)
  strata = np.zeros(radius.shape) if strata is None else np.asarray(strata)

  keep = np.isfinite(radius) & np.isfinite(drop)
  radius, drop, clusters, strata = (radius[keep], drop[keep], clusters[keep],
                                    strata[keep])
  uniq = list(dict.fromkeys(clusters.tolist()))
  rho = _stratified_spearman(radius, drop, strata)
  if radius.size < 4 or len(uniq) < SIZE_MIN_CLUSTERS_CI or not np.isfinite(rho):
    return rho, np.nan, np.nan, np.nan, len(uniq)

  idx_by_cluster = [np.where(clusters == u)[0] for u in uniq]
  rng = np.random.default_rng(seed)
  vals = np.empty(n_boot)
  for b in range(n_boot):
    pick = rng.integers(0, len(uniq), len(uniq))
    sel = np.concatenate([idx_by_cluster[i] for i in pick])
    vals[b] = _stratified_spearman(radius[sel], drop[sel], strata[sel])
  vals = vals[np.isfinite(vals)]
  if vals.size < n_boot // 10:
    return rho, np.nan, np.nan, np.nan, len(uniq)
  lo, hi = np.percentile(vals, [2.5, 97.5])
  frac = float(np.mean(vals <= 0.0))
  p = float(min(1.0, 2.0 * min(frac, 1.0 - frac)))
  return rho, float(lo), float(hi), p, len(uniq)


def report_cortex_size_dependence(df: pd.DataFrame, output_dir: str):
  """Cortex peak lost against vesicle radius, within and across field strengths.

  Reads peak_drop_final, so it runs on the output of add_cortex_peak_metrics
  and inherits the matched endpoint: vesicles whose record stops short carry
  NaN there and are absent here rather than being read at their own end.
  """
  out_path = sub_dir(output_dir, "cortex_breakdown")
  if df.empty or "peak_drop_final" not in df or "radius_um" not in df:
    print("Cortex size dependence skipped: no peak metrics or no radius.")
    return pd.DataFrame()

  sub = cortex_stage_population(df, verbose=False)
  sub = sub[np.isfinite(sub["peak_drop_final"])
            & np.isfinite(sub["radius_um"])]
  if len(sub) < 4:
    print(f"Cortex size dependence skipped: {len(sub)} measurable GUV(s).")
    return pd.DataFrame()

  voltages = _voltage_order(sub)
  rows = []
  for v in voltages:
    cell = sub[sub["voltage"] == v]
    rho, lo, hi, p, n_c = _boot_corr_ci(cell["radius_um"],
                                        cell["peak_drop_final"],
                                        cell["experiment"])
    rows.append({
        "voltage": v, "pooled": False, "n_guv": len(cell),
        "n_experiments": n_c,
        "radius_median_um": float(cell["radius_um"].median()),
        "radius_min_um": float(cell["radius_um"].min()),
        "radius_max_um": float(cell["radius_um"].max()),
        "rho": rho, "ci_lo": lo, "ci_hi": hi, "p": p,
    })

  # Pooled over every field, on within-field ranks. This is the test; the
  # rows above are 2 to 6 chambers each and cannot support a verdict.
  rho, lo, hi, p, n_c = _boot_corr_ci(sub["radius_um"], sub["peak_drop_final"],
                                      sub["experiment"], strata=sub["voltage"])
  rows.append({
      "voltage": "all fields (within-field ranks)", "pooled": True,
      "n_guv": len(sub), "n_experiments": n_c,
      "radius_median_um": float(sub["radius_um"].median()),
      "radius_min_um": float(sub["radius_um"].min()),
      "radius_max_um": float(sub["radius_um"].max()),
      "rho": rho, "ci_lo": lo, "ci_hi": hi, "p": p,
  })

  tbl = pd.DataFrame(rows)
  per_field = ~tbl["pooled"]
  tbl["p_holm"] = np.nan
  tbl.loc[per_field, "p_holm"] = process.holm(tbl.loc[per_field, "p"].values)
  tbl["ci_reliable"] = np.where(
      tbl["n_experiments"] >= SIZE_MIN_CLUSTERS_RESOLVED, "yes",
      np.where(tbl["n_experiments"] >= SIZE_MIN_CLUSTERS_CI,
               f"few clusters (<{SIZE_MIN_CLUSTERS_RESOLVED})", "not reported"))
  # Parenthesised deliberately: & binds tighter than >=, so without the
  # brackets this reads n_experiments >= (12 & isfinite & excludes_zero) and
  # marks everything resolved, including cells whose interval spans zero.
  tbl["resolved"] = ((tbl["n_experiments"] >= SIZE_MIN_CLUSTERS_RESOLVED)
                     & np.isfinite(tbl["ci_lo"])
                     & ((tbl["ci_lo"] > 0) | (tbl["ci_hi"] < 0)))

  csv_path = out_path / "guv_cortex_size_dependence.csv"
  tbl.to_csv(csv_path, index=False)

  # --- figure: one panel per field, radius against loss ---------------------
  n_cols = min(3, len(voltages))
  n_rows = int(np.ceil(len(voltages) / n_cols))
  fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True,
                           sharey=True, layout="constrained",
                           figsize=fig_size("cortex_size_dependence",
                                            4.2 * n_cols, 3.6 * n_rows,
                                            len(voltages)))
  for ax in axes.ravel():
    ax.set_visible(False)
  for k, v in enumerate(voltages):
    ax = axes[k // n_cols][k % n_cols]
    ax.set_visible(True)
    cell = sub[sub["voltage"] == v]
    colour = (PALETTE["dark_red"] if v == CONTROL_VOLTAGE
              else PALETTE["dark_blue"])
    ax.scatter(cell["radius_um"], cell["peak_drop_final"], s=16, alpha=0.5,
               color=colour, edgecolors="none", zorder=3)
    # Median loss either side of this field's own median radius. A summary
    # rather than a fitted line: a slope drawn through a dozen points implies
    # a model the data cannot support, and the split is what the correlation
    # is actually detecting.
    r_split = float(cell["radius_um"].median())
    for lo_r, hi_r, x0, x1 in ((-np.inf, r_split, cell["radius_um"].min(),
                                r_split),
                               (r_split, np.inf, r_split,
                                cell["radius_um"].max())):
      half = cell[(cell["radius_um"] > lo_r) & (cell["radius_um"] <= hi_r)]
      if len(half) >= 3:
        ax.plot([x0, x1], [half["peak_drop_final"].median()] * 2,
                color=colour, lw=2.0, zorder=5)
    ax.axvline(r_split, color=ANNOTATION_TEXT, lw=0.8, ls=":", zorder=1)
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
    r = tbl[(tbl["voltage"] == v) & ~tbl["pooled"]].iloc[0]
    lab = ("rho n/a" if not np.isfinite(r["rho"])
           else f"$\\rho$ = {r['rho']:+.2f}"
                + ("" if not np.isfinite(r["ci_lo"])
                   else f" [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"))
    # Per-panel identifier drawn as an annotation, not an axes title: with
    # one panel per field strength, each still needs to say which field
    # (and its correlation) it is, but that is not a plot title.
    ax.text(0.5, 1.0, f"{field_label(v)} kV/cm  (n={len(cell)}, "
            f"{int(r['n_experiments'])} chambers)\n{lab}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")
    if k % n_cols == 0:
      ax.set_ylabel("cortex peak height lost")
    if k // n_cols == n_rows - 1:
      ax.set_xlabel(r"pre-pulse radius ($\mu$m)")

  style_figure("cortex_size_dependence")
  pooled = tbl[tbl["pooled"]].iloc[0]
  pdf_path = out_path / "guv_cortex_size_dependence.pdf"
  fig.savefig(pdf_path, format="pdf", dpi=300)
  plt.close(fig)

  print(f"\nSaved cortex size dependence: {pdf_path}")
  print(f"Saved {csv_path}")
  print("rho > 0 means LARGER vesicles lost more cortex peak. Within each "
        "field, so the field is held fixed; pooled on within-field ranks.")
  show = ["voltage", "n_guv", "n_experiments", "radius_min_um",
          "radius_max_um", "rho", "ci_lo", "ci_hi", "p", "p_holm",
          "ci_reliable"]
  print(tbl[show].to_string(index=False))
  if not np.isfinite(pooled["ci_lo"]):
    print("  The pooled interval is not reported: fewer than "
          f"{SIZE_MIN_CLUSTERS_CI} chambers contributed.")
  elif pooled["ci_lo"] <= 0 <= pooled["ci_hi"]:
    print(f"  The pooled interval spans zero, so no size dependence is "
          f"resolved. It rules out anything stronger than "
          f"rho = {max(abs(pooled['ci_lo']), abs(pooled['ci_hi'])):.2f}, "
          "which is a bound rather than a null.")
  else:
    print("  The pooled interval excludes zero. Before reading that as a "
          "size effect: dV_m scales with radius, so this is what a pure dose "
          "response looks like plotted against size.")
  print()
  return tbl


# -----------------------------------------------------------------------------
# --- THE BLEACH REFERENCE ---
# -----------------------------------------------------------------------------


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
          "to a per-exposure basis. Nothing here divides by it -- this is a "
          "check on the reference, not a correction applied to anything.")
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


# -----------------------------------------------------------------------------
# --- THE STAGE ---
# -----------------------------------------------------------------------------


def cortex_analysis(df: pd.DataFrame, outputs_root, results_dir):
  """Every cortex figure that rests on the radial profiles, in order.

  df is the aggregated bulk table with intensity gainers already dropped, NOT
  size-windowed: cortex_stage_population is the filter these stages use, and
  it deliberately does not apply the size window (see its docstring).

  The onset fits and the pole/equator test still live in process.py -- they
  are not peak measurements -- but both are called from here because both
  need peak_drop_final, which now comes from the profiles rather than from
  the exported cortex_peak traces.
  """
  root = str(outputs_root)
  results = str(results_dir)

  # Independent of the cortex-only collection below: needs all three
  # phenotypes (CORTEX, NO_CORTEX, AMBIGUOUS), not just cortex-bearing GUVs,
  # so it cannot be built from collect_evolution_peaks / traces.
  plot_representative_radial_profiles(df, root, results)

  # Read once. Everything below is a different view of these traces, so
  # re-reading the profile files per stage would only invite the views
  # drifting apart.
  traces, profiles, grid_x, edges = collect_evolution_peaks(df, root)
  if not traces:
    print("Cortex stages skipped: no *_actin_evolution.csv files matched the "
          "cortex-stage population.")
    return df.iloc[0:0]
  # Measured once and passed to both consumers, so the figure and the table
  # cannot end up reporting different endpoints for the same vesicle.
  per_guv = evolution_peak_metrics(traces)
  if edges is not None:
    plot_mean_actin_evolution(traces, profiles, grid_x, edges, results)
  plot_evolution_peak_timecourse(traces, per_guv, results)

  df_peak = add_cortex_peak_metrics(df, per_guv)
  plot_cortex_peak_breakdown(df_peak, results)
  report_cortex_size_dependence(df_peak, results)
  # Fitted on df merged with peak_drop_final rather than on df_peak itself,
  # because df_peak is an inner merge and so contains only vesicles with an
  # actin trace -- Bare GUVs have none, and dropping them would leave the
  # area-loss onset with one population to fit instead of two. The merge is
  # left, so bare vesicles carry NaN for the cortex measure and are simply
  # absent from that fit.
  #
  # One thing this does NOT yet do, an open decision rather than an oversight:
  # the two measures do not have to rest on the same vesicles. Area loss is
  # available for every GUV with a terminal radius, the cortex drop only for
  # those whose actin trace reached ENDPOINT_MATCHED_T_S. Check n_guv per
  # measure in onset_fields.csv before reading the two midpoints as an
  # ordering. The cortex split is applied inside report_onset_fields, so the
  # frame passed here is deliberately the unsplit one.
  process.report_onset_fields(
      df.merge(df_peak[["experiment", "guv_id", "peak_drop_final"]],
               on=["experiment", "guv_id"], how="left")
      if not df_peak.empty else df,
      results)
  # Not a correction, a check: whether the 30 V reference was even sampled on
  # a comparable schedule. Nothing here is divided by it.
  report_bleach_comparability(root)
  # Directionality of that loss, pooled across vesicles. Reads the
  # per-experiment *_pole_vs_equator_traces.csv files, so it needs
  # EXPORT_ACTIN_KYMOGRAPH = True on the run that produced them.
  #
  # Given df_peak, not df: the conditioned test ("of the cortices that broke
  # down, did they break down directionally?") needs peak_drop_final, which
  # only exists after the actin merge. Falls back to df so the pooled test
  # still runs when no actin traces were found.
  process.plot_pole_equator_pooled(
      df_peak if not df_peak.empty else df, root, results)
  return df_peak


if __name__ == "__main__":
  root = outputs_root()
  results = root / RESULTS_SUBFOLDER
  results.mkdir(parents=True, exist_ok=True)
  if RUN_CORTEX_FIGURES:
    # Aggregated here rather than read from guv_bulk_summary.csv: that file has
    # the size window applied, and these stages run on the population before
    # it. run_bulk.py passes its own table instead of re-aggregating.
    df_all = process.aggregate_pipeline_results(str(root))
    df_all = process.drop_intensity_gainers(df_all, results)
    cortex_analysis(df_all, root, results)
  if RUN_CROP_MONTAGES:
    main()