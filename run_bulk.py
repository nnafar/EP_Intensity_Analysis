"""Actin-channel montages of GUVs with and without a cortex.

Two figures: one panel per GUV classified CORTEX, one per GUV classified
NO_CORTEX (actin present but never polymerised onto the membrane). This is the
visual check on the classifier -- guv_cortex_contrast_histogram.pdf says
whether the threshold sits in a trough, and these say whether the objects
either side of it actually look different.

Panels are sorted by cortex_contrast, ascending. Read left to right, the
montage should show a progression; if the two figures look interchangeable at
their shared boundary, the threshold is not separating anything real.

Requires the raw ND2s, so DATA_ROOT below must point at them.

In Spyder: open and press F5.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palette import PALETTE

try:
  import config as cfg
except Exception:
  cfg = None

# -----------------------------------------------------------------------------
# --- SPYDER SETTINGS ---
# -----------------------------------------------------------------------------

DATA_ROOT = r"D:\Data\EP"          # tree containing the .nd2 files
OUTPUTS_ROOT = None                # None = config.PARENT_OUTPUT_FOLDER

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

RESULTS_SUBFOLDER = "Bulk_Analysis_Results"


# -----------------------------------------------------------------------------


def outputs_root() -> Path:
  if OUTPUTS_ROOT:
    return Path(OUTPUTS_ROOT)
  return Path(cfg.PARENT_OUTPUT_FOLDER)


def index_nd2(data_root: Path) -> dict:
  """Map ND2 stem -> path, so an output folder can find its raw movie."""
  out = {}
  for p in data_root.glob("**/*.nd2"):
    if "Outputs" in p.parts:
      continue
    out.setdefault(p.stem, p)
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
  rows = []
  for status_csv in sorted(root.glob("**/*_actin_cortex_status.csv")):
    if RESULTS_SUBFOLDER in status_csv.parts:
      continue
    exp_dir = status_csv.parents[1]
    base = status_csv.name.replace("_actin_cortex_status.csv", "")

    nd2_path = nd2_index.get(base)
    if nd2_path is None:
      print(f"  no ND2 found for {base}; skipped")
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
      rows.append({
          "experiment": exp_dir.name,
          "base": base,
          "nd2": nd2_path,
          "frame": use_frame,
          "guv_id": gid,
          "status": str(r.get("cortex_status", "UNKNOWN")),
          "contrast": r.get("cortex_contrast", np.nan),
          "x": float(t["x"]), "y": float(t["y"]), "radius": float(t["radius"]),
      })
  return pd.DataFrame(rows)


def cut_crops(sel: pd.DataFrame, ch: int) -> list:
  """Read each GUV's crop, opening every ND2 once."""
  crops = []
  for nd2_path, grp in sel.groupby("nd2", sort=False):
    for frame, sub in grp.groupby("frame", sort=False):
      try:
        img = load_actin_frame(Path(nd2_path), int(frame), ch)
      except Exception as e:
        print(f"  could not read {Path(nd2_path).name} frame {frame}: {e}")
        continue
      h, w = img.shape[:2]
      for _, r in sub.iterrows():
        half = int(round(r["radius"] * CROP_RADII))
        if half < 3:
          continue
        x0, x1 = int(round(r["x"])) - half, int(round(r["x"])) + half
        y0, y1 = int(round(r["y"])) - half, int(round(r["y"])) + half
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
          continue                      # partially outside the frame
        crops.append({**r.to_dict(), "img": img[y0:y1, x0:x1]})
  return crops


def short_tag(experiment: str) -> str:
  """Voltage plus experiment number, e.g. '400V E2'. Falls back to the name."""
  v = re.search(r"-(\d+V)-", experiment)
  e = re.search(r"Experiment(\d+)", experiment)
  if v and e:
    return f"{v.group(1)} E{e.group(1)}"
  return experiment[:14]


def montage(crops: list, title: str, out_path: Path, vmin=None, vmax=None):
  if not crops:
    print(f"  nothing to draw for {title}")
    return
  crops = sorted(crops, key=lambda c: (np.inf if pd.isna(c["contrast"])
                                       else c["contrast"]))
  if len(crops) > MAX_PANELS:                 # even thinning, keeps the range
    idx = np.linspace(0, len(crops) - 1, MAX_PANELS).round().astype(int)
    crops = [crops[i] for i in idx]

  n_rows = int(np.ceil(len(crops) / N_COLS))
  # The header is given its own strip of the figure rather than a fraction of
  # it, so the two-line suptitle does not creep onto the first row of panel
  # titles as the montage gets taller or shorter.
  header_in = 1.0
  fig_h = 2.1 * n_rows + header_in
  fig, axes = plt.subplots(n_rows, N_COLS,
                           figsize=(1.9 * N_COLS, fig_h), squeeze=False)

  for ax in axes.ravel():
    ax.axis("off")

  for i, c in enumerate(crops):
    ax = axes[i // N_COLS][i % N_COLS]
    kw = {} if SCALING == "per_panel" else {"vmin": vmin, "vmax": vmax}
    ax.imshow(c["img"], cmap="gray", interpolation="nearest",
              aspect="equal", **kw)
    # GUV ids restart at 1 in every experiment, so the id alone is ambiguous
    # across a pooled montage; the voltage tag disambiguates it.
    label = f"{short_tag(c['experiment'])} G{c['guv_id']}"
    if pd.notna(c["contrast"]):
      label += f"  c={c['contrast']:.2f}"
    ax.set_title(label, fontsize=6.5, color=PALETTE["grey"], pad=3)
    ax.axis("off")

  fig.suptitle(title, fontsize=11, y=1 - 0.12 / fig_h)
  # tight_layout packs the rows until the titles collide with the panel above,
  # so the spacing is set explicitly instead.
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
  results = (Path(results_dir) if results_dir
             else outputs_root() / RESULTS_SUBFOLDER)
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
  print(f"Classified GUVs: {len(df)}")
  print(df["status"].value_counts().to_string())

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
    montage(sel,
            f"Actin channel, pre-pulse — GUVs {label}\n"
            f"(sorted by cortex contrast; {SCALING} intensity scaling)",
            results / name, vmin, vmax)


if __name__ == "__main__":
  main()