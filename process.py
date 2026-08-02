import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load pipeline configuration
import config as cfg


def extract_voltage(path_str: str) -> str:
  """Extract voltage label (e.g., '400V') from directory name."""
  match = re.search(r"(\d+)\s*V", path_str, re.IGNORECASE)
  return f"{match.group(1)}V" if match else "Unknown"


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


def aggregate_pipeline_results(outputs_root: str) -> pd.DataFrame:
  """Parse tracking fates, unnormalized radius, and pre-pulse vs. last 3 frames intensity traces across output folders."""
  root = Path(outputs_root)
  records = []

  fate_files = list(root.glob("**/*_guv_fate_classification.csv"))

  for fate_file in fate_files:
    exp_dir = fate_file.parents[1]
    voltage = extract_voltage(exp_dir.name)

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

    pre_mask = None
    if df_norm is not None:
      if "phase" in df_norm.columns:
        pre_mask = df_norm["phase"] == "pre_pulse"
      elif "Time (s)" in df_norm.columns:
        pre_mask = df_norm["Time (s)"] < 0
      else:
        pre_mask = df_norm.index < 5

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

      # Get raw unnormalized radius_um and bin it
      radius_um = np.nan
      if df_fit is not None:
        guv_fit = df_fit[df_fit["guv_id"].astype(str) == gid]
        if not guv_fit.empty:
          radius_um = guv_fit.iloc[0].get("radius_um", np.nan)

      radius_bin = assign_size_bin(radius_um)

      intensity_cat = None
      i_pre = np.nan
      i_last3_avg = np.nan
      diff = np.nan

      if size_cat == "Stagnate" and df_norm is not None:
        target_col = f"GUV_{gid}_I_retained"
        if target_col in df_norm.columns:
          trace = df_norm[target_col].dropna().values
          if len(trace) > 0:
            pre_vals = df_norm.loc[pre_mask, target_col].dropna().values
            i_pre = np.mean(pre_vals) if len(pre_vals) > 0 else trace[0]

            # Average of last 3 valid frames
            last_3_vals = trace[-3:]
            i_last3_avg = np.mean(last_3_vals)
            diff = i_last3_avg - i_pre

            threshold = 0.05
            if diff > threshold:
              intensity_cat = "Increase Intensity"
            elif abs(diff) <= threshold:
              intensity_cat = "Flatline"
            else:
              intensity_cat = "Lose Intensity"

      records.append({
          "voltage": voltage,
          "guv_id": gid,
          "size_category": size_cat,
          "radius_um": radius_um,
          "radius_bin": radius_bin,
          "i_pre": i_pre,
          "i_last3_avg": i_last3_avg,
          "diff": diff,
          "intensity_category": intensity_cat,
      })

  return pd.DataFrame(records)


def generate_separate_pdf_plots(df: pd.DataFrame, output_dir: str):
  """Generate graphs in separate PDF files, including individual per-voltage binned size charts."""
  if df.empty:
    print("No valid GUV records found. Please check output directory path.")
    return

  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  voltages = sorted(
      df["voltage"].unique(),
      key=lambda v: int(re.search(r"\d+", v).group())
      if re.search(r"\d+", v)
      else 0,
  )

  # Summaries
  size_categories = ["Grow", "Stagnate", "Reduce", "Rupture/Collapse"]
  size_summary = (
      df.groupby(["voltage", "size_category"]).size().unstack(fill_value=0)
  )
  size_summary = size_summary.reindex(
      index=voltages, columns=size_categories, fill_value=0
  )

  # Filter Stagnate GUVs: exclude intensity increases and values > 1.0
  df_stagnate = df[
      (df["size_category"] == "Stagnate")
      & (df["intensity_category"].isin(["Lose Intensity", "Flatline"]))
      & (df["i_pre"] <= 1.0)
      & (df["i_last3_avg"] <= 1.0)
  ].copy()

  intensity_categories = ["Lose Intensity", "Flatline"]
  if df_stagnate.empty:
    intensity_summary = pd.DataFrame(
        0, index=voltages, columns=intensity_categories
    )
  else:
    intensity_summary = (
        df_stagnate.groupby(["voltage", "intensity_category"])
        .size()
        .unstack(fill_value=0)
    )
    intensity_summary = intensity_summary.reindex(
        index=voltages, columns=intensity_categories, fill_value=0
    )

  # -------------------------------------------------------------------------
  # GRAPH 1: Size Category Distribution (Standalone PDF)
  # -------------------------------------------------------------------------
  fig1, ax1 = plt.subplots(figsize=(7, 5))
  size_summary.plot(
      kind="bar",
      stacked=True,
      ax=ax1,
      colormap="viridis",
      edgecolor="black",
  )
  ax1.set_title("GUV Size Category Distribution")
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
  # GRAPH 2: Stagnate Intensity Breakdown (Standalone PDF)
  # -------------------------------------------------------------------------
  fig2, ax2 = plt.subplots(figsize=(7, 5))
  intensity_summary.plot(
      kind="bar",
      stacked=False,
      ax=ax2,
      color=["#e74c3c", "#3498db"],
      edgecolor="black",
  )
  ax2.set_title(
      "Stagnate GUV Intensity Breakdown\n(Pre-Pulse vs. Last 3 Frames)"
  )
  ax2.set_xlabel("Voltage")
  ax2.set_ylabel("GUV Count")
  ax2.legend(
      title="Intensity Behavior", bbox_to_anchor=(1.02, 1), loc="upper left"
  )
  ax2.grid(axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()

  pdf_path2 = out_path / "guv_stagnate_intensity_counts.pdf"
  plt.savefig(pdf_path2, format="pdf", dpi=300)
  plt.close(fig2)
  print(f"Saved Graph 2: {pdf_path2}")

  # -------------------------------------------------------------------------
  # GRAPH 3: Voltage Trajectory Plot (Extra-Wide Standalone PDF)
  # -------------------------------------------------------------------------
  fig3, ax3 = plt.subplots(figsize=(14, 5.5))
  x_coords = np.arange(len(voltages))
  width = 0.22

  for idx, v in enumerate(voltages):
    sub = df_stagnate[df_stagnate["voltage"] == v]
    x_pre = idx - width
    x_post = idx + width

    # Individual GUV Trajectory Lines
    for _, r in sub.iterrows():
      if pd.notna(r["i_pre"]) and pd.notna(r["i_last3_avg"]):
        ax3.plot(
            [x_pre, x_post],
            [r["i_pre"], r["i_last3_avg"]],
            "-o",
            color="gray",
            alpha=0.35,
            ms=4,
        )

    # Voltage Mean Trajectory
    if not sub.empty:
      m_pre = sub["i_pre"].mean()
      m_post = sub["i_last3_avg"].mean()
      ax3.plot(
          [x_pre, x_post],
          [m_pre, m_post],
          "-s",
          color="#d63031",
          lw=1.8,
          ms=5,
          alpha=0.8,
          zorder=5,
          label="Mean Drop" if idx == 0 else "",
      )

  ax3.set_xticks(x_coords)
  ax3.set_xticklabels(voltages, fontsize=10)
  ax3.set_xlabel("Voltage")
  ax3.set_ylabel("Normalized Intensity")
  ax3.set_title(
      "Intensity Drop Trajectories per Voltage\n(Pre-Pulse → Final 3-Frame"
      " Average)"
  )
  ax3.set_ylim(bottom=0)
  ax3.legend(loc="lower left", fontsize=9)
  ax3.grid(axis="y", linestyle="--", alpha=0.5)
  plt.tight_layout()

  pdf_path3 = out_path / "guv_intensity_drop_trajectories.pdf"
  plt.savefig(pdf_path3, format="pdf", dpi=300)
  plt.close(fig3)
  print(f"Saved Graph 3: {pdf_path3}")

  # -------------------------------------------------------------------------
  # GRAPH 4+: Binned Size vs. Intensity Drop (STANDALONE PDF PER VOLTAGE)
  # -------------------------------------------------------------------------
  bin_order = ["< 5 µm", "5-8 µm", "9-12 µm", "> 13 µm"]
  df_binned = df_stagnate[df_stagnate["radius_bin"].isin(bin_order)].copy()

  for v in voltages:
    sub = df_binned[df_binned["voltage"] == v]

    fig_v, ax_v = plt.subplots(figsize=(5.5, 5))

    for b_idx, b_label in enumerate(bin_order):
      bin_sub = sub[sub["radius_bin"] == b_label]
      x_pre = b_idx - width
      x_post = b_idx + width

      # Individual GUV Trajectories
      for _, r in bin_sub.iterrows():
        ax_v.plot(
            [x_pre, x_post],
            [r["i_pre"], r["i_last3_avg"]],
            "-o",
            color="gray",
            alpha=0.4,
            ms=4,
        )

      # Size-group Mean Drop
      if not bin_sub.empty:
        m_pre = bin_sub["i_pre"].mean()
        m_post = bin_sub["i_last3_avg"].mean()
        ax_v.plot(
            [x_pre, x_post],
            [m_pre, m_post],
            "-s",
            color="#d63031",
            lw=1.8,
            ms=5,
            alpha=0.8,
            zorder=5,
        )

    ax_v.set_xticks(np.arange(len(bin_order)))
    ax_v.set_xticklabels(bin_order, fontsize=10)
    ax_v.set_xlabel("Size Group")
    ax_v.set_ylabel("Normalized Intensity")
    ax_v.set_title(f"{v} Voltage\n(Size Group vs. Intensity Drop)")
    ax_v.set_ylim(bottom=0, top=1.05)
    ax_v.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save each voltage chart as its own individual PDF
    pdf_v_path = out_path / f"guv_binned_size_vs_intensity_{v}.pdf"
    plt.savefig(pdf_v_path, format="pdf", dpi=300)
    plt.close(fig_v)
    print(f"Saved Voltage Graph ({v}): {pdf_v_path}")


if __name__ == "__main__":
  outputs_root = getattr(cfg, "PARENT_OUTPUT_FOLDER", r"D:\EP\Outputs")

  df_summary = aggregate_pipeline_results(outputs_root)
  generate_separate_pdf_plots(df_summary, outputs_root)