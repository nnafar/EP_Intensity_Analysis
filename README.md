1. Setup and validation

Reads config.py, checks parameter ranges, and resolves channel roles. Channel indices raise if missing rather than defaulting — the old fallbacks pointed actin at your membrane channel and the selector at your actin channel.

2. Load GUV positions

Loads guv_circles.json, or launches the interactive selector if it's absent.

3. Load image stacks

Opens the ND2, extracts the three channels into memory-mapped stacks, builds the timestamp array from FRAME_INTERVAL_SCHEDULE (1 s pre-pulse → 0.1 s fast window → 5 s tail). Optionally checks empirically that the channel labelled "membrane" really is the one with ring structure.

4. Track each GUV (parallel)

Per frame, per vesicle: ring-score grid search for the centre, ellipse fit for the boundary, radius clamped to ±20% change per frame. Builds three masks — interior, membrane ring, background annulus. Records a provisional lumen mean and background. Radius stays float now (was integer pixels).

5. Recompute background properly

For every GUV and frame, from the annulus:

Level = 15th percentile
σ = spread between the p15 and p40 quantiles, on unblurred pixels — both sit in the dark part of the distribution so the bright tail never enters
Crowding check: how far the annulus median sits above the quantile anchor

This runs before pulse detection so the same background feeds everything downstream.

6. Detect the pulse frame

From the time-normalised derivative dI/dt within the fast-acquisition window, cross-checked against the frameN filename tag.

7. Classify vesicle fates

Ruptured / shrunk (≥15% radius loss) / out-of-frame / intact, from ring-score collapse and radius trajectory.

8. Normalise

I_retained(t) = (I_bg,t − I_dye,t) / (I_bg,t − I_dye,0), with I_dye,0 the mean of the 5 pre-pulse frames.

A frame is blanked only if the denominator falls below 5× the standard error of the background estimate (≈0.058σ at n≈700 pixels) — not per-pixel σ, which is ~17× larger and blanked almost everything. Writes dye_qc.csv with sep0/σ for every GUV including any the filter would reject.

9. Fit

For each GUV, both models with 6 seeded multi-starts:

1EXP: Iinf + A·exp(−t/τ)
2EXP: Iinf + a₁·exp(−t/τ₁) + a₂·exp(−t/τ₂)

with A, a ≥ 0 and Iinf ∈ [0, 1.2]. Those bounds make 1EXP a strict subset of 2EXP, so RSS_2EXP ≤ RSS_1EXP holds by construction — asserted before any AICc is computed. Drift term off.

Then the four flags: is_responding, tau_identifiable, has_step, radius_growth_flag. Model-free descriptors (t50_s with left-censoring flag, frac_remaining_*, initial_rate_per_s) computed for every GUV regardless.

10. Export

fit_parameters.csv, model_comparison.csv, normalized_curves.csv (with pre-pulse frames at negative time), dye_qc.csv, background_diagnostics.csv, the summary plot, the fits grid, and boxplots of Iinf/A/τ for identifiable fits only.