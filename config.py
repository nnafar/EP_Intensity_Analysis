"""
GUV ANALYSIS CONFIGURATION

Workflow
--------
1. Run  `python interactive_select.py`  to draw circles on the first ROI
   frame.  This saves  CIRCLES_JSON_PATH  (e.g. "guv_circles.json").
2. Run  `python run_analysis.py`.  If CIRCLES_JSON_PATH is missing the
   selector is launched automatically.
"""

import numpy as np
import os
import re

# -----------------------------------------------------------------------------
# --- 1. FILE & EXPERIMENT IDENTIFICATION ---
# -----------------------------------------------------------------------------

INPUT_FORMAT = 'ND2' 

# Experiment identity. Both may be overridden by environment variables so
# that batch_run.py can drive many experiments through one code base
# without editing this file. A plain single run ignores the environment
# entirely and uses the literals below, exactly as before.
#
# Environment override is used rather than mutating cfg at runtime
# because the worker pool uses the 'spawn' start method: each worker
# re-imports this module from disk, so in-process attribute edits would
# silently fail to reach them.
DATA_FOLDER = os.environ.get(
    "EP_DATA_FOLDER",
    r"D:\Data\EP\260331_Trial4_InvE_BranchedCortex_SRB_400V",
)
EXPERIMENT_BASE_NAME = os.environ.get(
    "EP_EXPERIMENT_BASE_NAME",
    "DOPC_BranchedCortex_Experiment2-400V-500us-frame6",
)

# --- NEW DIRECTORY ROUTING ---
# Define the master parent directory where all analyses will be stored
PARENT_OUTPUT_FOLDER = r"D:\Data\EP\Outputs"

# Extract the 6-digit YYMMDD date from the DATA_FOLDER name
_data_dir_name = os.path.basename(os.path.normpath(DATA_FOLDER))
_match = re.search(r'^(\d{6})', _data_dir_name)
_yymmdd = _match.group(1) if _match else "000000"

# Construct the specific output subfolder: Parent\YYMMDD_EXPERIMENT_BASE_NAME
OUTPUT_IMAGE_FOLDER = os.path.join(PARENT_OUTPUT_FOLDER, f"{_yymmdd}_{EXPERIMENT_BASE_NAME}")

# Route the circle definitions JSON into the new output subfolder
CIRCLES_JSON_PATH = os.path.join(OUTPUT_IMAGE_FOLDER, f"{EXPERIMENT_BASE_NAME}_guv_circles.json")

# Required if INPUT_FORMAT = 'ND2'
ND2_FILE_NAME = f"{EXPERIMENT_BASE_NAME}.nd2"

# --- CHANNEL IDENTITY -------------------------------------------------------
# WARNING: the ND2 and TIFF branches use DIFFERENT channel orders. Switching
# INPUT_FORMAT therefore remaps every channel. A silent off-by-one here is
# unrecoverable and does not announce itself: for an Empty (no-cortex)
# sample the actin channel is featureless noise, so mistaking it for the
# membrane channel looks like a tracking failure, not a config error.
#
# These are validated at startup by utils.validate_channel_config(), which
# raises rather than falling back to a default. Do NOT reintroduce
# getattr(cfg, 'ND2_CHANNEL_IDX_*', 0) style defaults anywhere — a default
# of 0 silently selects the ACTIN channel.
#
# ND2 dimensions (typically T, C, Y, X). Acquisition order here is
# DESCENDING wavelength: 640=Actin, 561=Dye, 488=Membrane.
ND2_CHANNEL_IDX_ACTIN    = 0
ND2_CHANNEL_IDX_DYE      = 1
ND2_CHANNEL_IDX_MEMBRANE = 2

# TIFF Prefixes (typically T, C, Y, X). NOTE the different order to ND2.
ROI_CHANNEL_PREFIX    = "C1_"    # Guide / membrane channel (used for tracking)
ACTIN_CHANNEL_PREFIX  = "C2_"    # Actin channel (for future implementation)
DYE_CHANNEL_PREFIX    = "C3_"    # Dye / measurement channel

# Optional startup sanity check on the ND2 channel assignment. When True the
# pipeline inspects frame 0 of each ND2 channel and warns if the channel
# nominated as MEMBRANE is not the one with the strongest ring-like radial
# structure. Catches a transposed acquisition order before it costs a run.
VALIDATE_CHANNEL_ASSIGNMENT = True

# The new frame index pattern
TIF_SUFFIX            = "-f*.tif"

# Variable frame rate schedule: (start_frame_index, end_frame_index, interval_seconds)
# Set to `None` to attempt metadata extraction.
FRAME_INTERVAL_SCHEDULE = [
    (0,  5,  1.0),   # Pre-pulse: 5 frames, 1 s interval
    (5,  22, 0.1),   # Pulse: 17 frames, 100 ms (0.1 s) interval
    (22, None, 5.0)  # Post-pulse: remaining frames, 5 s interval
]

# --- Per-experiment frame truncation -----------------------------------------
# Keys are EXPERIMENT_BASE_NAME, values the number of frames to KEEP
# (indices 0 .. n-1). Applied in run_analysis.load_input_data by shrinking the
# declared memmap shape, so the .dat files themselves are untouched.
#
# Used where the synchronized-exit QC fires and the video confirms the
# vesicles are intact after the event: cutting the record at the last good
# frame recovers the GUVs for analysis, whereas discarding the field of view
# loses them. Note that t_final shortens accordingly, so these GUVs will have
# a shorter record than their siblings -- check any fixed-time descriptor
# (frac_remaining_300s and later) is NaN rather than extrapolated for them.
FRAME_TRUNCATION = {
    # Focal-plane excursion at frames 78-91: all 25 GUVs exit within 3 frames
    # of frame 79 (t = 291.6 s) and are visibly intact at frame 95. Confirmed
    # from the ALL_TRACKING video and from a synchronized dye step at
    # 291.6 s in 24/24 GUVs (std = 0.00).
    'DOPC_Empty_Experiment2-360V-500us-frame5': 78,
}

# -----------------------------------------------------------------------------
# --- 2. ANALYSIS & MASKING PARAMETERS ---
# -----------------------------------------------------------------------------

# 'LABELED'   – bright ring (fluorescent membrane)
# 'UNLABELED' – dark ring  (brightfield / phase contrast)
MEMBRANE_DETECTION_MODE = 'LABELED'

# Search window for radius detection: look ± this fraction of current radius.
MEMBRANE_SEARCH_FACTOR    = 0.35

# Half-width of the membrane mask in pixels (mask = peak ± this).
MEMBRANE_FIXED_HALF_WIDTH = 5   # Increased from 3

# scipy.signal.find_peaks parameters
PEAK_FIND_MIN_DISTANCE   = 5
PEAK_FIND_MIN_PROMINENCE = 0.05

# Photometry smoothing
# Whether to Gaussian-blur (5x5, sigma=1.5) each frame before sampling the
# lumen and background masks.
#
# Default is now False. Blurring before a MEAN over a large mask is close to
# a no-op, but blurring before a PERCENTILE or a STD is not: it destroys the
# per-pixel noise statistic that the dead-GUV filter divides by. Whatever
# denoising is wanted should happen after photometry, on the trace, not on
# the pixels feeding a variance estimate. Set True only to reproduce the
# historical (pre-fix) numbers.
PHOTOMETRY_BLUR = False
PHOTOMETRY_BLUR_KERNEL = 5
PHOTOMETRY_BLUR_SIGMA  = 1.5

# Background annulus geometry
# NOTE ON THE CURRENT VALUES: with MEMBRANE_FIXED_HALF_WIDTH = 5 these place
# the sampling ring at r+8 to r+10 px, i.e. 0.88-1.10 um outside the
# membrane and only 2 px wide. Confocal PSF tails deposit lumen signal at
# that distance, so the ring is expected to carry some vesicle signal and
# the background is biased HIGH (which biases the normalisation denominator
# |I_bg - I_dye,0| LOW, amplifying its noise).
# These are left at their historical values so this fix does not silently
# change your numbers. Recommended: widen to BG_BUFFER_PIXELS = 6 and
# BG_RING_WIDTH_PIXELS = 6, then confirm on a sparse field that the
# background median stops depending on the buffer distance.
BG_BUFFER_PIXELS = 6        # gap (px) between outer membrane edge and bg ring start
BG_RING_WIDTH_PIXELS = 4    # width (px) of background sampling ring

# One-off QC: measures the bath level far from any tracked vesicle, to confirm
# the annulus is not sitting in a locally bright region. Off by default — it
# costs ~50 s per run and tells you nothing about kinetics. Worth running once
# per new imaging condition.
# One-off QC: measures the bath level far from any tracked vesicle, to
# confirm the annulus is not sitting in a locally bright region. Off by
# default - it costs ~50 s per run and tells you nothing about kinetics.
# Worth running once per new imaging condition.
# NOTE ON THE LEVEL (affects plateaus, NOT timescales)
# A percentile is robust but is not an unbiased estimate of the background
# MEAN: it sits |z(p)|*sigma below it (1.036 sigma at p15). This shifts the
# normalised curve by an affine constant, so the fitted TAU is unaffected -
# only the apparent plateau Iinf is offset upward. Measured on the 260422
# Empty run: annulus p15 = 97.0 vs far-field mean = 102.1, i.e. 5.1 counts
# = 1.00 sigma, matching theory. A fully-emptied vesicle therefore reads
# about delta/sep0 (~0.4 there) rather than 0. Do NOT read a fitted Iinf as
# 'fraction of dye retained' without accounting for this.

# Use the low-quantile anchor instead of the median to estimate background
BG_LEVEL_ESTIMATOR = 'percentile'

# Which low quantile of the annulus is taken as THE background level.
# RESTORED: this constant had gone missing from config, so every call site
# was silently falling back on its getattr default of 15.0 and the value
# could not be changed from here. 15.0 reproduces that behaviour exactly, so
# restoring it changes no results — it only makes the knob reachable again
# and lets validate_config() see the value it checks.
# Must satisfy 0 < BG_PERCENTILE < BG_SIGMA_UPPER_PERCENTILE < 50.
BG_PERCENTILE = 15.0

# A fitted Iinf therefore mixes genuine trapped dye with this offset and the
# curve alone cannot separate them. No correction is applied: do not read a
# fitted Iinf as 'fraction of dye retained'.

# -----------------------------------------------------------------------------
# --- 3. FITTING & MODELING ---
# -----------------------------------------------------------------------------

# Dye Efflux: ['EFFLUX-1EXP' or 'EFFLUX-2EXP'] 
# Dye Influx: ['INFLUX-1EXP' or 'INFLUX-2EXP']
MODEL_TO_USE        = 'EFFLUX-1EXP' 

# Fraction of the post-pulse record actually passed to the fitter, measured
# in TIME (not frame count), so the variable frame-rate schedule doesn't
# make this mean different things in different segments.
# 1.0 = fit the whole post-pulse record.
# NOTE: this knob existed previously but was never read by any code path —
# every fit silently used 100% of the record. It is now honoured. Leave it
# at 1.0 to reproduce the historical behaviour.
FIT_DATA_PERCENTAGE = 1.0

# --- Kinetic model parameter bounds -----------------------------------------
# The efflux models are parameterised as
#     1EXP:  I(t) = Iinf + A  * exp(-t/tau)                [+ D*t]
#     2EXP:  I(t) = Iinf + a1 * exp(-t/tau1) + a2 * exp(-t/tau2)  [+ D*t]
# with A, a1, a2 >= 0 and Iinf >= 0. Under these bounds the 1EXP feasible
# set is a strict SUBSET of the 2EXP feasible set, so RSS_2EXP <= RSS_1EXP
# holds by construction and the nesting assertion below can only ever be
# tripped by an optimiser failure.
#
# Historical note: the old bounds allowed Iinf in [-2, 2] with a free I0,
# which admitted a *rising* exponential (Iinf > I0, i.e. negative
# amplitude) for an efflux model, while the 2EXP amplitudes were pinned
# >= 0. That asymmetry broke nesting and let ~half of all GUVs settle on
# a degenerate "rise toward Iinf=2, cancelled by a negative linear drift"
# solution with tau far longer than the recording.
FIT_IINF_MAX     = 1.2     # upper bound on the plateau (1.0 + headroom for noise)
FIT_AMPLITUDE_MAX = 2.0    # upper bound on each exponential amplitude
FIT_TAU_MIN      = 0.01    # s
FIT_TAU_MAX      = 5000.0  # s

# --- Linear drift term D ----------------------------------------------------
# Set True to include a +D*t term in every kinetic model.
#
# Default is now False. D was previously always active with bounds of
# +/-0.1 per second, i.e. +/-72 over a 725 s record on data that spans 0-1 —
# not a drift correction but a free linear ramp able to cancel almost any
# curvature. Measured drift in non-responding GUVs (which by definition
# have no efflux to explain) is ~1.5% over a full record, so there is
# nothing for D to absorb. Only enable this if you have independently
# demonstrated bleaching in YOUR data, and if you do, keep the bound tight.
FIT_INCLUDE_DRIFT = False
FIT_DRIFT_ABS_MAX = 5e-5   # per second; used only when FIT_INCLUDE_DRIFT

# --- Multi-start ------------------------------------------------------------
# Number of additional randomised initial guesses tried per model, on top of
# the deterministic heuristic guess. The best (lowest-RSS) converged fit is
# kept. Guards against curve_fit stalling at a bound, which previously left
# a subset of 2EXP fits returning their starting point unchanged.
FIT_N_MULTISTART = 6
FIT_MULTISTART_SEED = 0     # set for reproducible fits; None for nondeterministic

# --- Nesting assertion ------------------------------------------------------
# A 2EXP fit whose RSS exceeds the 1EXP RSS by more than this tolerance is
# mathematically impossible at a true optimum and indicates the optimiser
# failed. Such GUVs get their model comparison written as NaN with a
# 'FIT_FAILED' flag rather than a spurious "1EXP preferred" verdict.
FIT_NESTING_TOLERANCE = 1e-6

# --- Identifiability gate ---------------------------------------------------
# tau cannot be recovered from a record that ends long before the decay
# completes: the data then constrain only the initial slope, and Iinf/tau
# trade off freely. Fits violating either criterion below are retained in
# the output but flagged `tau_identifiable = False`, and their tau/Iinf are
# NOT to be used for population statistics. Use the model-free metrics
# (t50, frac_remaining_*) exported alongside them instead.
TAU_SE_RATIO_MAX          = 0.5   # reject if tau_SE/tau exceeds this
TAU_MAX_FRACTION_OF_RECORD = 1/3  # reject if tau > this fraction of the record

# --- Normalisation denominator guard ----------------------------------------
# I_retained(t) = (I_bg,t - I_dye,t) / (I_bg,t - I_dye,0)
# The denominator is a DIFFERENCE OF TWO SIMILAR NUMBERS, so it must be
# guarded against noise pushing it through zero and flipping the sign.
#
# THE RELEVANT SCALE IS THE STANDARD ERROR OF THE BACKGROUND ESTIMATE,
# NOT PER-PIXEL SIGMA. I_bg,t is a percentile of ~700 annulus pixels, so its
# sampling error is
#       SE = sqrt(p(1-p)/n) / phi(z_p) * sigma
# which is 0.058*sigma at p=0.15, n=696 — about 17x smaller than sigma itself.
#
# An earlier version of this guard compared |denominator| against per-pixel
# sigma and demanded > 3 sigma. The measured contrast in the 260422 Empty run
# is sep0/sigma ~= 2.3, so that condition blanked 120-166 of 166 frames on
# most GUVs and removed 17 vesicles from the analysis entirely — despite
# those denominators sitting roughly 40 standard errors clear of zero.
# Frames are now blanked only when the denominator is genuinely unresolvable.
NORM_DENOM_MIN_SE = 5.0

# --- Size grouping ----------------------------------------------------------
# Bin edges (um) used for the parameter boxplots and the size_group column.
# NOTE: a single field of view will not give balanced bins at any choice of
# edges. For size-dependence claims, pool across experiments in
# process_bulk_data.py and use quantile bins, or better, regress the
# parameter against continuous radius and skip binning entirely — bin edges
# are an arbitrary analysis choice that has to be defended.
SIZE_GROUP_BINS   = [0, 5.0, 8.0, 11.0, 100.0]
SIZE_GROUP_LABELS = ['<5.0', '5.0-8.0', '8.0-11.0', '>11.0']

# Warn when a size bin holds fewer than this many GUVs. A bin of n=1 is not
# a population and should not be presented as a box.
SIZE_GROUP_MIN_N = 3

# Times (s, relative to pulse) at which model-free fractional dye retention
# is reported for every GUV, identifiable or not.
MODEL_FREE_REPORT_TIMES_S = [10, 30, 60, 120, 300, 600]

# --- Response detection -----------------------------------------------------
# A GUV below the permeabilisation threshold has a FLAT trace: the fit drives
# the amplitude to ~0, at which point tau is arbitrary and its standard error
# can come out deceptively small, sneaking a meaningless tau past the
# identifiability gate. Such GUVs are valid data points — locating the voltage
# threshold is the point of the experiment — they simply do not carry a tau.
# A vesicle counts as having RELEASED dye when its drop, measured against its
# own PRE-pulse mean at the matched endpoint, exceeds both an absolute floor
# and a multiple of its own frame-to-frame noise, so the test adapts to each
# trace instead of using one fixed cutoff.
#
# The call is made in process.aggregate_pipeline_results, not at fit time.
# It used to be made in run_analysis against a baseline taken from the first
# few POST-pulse frames, which is wrong here: poration onset falls inside the
# first imaging frame, so that baseline had already absorbed part of the
# release and the test saw only the remainder. The two thresholds below are
# unchanged; what changed is the baseline they are applied to.
FIT_RESPONSE_AMPLITUDE_SIGMA = 3.0   # read by process.RELEASE_NOISE_SIGMA
FIT_MIN_RESPONSE_AMPLITUDE   = 0.05  # see process.INTENSITY_DIFF_THRESHOLD

# --- Stepwise traces --------------------------------------------------------
# The efflux formula assumes ONE permeabilisation event followed by a
# monotonic relaxation. Some traces instead decline slowly and then drop
# abruptly partway through — a second event, or a tracking/mask artefact.
# Neither a 1EXP nor a 2EXP can describe that: the fit compromises between the
# segments and returns a tau belonging to neither. Steps are detected from the
# time-normalised derivative (robust MAD scale) and flagged in the fit table
# as has_step / n_steps / first_step_t_s / largest_step_drop.
STEP_DETECT_SIGMA    = 6.0    # drops beyond this many robust sigmas of dI/dt
STEP_DETECT_MIN_DROP = 0.10   # and at least this much normalised intensity

# Restrict the fit to data BEFORE the first detected step, so the efflux
# formula describes a single relaxation. Off by default because it silently
# shortens the record; turn on once you have looked at which GUVs are flagged.
FIT_TRUNCATE_AT_FIRST_STEP = False

# --- z-drift / swelling -----------------------------------------------------
# A vesicle whose equatorial radius GROWS over the record is either genuinely
# swelling or drifting through the confocal plane toward its equator. Either
# way the interior mask samples a changing volume and an apparent intensity
# change is not necessarily efflux. Shrinkage is already handled by the fate
# classifier; growth was not flagged anywhere.
RADIUS_GROWTH_FLAG_FRACTION = 0.10

# Pulse-frame detection
# Set to an integer to hard-pin the pulse frame and skip auto-detection.
# Set to None to let the pipeline detect it from the dye signal drop/rise.
PULSE_FRAME_OVERRIDE = None

# Gaussian σ (frames) used to smooth the mean trace before differentiation.
# Increase if the signal is noisy; decrease if pulses are very abrupt.
PULSE_DETECT_SMOOTH_SIGMA = 2.0

# Dead-GUV filter: a GUV is excluded from dye-channel analysis if its
# pre-pulse interior signal is not separated from background by at least
# this many multiples of the background per-pixel std:
#   |I_dye,0 - I_bg,0|  <  MIN_PREPULSE_SEPARATION_SIGMA * bg_std0
#
# Set to None to disable exclusion entirely and only REPORT the distribution.
# That is the default, deliberately: see below.
#
# WHY THE OLD VALUE OF 1.1 WAS NOT A DATA PROPERTY
# ------------------------------------------------
# 1.1 was tuned against a bg_std0 that was not a per-pixel sigma. The old
# estimator was np.std(pixels[pixels <= median]) on a GAUSSIAN-BLURRED frame,
# which understates sigma twice over (the blur suppresses per-pixel variance;
# the lower-half truncation costs a further ~0.6x). The observed "no natural
# pass/fail gap around 1.1" was a property of that broken statistic, not of
# the vesicles.
#
# Independent check from your own refitted data: with the p15 anchor the
# apparent plateau floor is  artefact = |z(0.15)| / (sep0/sigma) = 1.0364 /
# (sep0/sigma). A measured Iinf is artefact PLUS any genuine retained dye, so
# Iinf >= artefact, hence sep0/sigma >= 1.0364/Iinf. The four identifiable
# responders refit to Iinf = 0.085-0.133, which forces
#       sep0/sigma  >=  7.8   (and >= 12 for the cleanest vesicle)
# A ratio of 1.1 would imply a 94% plateau floor -- curves could not fall to
# 0.1 at all. So the true contrast is ~7-11x higher than the old number said.
#
# HOW TO CALIBRATE (2 minutes, once)
# ----------------------------------
# Every run now writes {experiment}_dye_qc.csv containing sep0, bg_std0 and
# sep0_over_sigma for EVERY GUV -- including the ones the filter rejects, which
# previously never reached any export, so the distribution you needed to
# threshold on was invisible. The log also prints the quantiles and the largest
# gap in the sorted ratios.
#   1. run one known-good experiment with this set to None,
#   2. open dye_qc.csv and look at sep0_over_sigma,
#   3. if there is a clean gap, put the threshold in it;
#      if the distribution is smooth, use the artefact table below instead.
#
MIN_PREPULSE_SEPARATION_SIGMA = None


# Maximum number of frames to evaluate for the pulse within the fast-acquisition window.
# Set to None to search the entire fast window.
PULSE_SEARCH_MAX_FRAMES = 5

# -----------------------------------------------------------------------------
# --- 4. OUTPUT & VISUALIZATION ---
# -----------------------------------------------------------------------------

# Output sub-folders (created automatically at runtime inside the new directory)
FOLDER_TRACKING = os.path.join(OUTPUT_IMAGE_FOLDER, "tracking")   # track PNGs/AVIs, metrics, CSVs
FOLDER_MASKS    = os.path.join(OUTPUT_IMAGE_FOLDER, "masks")      # mask overlay AVI
FOLDER_DYE      = os.path.join(OUTPUT_IMAGE_FOLDER, "dye")        # dye-channel root: diagnostics + snapshots

# The dye folder is split so model-dependent and model-free products never
# sit side by side. Anything under fitting/ inherits the assumptions of
# MODEL_TO_USE and is only meaningful when the post-pulse sampling actually
# resolves the transient; anything under intensity/ is a direct measurement
# that stands on its own. Diagnostics belonging to neither (pulse-frame
# detection, background diagnostics, contrast QC, dye snapshots) stay in the
# dye/ root.
FOLDER_DYE_FITTING   = os.path.join(FOLDER_DYE, "fitting")     # fit grids, fit params, model comparison, boxplots
FOLDER_DYE_INTENSITY = os.path.join(FOLDER_DYE, "intensity")   # normalised traces, endpoint figure, crops
FOLDER_ACTIN    = os.path.join(OUTPUT_IMAGE_FOLDER, "actin")      # actin cortex plots, traces, profiles

EXPORT_TIME_POINTS_S     = [0, 50, 100, 200, 300]
EXPORT_DEBUG_PLOTS       = True

# -----------------------------------------------------------------------------
# --- BULK ANALYSIS: SIZE WINDOW ---
#
# Radius range in micrometres that a vesicle must fall inside to enter the
# bulk comparison between preparations. None disables the restriction.
#
# THIS IS THE ONLY PLACE THE WINDOW IS SET. process.py reads it from here and
# every other bulk script reads it from process.py, so changing this number
# changes the whole pipeline. Nothing else should define a radius range.
#
# Why it exists. The two preparations differ by 2.09 um in median radius
# (6.58 for Bare against 4.49 for Branched cortex), and the induced
# transmembrane potential scales with radius, so a comparison at matched
# applied field is not a comparison at matched dose. The alternative is to
# divide the dose axis by radius, which assumes the inverse radius-field
# relationship Schwan's equation predicts; Mercadal et al. (2016) report that
# this is often not observed experimentally, or is much shallower than
# predicted. Restricting on size assumes nothing about how dose scales.
#
# Choice of width, measured on the stagnate vesicles with an endpoint:
#
#   window      Bare  cortex   median dR   p(size)   responding      p
#   4.5-5.5       92      82     +0.03 um     0.27   4/92, 11/82   0.055
#   4.25-5.75    124     126     +0.18 um    0.008   5/124, 19/126 0.004
#   4.0-6.0      159     154     +0.41 um   <0.001   6/159, 20/154 0.004
#
# 4.0-6.0 is used. The narrow window matches almost exactly but leaves too
# few vesicles per field to read most conditions. The residual 0.41 um at
# 4.0-6.0 is detectable and must be reported rather than described as a
# match, but it runs conservatively: Bare vesicles remain the larger, reach
# the higher induced potential, and should porate more readily than they do.
#
# WHAT IS EXEMPT. The cortex-breakdown stages -- radial peak height, its time
# course, and the pole-versus-equator index -- run on every vesicle regardless
# of this setting. They describe the cortex-bearing population on its own and
# contain no Bare-versus-Branched comparison, so the confound this window
# removes is not present in them; inside the window they would lose three
# quarters of their vesicles and the 30 V bleach reference they are read
# against. run_bulk.py routes those stages to the unrestricted frame.
# -----------------------------------------------------------------------------
SIZE_WINDOW_UM = (4.0, 6.0)

MICRONS_PER_PIXEL        = 0.11 # Plan Apo λ 60x Oil
SCALE_BAR_LENGTH_MICRONS = 10

# -----------------------------------------------------------------------------
# --- MODEL-FREE ENDPOINT OUTPUTS (dye/intensity/) ---
# -----------------------------------------------------------------------------

# Per-GUV multipanel plot (and CSV) of normalised dye intensity BEFORE the
# pulse vs. at the END of the movie. Intended for datasets where the
# post-pulse acquisition is too coarse to resolve the efflux transient, so a
# fitted tau is set by the sampling interval rather than by membrane
# permeability. The endpoint contrast assumes nothing about the shape of the
# trajectory between the two time points.
EXPORT_PREPOST_INTENSITY_PLOT = True

# Frames averaged for the pre-pulse baseline level. The window ends at the
# frame immediately before the detected pulse frame — the same window the
# normalisation uses, so I_pre comes out at ~1 by construction.
PREPOST_N_PRE_FRAMES = 5

# Frames averaged at the END of each GUV's own trace. Taken per-GUV from its
# last FINITE post-pulse frames, so a vesicle whose tracking ended early
# reports its own true endpoint (with t_final_s in the CSV recording when
# that was) rather than being padded or dropped. Set to 1 for the strict
# last frame; >1 averages down endpoint noise at the cost of reaching
# slightly further back in time.
PREPOST_N_FINAL_FRAMES = 3

# --- Endpoint QC gates ---
# Two ways an endpoint can be numerically fine but scientifically wrong.
# GUVs failing either gate are FLAGGED, never deleted: they keep their row in
# the CSV (column `endpoint_qc`, with `qc_pass` False) and their panel in the
# figure, but they are left out of the reported population mean.

# Minimum length of a GUV's own observation window, as a fraction of the
# post-pulse duration of the movie. Below this, its "final" value is an early
# post-pulse sample rather than an endpoint, and pooling it with full-length
# traces silently averages over two different observation windows. A vesicle
# that ruptures or leaves the field a few seconds after the pulse typically
# lands here. Set to 0.0 to disable the gate.
PREPOST_MIN_FINAL_TIME_FRAC = 0.5

# Maximum permitted RISE of the endpoint above the pre-pulse baseline, in
# units of that GUV's own pre-pulse SD (floored at the population median SD,
# so an unusually quiet baseline cannot manufacture a large sigma from a
# trivial excursion). Efflux only ever lowers lumen intensity, so an endpoint
# significantly above baseline is not dye loss — it is the ROI having come to
# enclose brighter surroundings instead of a lumen, which is what happens
# after rupture. Left unflagged, such a GUV reports a large NEGATIVE release
# and drags the population mean down while inflating its SD.
#
# The test is one-sided by design: a very low endpoint is the expected signal
# and is never flagged. Set high (e.g. np.inf) to disable the gate.
PREPOST_MAX_ENDPOINT_RISE_SIGMA = 5.0

# Single-vesicle crops accompanying the endpoint figure: membrane + dye
# (+ actin when ANALYZE_ACTIN_CHANNEL is on) at the last pre-pulse frame and
# at each GUV's own final frame.
EXPORT_GUV_CROPS = True

# Half-width of the crop box in units of the pre-pulse radius. 2.0 spans four
# radii, leaving roughly one radius of surroundings each side. The box is
# sized from the PRE-pulse radius and held constant for the final frame, so
# deflation shows as the vesicle shrinking inside a fixed field rather than
# being re-zoomed to fill it.
CROP_FACTOR = 2.0

# Also write each crop as a standalone 8-bit PNG into dye/intensity/crops/
# (one file per GUV per channel per timepoint), for assembling figures by
# hand. The montage PNG is written regardless.
CROP_SAVE_INDIVIDUAL = True

# Use ONE crop box size for every GUV in the montage, sized from the largest
# pre-pulse radius. Matplotlib stretches each panel to fill its axes, so
# per-GUV box sizes would make a 10 um and a 25 um vesicle look identical; a
# single box keeps the whole montage at one pixel scale. False gives tighter
# framing of small vesicles at the cost of between-GUV size comparison.
CROP_UNIFORM_BOX = True

# Display colours for the black-to-colour lookup applied to each channel.
CROP_CHANNEL_COLORS = {
    'membrane': '#00FF66',
    'dye':      '#FF3355',
    'actin':    '#33CCFF',
}

# Percentiles for contrast boosting (Lower p_low or lower p_high = more contrast)
CONTRAST_P_LOW  = 0.5  # Ignore bottom 0.5% of pixels
CONTRAST_P_HIGH = 98.0 # Ignore top 2.0% of pixels

EXPORT_MASK_VISUALIZATION     = False
MASK_VIZ_OVERLAY_ALPHA        = 0.7
VIZ_MEMBRANE_THICKNESS_PIXELS = None

# Consolidated video export:
EXPORT_CONSOLIDATED_TRACK_VIDEO = True
EXPORT_CONSOLIDATED_MASK_VIDEO  = True
VIDEO_EXPORT_FPS                = 10.0


# -----------------------------------------------------------------------------
# --- 5. TRACKING PARAMETERS ---
# -----------------------------------------------------------------------------

# Add this flag to disable dye analysis
TRACKING_ONLY_MODE = False

# Centre grid search
# Search radius = previous_radius × TRACKING_SEARCH_WINDOW_FACTOR.
# Increase if GUVs move more than ~70 % of their radius between frames.
TRACKING_SEARCH_WINDOW_FACTOR = 0.7 

# Grid step = previous_radius × TRACKING_GRID_STEP_FACTOR.
# Smaller = finer / slower. 0.12–0.20 is a good balance.
TRACKING_GRID_STEP_FACTOR = 0.20 

# Radius change clamp 
# Maximum fractional radius change allowed per frame.
# 0.20 = up to 20 % shrink or grow per frame.
TRACKING_MAX_RADIUS_CHANGE_FACTOR = 0.20

# Rupture detection
# GUV is declared ruptured when its ring-quality score < this threshold.
# Typical good-frame scores are 10–30; set this to ~20–30 % of that.
RUPTURE_SCORE_THRESHOLD = 4.0

# Number of consecutive frames below RUPTURE_SCORE_THRESHOLD required
# to declare rupture (reduces false positives from single bad frames).
RUPTURE_DETECTION_CONSECUTIVE_FAILS = 3

# Exit classification: OUT_OF_FRAME vs RUPTURED is decided by the geometry
# of the vesicle at the exit, not by a majority vote over the failing
# streak. "Clearance" is the distance from the vesicle to the nearest frame
# edge MINUS the extent that has to be visible, expressed in units of the
# vesicle's own mean radius. Negative means part of what we need to see
# lies outside the frame, so the vesicle is out of frame rather than burst.
#
# Two clearances are recorded per exit in *_detection_quality.csv:
#   exit_clearance_profile_r  - against the window the ring scorer actually
#                               reads, MEMBRANE_SEARCH_FACTOR beyond the
#                               membrane plus a 5 px margin (1.35 r + 5 at
#                               the default). This is the one that decides
#                               the label: a vesicle whose profile window is
#                               clipped cannot be scored, so its ring
#                               disappears for a geometric reason.
#   exit_clearance_vesicle_r  - against the fitted ellipse itself (bounding
#                               circle on the semi-major axis). Recorded for
#                               audit only. A GUV with this positive and the
#                               profile clearance negative is one whose
#                               outline was visible but not measurable.
#
# 0.0 means "any clipping at all is out of frame". Calibrate from the two
# columns once a full run exists rather than trusting this default.
OUT_OF_FRAME_CLEARANCE_R = 0.0

# GUV fate refinement: distinguishes gradual SHRINKAGE (deflation) from
# abrupt RUPTURE. A GUV whose radius has declined by at least this fraction
# from its own pre-pulse baseline is classified SHRUNK regardless of how
# tracking ended — including overriding a raw RUPTURED tag, since a vesicle
# that deflates below the ring-detector's minimum resolvable size will also
# trip the tracker's rupture-score exit, but that's a shrinkage artifact of
# detection, not a membrane burst. 0.15 = 15% radius loss.
SHRINKAGE_FRACTION_THRESHOLD = 0.15

# Number of the LAST valid tracked frames (per GUV) averaged to get its
# terminal radius for the shrinkage/growth check. Smooths single-frame ring
# detection noise right at the endpoint rather than keying the whole
# fate classification off one potentially noisy frame.
SHRINKAGE_TERMINAL_N_FRAMES = 3

# --- Synchronized-exit QC ----------------------------------------------------
# Fails a field of view when this fraction or more of its GUVs stop tracking
# within QC_SYNC_EXIT_WINDOW_FRAMES of each other, away from the pulse.
# Vesicles porate independently, so their exit frames scatter; simultaneous
# loss of the whole field is a focal-plane excursion, stage bump or
# illumination dropout, not lysis. Every affected GUV is tagged RUPTURED and
# never re-acquired, because RUPTURED is a terminal exit state.
#
# 0.5 is deliberately permissive. The failure mode produces fractions at or
# near 1.0, so a tighter gate buys nothing and would start flagging genuine
# high-field poration. Clusters inside the pulse window are reported but not
# failed: mass poration at the pulse is the experiment working.
QC_SYNC_EXIT_MIN_FRACTION  = 0.5
QC_SYNC_EXIT_WINDOW_FRAMES = 3
QC_SYNC_EXIT_MIN_GUVS      = 3

# GUV fate: GROWTH threshold. A GUV whose radius has increased by at least
# this fraction above its own pre-pulse baseline, with tracking intact
# throughout, is classified GROWN. 0.15 = 15% radius gain.
#
# Kept as a SEPARATE constant from SHRINKAGE_FRACTION_THRESHOLD rather than
# reusing one symmetric value: equal fractional radius changes are not
# equivalent in area or volume (-15% radius = -39% volume, but +15% radius =
# +52% volume), and the two directions have different physical ceilings.
# A bilayer cannot stretch more than a few percent in AREA before lysing, so
# a >15% radius gain (~32% area) is impossible by inflation of an already-
# taut vesicle. Real GROWN calls therefore mean an initially floppy vesicle
# with excess membrane area rounding up as it tenses (the interesting case),
# fusion with a neighbour, or the ring detector jumping to another object.
# Cross-check the terminal_norm_circularity column in the fate CSV: case one
# is accompanied by circularity rising toward 1, the artifacts usually not.
GROWTH_FRACTION_THRESHOLD = 0.15

# Whether the growth test uses the PEAK post-pulse radius (True) instead of
# the terminal radius (False). Post-electroporation swelling is often
# transient — a vesicle can inflate, reseal, and re-equilibrate back toward
# baseline, which a terminal-frame criterion misses entirely. Set True to
# count transient swelling as GROWN; leave False to require the expansion to
# persist to the end of the movie (the conservative definition, symmetric
# with how SHRUNK is scored). Either way, peak_norm_radius_post is exported
# in the fate CSV, so transient swelling can be checked without re-running.
GROWTH_USE_PEAK_RADIUS = False

# -----------------------------------------------------------------------------
# Parameter boxplots: which FITTED parameters get a panel.
# -----------------------------------------------------------------------------
# Deliberately an explicit shortlist, not "everything numeric in
# fit_parameters.csv". The sweep it replaces produced ~24 panels — flags,
# backing diagnostics, rss, model-free metrics — in one 36000 px-wide figure,
# burying the three quantities the model actually estimates among bookkeeping
# columns. Everything else stays in fit_parameters.csv, which is where
# per-GUV numbers belong; this figure exists only to show how the fitted
# parameters vary with vesicle size.
#
# For a 2EXP model the parameters are named differently, so use
#   ['Iinf', 'a1', 'tau1', 'a2', 'tau2']
# Names not present for the active model are skipped with a log note.
BOXPLOT_PARAMS = ['Iinf', 'A', 'tau']

# -----------------------------------------------------------------------------
# --- RESPONSE PHENOTYPE CLASSIFICATION ---
# -----------------------------------------------------------------------------
# Sorts responding GUVs into EXPONENTIAL / DELAYED / GRADUAL. The point of the
# split is that "it went down" covers at least three physically different
# things, and only one of them supports a tau.
#
# EXPONENTIAL threshold: exp_vs_linear = RSS(straight line) / RSS(model),
# both fitted to the same window. An exponential whose tau far exceeds the
# record is, within that record, a straight line — the fit still returns a
# tau and the panel still looks like a decay, but the data do not contain the
# curvature that would justify calling it one.
#
# On the 260422 Empty 400 V run the distribution of exp_vs_linear has a clean
# gap: fifteen GUVs at 0.61-1.98, then four at 5.17-11.26 and nothing in
# between. Any threshold in 2.0-5.0 gives the identical split, so 3.0 sits in
# the middle of the gap rather than on the edge of it. Re-check that gap on a
# new imaging condition before trusting the number.
# NOTE: this no longer drives a response_class label -- that column has
# been removed. It survives as the input to exp_vs_linear /
# onset_lag_frac in the fit table, which are descriptive only.
RESPONSE_SHAPE_MIN_RSS_RATIO = 3.0

# DELAYED threshold: onset_lag_frac = t10 / t90, where t10 and t90 are the
# times at which the trace has completed 10 % and 90 % of its OWN total
# decline. A vesicle permeabilised by the pulse itself starts leaking at once
# (ratio ~0.00); one that sits flat and only then falls gives a large ratio.
#
# Same dataset: fourteen GUVs at 0.001-0.037, then six at 0.096-0.469. The
# gap runs 0.04-0.09, so 0.07 is its midpoint.
#
# NOTE this is NOT the same thing as STEP_DETECT_* below. A step is one abrupt
# frame-to-frame drop; a delayed onset can be a perfectly smooth decline that
# simply begins late. On this dataset the two flag almost disjoint sets of
# GUVs, so both are worth keeping.
# NOTE: this no longer drives a response_class label -- that column has
# been removed. It survives as the input to exp_vs_linear /
# onset_lag_frac in the fit table, which are descriptive only.
RESPONSE_ONSET_LAG_FRAC = 0.07

# Tracking outputs
# CSV: per-frame (guv_id, frame, x, y, radius, ring_score)
EXPORT_TRACKING_DATA = True

# PNG: centre trajectory overlaid on the last valid ROI frame
EXPORT_TRACK_VISUALIZATION = True


# -----------------------------------------------------------------------------
# --- 7. ACTIN CORTEX ANALYSIS ---
# -----------------------------------------------------------------------------

# Enable actin cortex analysis on the C2 channel
def _actin_channel_default() -> bool:
    """Actin analysis on for corticated samples, off for Empty ones.

    Derived from EXPERIMENT_BASE_NAME so a batch run needs no
    per-experiment edit. EP_ANALYZE_ACTIN overrides it when set.

    NOTE: this switches ANALYSIS only, not channel indices. Empty
    acquisitions are still assumed to contain three channels in the
    same order, with the actin channel simply featureless. If an Empty
    ND2 was acquired with only two channels the ND2_CHANNEL_IDX_*
    constants are wrong for it and must be set separately.
    """
    env = os.environ.get('EP_ANALYZE_ACTIN')
    if env is not None:
        return env.strip().lower() in ('1', 'true', 'yes', 'on')
    return 'empty' not in EXPERIMENT_BASE_NAME.lower()


ANALYZE_ACTIN_CHANNEL = _actin_channel_default()

# Which mask to use for the cortex signal.
# 'membrane' = the ring mask (peak ± MEMBRANE_FIXED_HALF_WIDTH).
# 'inner'    = the inner lumen mask (gives lumenal actin only).
# Both are always extracted; this controls the *cortex* definition.
ACTIN_CORTEX_MASK = 'membrane'

# Half-width (pixels) for the cortex ring mask in C2.
# Increase slightly if the cortex ring is thicker than the membrane label.
ACTIN_CORTEX_HALF_WIDTH = 5   # px  (~0.44 µm at 0.11 µm/px)

# Minimum peak prominence for cortex detection.
# The peak-to-baseline amplitude must exceed this fraction of the total
# radial profile range (max − min) to be accepted as a real cortex peak.
# Raise this value if flat-profile GUVs (no cortex) are being falsely detected;
# lower it if faint-but-real cortices are being missed.  Default = 0.10 (10 %).
ACTIN_PEAK_MIN_PROMINENCE = 0.10

# --- Cortex presence classification (pre-pulse) ---
# cortex_contrast = (I_peak - I_lumen) / (I_lumen - I_bg), measured on
# pre-pulse frames. Separates a corticated vesicle from one merely filled
# with unpolymerised actin. PROVISIONAL: set these from the observed
# contrast distribution across your BranchedCortex experiments before use.
CORTEX_CONTRAST_HIGH = 0.35   # >= this -> CORTEX
CORTEX_CONTRAST_LOW  = 0.15   # <= this -> NO_CORTEX; between -> AMBIGUOUS
CORTEX_MAX_FWHM_UM   = 1.5    # a peak broader than this is not a thin shell

# Number of angular samples used for the cortex angular profile and Gini index.
# Higher values give a finer angular map but increase per-frame compute time.
# 72 = 5° resolution (matches the tracking grid search default).
ACTIN_N_ANGLES = 72

# Padding (pixels) added outward to each side of the membrane FWHM border before
# it is used as the search window for the actin peak.  Matches the ±3 px padding
# in skeleton.py's membrane_detection().  Increase if the actin peak is being
# clipped at the membrane edge; decrease (or set to 0) for tighter confinement.
ACTIN_MEMBRANE_BORDER_PADDING = 3

# Export a per-GUV CSV and plot of cortex / lumen actin over time
EXPORT_ACTIN_TRACES = True

# Smoothing sigma (frames) for the actin ratio plot; set 0 to disable
ACTIN_PLOT_SMOOTH_SIGMA = 1.5

# Electrode axis orientation, in degrees, in the IMAGE FRAME (0 = along +x /
# horizontal, matching the angular sampling convention used for
# 'angular_profiles'). The electrode-facing poles are drawn at this angle
# and this angle + 180 on the actin angular kymograph. This is the
# CODE-frame angle used internally for all pole/equator masking and the
# polarization index — do not change this to relabel the plot axis; see
# ANGLE_DISPLAY_OFFSET_DEG below for that.
ELECTRODE_ANGLE_DEG = 0.0

# Display-only rotation applied to the kymograph angle axis at plot time:
#   display_angle = (code_angle + ANGLE_DISPLAY_OFFSET_DEG) mod 360
# Does NOT affect pole/equator masking, the polarization index, or any other
# computation — those stay in code-frame (0 deg = directly right of center,
# 90 deg = directly below, matching standard image row/column axes).
# Default 90 deg matches the lab convention: 0 deg at the top of the vesicle
# (a reference point on the equator, not a pole), sweeping clockwise so that
# 90 deg = cathode-facing pole (right) and 270 deg = anode-facing pole
# (left) — i.e. code 270 -> display 0, code 0 -> display 90, code 180 ->
# display 270. Must be a multiple of (360 / ACTIN_N_ANGLES) so the axis
# roll lands exactly on a sampled angle.
ANGLE_DISPLAY_OFFSET_DEG = 90.0

# Generate per-GUV actin angular kymographs (angle vs. time post-pulse,
# cortex intensity as color) plus one population-average kymograph per
# experiment, to check whether cortex breakdown after electroporation is
# directional (concentrated at the electrode-facing poles) or uniform
# (all around the vesicle). Requires ANALYZE_ACTIN_CHANNEL = True.
EXPORT_ACTIN_KYMOGRAPH = True

# Half-width (degrees) of the angular window averaged around each pole and
# each equator direction for the quantitative pole-vs-equator trace and
# polarization index. 22.5 deg means the 4 windows (2 poles + 2 equator
# points) exactly tile the full 360 deg with no gaps/overlap.
POLE_EQUATOR_HALF_WIDTH_DEG = 22.5

# -----------------------------------------------------------------------------
# --- 6. ADVANCED ---
# -----------------------------------------------------------------------------

N_WORKERS    = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
FALLBACK_FPS = 1.0 # second(s)