"""Shared colour palette for the bulk figures.

One definition, imported by process.py and tau_identifiability_report.py, so a
population keeps the same colour across every panel of the chapter.

The ramp is a blue-grey-red diverging scheme. Where a variable is ORDERED
(size groups) the blues are used as a sequential ramp; where it is
CATEGORICAL AND SIGNED (grew / unchanged / shrank / ruptured) the full
diverging range is used with grey at the neutral centre.
"""

PALETTE = {
    'dark_blue':   '#1065AB',
    'medium_blue': '#3A93C3',
    'light_blue':  '#8EC4DE',
    'pale_blue':   '#D1E5F0',
    'light_grey':  '#DBD9D4',
    'grey':        '#868684',
    'white':       '#F9F9F9',
    'pale_red':    '#FDDBC7',
    'light_red':   '#F6A482',
    'medium_red':  '#D75F4C',
    'dark_red':    '#B31529',
}

# --- populations -------------------------------------------------------------
POPULATION_COLORS = {
    'BranchedCortex': PALETTE['dark_blue'],
    'Empty':          PALETTE['grey'],
}

# --- size / fate categories --------------------------------------------------
# Diverging about "Stagnate": growth to the blue side, loss to the red side,
# so the direction of a change is readable from the colour alone.
SIZE_CATEGORY_COLORS = {
    'Grow':             PALETTE['medium_blue'],
    'Stagnate':         PALETTE['light_grey'],
    'Reduce':           PALETTE['light_red'],
    'Rupture/Collapse': PALETTE['dark_red'],
}

# --- size bins ---------------------------------------------------------------
# Ordered variable, so a sequential ramp: small = pale, large = dark.
SIZE_BIN_COLORS = {
    '< 5 \u00b5m':  PALETTE['pale_blue'],
    '5-8 \u00b5m':  PALETTE['light_blue'],
    '9-12 \u00b5m': PALETTE['medium_blue'],
    '> 13 \u00b5m': PALETTE['dark_blue'],
}

# --- cortex presence ---------------------------------------------------------
CORTEX_STATUS_COLORS = {
    'CORTEX':    PALETTE['dark_blue'],
    'AMBIGUOUS': PALETTE['light_grey'],
    'NO_CORTEX': PALETTE['medium_red'],
    'UNKNOWN':   PALETTE['grey'],
}

# --- analysis groups ---------------------------------------------------------
# BranchedCortex splits in two: vesicles with an actual rim, and vesicles whose
# actin never polymerised onto the membrane and sits in the lumen instead.
# Those are different objects and should not share a colour.
GROUP_COLORS = {
    'Empty':                  PALETTE['grey'],
    'Branched, cortex':       PALETTE['dark_blue'],
    'Branched, lumenal only': PALETTE['light_blue'],
    'Branched, ambiguous':    PALETTE['light_grey'],
}

GROUP_ORDER = ['Empty', 'Branched, cortex', 'Branched, lumenal only',
               'Branched, ambiguous']

# --- plot furniture ----------------------------------------------------------
# Individual per-GUV traces sit behind the population summary, so they take the
# palest grey; the summary line takes dark red, which appears nowhere else as a
# category colour and therefore cannot be misread as a population.
INDIVIDUAL_TRACE = PALETTE['light_grey']
SUMMARY_LINE     = PALETTE['dark_red']
ANNOTATION_TEXT  = PALETTE['grey']