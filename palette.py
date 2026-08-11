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

POPULATION_COLORS = {
    'BranchedCortex': PALETTE['dark_blue'],
    'Bare':           PALETTE['grey'],
    'Empty':          PALETTE['grey'],   
}

SIZE_CATEGORY_COLORS = {
    'Grow':             PALETTE['medium_blue'],
    'Stagnate':         PALETTE['light_grey'],
    'Reduce':           PALETTE['light_red'],
    'Rupture/Collapse': PALETTE['dark_red'],
}

SIZE_BIN_COLORS = {
    '< 5 \u00b5m':  PALETTE['pale_blue'],
    '5-8 \u00b5m':  PALETTE['light_blue'],
    '9-12 \u00b5m': PALETTE['medium_blue'],
    '> 13 \u00b5m': PALETTE['dark_blue'],
}

CORTEX_STATUS_COLORS = {
    'CORTEX':    PALETTE['dark_blue'],
    'AMBIGUOUS': PALETTE['light_grey'],
    'NO_CORTEX': PALETTE['medium_red'],
    'UNKNOWN':   PALETTE['grey'],
}

GROUP_COLORS = {
    'Bare':                   PALETTE['grey'],
    'Branched, cortex':       PALETTE['dark_blue'],
    'Branched, lumenal only': PALETTE['light_blue'],
    'Branched, ambiguous':    PALETTE['light_grey'],
    'Branched, unclassified': PALETTE['pale_red'],
}

GROUP_ORDER = ['Bare', 'Branched, cortex', 'Branched, lumenal only',
               'Branched, ambiguous', 'Branched, unclassified']

POPULATION_COLORS.update(GROUP_COLORS)

INDIVIDUAL_TRACE = PALETTE['light_grey']
SUMMARY_LINE     = PALETTE['dark_red']
ANNOTATION_TEXT  = PALETTE['grey']