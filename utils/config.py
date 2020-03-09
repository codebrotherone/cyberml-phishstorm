"""

THIS FILE CONTAINS GLOBALS FOR CYBERML-URL PHISHING MODEL


"""

#
# Globals
#
DTYPES={'domain': 'str',
    'ranking': 'float',
    'mld_res': 'float',
    'mld.ps_res': 'float',
    'card_rem': 'float',
    'ratio_Rrem': 'float',
    'ratio_Arem': 'float',
    'jaccard_RR': 'float',
    'jaccard_RA': 'float',
    'jaccard_AR': 'float',
    'jaccard_AA': 'float',
    'jaccard_ARrd': 'float',
    'jaccard_ARrem': 'float',
    'label': 'float'}

# index is 0 for domain in csv file
CAT_COLS = ['domain']
# indices 1-13 for values in csv file
NUM_COLS = ['ranking', 'mld_res', 'mld.ps_res', 'card_rem', 'ratio_Rrem',
    'ratio_Arem', 'jaccard_RR', 'jaccard_RA', 'jaccard_AR',
    'jaccard_AA', 'jaccard_ARrd', 'jaccard_ARrem']