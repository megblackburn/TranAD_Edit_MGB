from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
        'GAUSS': [(0.98, 1), (0.99935, 1)],
        'SPIKE': [(0.98, 1), (0.99935, 1)],
}

lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
      'GAUSS': 0.006,
      'SPIKE': 0.006,
}

lr = lr_d[args.dataset]

percentiles = {
        'GAUSS': (98,2),
        'SPIKE': (98,2),
}

percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
#debug = 9
