from numpy import pi
import torch.optim.lr_scheduler

from isolated_utils import FC1d
from FixedReduceLROnPlateau import FixedReduceLROnPlateau

# model hyperparameters
NUM_MODES = 32
WIDTH = 200

# model depth must be 4; this option is just here for clarity
DEPTH = 4

# set to either False or a float (the c0 parameter is initialized to this value)
C0_INITIAL = False

CONTINUATION_FUNC = lambda x : FC1d(x, 3)
CONTINUATION_FUNC_STRING = "lambda x : FC1d(x, 3)"
CONTINUATION_GRIDPOINTS = 25

# Options: "End only", "Each layer", "Entire intermediate", "Padding", "No continuation".
FC_PINO_TYPE = "End only"

# training parameters
m = 400
HIGHER_RES_M = 900
NUM_EPOCHS = 60001

INTERIOR_WEIGHT = 5
BOUNDARY_WEIGHT = 1
SMOOTHNESS_WEIGHT = 0.1
TRACK_SMOOTHNESS_LOSS = True

# optimizer
# set to either False or an int (0 for only LBFGS)
LBFGS_AFTER_EPOCH = False
# set to None to use Adam learning rate at the switch epoch
LBFGS_LR = 10**-2
LBFGS_HISTORY_SIZE = 5

# scheduler
LEARNING_RATE = 10**-4
SCHEDULER_TYPE = FixedReduceLROnPlateau

# StepLR parameters
STEP_SIZE = 1000
LR_DECAY_FACTOR = 0.5

# ReduceLROnPlateau parameters
PLATEAU_DECAY_FACTOR = 0.5
PATIENCE = 500
THRESHOLD = 0.01
MIN_LR = 0
EPSILON = 0

# logging parameters
WANDB = True
PLOT_POPUPS = False

# Plotting model output against Burgers' equation exact solution
BURGER_OVERLAY_PLOTS = False

WANDB_USERNAME = 'caltech-anima-group'
WANDB_PROJECT = 'Legendre_search'
WANDB_DIR = '/central/groups/tensorlab/aganeshram/FC/wandb_logs'

# training set description
INTERVAL = (-2, 2)
D = INTERVAL[1] - INTERVAL[0]

# lambda value used in Burgers' equation
l = 0.5

# Expression used to compute interior loss
INTERIOR_TERM = lambda U, Uy, y: -l*U + ((1+l)*y + U) * Uy  
INTERIOR_TERM_STRING = "-l*U + ((1+l)*y + U) * Uy"

# Expression used to compute smoothness loss
INTERIOR_TERM_2 = lambda U, Uy, Uyy, y : ((1+l)*y + U) * Uyy + (1 + Uy) * Uy
INTERIOR_TERM_2_STRING = "((1+l)*y + U) * Uyy + (1 + Uy) * Uy"

# Expression used to compute boundary loss
BOUNDARY_FUNC = lambda U: torch.norm(U[0] - 1) ** 2 \
        + torch.norm(U[(m+1)//2]) ** 2 \
        + torch.norm(U[-1] + 1) ** 2
BOUNDARY_FUNC_STRING = "U(-2) = 1; U(0) = 0; U(2) = -1"


BOUNDARY_FUNC_HIGHER_RES = lambda U: torch.norm(U[0] - 1) ** 2 \
        + torch.norm(U[(HIGHER_RES_M+1)//2]) ** 2 \
        + torch.norm(U[-1] + 1) ** 2
