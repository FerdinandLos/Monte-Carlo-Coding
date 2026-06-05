import numpy as np
import os

# ---------------- Reproducibility / Seeding ----------------
# Set a global seed and a few environment variables to make
# pseudorandom draws repeatable across runs on the same setup.
# Note: For strict byte-for-byte reproducibility also set
# PYTHONHASHSEED before interpreter start and pin package versions.
SEED = 12345
# Set Python hash seed (best set before interpreter start to be fully deterministic)
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
# Limit threaded BLAS behaviour to reduce nondeterminism across runs
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Apply seeds to Python RNGs used in this script
random.seed(SEED)
np.random.seed(SEED)

# 1. Setup the path
# 1. Get the exact folder where this specific .py file lives
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# 2. Resolve the Kilian & Murphy folder consistently regardless of how
# the script is launched (debugger vs manual). If `script_dir` already
# points at the 'Kilian and Murphy (2014)' folder, use it directly;
# otherwise assume `script_dir` is the repo root and append the folder.
KM_FOLDER = 'Kilian and Murphy (2014)'
if os.path.basename(script_dir) == KM_FOLDER:
    km_base = script_dir
else:
    km_base = os.path.join(script_dir, KM_FOLDER)


# Get dgp and dataset
dgp_path = os.path.join(km_base, 'DGP files', 'true_dgp_parameters_2_lags.npz')
km_data_path = os.path.join(km_base, 'km-ascii-data', 'kmData.txt')
# 2. Load the parameters
dgp = np.load(dgp_path)
km_data_array = np.loadtxt(km_data_path)

A_true = dgp['A_true']
SIGMA_true = dgp['SIGMA_true']
B_tilde_true = dgp['B_tilde_true']
True_IRF = dgp['True_IRF']
V_true = dgp['V_true']
p_true = dgp['p_true']

print("Loaded True DGP Parameters!")
print("True IRF Shape:", True_IRF.shape)

# ... Start your Monte Carlo loop down here ...