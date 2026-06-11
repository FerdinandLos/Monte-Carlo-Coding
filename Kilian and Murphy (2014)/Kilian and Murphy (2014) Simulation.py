import numpy as np
import os
import random

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

print(dgp.files)

A_true = dgp['A_true']
SIGMA_true = dgp['SIGMA_true']
B_tilde_true = dgp['B_tilde_true']
True_IRF = dgp['True_IRF']
V_true = dgp['V_true']
p_true = dgp['p_true']

print("Loaded True DGP Parameters!")
print("True IRF Shape:", True_IRF.shape)

# ... Start your Monte Carlo loop down here ...

# 1. Define a function to create the random samples from the DGP parameters
def simulate_var_dgp(A, V, B_tilde, p, T_target, burn_in=100):
    """
    Simulates a random path from the true VAR Data Generating Process.
    
    Parameters:
    A        : VAR slope coefficients (K x K*p)
    V        : Deterministic terms (Constant + 11 Seasonal Dummies) (K x 12)
    B_tilde  : True structural impact matrix from the median target (K x K)
    p        : Lag order
    T_target : The target length of the simulated time series (usually matches your real data length)
    burn_in  : Number of initial periods to simulate and discard to remove initial condition bias
    """
    K = A.shape[0]
    T_total = T_target + burn_in
    
    # 1. Generate True Structural Shocks
    # Standard normal distribution: N(0, 1)
    structural_shocks = np.random.randn(K, T_total)
    
    # 2. Convert to Reduced-Form Residuals
    # u_t = B_tilde @ epsilon_t
    reduced_residuals = B_tilde @ structural_shocks
    
    # 3. Initialize the simulated data matrix with zeros
    y_sim = np.zeros((K, T_total))
    
    # 4. Recursively build the time series
    for t in range(p, T_total):
        # --- A. Calculate deterministic component (V @ d_t) ---
        d_t = np.zeros((12, 1))
        d_t[0, 0] = 1.0  # The Constant is always 1
        
        # Figure out the seasonal month index (0 to 11)
        month_idx = t % 12
        if month_idx < 11:
            # Activate the specific Jan-Nov dummy
            d_t[month_idx + 1, 0] = 1.0
            
        deterministic_term = V @ d_t
        
        # --- B. Calculate the autoregressive (lag) component ---
        lag_stack = []
        for lag in range(1, p + 1):
            lag_stack.append(y_sim[:, t - lag])
            
        # Reshape the stacked lags to (K*p, 1) to multiply with A
        y_lags = np.hstack(lag_stack).reshape(-1, 1) 
        autoregressive_term = A @ y_lags
        
        # --- C. Combine everything to get y_t ---
        y_t = deterministic_term + autoregressive_term + reduced_residuals[:, t].reshape(-1, 1)
        
        # Store the calculated time step
        y_sim[:, t] = y_t.flatten()
        
    # 5. Discard the burn-in period and return the target length
    return y_sim[:, burn_in:].T # Transposed to return shape (T_target, K) to match your input format


# -------------------------------------------------------------------
# 5. MONTE CARLO SIMULATION LOOP
# -------------------------------------------------------------------
N_iterations = 1000
T_real = km_data_array.shape[0]

print("\n" + "="*60)
print(f" STARTING MONTE CARLO SIMULATION ({N_iterations} ITERATIONS)")
print("="*60)

# We will use the 24-lag DGP as the "True" reality for this example
config_name = "24_lags"
A_true = dgp_results[config_name]["A"]
V_true = dgp_results[config_name]["V"]
B_tilde_true = dgp_results[config_name]["B_tilde_true"]
p_true = dgp_results[config_name]["p"]

# Initialize storage for your evaluation metrics (e.g., MSE)
var_mse_scores = []
bvar_mse_scores = []

for i in range(N_iterations):
    # 1. Generate an alternate reality dataset
    simulated_data = simulate_var_dgp(A_true, V_true, B_tilde_true, p_true, T_target=T_real)
    
    # 2. Fit Standard VAR (acting as if we don't know the true parameters)
    # A_est, B_est, X_est, SIGMA_est, U_est, V_est = lsvarcSA2(simulated_data, p_true)
    
    # 3. Fit BVAR on the same simulated_data
    # ... (Your BVAR implementation here) ...
    
    # 4. Calculate Mean Squared Error for both models' forecasts
    # ... 
    
    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1} / {N_iterations} Monte Carlo iterations...")