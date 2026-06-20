# Simple file to execute the VAR fitting
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ---------------- Reproducibility / Seeding ----------------
SEED = 12345
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit # Optimization 2: JIT Compiler

# Apply seeds to main process
random.seed(SEED)
np.random.seed(SEED)
# -----------------------------------------------------------

def lsvarcSA2(y, p):
    """
    Estimates a VAR(p) by Ordinary Least Squares (OLS).
    """
    t, K = y.shape
    print(f"lsvarcSA2 start: t={t}, K={K}, p={p}", flush=True)
    
    y = y.T
    Y = y[:, p-1:t]
    
    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i : t-i]])
        
    x = np.vstack([np.eye(11), np.zeros((1, 11))])
    n_years = int((t - p) // 12)
    remainder = int((t - p) % 12)
    
    if n_years > 0:
        X2 = np.tile(x, (n_years, 1))
    else:
        X2 = np.empty((0, 11))
        
    if remainder > 0:
        last = np.hstack([np.eye(remainder), np.zeros((remainder, 11 - remainder))])
        X2 = np.vstack([X2, last])
        
    X2 = np.hstack([np.ones((t - p, 1)), X2])
    
    X = np.vstack([X2.T, Y[:, :t-p]])
    Y2 = y[:, p:t]
    
    # --- OPTIMIZATION 4: Fast Linear Solver ---
    # Replaced np.linalg.pinv with np.linalg.solve for massive speedup
    # We solve (X @ X.T) @ B.T = X @ Y2.T
    B = np.linalg.solve(X @ X.T, X @ Y2.T).T
    
    U = Y2 - B @ X
    SIGMA = (U @ U.T) / (t - p - p * K - 12)
    
    # FORCE CONTIGUOUS MEMORY HERE:
    V = np.ascontiguousarray(B[:, 0:12]) 
    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    
    return A, B, X, SIGMA, U, V

# -------------------------------------------------------------------
# OPTIMIZATION 2 & 3: JIT-COMPILED CORE WITH "FAIL FAST" LOGIC
# -------------------------------------------------------------------
@njit
def compute_structural_irf_numba(A_3d, B_tilde, h_max, K, p):
    """JIT-compiled IRF generator taking a pre-sliced 3D A-matrix."""
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            # Simply access the pre-sliced, contiguous 2D array!
            Phi[h] += A_3d[j-1] @ Phi[h-j]
            
    IRF = np.zeros((h_max, K, K))
    for h in range(h_max):
        IRF[h] = Phi[h] @ B_tilde
        
    return IRF

@njit
def fast_draw_core(A, P, signs, p, K, h_max, target_draws):
    """
    JIT-compiled inner loop. 
    Implements 'Fail Fast' by rejecting matrices before calculating full IRFs,
    and enforces dynamic sign restrictions for the first 3 months.
    """
    valid_IRFs = np.zeros((target_draws, h_max, K, K))
    valid_B_tildes = np.zeros((target_draws, K, K))
    
    attempts = 0
    accepted = 0
    
    while accepted < target_draws:
        attempts += 1
        
        # 1. Draw and rotate
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)
        
        # Normalize Q to ensure uniform Haar prior distribution
        for i in range(K):
            if R[i, i] < 0:
                Q[:, i] = -Q[:, i]
                
        # --- NEW FIX: Force Q into C-contiguous memory ---
        Q = np.ascontiguousarray(Q)
        
        B_tilde = P @ Q
        
        # 2. FAIL FAST 1: Check Impact Sign Restrictions
        match = True
        for i in range(K):
            for j in range(K):
                if not np.isnan(signs[i, j]):
                    if np.sign(B_tilde[i, j]) != signs[i, j]:
                        match = False
                        break
            if not match:
                break
                
        if not match:
            continue # Throw away the matrix immediately. Do not compute IRF!

        # 3. Pass it into the function
        irf = compute_structural_irf_numba(A, B_tilde, h_max, K, p)
        
        
        irf_cumulative = irf.copy()
        # Explicit loop for cumulative sum (safest approach in Numba 3D arrays)
        for h in range(1, h_max):
            irf_cumulative[h, 0, :] = irf_cumulative[h-1, 0, :] + irf[h, 0, :]
            irf_cumulative[h, 3, :] = irf_cumulative[h-1, 3, :] + irf[h, 3, :]

        # 4. FAIL FAST 2: Dynamic Sign Restrictions (First 3 months)
        # Flow Supply Shock = Column 0
        dynamic_match = True
        for h in range(3): # Horizons 0 through 11
            
            # Oil Production level (cumulative) must be negative
            if irf_cumulative[h, 0, 0] >= 0:
                dynamic_match = False
                break
                
            # Real Activity level (standard IRF) must be negative
            if irf[h, 1, 0] >= 0:
                dynamic_match = False
                break
                
            # Real Oil Price level (standard IRF) must be positive
            if irf[h, 2, 0] <= 0:
                dynamic_match = False
                break
                
        if not dynamic_match:
            continue # Reject model if any dynamic restriction fails
            
        # 5. Store Valid Models
        valid_IRFs[accepted] = irf_cumulative
        valid_B_tildes[accepted] = B_tilde
        accepted += 1
            
    return valid_IRFs, valid_B_tildes, attempts

# -------------------------------------------------------------------
# OPTIMIZATION 1: PARALLEL WORKER WRAPPER
# -------------------------------------------------------------------
def worker_draw(args):
    """Wrapper function to assign a unique seed to each parallel core."""
    A_3d, P, signs, p, K, h_max, target_draws, seed = args
    np.random.seed(seed) # Crucial for independent random streams
    return fast_draw_core(A_3d, P, signs, p, K, h_max, target_draws)

def draw_sign_restrictions_parallel(A, SIGMA, signs, p, K, h_max=24, n_draws=1000):
    P = np.linalg.cholesky(SIGMA)
    # Ensure P is contiguous before sending it to the workers
    P = np.ascontiguousarray(P)
    
    # NEW: Pre-slice A into a contiguous 3D array (p, K, K)
    A_3d = np.zeros((p, K, K))
    for j in range(1, p + 1):
        A_3d[j-1] = np.ascontiguousarray(A[:, (j-1)*K : j*K])
        
    n_cores = multiprocessing.cpu_count()
    draws_per_core = n_draws // n_cores
    remain = n_draws % n_cores
    
    tasks = []
    for i in range(n_cores):
        td = draws_per_core + (1 if i < remain else 0)
        if td > 0:
            # Pass A_3d instead of A
            tasks.append((A_3d, P, signs, p, K, h_max, td, SEED + i + p*100))
            
    valid_IRFs_list = []
    valid_B_tildes_list = []
    total_attempts = 0
    
    print(f"Parallelizing {n_draws} successful draws across {n_cores} CPU cores...", flush=True)
    
    # Execute heavily intensive rejection sampling in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        for irfs, b_tildes, attempts in executor.map(worker_draw, tasks):
            valid_IRFs_list.append(irfs)
            valid_B_tildes_list.append(b_tildes)
            total_attempts += attempts
            
    valid_IRFs = np.vstack(valid_IRFs_list)
    valid_B_tildes = np.vstack(valid_B_tildes_list)
    
    print(f"Success! Found {n_draws} valid models out of {total_attempts} random draws.")
    print(f"Acceptance rate: {(n_draws/total_attempts)*100:.4f}%")
    
    return valid_IRFs, valid_B_tildes

# -------------------------------------------------------------------
# MAIN EXECUTION BLOCK (Required for Windows/Mac parallel processing)
# -------------------------------------------------------------------
def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = script_dir if os.path.basename(script_dir) == KM_FOLDER else os.path.join(script_dir, KM_FOLDER)

    km_data_path = os.path.join(km_base, 'km-ascii-data', 'kmData.txt')

    km_data_array = np.loadtxt(km_data_path)

    print("Data loaded successfully!")
    var_names = ["Oil Production", "Real Activity", "Real Oil Price", "Inventories"]

    dgp_configs = [
        {"p": 4, "name": "4_lags"},
        {"p": 6, "name": "6_lags"},
        {"p": 8, "name": "8_lags"},
        {"p": 10, "name": "10_lags"}
    ]

    dgp_results = {}

    for config in dgp_configs:
        p = config["p"]
        config_name = config["name"]
        
        print("\n" + "="*60)
        print(f" VAR MODEL ESTIMATION SUMMARY (p={p})")
        print("="*60)
        
        A, B, X, SIGMA, U, V = lsvarcSA2(km_data_array, p)
        
        dgp_results[config_name] = {
            "A": A, "B": B, "X": X, "SIGMA": SIGMA, "U": U, "V": V, "p": p
        }
        
        sigma_df = pd.DataFrame(SIGMA, index=var_names, columns=var_names)
        print("\n### 1. Residual Covariance Matrix (SIGMA) ###")
        print(sigma_df.round(4)) 

        v_cols = ["Constant"] + [f"Month_{i}_Dummy" for i in range(1, 12)]
        v_df = pd.DataFrame(V, index=var_names, columns=v_cols)
        print("\n### 2. Deterministic Terms (V) ###")
        print(v_df.iloc[:, :4].round(4), "...\n(Showing first 4 of 12 columns)")

        lag_cols = []
        for lag in range(1, p + 1):
            for var in var_names:
                short_name = var.replace("Production", "Prod").replace("Activity", "Act")
                lag_cols.append(f"Lag{lag}_{short_name}")
        a_df = pd.DataFrame(A, index=var_names, columns=lag_cols)
        print("\n### 3. VAR Slope Coefficients (A) ###")
        print(a_df.T.round(4)) 
        print("="*60 + "\n")

    sign_matrix = np.array([
        [-1,       1,      1,      np.nan],  
        [-1,       1,     -1,      np.nan],  
        [ 1,       1,      1,      np.nan],  
        [np.nan, np.nan,  1,       np.nan]   
    ], dtype=np.float64)

    h_max = 24  
    n_draws = 100000 

    for config in dgp_configs:
        p = config["p"]
        config_name = config["name"]
        
        print(f"\n{'='*60}")
        print(f" PROCESSING {p}-LAG DGP")
        print(f"{'='*60}\n")
        
        A = dgp_results[config_name]["A"]
        SIGMA = dgp_results[config_name]["SIGMA"]
        V = dgp_results[config_name]["V"]
        K = SIGMA.shape[0]
        
        accepted_irfs, accepted_B_tildes = draw_sign_restrictions_parallel(
            A=A, SIGMA=SIGMA, signs=sign_matrix, p=p, K=K, h_max=h_max, n_draws=n_draws
        )

        pointwise_median_irf = np.median(accepted_irfs, axis=0)

        # Fry-Pagan standardization: divide deviations by pointwise std across draws
        pointwise_std_irf = np.std(accepted_irfs, axis=0)
        # Guard against division by zero (e.g. IRF[0] identity-driven zeros)
        pointwise_std_irf = np.where(pointwise_std_irf == 0, np.nan, pointwise_std_irf)

        standardized_dev = (accepted_irfs - pointwise_median_irf) / pointwise_std_irf
        # NaN entries (zero-variance cells) contribute nothing to the sum
        distances = np.nansum(standardized_dev**2, axis=(1, 2, 3))
        best_model_idx = np.argmin(distances)

        B_tilde_true = accepted_B_tildes[best_model_idx]
        True_IRF = accepted_irfs[best_model_idx]

        print(f"\n=== DGP PARAMETERS ESTABLISHED ({p} LAGS) ===")
        
        export_folder = os.path.join(km_base, 'DGP files')
        os.makedirs(export_folder, exist_ok=True)
        export_path = os.path.join(export_folder, f'true_dgp_parameters_{config_name}.npz')

        # Save the DGP
        np.savez(export_path, 
                 A_true=A, 
                 SIGMA_true=SIGMA, 
                 B_tilde_true=B_tilde_true, 
                 True_IRF=True_IRF, 
                 V_true=V, 
                 p_true=p,
                 empirical_data=km_data_array.T,    # <-- ADD THIS (Shape K x T)
                 empirical_residuals=U)             # <-- ADD THIS (Shape K x T_eff)         
    

        dgp_results[config_name]["B_tilde_true"] = B_tilde_true
        dgp_results[config_name]["True_IRF"] = True_IRF

    export_folder_viz = os.path.join(km_base, 'DGP files', 'Visualizations')
    os.makedirs(export_folder_viz, exist_ok=True)

    for config in dgp_configs:
        config_name = config["name"]
        p = config["p"]
        
        A = dgp_results[config_name]["A"]
        V = dgp_results[config_name]["V"]
        SIGMA = dgp_results[config_name]["SIGMA"]
        B_tilde_true = dgp_results[config_name]["B_tilde_true"]
        True_IRF = dgp_results[config_name]["True_IRF"]
        K = SIGMA.shape[0]
        
        fig_irf, axes_irf = plt.subplots(nrows=K, ncols=K, figsize=(15, 12))
        shock_names = ["Flow Supply Shock", "Flow Demand Shock", "Speculative Demand Shock", "Residual Shock"]
        var_names_plot = ["Oil Production", "Real Activity", "Real Oil Price", "Inventories"]

        for i in range(K): 
            for j in range(K): 
                axes_irf[i, j].plot(True_IRF[:, i, j], color='darkblue', linewidth=2)
                axes_irf[i, j].axhline(0, color='black', linestyle='--', linewidth=1)
                if i == 0:
                    axes_irf[i, j].set_title(f"{shock_names[j]}", fontweight='bold')
                if j == 0:
                    axes_irf[i, j].set_ylabel(f"{var_names_plot[i]}", fontweight='bold')
                axes_irf[i, j].grid(alpha=0.3)

        fig_irf.suptitle(f"True Structural Impulse Responses ({p}-Lag Model, Median Target Model)", fontsize=16, y=1.02)
        plt.tight_layout()
        irf_plot_path = os.path.join(export_folder_viz, f'True_IRF_plot_Cumulative_{config_name}.png')
        plt.savefig(irf_plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        print("\n" + "="*60)
        print(f" TRUE DGP PARAMETERS EXPORT SUMMARY ({p} LAGS)")
        print("="*60)

        sigma_true_df = pd.DataFrame(SIGMA, index=var_names, columns=var_names)
        print("\n### 1. True Residual Covariance Matrix (SIGMA_true) ###")
        print(sigma_true_df.round(4)) 

        v_true_df = pd.DataFrame(V, index=var_names, columns=v_cols)
        print("\n### 2. True Deterministic Terms (V_true) ###")
        print(v_true_df.iloc[:, :4].round(4), "...")

        # Reset the lag_ols matrix
        lag_cols = []
        for lag in range(1, p + 1):
            for var in var_names:
                short_name = var.replace("Production", "Prod").replace("Activity", "Act")
                lag_cols.append(f"Lag{lag}_{short_name}")


        a_true_df = pd.DataFrame(A, index=var_names, columns=lag_cols)
        print("\n### 3. True VAR Slope Coefficients (A_true) ###")
        print(a_true_df.T.iloc[:20].round(4))  
        print(f"... ({len(a_true_df.T)} total rows)")

        b_tilde_df = pd.DataFrame(B_tilde_true, index=var_names, columns=shock_names)
        print("\n### 4. True Structural Impact Matrix (B_tilde_true) ###")
        print(b_tilde_df.round(4))

if __name__ == '__main__':
    main()