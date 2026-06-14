import os
import multiprocessing
import numpy as np
import pandas as pd
from numba import njit
import time
from joblib import Parallel, delayed

# ---------------- Reproducibility & Environment ----------------
SEED = 12345
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

# -------------------------------------------------------------------
# 1. NUMBA-OPTIMIZED DATA GENERATING PROCESS (DGP)
# -------------------------------------------------------------------
@njit
def simulate_var_dgp_fast(A, V, B_tilde, p, T_target, burn_in, seed_val):
    np.random.seed(seed_val)
    
    K = A.shape[0]
    T_total = T_target + burn_in
    
    structural_shocks = np.random.randn(K, T_total)
    reduced_residuals = B_tilde @ structural_shocks
    y_sim = np.zeros((K, T_total))
    
    for t in range(p, T_total):
        d_t = np.zeros((12, 1))
        d_t[0, 0] = 1.0  
        month_idx = t % 12
        if month_idx < 11:
            d_t[month_idx + 1, 0] = 1.0
            
        deterministic_term = V @ d_t
        
        y_lags = np.zeros((K * p, 1))
        for lag in range(1, p + 1):
            for k in range(K):
                y_lags[(lag-1)*K + k, 0] = y_sim[k, t - lag]
            
        autoregressive_term = A @ y_lags
        y_t = deterministic_term + autoregressive_term + reduced_residuals[:, t:t+1]
        y_sim[:, t] = y_t[:, 0]
        
    return np.ascontiguousarray(y_sim[:, burn_in:].T)

# -------------------------------------------------------------------
# 2. FAST VAR ESTIMATOR
# -------------------------------------------------------------------
def lsvarcSA2_silent(y, p):
    t, K = y.shape
    y = y.T
    Y = y[:, p-1:t]
    
    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i : t-i]])
        
    x = np.vstack([np.eye(11), np.zeros((1, 11))])
    n_years = int((t - p) // 12)
    remainder = int((t - p) % 12)
    
    X2 = np.tile(x, (n_years, 1)) if n_years > 0 else np.empty((0, 11))
    if remainder > 0:
        last = np.hstack([np.eye(remainder), np.zeros((remainder, 11 - remainder))])
        X2 = np.vstack([X2, last])
        
    X2 = np.hstack([np.ones((t - p, 1)), X2])
    X = np.vstack([X2.T, Y[:, :t-p]])
    Y2 = y[:, p:t]
    
    B = np.linalg.lstsq(X.T, Y2.T, rcond=None)[0].T
    U = Y2 - B @ X
    SIGMA = np.ascontiguousarray((U @ U.T) / (t - p - p * K - 12))
    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    
    return A, SIGMA

# -------------------------------------------------------------------
# 3. HIGH-EFFICIENCY PURE SIGN RESTRICTION CORE
# -------------------------------------------------------------------
@njit
def compute_structural_irf_inplace(A, B_tilde, h_max, K, p, Phi_buffer, IRF_buffer):
    """Computes IRF by overwriting memory buffers to prevent RAM allocation overhead."""
    Phi_buffer.fill(0.0)
    
    for i in range(K):
        Phi_buffer[0, i, i] = 1.0
        
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            A_j = A[:, (j-1)*K : j*K]
            Phi_buffer[h] += A_j @ Phi_buffer[h-j]
            
    for h in range(h_max):
        IRF_buffer[h] = Phi_buffer[h] @ B_tilde

@njit
def fast_draw_core(A, P, signs, p, K, h_max, target_draws, max_loops, seed_val):
    np.random.seed(seed_val)
    
    valid_IRFs = np.zeros((target_draws, h_max, K, K))
    attempts = 0
    accepted = 0
    
    # ALLOCATE MEMORY EXACTLY ONCE
    Phi_buffer = np.zeros((h_max, K, K))
    IRF_buffer = np.zeros((h_max, K, K))
    
    while accepted < target_draws and attempts < max_loops:
        attempts += 1
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)
        for i in range(K):
            if R[i, i] < 0:
                Q[:, i] = -Q[:, i]
                
        B_tilde = P @ Q
        
        match = True
        for i in range(K):
            for j in range(K):
                if not np.isnan(signs[i, j]):
                    if np.sign(B_tilde[i, j]) != signs[i, j]:
                        match = False
                        break
            if not match: break
        if not match: continue 
            
        # OVERWRITE BUFFERS INSTEAD OF ALLOCATING NEW ARRAYS
        compute_structural_irf_inplace(A, B_tilde, h_max, K, p, Phi_buffer, IRF_buffer)
        
        # Copy directly into the final array matrix
        for h in range(h_max):
            for i in range(K):
                for j in range(K):
                    valid_IRFs[accepted, h, i, j] = IRF_buffer[h, i, j]
        
        # Compute Cumulative sums in-place
        for h in range(1, h_max):
            for k in range(K):
                valid_IRFs[accepted, h, 0, k] += valid_IRFs[accepted, h-1, 0, k]
                valid_IRFs[accepted, h, 3, k] += valid_IRFs[accepted, h-1, 3, k]
        
        accepted += 1
            
    return valid_IRFs, attempts, accepted

# -------------------------------------------------------------------
# 4. THE ISOLATED AGGREGATION COMPARISON
# -------------------------------------------------------------------
def single_monte_carlo_iteration(args):
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, h_max, mc_draws, max_loops, T_real = args
    
    try:
        K = true_A.shape[0]
        
        # 1. Generate Data
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 100, iteration_seed)
        
        # 2. Estimate Model strictly at True DGP Lag Order
        A_est, SIGMA_est = lsvarcSA2_silent(simulated_data, true_p)
        
        try:
            P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
        except np.linalg.LinAlgError:
            SIGMA_est += np.eye(K) * 1e-8 
            P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            
        # 3. Draw Structural Models
        seed_for_matrices = iteration_seed + true_p
        valid_irfs, attempts, accepted = fast_draw_core(A_est, P_est, signs, true_p, K, h_max, mc_draws, max_loops, seed_for_matrices)
        
        if accepted < mc_draws:
            return iter_idx, None, None, 0, "Empty Set"
            
        valid_irfs_sliced = valid_irfs[:accepted]
        
        # -----------------------------------------------------------
        # AGGREGATION METHOD 1: POINTWISE MEDIAN (Synthetic Model)
        # -----------------------------------------------------------
        pointwise_median_irf = np.median(valid_irfs_sliced, axis=0)
        SE_pointwise = (pointwise_median_irf - true_IRF_target)**2 
        
        # -----------------------------------------------------------
        # AGGREGATION METHOD 2: MEDIAN TARGET (Fry & Pagan Fix)
        # -----------------------------------------------------------
        distances = np.sum((valid_irfs_sliced - pointwise_median_irf)**2, axis=(1, 2, 3))
        best_model_idx = np.argmin(distances)
        target_irf = valid_irfs_sliced[best_model_idx]
        SE_target = (target_irf - true_IRF_target)**2
            
        return iter_idx, SE_pointwise, SE_target, attempts, "Success"
        
    except Exception as e:
        print(f"\n[WORKER CRASH] Iteration {iter_idx}: {type(e).__name__} - {str(e)}")
        return iter_idx, None, None, 0, f"Python Error: {str(e)}"

# -------------------------------------------------------------------
# 5. PARALLEL ORCHESTRATION WITH NESTED LOOP (p0 and T)
# -------------------------------------------------------------------
def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = script_dir if os.path.basename(script_dir) == KM_FOLDER else os.path.join(script_dir, KM_FOLDER)
    
    # Global Target Parameters
    h_max = 24
    N_iterations = 1000   
    mc_draws = 100        
    max_loops = 5000000 
    
    sign_matrix = np.ascontiguousarray(np.array([
        [-1,       1,      1,      np.nan],  
        [-1,       1,     -1,      np.nan],  
        [ 1,       1,      1,      np.nan],  
        [np.nan, np.nan,   1,      np.nan]   
    ], dtype=np.float64))

    # The True Lag Orders and Sample Sizes we want to test
    dgp_lag_orders = [4, 6, 8, 10]
    sample_sizes = [80, 100, 120, 160, 200]
    
    master_results_list = []
    
    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    
    print("\n" + "="*80)
    print(f" STARTING MASTER MONTE CARLO: POINTWISE vs TARGET")
    print(f" Lags (p0) = {dgp_lag_orders} | Sample Sizes (T) = {sample_sizes}")
    print(f" Iterations: {N_iterations} | Draws: {mc_draws} | Utilizing {n_cores} CPU Cores")
    print("="*80)

    # OUTER LOOP: Iterate over the Data Generating Processes (True Lags)
    for current_p0 in dgp_lag_orders:
        dgp_path = os.path.join(km_base, 'DGP files', f'true_dgp_parameters_{current_p0}_lags.npz')
        
        if not os.path.exists(dgp_path):
            print(f"\n[WARNING] Could not find {dgp_path}. Skipping p0={current_p0}...")
            continue
            
        print(f"\n---> Loading DGP for p0 = {current_p0} ...")
        
        dgp = np.load(dgp_path)
        true_A = np.ascontiguousarray(dgp['A_true'])
        true_V = np.ascontiguousarray(dgp['V_true'])
        true_B_tilde = np.ascontiguousarray(dgp['B_tilde_true'])
        true_IRF = np.ascontiguousarray(dgp['True_IRF'])
        true_p = int(dgp['p_true'])
        
        # INNER LOOP: Iterate over the Sample Sizes
        for current_T in sample_sizes:
            print(f"     Running Simulation for T = {current_T} ...", end="", flush=True)
            
            tasks = []
            for i in range(N_iterations):
                # Seed varies by iteration, p0, AND T to ensure absolute statistical independence
                iteration_seed = SEED + i + (current_p0 * 10000) + (current_T * 100000)
                tasks.append((
                    i, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF, 
                    sign_matrix, h_max, mc_draws, max_loops, current_T
                ))

            results = Parallel(n_jobs=n_cores, backend='loky')(
                delayed(single_monte_carlo_iteration)(task) for task in tasks
            )

            all_SE_pointwise = []
            all_SE_target = []
            discarded_draws = 0

            for iter_idx, SE_pointwise, SE_target, attempts, status in results:
                if status != "Success":
                    discarded_draws += 1
                    continue
                    
                all_SE_pointwise.append(SE_pointwise)
                all_SE_target.append(SE_target)

            if len(all_SE_pointwise) == 0:
                print(f" [FAILED] All iterations discarded.")
                continue
                
            print(f" [SUCCESS] ({len(all_SE_pointwise)}/{N_iterations})")

            # -------------------------------------------------------------------
            # EMPIRICAL DISTRIBUTION AGGREGATION
            # -------------------------------------------------------------------
            MSE_pointwise_per_iter = np.sum(all_SE_pointwise, axis=(1, 2, 3))
            MSE_target_per_iter = np.sum(all_SE_target, axis=(1, 2, 3))
            
            Ratio_Distribution = MSE_pointwise_per_iter / (MSE_target_per_iter + 1e-12)
            
            geom_mean_ratio = np.exp(np.mean(np.log(Ratio_Distribution)))
            percentile_05 = np.percentile(Ratio_Distribution, 5)
            percentile_95 = np.percentile(Ratio_Distribution, 95)
            pointwise_win_rate = np.mean(Ratio_Distribution < 1.0) * 100

            # Append to Master List including BOTH p0 and T
            master_results_list.append({
                "True DGP (p0)": current_p0,
                "Sample Size (T)": current_T,
                "Geom Mean Ratio (P/T)": round(geom_mean_ratio, 6),
                "5th Percentile": round(percentile_05, 6),
                "95th Percentile": round(percentile_95, 6),
                "Pointwise Win Rate (%)": round(pointwise_win_rate, 2),
                "Verdict": "Target is Better" if geom_mean_ratio > 1 else "Pointwise is Better"
            })

    total_time = time.time() - global_start_time

    # -------------------------------------------------------------------
    # SAVE MASTER RESULTS TO DISK
    # -------------------------------------------------------------------
    if len(master_results_list) > 0:
        final_df = pd.DataFrame(master_results_list)
        
        print("\n" + "="*80)
        print(" MASTER SIMULATION RESULTS: POINTWISE vs TARGET")
        print("="*80)
        print(final_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print(f"Total Master Execution Time: {round(total_time, 2)} Seconds")
        print("-"*80)

        results_dir = os.path.join(km_base, 'Results')
        os.makedirs(results_dir, exist_ok=True)
        filename = f"Master_Pointwise_vs_Target_Multi_p0_T_iters{N_iterations}_draws{mc_draws}.csv"
        save_path = os.path.join(results_dir, filename)
        
        final_df.to_csv(save_path, index=False)
        print(f"\n[SUCCESS] Master Dataframe saved to: {save_path}")
    else:
        print("\n[ERROR] No data generated across any DGP. Check file paths and errors.")

if __name__ == '__main__':
    main()