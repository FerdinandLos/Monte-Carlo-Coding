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
def simulate_var_dgp_fast(A, V, B_tilde, p, T_target, burn_in):
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
# 3. PURE SIGN RESTRICTION CORE (ELASTICITY REMOVED)
# -------------------------------------------------------------------
@njit
def compute_structural_irf_numba(A, B_tilde, h_max, K, p):
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            A_j = A[:, (j-1)*K : j*K]
            Phi[h] += A_j @ Phi[h-j]
            
    IRF = np.zeros((h_max, K, K))
    for h in range(h_max):
        IRF[h] = Phi[h] @ B_tilde
    return IRF

@njit
def fast_draw_core(A, P, signs, p, K, Q_avg, h_max, target_draws, max_loops):
    valid_IRFs = np.zeros((target_draws, h_max, K, K))
    attempts = 0
    accepted = 0
    
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
            
        # ---------------------------------------------------------
        # PURE SIGN RESTRICTIONS: Elasticity Checks Removed!
        # If it reaches here, the signs match, so we instantly accept it.
        # ---------------------------------------------------------
        irf = compute_structural_irf_numba(A, B_tilde, h_max, K, p)
        irf_cumulative = irf.copy()
        
        for h in range(1, h_max):
            for k in range(K):
                irf_cumulative[h, 0, k] = irf_cumulative[h-1, 0, k] + irf[h, 0, k]
                irf_cumulative[h, 3, k] = irf_cumulative[h-1, 3, k] + irf[h, 3, k]
        
        valid_IRFs[accepted] = irf_cumulative
        accepted += 1
            
    return valid_IRFs, attempts, accepted

def get_median_target_model(valid_irfs, accepted_count):
    valid_irfs_sliced = valid_irfs[:accepted_count]
    pointwise_median_irf = np.median(valid_irfs_sliced, axis=0)
    distances = np.sum((valid_irfs_sliced - pointwise_median_irf)**2, axis=(1, 2, 3))
    best_model_idx = np.argmin(distances)
    return valid_irfs_sliced[best_model_idx]

# -------------------------------------------------------------------
# 4. THE SINGLE MONTE CARLO ITERATION (WITH STABILIZATION)
# -------------------------------------------------------------------
def single_monte_carlo_iteration(args):
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, Q_avg, h_max, mc_draws, max_loops, T_real, p_max = args
    
    try:
        np.random.seed(iteration_seed)
        K = true_A.shape[0]
        
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 100)
        
        best_aic = float('inf')
        p_hat = 1
        N_eff = T_real - p_max 
        
        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            _, SIGMA_temp = lsvarcSA2_silent(y_slice, p_test)
            
            if not np.all(np.isfinite(SIGMA_temp)):
                continue
                
            sign_det, logdet_ols = np.linalg.slogdet(SIGMA_temp)
            if sign_det > 0: 
                # ---- THE FIX: Remove the Ghost Penalty ----
                # Convert the OLS determinant to the ML determinant
                df_correction = N_eff - (p_test * K) - 12
                logdet_ml = logdet_ols + K * np.log(df_correction / N_eff)
                
                # Standard AIC formula
                aic_val = logdet_ml + (2.0 / N_eff) * (K**2 * p_test)
                
                if aic_val < best_aic:
                    best_aic = aic_val
                    p_hat = p_test
                    
        # 1. Evaluate AIC Selected Model
        A_aic, SIGMA_aic = lsvarcSA2_silent(simulated_data, p_hat)
        
        # Robust Cholesky: Adds a tiny ridge penalty to force non-singular matrices to behave
        try:
            P_aic = np.ascontiguousarray(np.linalg.cholesky(SIGMA_aic))
        except np.linalg.LinAlgError:
            SIGMA_aic += np.eye(K) * 1e-8 
            P_aic = np.ascontiguousarray(np.linalg.cholesky(SIGMA_aic))
            
        valid_irfs_aic, attempts_aic, accepted_aic = fast_draw_core(A_aic, P_aic, signs, p_hat, K, Q_avg, h_max, mc_draws, max_loops)
        
        if accepted_aic < mc_draws:
            return iter_idx, None, None, 0, -1, f"Empty Set (AIC p={p_hat})"
            
        Target_IRF_aic = get_median_target_model(valid_irfs_aic, accepted_aic)
        SE_aic = (Target_IRF_aic - true_IRF_target)**2 
        
        # 2. Evaluate True p0 Model
        if p_hat == true_p:
            SE_p0 = SE_aic.copy()
            attempts_p0 = 0
        else:
            A_p0, SIGMA_p0 = lsvarcSA2_silent(simulated_data, true_p)
            
            try:
                P_p0 = np.ascontiguousarray(np.linalg.cholesky(SIGMA_p0))
            except np.linalg.LinAlgError:
                SIGMA_p0 += np.eye(K) * 1e-8 
                P_p0 = np.ascontiguousarray(np.linalg.cholesky(SIGMA_p0))
                
            valid_irfs_p0, attempts_p0, accepted_p0 = fast_draw_core(A_p0, P_p0, signs, true_p, K, Q_avg, h_max, mc_draws, max_loops)
            
            if accepted_p0 < mc_draws:
                return iter_idx, None, None, 0, -1, f"Empty Set (True p={true_p})"
                
            Target_IRF_p0 = get_median_target_model(valid_irfs_p0, accepted_p0)
            SE_p0 = (Target_IRF_p0 - true_IRF_target)**2
            
        return iter_idx, SE_aic, SE_p0, attempts_aic + attempts_p0, p_hat, "Success"
        
    except Exception as e:
        # LOUD ERROR LOGGING: If something fails, it screams it to the console!
        print(f"\n[WORKER CRASH] Iteration {iter_idx}: {type(e).__name__} - {str(e)}")
        return iter_idx, None, None, 0, -1, f"Python Error: {str(e)}"

# -------------------------------------------------------------------
# 5. PARALLEL ORCHESTRATION 
# -------------------------------------------------------------------
def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = script_dir if os.path.basename(script_dir) == KM_FOLDER else os.path.join(script_dir, KM_FOLDER)
    
    dgp_path = os.path.join(km_base, 'DGP files', 'true_dgp_parameters_2_lags.npz')
    
    if not os.path.exists(dgp_path):
        print(f"ERROR: Could not find DGP file at {dgp_path}")
        return

    dgp = np.load(dgp_path)
    true_A = np.ascontiguousarray(dgp['A_true'])
    true_V = np.ascontiguousarray(dgp['V_true'])
    true_B_tilde = np.ascontiguousarray(dgp['B_tilde_true'])
    true_IRF = np.ascontiguousarray(dgp['True_IRF'])
    true_p = int(dgp['p_true'])
    
    K = 4
    T_real = 414 
    Q_avg = 72.3 
    h_max = 24
    p_max = max(6, true_p) 
    
    N_iterations = 100  
    mc_draws = 10           # Look for 10 valid models
    max_loops = 5000000     
    
    sign_matrix = np.ascontiguousarray(np.array([
        [-1,       1,      1,      np.nan],  
        [-1,       1,     -1,      np.nan],  
        [ 1,       1,      1,      np.nan],  
        [np.nan, np.nan,   1,      np.nan]   
    ], dtype=np.float64))

    print("\n" + "="*60)
    print(f" STARTING MONTE CARLO (PURE SIGN RESTRICTIONS)")
    print(f" True DGP: {true_p} Lags | Max Evaluated Lag: {p_max}")
    print(f" Iterations: {N_iterations} | Draws per iteration target: {mc_draws}")
    print("="*60)

    tasks = []
    for i in range(N_iterations):
        tasks.append((
            i, SEED + i, true_A, true_V, true_B_tilde, true_p, true_IRF, 
            sign_matrix, Q_avg, h_max, mc_draws, max_loops, T_real, p_max
        ))

    n_cores = multiprocessing.cpu_count()
    start_time = time.time()
    
    print(f"Dispatching to {n_cores} CPU cores...\n")
    results = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(single_monte_carlo_iteration)(task) for task in tasks
    )

    all_SE_aic = []
    all_SE_p0 = []
    all_p_hats = []
    total_rejection_attempts = 0
    discarded_draws = 0
    failure_log = {}

    for iter_idx, SE_aic, SE_p0, attempts, p_hat, status in results:
        if status != "Success":
            discarded_draws += 1
            failure_log[status] = failure_log.get(status, 0) + 1
            continue
            
        all_SE_aic.append(SE_aic)
        all_SE_p0.append(SE_p0)
        all_p_hats.append(p_hat)
        total_rejection_attempts += attempts

    exec_time = time.time() - start_time

    # -------------------------------------------------------------------
    # DIAGNOSTIC REPORTING
    # -------------------------------------------------------------------
    print("\n" + "="*60)
    print(" MONTE CARLO DIAGNOSTIC REPORT")
    print("="*60)
    print(f"Successful Iterations: {len(all_SE_aic)}")
    print(f"Discarded Iterations:  {discarded_draws}")
    
    if discarded_draws > 0:
        print("\nBreakdown of Failures:")
        for reason, count in failure_log.items():
            print(f" - {count} times: {reason}")
            
    if len(all_SE_aic) == 0:
        print("\nCONCLUSION: Failed entirely. Check terminal for [WORKER CRASH] messages.")
        return

    # -------------------------------------------------------------------
    # RELATIVE MSE AGGREGATION (IVANOV & KILIAN)
    # -------------------------------------------------------------------
    MSE_aic = np.mean(all_SE_aic, axis=0)
    MSE_p0 = np.mean(all_SE_p0, axis=0)
    
    MSE_Ratio = MSE_aic / (MSE_p0 + 1e-12)
    standardized_relative_mse = np.exp(np.mean(np.log(MSE_Ratio)))
    correct_lag_percent = (all_p_hats.count(true_p) / len(all_p_hats)) * 100

    print("\n" + "="*60)
    print(" SIMULATION RESULTS (AIC vs TRUE DGP)")
    print("="*60)
    
    summary_data = {
        "Metric": [
            "AIC True Lag Detection Rate (%)",
            "Average MSE Ratio (AIC / p0) [Geometric Mean]", 
            "Total Rejection Draws Executed",
            "Execution Time (Seconds)"
        ],
        "Value": [
            round(correct_lag_percent, 2),
            round(standardized_relative_mse, 6),
            total_rejection_attempts,
            round(exec_time, 2)
        ]
    }
    
    results_df = pd.DataFrame(summary_data).set_index("Metric")
    print(results_df)

if __name__ == '__main__':
    main()