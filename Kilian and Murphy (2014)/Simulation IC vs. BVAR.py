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
# 2. FAST VAR ESTIMATORS (OLS & BVAR)
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

def bvar_minnesota_silent(y, p, tau=0.2, c=1e5):
    """
    Estimates a Bayesian VAR with a Minnesota Prior using Dummy Observations.
    Shrinks coefficients towards zero (White Noise prior) with standard lag decay.
    """
    t, K = y.shape
    y_T = y.T
    Y_mat = y_T[:, p-1:t]
    
    for i in range(1, p):
        Y_mat = np.vstack([Y_mat, y_T[:, p-1-i : t-i]])
        
    x = np.vstack([np.eye(11), np.zeros((1, 11))])
    n_years = int((t - p) // 12)
    remainder = int((t - p) % 12)
    
    X2 = np.tile(x, (n_years, 1)) if n_years > 0 else np.empty((0, 11))
    if remainder > 0:
        last = np.hstack([np.eye(remainder), np.zeros((remainder, 11 - remainder))])
        X2 = np.vstack([X2, last])
        
    X2 = np.hstack([np.ones((t - p, 1)), X2])
    
    # Standard OLS matrices
    X_ols = np.vstack([X2.T, Y_mat[:, :t-p]]).T 
    Y_ols = y_T[:, p:t].T 
    
    # Calculate variable standard deviations for prior scaling
    s = np.std(y, axis=0) 
    
    # Construct Dummy Observations
    # 1. Lags (Shrinks AR coefficients to zero, tighter for longer lags)
    Y_d1 = np.zeros((K * p, K))
    X_d1 = np.zeros((K * p, 12 + K * p))
    row = 0
    for lag in range(1, p + 1):
        for j in range(K):
            X_d1[row, 12 + (lag - 1) * K + j] = (s[j] * lag) / tau
            row += 1
            
    # 2. Deterministic Terms (Diffuse prior)
    Y_d2 = np.zeros((12, K))
    X_d2 = np.zeros((12, 12 + K * p))
    for i in range(12):
        X_d2[i, i] = 1.0 / c
        
    # 3. Covariance
    Y_d3 = np.diag(s)
    X_d3 = np.zeros((K, 12 + K * p))
    
    # Append dummies to the real data
    Y_aug = np.vstack([Y_ols, Y_d1, Y_d2, Y_d3])
    X_aug = np.vstack([X_ols, X_d1, X_d2, X_d3])
    
    # OLS on augmented data mathematically equals the Bayesian Posterior Mean
    B = np.linalg.lstsq(X_aug, Y_aug, rcond=None)[0].T 
    U = Y_ols.T - B @ X_ols.T 
    
    SIGMA = np.ascontiguousarray((U @ U.T) / (t - p - p * K - 12))
    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    
    return A, SIGMA

# -------------------------------------------------------------------
# 3. PURE SIGN RESTRICTION CORE 
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
def fast_draw_core(A, P, signs, p, K, Q_avg, h_max, target_draws, max_loops, seed_val):
    np.random.seed(seed_val)
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
# 4. THE SINGLE MONTE CARLO ITERATION (IC vs BVAR)
# -------------------------------------------------------------------
def single_monte_carlo_iteration(args):
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, Q_avg, h_max, mc_draws, max_loops, T_real, p_max = args
    
    try:
        K = true_A.shape[0]
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 100, iteration_seed)
        
        best_aic, best_sic, best_hqc = float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_sic, p_hat_hqc = 1, 1, 1
        N_eff = T_real - p_max 
        
        # 1. EVALUATE LAG ORDERS (Information Criteria)
        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            _, SIGMA_temp = lsvarcSA2_silent(y_slice, p_test)
            
            if not np.all(np.isfinite(SIGMA_temp)):
                continue
                
            sign_det, logdet_ols = np.linalg.slogdet(SIGMA_temp)
            if sign_det > 0: 
                df_correction = N_eff - (p_test * K) - 12
                logdet_ml = logdet_ols + K * np.log(df_correction / N_eff)
                
                num_params = K**2 * p_test
                
                aic_val = logdet_ml + (2.0 / N_eff) * num_params
                sic_val = logdet_ml + (np.log(N_eff) / N_eff) * num_params
                hqc_val = logdet_ml + (2.0 * np.log(np.log(N_eff)) / N_eff) * num_params
                
                if aic_val < best_aic:
                    best_aic = aic_val
                    p_hat_aic = p_test
                if sic_val < best_sic:
                    best_sic = sic_val
                    p_hat_sic = p_test
                if hqc_val < best_hqc:
                    best_hqc = hqc_val
                    p_hat_hqc = p_test
                    
        # 2. CACHE-BASED OLS EVALUATION
        computed_SEs = {}
        total_attempts = 0
        
        def get_se_for_p(p_target):
            nonlocal total_attempts
            if p_target in computed_SEs:
                return computed_SEs[p_target].copy()
                
            A_est, SIGMA_est = lsvarcSA2_silent(simulated_data, p_target)
            
            try:
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8 
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                
            seed_for_matrices = iteration_seed + p_target
            valid_irfs, attempts, accepted = fast_draw_core(A_est, P_est, signs, p_target, K, Q_avg, h_max, mc_draws, max_loops, seed_for_matrices)
            total_attempts += attempts
            
            if accepted < mc_draws:
                raise ValueError(f"Empty Set (OLS p={p_target})")
                
            target_irf = get_median_target_model(valid_irfs, accepted)
            se = (target_irf - true_IRF_target)**2 
            computed_SEs[p_target] = se
            return se

        try:
            SE_aic = get_se_for_p(p_hat_aic)
            SE_sic = get_se_for_p(p_hat_sic)
            SE_hqc = get_se_for_p(p_hat_hqc)
            SE_p0 = get_se_for_p(true_p)
        except ValueError as e:
            return iter_idx, None, None, None, None, None, 0, (None, None, None), str(e)
            
        # 3. BVAR MINNESOTA EVALUATION (Fixed at p_max)
        A_bvar, SIGMA_bvar = bvar_minnesota_silent(simulated_data, p_max, tau=0.2)
        
        try:
            P_bvar = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar))
        except np.linalg.LinAlgError:
            SIGMA_bvar += np.eye(K) * 1e-8 
            P_bvar = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar))
            
        # Give BVAR a distinct deterministic seed so it doesn't cross-contaminate OLS PRNG streams
        seed_for_bvar = iteration_seed + 1000 
        valid_irfs_bvar, attempts_bvar, accepted_bvar = fast_draw_core(
            A_bvar, P_bvar, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, seed_for_bvar
        )
        total_attempts += attempts_bvar
        
        if accepted_bvar < mc_draws:
            return iter_idx, None, None, None, None, None, 0, (None, None, None), "Empty Set (BVAR)"
            
        target_irf_bvar = get_median_target_model(valid_irfs_bvar, accepted_bvar)
        SE_bvar = (target_irf_bvar - true_IRF_target)**2 
            
        return iter_idx, SE_aic, SE_sic, SE_hqc, SE_bvar, SE_p0, total_attempts, (p_hat_aic, p_hat_sic, p_hat_hqc), "Success"
        
    except Exception as e:
        print(f"\n[WORKER CRASH] Iteration {iter_idx}: {type(e).__name__} - {str(e)}")
        return iter_idx, None, None, None, None, None, 0, (None, None, None), f"Python Error: {str(e)}"

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
    
    dgp_path = os.path.join(km_base, 'DGP files', 'true_dgp_parameters_4_lags.npz')
    
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
    T_real = 100 
    Q_avg = 72.3 
    h_max = 24
    p_max = max(12, true_p) 
    
    N_iterations = 1000  
    mc_draws = 50           
    max_loops = 5000000     
    
    sign_matrix = np.ascontiguousarray(np.array([
        [-1,       1,      1,      np.nan],  
        [-1,       1,     -1,      np.nan],  
        [ 1,       1,      1,      np.nan],  
        [np.nan, np.nan,   1,      np.nan]   
    ], dtype=np.float64))

    print("\n" + "="*60)
    print(f" STARTING MONTE CARLO: OLS (AIC/SIC/HQC) vs BVAR (Minnesota)")
    print(f" Sample Size (T): {T_real} | True DGP: {true_p} Lags")
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

    all_SE_aic, all_SE_sic, all_SE_hqc, all_SE_bvar, all_SE_p0 = [], [], [], [], []
    all_p_hats_aic, all_p_hats_sic, all_p_hats_hqc = [], [], []
    total_rejection_attempts = 0
    discarded_draws = 0
    failure_log = {}

    for iter_idx, SE_aic, SE_sic, SE_hqc, SE_bvar, SE_p0, attempts, p_hats, status in results:
        if status != "Success":
            discarded_draws += 1
            failure_log[status] = failure_log.get(status, 0) + 1
            continue
            
        all_SE_aic.append(SE_aic)
        all_SE_sic.append(SE_sic)
        all_SE_hqc.append(SE_hqc)
        all_SE_bvar.append(SE_bvar)
        all_SE_p0.append(SE_p0)
        
        all_p_hats_aic.append(p_hats[0])
        all_p_hats_sic.append(p_hats[1])
        all_p_hats_hqc.append(p_hats[2])
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
    # RELATIVE MSE AGGREGATION
    # -------------------------------------------------------------------
    MSE_aic = np.mean(all_SE_aic, axis=0)
    MSE_sic = np.mean(all_SE_sic, axis=0)
    MSE_hqc = np.mean(all_SE_hqc, axis=0)
    MSE_bvar = np.mean(all_SE_bvar, axis=0)
    MSE_p0  = np.mean(all_SE_p0, axis=0)
    
    MSE_Ratio_aic = MSE_aic / (MSE_p0 + 1e-12)
    MSE_Ratio_sic = MSE_sic / (MSE_p0 + 1e-12)
    MSE_Ratio_hqc = MSE_hqc / (MSE_p0 + 1e-12)
    MSE_Ratio_bvar = MSE_bvar / (MSE_p0 + 1e-12)
    
    std_rel_mse_aic = np.exp(np.mean(np.log(MSE_Ratio_aic)))
    std_rel_mse_sic = np.exp(np.mean(np.log(MSE_Ratio_sic)))
    std_rel_mse_hqc = np.exp(np.mean(np.log(MSE_Ratio_hqc)))
    std_rel_mse_bvar = np.exp(np.mean(np.log(MSE_Ratio_bvar)))
    
    lag_rate_aic = (all_p_hats_aic.count(true_p) / len(all_p_hats_aic)) * 100
    lag_rate_sic = (all_p_hats_sic.count(true_p) / len(all_p_hats_sic)) * 100
    lag_rate_hqc = (all_p_hats_hqc.count(true_p) / len(all_p_hats_hqc)) * 100

    print("\n" + "="*60)
    print(" SIMULATION RESULTS: OLS (INFO CRITERIA) vs BVAR")
    print("="*60)
    
    summary_data = {
        "Metric": [
            "True Lag Detection Rate (%)",
            "Average MSE Ratio (Model / True DGP)", 
            "Mean Evaluated Lag Order"
        ],
        "AIC": [
            round(lag_rate_aic, 2),
            round(std_rel_mse_aic, 6),
            round(np.mean(all_p_hats_aic), 2)
        ],
        "SIC (BIC)": [
            round(lag_rate_sic, 2),
            round(std_rel_mse_sic, 6),
            round(np.mean(all_p_hats_sic), 2)
        ],
        "HQC": [
            round(lag_rate_hqc, 2),
            round(std_rel_mse_hqc, 6),
            round(np.mean(all_p_hats_hqc), 2)
        ],
        "BVAR (Minn.)": [
            "N/A", # BVAR doesn't select lags, it shrinks them
            round(std_rel_mse_bvar, 6),
            float(p_max) # Always evaluated at maximum lag length
        ]
    }
    
    results_df = pd.DataFrame(summary_data).set_index("Metric")
    print(results_df)
    
    print("\n" + "-"*60)
    print(f"Total Rejection Draws Executed: {total_rejection_attempts:,}")
    print(f"Total Execution Time:           {round(exec_time, 2)} Seconds")
    print("-"*60)

# -------------------------------------------------------------------
    # 6. SAVE RESULTS TO DISK
    # -------------------------------------------------------------------
    # Construct the path to the Results folder
    results_dir = os.path.join(km_base, 'Results')
    
    # Create the folder if it does not exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Dynamically generate the filename using the simulation parameters
    filename = f"Simulation_Results_p{true_p}_iters{N_iterations}_draws{mc_draws}.csv"
    save_path = os.path.join(results_dir, filename)
    
    # Save the dataframe to CSV
    results_df.to_csv(save_path)
    
    print(f"\n[SUCCESS] Matrix saved to: {save_path}")

if __name__ == '__main__':
    main()