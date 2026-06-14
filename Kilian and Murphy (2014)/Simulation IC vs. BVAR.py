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
# 2. FAST VAR ESTIMATORS (OLS & BVARs)
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
    X_ols = np.vstack([X2.T, Y_mat[:, :t-p]]).T 
    Y_ols = y_T[:, p:t].T 
    s = np.std(y, axis=0) 
    
    Y_d1 = np.zeros((K * p, K))
    X_d1 = np.zeros((K * p, 12 + K * p))
    row = 0
    for lag in range(1, p + 1):
        for j in range(K):
            X_d1[row, 12 + (lag - 1) * K + j] = (s[j] * lag) / tau
            row += 1
            
    Y_d2 = np.zeros((12, K))
    X_d2 = np.zeros((12, 12 + K * p))
    for i in range(12):
        X_d2[i, i] = 1.0 / c
        
    Y_d3 = np.diag(s)
    X_d3 = np.zeros((K, 12 + K * p))
    
    Y_aug = np.vstack([Y_ols, Y_d1, Y_d2, Y_d3])
    X_aug = np.vstack([X_ols, X_d1, X_d2, X_d3])
    
    B = np.linalg.lstsq(X_aug, Y_aug, rcond=None)[0].T 
    U = Y_ols.T - B @ X_ols.T 
    
    SIGMA = np.ascontiguousarray((U @ U.T) / (t - p - p * K - 12))
    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    
    return A, SIGMA

def bvar_conjugate_silent(y, p, tau=0.2, c=1e5, mu=1.0, delta=1.0):
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
    X_ols = np.vstack([X2.T, Y_mat[:, :t-p]]).T 
    Y_ols = y_T[:, p:t].T 
    
    s = np.std(y, axis=0) 
    y_bar = np.mean(y[:p, :], axis=0) 
    
    Y_d1 = np.zeros((K * p, K))
    X_d1 = np.zeros((K * p, 12 + K * p))
    row = 0
    for lag in range(1, p + 1):
        for j in range(K):
            X_d1[row, 12 + (lag - 1) * K + j] = (s[j] * lag) / tau
            row += 1
            
    Y_d2 = np.zeros((12, K))
    X_d2 = np.zeros((12, 12 + K * p))
    for i in range(12):
        X_d2[i, i] = 1.0 / c
        
    Y_d3 = np.diag(s)
    X_d3 = np.zeros((K, 12 + K * p))
    
    Y_d4 = np.zeros((K, K))
    X_d4 = np.zeros((K, 12 + K * p))
    for i in range(K):
        Y_d4[i, i] = y_bar[i] / mu
        for lag in range(1, p + 1):
            X_d4[i, 12 + (lag - 1) * K + i] = y_bar[i] / mu
            
    Y_d5 = np.zeros((1, K))
    X_d5 = np.zeros((1, 12 + K * p))
    for i in range(K):
        Y_d5[0, i] = y_bar[i] / delta
        for lag in range(1, p + 1):
            X_d5[0, 12 + (lag - 1) * K + i] = y_bar[i] / delta
    X_d5[0, 0] = 1.0 / delta 
    
    Y_aug = np.vstack([Y_ols, Y_d1, Y_d2, Y_d3, Y_d4, Y_d5])
    X_aug = np.vstack([X_ols, X_d1, X_d2, X_d3, X_d4, X_d5])
    
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
    if target_draws <= 0:
        return np.zeros((1, h_max, K, K)), 0, 0
        
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
# 4. THE SINGLE MONTE CARLO ITERATION
# -------------------------------------------------------------------
def single_monte_carlo_iteration(args):
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, Q_avg, h_max, mc_draws, max_loops, T_real, p_max = args
    
    try:
        K = true_A.shape[0]
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 100, iteration_seed)
        
        best_aic, best_sic, best_hqc = float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_sic, p_hat_hqc = 1, 1, 1
        N_eff = T_real - p_max 
        
        ols_cache = {}
        bic_scores = {} # Store raw BIC scores for BMA weighting
        
        # 1. EVALUATE LAG ORDERS 
        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            A_temp, SIGMA_temp = lsvarcSA2_silent(y_slice, p_test)
            
            if not np.all(np.isfinite(SIGMA_temp)):
                continue
                
            ols_cache[p_test] = (A_temp.copy(), SIGMA_temp.copy())
                
            sign_det, logdet_ols = np.linalg.slogdet(SIGMA_temp)
            if sign_det > 0: 
                df_correction = N_eff - (p_test * K) - 12
                logdet_ml = logdet_ols + K * np.log(df_correction / N_eff)
                
                num_params = K**2 * p_test
                
                aic_val = logdet_ml + (2.0 / N_eff) * num_params
                sic_val = logdet_ml + (np.log(N_eff) / N_eff) * num_params
                hqc_val = logdet_ml + (2.0 * np.log(np.log(N_eff)) / N_eff) * num_params
                
                bic_scores[p_test] = sic_val # Store for BMA
                
                if aic_val < best_aic:
                    best_aic = aic_val
                    p_hat_aic = p_test
                if sic_val < best_sic:
                    best_sic = sic_val
                    p_hat_sic = p_test
                if hqc_val < best_hqc:
                    best_hqc = hqc_val
                    p_hat_hqc = p_test
                    
        # -------------------------------------------------------------
        # 2. BAYESIAN MODEL AVERAGING (BIC WEIGHTS)
        # -------------------------------------------------------------
        if not bic_scores:
            raise ValueError("No valid OLS estimations found.")
            
        min_bic = min(bic_scores.values())
        raw_weights = {p: np.exp(-0.5 * (score - min_bic)) for p, score in bic_scores.items()}
        weight_sum = sum(raw_weights.values())
        
        # Filter out negligible models (weight < 1%) to optimize speed
        kept_lags = [p for p, w in raw_weights.items() if (w / weight_sum) > 0.01]
        kept_weight_sum = sum(raw_weights[p] for p in kept_lags)
        final_bic_weights = {p: raw_weights[p] / kept_weight_sum for p in kept_lags}

        # -------------------------------------------------------------
        # 3. CACHE-BASED OLS EVALUATION (Standard ICs)
        # -------------------------------------------------------------
        computed_SEs = {}
        total_attempts = 0
        
        def get_se_for_p(p_target):
            nonlocal total_attempts
            if p_target in computed_SEs:
                return computed_SEs[p_target].copy()
                
            A_est, SIGMA_est = ols_cache[p_target]
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
            return iter_idx, None, None, None, None, None, None, None, 0, (None, None, None), str(e)
            
        # -------------------------------------------------------------
        # 4. DRAW BMA POOL
        # -------------------------------------------------------------
        bma_pool = []
        bma_accepted_total = 0
        
        for p_bma in kept_lags:
            # Proportional drawing based on BIC weight
            target_draws_p = int(round(mc_draws * final_bic_weights[p_bma]))
            if target_draws_p == 0:
                continue
                
            A_est, SIGMA_est = ols_cache[p_bma]
            try:
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8 
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                
            seed_bma = iteration_seed + 5000 + p_bma
            v_irfs, attempts, acc = fast_draw_core(A_est, P_est, signs, p_bma, K, Q_avg, h_max, target_draws_p, max_loops, seed_bma)
            total_attempts += attempts
            
            if acc > 0:
                bma_pool.append(v_irfs[:acc])
                bma_accepted_total += acc
                
        if bma_accepted_total == 0:
            return iter_idx, None, None, None, None, None, None, None, 0, (None, None, None), "Empty Set (BMA)"
            
        # Stack all drawn proportional models into a single pooled universe
        pooled_bma_irfs = np.vstack(bma_pool)
        target_irf_bma = get_median_target_model(pooled_bma_irfs, bma_accepted_total)
        SE_bma = (target_irf_bma - true_IRF_target)**2

        # -------------------------------------------------------------
        # 5. BVAR EVALUATIONS
        # -------------------------------------------------------------
        A_bvar_minn, SIGMA_bvar_minn = bvar_minnesota_silent(simulated_data, p_max, tau=0.2)
        try:
            P_bvar_minn = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar_minn))
        except np.linalg.LinAlgError:
            SIGMA_bvar_minn += np.eye(K) * 1e-8 
            P_bvar_minn = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar_minn))
            
        valid_irfs_bvar_minn, att_minn, acc_minn = fast_draw_core(
            A_bvar_minn, P_bvar_minn, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + 1000
        )
        total_attempts += att_minn
        if acc_minn < mc_draws:
            return iter_idx, None, None, None, None, None, None, None, 0, (None, None, None), "Empty Set (BVAR Minn)"
        SE_bvar_minn = (get_median_target_model(valid_irfs_bvar_minn, acc_minn) - true_IRF_target)**2 

        A_bvar_conj, SIGMA_bvar_conj = bvar_conjugate_silent(simulated_data, p_max, tau=0.2, mu=1.0, delta=1.0)
        try:
            P_bvar_conj = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar_conj))
        except np.linalg.LinAlgError:
            SIGMA_bvar_conj += np.eye(K) * 1e-8 
            P_bvar_conj = np.ascontiguousarray(np.linalg.cholesky(SIGMA_bvar_conj))
            
        valid_irfs_bvar_conj, att_conj, acc_conj = fast_draw_core(
            A_bvar_conj, P_bvar_conj, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + 2000
        )
        total_attempts += att_conj
        if acc_conj < mc_draws:
            return iter_idx, None, None, None, None, None, None, None, 0, (None, None, None), "Empty Set (BVAR Conj)"
        SE_bvar_conj = (get_median_target_model(valid_irfs_bvar_conj, acc_conj) - true_IRF_target)**2 
            
        return iter_idx, SE_aic, SE_sic, SE_hqc, SE_bma, SE_bvar_minn, SE_bvar_conj, SE_p0, total_attempts, (p_hat_aic, p_hat_sic, p_hat_hqc), "Success"
        
    except Exception as e:
        return iter_idx, None, None, None, None, None, None, None, 0, (None, None, None), f"Python Error: {str(e)}"

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
    
    Q_avg = 72.3 
    h_max = 24
    N_iterations = 1000   # <--- Bump to 1000 for final run
    mc_draws = 100        # <--- Target accepted models        
    max_loops = 5000000 
    
    sign_matrix = np.ascontiguousarray(np.array([
        [-1,       1,      1,      np.nan],  
        [-1,       1,     -1,      np.nan],  
        [ 1,       1,      1,      np.nan],  
        [np.nan, np.nan,   1,      np.nan]   
    ], dtype=np.float64))

    dgp_lag_orders = [4, 6, 8, 10]
    sample_sizes = [80, 100, 120, 160, 200]
    
    master_results_list = []
    
    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    
    print("\n" + "="*80)
    print(f" STARTING MASTER ASYMPTOTIC MONTE CARLO: 6 Estimators")
    print(f" Lags (p0) = {dgp_lag_orders} | Sample Sizes (T) = {sample_sizes}")
    print(f" Iterations: {N_iterations} | Draws: {mc_draws} | Utilizing {n_cores} CPU Cores")
    print("="*80)

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
        
        p_max = max(12, true_p)

        for current_T in sample_sizes:
            print(f"     Running Simulation for T = {current_T} ...", end="", flush=True)
            
            tasks = []
            for i in range(N_iterations):
                iteration_seed = SEED + i + (current_p0 * 10000) + (current_T * 100000)
                tasks.append((
                    i, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF, 
                    sign_matrix, Q_avg, h_max, mc_draws, max_loops, current_T, p_max
                ))

            results = Parallel(n_jobs=n_cores, backend='loky')(
                delayed(single_monte_carlo_iteration)(task) for task in tasks
            )

            all_SE_aic, all_SE_sic, all_SE_hqc, all_SE_bma, all_SE_p0 = [], [], [], [], []
            all_SE_bvar_minn, all_SE_bvar_conj = [], []
            all_p_hats_aic, all_p_hats_sic, all_p_hats_hqc = [], [], []
            discarded_draws = 0

            for iter_idx, SE_aic, SE_sic, SE_hqc, SE_bma, SE_bvar_minn, SE_bvar_conj, SE_p0, attempts, p_hats, status in results:
                if status != "Success":
                    discarded_draws += 1
                    continue
                    
                all_SE_aic.append(SE_aic)
                all_SE_sic.append(SE_sic)
                all_SE_hqc.append(SE_hqc)
                all_SE_bma.append(SE_bma)
                all_SE_bvar_minn.append(SE_bvar_minn)
                all_SE_bvar_conj.append(SE_bvar_conj)
                all_SE_p0.append(SE_p0)
                
                all_p_hats_aic.append(p_hats[0])
                all_p_hats_sic.append(p_hats[1])
                all_p_hats_hqc.append(p_hats[2])

            if len(all_SE_aic) == 0:
                print(f" [FAILED] All iterations discarded.")
                continue
                
            print(f" [SUCCESS] ({len(all_SE_aic)}/{N_iterations})")

            # -------------------------------------------------------------------
            # EMPIRICAL DISTRIBUTION AGGREGATION
            # -------------------------------------------------------------------
            MSE_aic_iter = np.sum(all_SE_aic, axis=(1, 2, 3))
            MSE_sic_iter = np.sum(all_SE_sic, axis=(1, 2, 3))
            MSE_hqc_iter = np.sum(all_SE_hqc, axis=(1, 2, 3))
            MSE_bma_iter = np.sum(all_SE_bma, axis=(1, 2, 3))
            MSE_bvar_minn_iter = np.sum(all_SE_bvar_minn, axis=(1, 2, 3))
            MSE_bvar_conj_iter = np.sum(all_SE_bvar_conj, axis=(1, 2, 3))
            MSE_p0_iter = np.sum(all_SE_p0, axis=(1, 2, 3))
            
            ratio_aic = MSE_aic_iter / (MSE_p0_iter + 1e-12)
            ratio_sic = MSE_sic_iter / (MSE_p0_iter + 1e-12)
            ratio_hqc = MSE_hqc_iter / (MSE_p0_iter + 1e-12)
            ratio_bma = MSE_bma_iter / (MSE_p0_iter + 1e-12)
            ratio_bvar_minn = MSE_bvar_minn_iter / (MSE_p0_iter + 1e-12)
            ratio_bvar_conj = MSE_bvar_conj_iter / (MSE_p0_iter + 1e-12)
            
            def calculate_metrics(ratios_array, p_hats_list, true_p_val):
                geom_mean = np.exp(np.mean(np.log(ratios_array)))
                perc_05 = np.percentile(ratios_array, 5)
                perc_95 = np.percentile(ratios_array, 95)
                
                if p_hats_list is not None:
                    lag_rate = (p_hats_list.count(true_p_val) / len(p_hats_list)) * 100
                    mean_lag = np.mean(p_hats_list)
                else:
                    lag_rate = np.nan
                    mean_lag = p_max 
                    
                return round(geom_mean, 4), round(perc_05, 4), round(perc_95, 4), round(lag_rate, 2), round(mean_lag, 2)

            metrics_aic = calculate_metrics(ratio_aic, all_p_hats_aic, true_p)
            metrics_sic = calculate_metrics(ratio_sic, all_p_hats_sic, true_p)
            metrics_hqc = calculate_metrics(ratio_hqc, all_p_hats_hqc, true_p)
            metrics_bma = calculate_metrics(ratio_bma, None, true_p) # BMA blends lags
            metrics_bvar_minn = calculate_metrics(ratio_bvar_minn, None, true_p)
            metrics_bvar_conj = calculate_metrics(ratio_bvar_conj, None, true_p)

            # Append to Master List 
            models = [
                ("AIC", metrics_aic), 
                ("SIC (BIC)", metrics_sic), 
                ("HQC", metrics_hqc), 
                ("BMA (BIC-Weighted)", metrics_bma),
                ("BVAR (Minn.)", metrics_bvar_minn),
                ("BVAR (Conj.)", metrics_bvar_conj)
            ]
            
            for model_name, m_data in models:
                master_results_list.append({
                    "True DGP (p0)": current_p0,
                    "Sample Size (T)": current_T,
                    "Estimator": model_name,
                    "Lag Detection Rate (%)": m_data[3] if not np.isnan(m_data[3]) else "N/A",
                    "Mean Evaluated Lag": m_data[4],
                    "Geom Mean MSE Ratio": m_data[0],
                    "5th Percentile": m_data[1],
                    "95th Percentile": m_data[2]
                })

    total_time = time.time() - global_start_time

    # -------------------------------------------------------------------
    # SAVE MASTER RESULTS TO DISK
    # -------------------------------------------------------------------
    if len(master_results_list) > 0:
        final_df = pd.DataFrame(master_results_list)
        
        print("\n" + "="*80)
        print(" MASTER ASYMPTOTIC SIMULATION RESULTS: 6 Estimators")
        print("="*80)
        print(final_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print(f"Total Master Execution Time: {round(total_time, 2)} Seconds")
        print("-"*80)

        results_dir = os.path.join(km_base, 'Results')
        os.makedirs(results_dir, exist_ok=True)
        filename = f"Master_Final_SVAR_Comparison_iters{N_iterations}_draws{mc_draws}.csv"
        save_path = os.path.join(results_dir, filename)
        
        final_df.to_csv(save_path, index=False)
        print(f"\n[SUCCESS] Master Dataframe saved to: {save_path}")
    else:
        print("\n[ERROR] No data generated across any DGP. Check file paths and errors.")

if __name__ == '__main__':
    main()