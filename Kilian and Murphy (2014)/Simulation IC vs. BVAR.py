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

def bvar_minnesota_silent(y, p, tau=0.2, c=1e5, delta_prior=None):
    t, K = y.shape
    y_T = y.T
    Y_mat = y_T[:, p-1:t]
    
    if delta_prior is None: delta_prior = np.zeros(K)
        
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
            if lag == 1: Y_d1[row, j] = (s[j] * delta_prior[j]) / tau
            row += 1
            
    Y_d2 = np.zeros((12, K))
    X_d2 = np.zeros((12, 12 + K * p))
    for i in range(12): X_d2[i, i] = 1.0 / c
        
    Y_d3 = np.diag(s)
    X_d3 = np.zeros((K, 12 + K * p))
    
    Y_aug = np.vstack([Y_ols, Y_d1, Y_d2, Y_d3])
    X_aug = np.vstack([X_ols, X_d1, X_d2, X_d3])
    
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
    if target_draws <= 0: return np.zeros((1, h_max, K, K)), 0, 0
        
    np.random.seed(seed_val)
    valid_IRFs = np.zeros((target_draws, h_max, K, K))
    attempts, accepted = 0, 0
    
    while accepted < target_draws and attempts < max_loops:
        attempts += 1
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)
        for i in range(K):
            if R[i, i] < 0: Q[:, i] = -Q[:, i]
                
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
    if accepted_count == 0: raise ValueError("Empty set.")
    valid_irfs_sliced = valid_irfs[:accepted_count]
    pointwise_median_irf = np.median(valid_irfs_sliced, axis=0)

    # Fry-Pagan standardization: divide deviations by pointwise std across draws
    pointwise_std_irf = np.std(valid_irfs_sliced, axis=0)
    pointwise_std_irf = np.where(pointwise_std_irf == 0, np.nan, pointwise_std_irf)

    standardized_dev = (valid_irfs_sliced - pointwise_median_irf) / pointwise_std_irf
    distances = np.nansum(standardized_dev**2, axis=(1, 2, 3))
    return valid_irfs_sliced[np.argmin(distances)]

# -------------------------------------------------------------------
# 4. THE SINGLE MONTE CARLO ITERATION
# -------------------------------------------------------------------
def single_monte_carlo_iteration(args):
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, Q_avg, h_max, mc_draws, max_loops, T_real, p_max = args
    
    try:
        K = true_A.shape[0]
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 100, iteration_seed)
        
        best_aic, best_aicc, best_sic, best_hqc = float('inf'), float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_aicc, p_hat_sic, p_hat_hqc = 1, 1, 1, 1
        N_eff = T_real - p_max 
        
        # -------------------------------------------------------------
        # PRIOR CONFIGURATIONS: White Noise (WN) is the default
        # -------------------------------------------------------------
        delta_wn = np.zeros(K)  # Default
        delta_rw = np.ones(K)   # Comparison benchmark
        
        # -------------------------------------------------------------
        # 1. EVALUATE LAG ORDERS (OLS)
        # -------------------------------------------------------------
        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            A_temp, SIGMA_temp = lsvarcSA2_silent(y_slice, p_test)
            
            if not np.all(np.isfinite(SIGMA_temp)): continue
            ols_cache[p_test] = (A_temp.copy(), SIGMA_temp.copy())
                
            sign_det, logdet_ols = np.linalg.slogdet(SIGMA_temp)
            if sign_det > 0: 
                df_correction = N_eff - (p_test * K) - 12
                logdet_ml = logdet_ols + K * np.log(df_correction / N_eff)
                
                num_params = K**2 * p_test
                aic_val = logdet_ml + (2.0 / N_eff) * num_params
                sic_val = logdet_ml + (np.log(N_eff) / N_eff) * num_params
                hqc_val = logdet_ml + (2.0 * np.log(np.log(N_eff)) / N_eff) * num_params
                
                k_eq = K * p_test + 12 
                if N_eff - k_eq - 1 > 0:
                    aicc_val = aic_val + (2.0 * k_eq * (k_eq + 1)) / (N_eff - k_eq - 1)
                else:
                    aicc_val = float('inf')
                
                bic_scores[p_test] = sic_val 
                
                if aic_val < best_aic: best_aic, p_hat_aic = aic_val, p_test
                if aicc_val < best_aicc: best_aicc, p_hat_aicc = aicc_val, p_test
                if sic_val < best_sic: best_sic, p_hat_sic = sic_val, p_test
                if hqc_val < best_hqc: best_hqc, p_hat_hqc = hqc_val, p_test
                    
        # -------------------------------------------------------------
        # 2. BMA WEIGHTING
        # -------------------------------------------------------------
        if not bic_scores: raise ValueError("No valid OLS estimations.")
        min_bic = min(bic_scores.values())
        raw_weights = {p: np.exp(-0.5 * (score - min_bic)) for p, score in bic_scores.items()}
        weight_sum = sum(raw_weights.values())
        kept_lags = [p for p, w in raw_weights.items() if (w / weight_sum) > 0.01]
        final_bic_weights = {p: raw_weights[p] / sum(raw_weights[p] for p in kept_lags) for p in kept_lags}

        total_attempts = 0

        # -------------------------------------------------------------
        # 3. OLS ESTIMATORS
        # -------------------------------------------------------------
        computed_ols_SEs = {}
        def get_ols_se_for_p(p_target):
            nonlocal total_attempts
            if p_target in computed_ols_SEs: return computed_ols_SEs[p_target]
                
            A_est, SIGMA_est = ols_cache[p_target]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8 
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                
            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_target, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + p_target)
            total_attempts += att
            if acc < mc_draws: raise ValueError(f"Empty Set (OLS p={p_target})")
                
            se = (get_median_target_model(v_irfs, acc) - true_IRF_target)**2 
            computed_ols_SEs[p_target] = se
            return se

        try:
            SE_aic = get_ols_se_for_p(p_hat_aic)
            SE_aicc = get_ols_se_for_p(p_hat_aicc)
            SE_sic = get_ols_se_for_p(p_hat_sic)
            SE_hqc = get_ols_se_for_p(p_hat_hqc)
            SE_p0 = get_ols_se_for_p(true_p)
        except ValueError as e:
            return iter_idx, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), str(e)
            
        ols_bma_pool, ols_bma_accepted = [], 0
        for p_bma in kept_lags:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            
            A_est, SIGMA_est = ols_cache[p_bma]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8 
                P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                
            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 5000 + p_bma)
            total_attempts += att
            if acc > 0: ols_bma_pool.append(v_irfs[:acc]); ols_bma_accepted += acc
                
        if ols_bma_accepted == 0: return iter_idx, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), "Empty Set BMA"
        SE_ols_bma = (get_median_target_model(np.vstack(ols_bma_pool), ols_bma_accepted) - true_IRF_target)**2
       
       # -------------------------------------------------------------
       # 4. MINNESOTA BVAR SENSITIVITY (WN Default + RW Benchmark)
       # -------------------------------------------------------------
        def eval_minn(tau_val, delta_array, seed_offset):
            nonlocal total_attempts
            A_m, SIGMA_m = bvar_minnesota_silent(simulated_data, p_max, tau=tau_val, delta_prior=delta_array)
            try: P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            except np.linalg.LinAlgError:
                SIGMA_m += np.eye(K) * 1e-8 
                P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            v_m, att_m, acc_m = fast_draw_core(A_m, P_m, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + seed_offset)
            total_attempts += att_m
            if acc_m < mc_draws: raise ValueError("Empty Minn")
            return (get_median_target_model(v_m, acc_m) - true_IRF_target)**2
        
        try:
            # WN Baseline (Standard)
            SE_minn_wn_std = eval_minn(0.20, delta_wn, 1000)
            # RW Benchmark Comparison
            SE_minn_rw_std = eval_minn(0.20, delta_rw, 1001)
        except ValueError as e:
            return iter_idx, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), str(e)
        
        # -------------------------------------------------------------
        # 5. DYNAMIC BVAR (Using White Noise Prior)
        # -------------------------------------------------------------
        def get_bvar_est(p_target):
            if p_target not in bvar_cache:
                bvar_cache[p_target] = bvar_minnesota_silent(simulated_data, p_target, tau=0.2, delta_prior=delta_wn)
            return bvar_cache[p_target]

        def eval_dyn_bvar(p_tgt, seed_offset, is_bma=False, draws=mc_draws):
            nonlocal total_attempts
            A_c, SIGMA_c = get_bvar_est(p_tgt)
            try: P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            except np.linalg.LinAlgError:
                SIGMA_c += np.eye(K) * 1e-8 
                P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            v_c, att_c, acc_c = fast_draw_core(A_c, P_c, signs, p_tgt, K, Q_avg, h_max, draws, max_loops, iteration_seed + seed_offset)
            total_attempts += att_c
            if is_bma: return v_c, acc_c
            if acc_c < draws: raise ValueError("Empty Dyn BVAR")
            return (get_median_target_model(v_c, acc_c) - true_IRF_target)**2

        try:
            SE_bvar_bic = eval_dyn_bvar(p_hat_sic, 3000 + p_hat_sic)
        except ValueError as e:
            return iter_idx, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), str(e)

        bvar_bma_pool, bvar_bma_acc = [], 0
        for p_bma in kept_lags:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            v_c, acc_c = eval_dyn_bvar(p_bma, 4000 + p_bma, is_bma=True, draws=td)
            if acc_c > 0: bvar_bma_pool.append(v_c[:acc_c]); bvar_bma_acc += acc_c
                
        if bvar_bma_acc == 0: return iter_idx, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), "Empty BVAR BMA"
        SE_bvar_bma = (get_median_target_model(np.vstack(bvar_bma_pool), bvar_bma_acc) - true_IRF_target)**2

        return iter_idx, SE_aic, SE_aicc, SE_sic, SE_hqc, SE_ols_bma, SE_minn_rw_tight, SE_minn_rw_std, SE_minn_rw_loose, SE_minn_wn_std, SE_bvar_bic, SE_bvar_bma, SE_p0, (p_hat_aic, p_hat_aicc, p_hat_sic, p_hat_hqc), "Success"
        
    except Exception as e:
        return iter_idx, None, None, None, None, None, None, None, None, None, None, None, None, 0, (None, None, None, None), f"Error: {str(e)}"

# -------------------------------------------------------------------
# 5. PARALLEL ORCHESTRATION WITH NESTED LOOP (p0 and T)
# -------------------------------------------------------------------
def main():
    try: script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: script_dir = os.getcwd()

    KM_FOLDER = 'Kilian and Murphy (2014)'
    km_base = script_dir if os.path.basename(script_dir) == KM_FOLDER else os.path.join(script_dir, KM_FOLDER)
    
    Q_avg, h_max = 72.3, 24
    N_iterations, mc_draws, max_loops = 100, 50, 5000000 
    
    sign_matrix = np.ascontiguousarray(np.array([
        [-1,  1,  1, np.nan],  
        [-1,  1, -1, np.nan],  
        [ 1,  1,  1, np.nan],  
        [np.nan, np.nan, 1, np.nan]   
    ], dtype=np.float64))

    dgp_lag_orders, sample_sizes = [4, 6, 8, 10], [96, 144, 240, 480]
    master_results_list = []
    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()
    
    print("\n" + "="*90)
    print(f" STARTING MASTER ASYMPTOTIC MONTE CARLO (Conjugate Removed, WN vs RW Test Added)")
    print("="*90)

    for current_p0 in dgp_lag_orders:
        dgp_path = os.path.join(km_base, 'DGP files', f'true_dgp_parameters_{current_p0}_lags.npz')
        if not os.path.exists(dgp_path): continue
        dgp = np.load(dgp_path)
        true_A, true_V, true_B_tilde = np.ascontiguousarray(dgp['A_true']), np.ascontiguousarray(dgp['V_true']), np.ascontiguousarray(dgp['B_tilde_true'])
        true_IRF, true_p = np.ascontiguousarray(dgp['True_IRF']), int(dgp['p_true'])
        p_max = max(12, true_p)

        for current_T in sample_sizes:
            print(f"     Running Simulation for T = {current_T} ...", end="", flush=True)
            tasks = [(i, SEED + i + (current_p0 * 10000) + (current_T * 100000), true_A, true_V, true_B_tilde, true_p, true_IRF, sign_matrix, Q_avg, h_max, mc_draws, max_loops, current_T, p_max) for i in range(N_iterations)]
            results = Parallel(n_jobs=n_cores, backend='loky')(delayed(single_monte_carlo_iteration)(task) for task in tasks)

            all_SE_aic, all_SE_aicc, all_SE_sic, all_SE_hqc, all_SE_ols_bma = [], [], [], [], []
            all_SE_minn_rw_t, all_SE_minn_rw_s, all_SE_minn_rw_l = [], [], []
            all_SE_minn_wn_s, all_SE_bvar_bic, all_SE_bvar_bma, all_SE_p0 = [], [], [], []
            all_p_hats_aic, all_p_hats_aicc, all_p_hats_sic, all_p_hats_hqc = [], [], [], []

            for res in results:
                if res[-1] != "Success": continue
                all_SE_aic.append(res[1]); all_SE_aicc.append(res[2]); all_SE_sic.append(res[3]); all_SE_hqc.append(res[4])
                all_SE_ols_bma.append(res[5])
                all_SE_minn_rw_t.append(res[6]); all_SE_minn_rw_s.append(res[7]); all_SE_minn_rw_l.append(res[8])
                all_SE_minn_wn_s.append(res[9]); all_SE_bvar_bic.append(res[10]); all_SE_bvar_bma.append(res[11])
                all_SE_p0.append(res[12])
                all_p_hats_aic.append(res[13][0]); all_p_hats_aicc.append(res[13][1])
                all_p_hats_sic.append(res[13][2]); all_p_hats_hqc.append(res[13][3])

            if len(all_SE_aic) == 0:
                print(" [FAILED]"); continue
            print(f" [SUCCESS] ({len(all_SE_aic)}/{N_iterations})")

            MSE_p0_iter = np.sum(all_SE_p0, axis=(1, 2, 3)) + 1e-12
            
            def calc_met(se_list, p_list, true_val):
                arr = np.sum(se_list, axis=(1, 2, 3)) / MSE_p0_iter
                lg = (p_list.count(true_val) / len(p_list)) * 100 if p_list else np.nan
                ml = np.mean(p_list) if p_list else p_max 
                return round(np.exp(np.mean(np.log(arr))), 4), round(np.percentile(arr, 5), 4), round(np.percentile(arr, 95), 4), round(lg, 2), round(ml, 2)

            models = [
                ("AIC", calc_met(all_SE_aic, all_p_hats_aic, true_p)), 
                ("AICc", calc_met(all_SE_aicc, all_p_hats_aicc, true_p)), 
                ("SIC (BIC)", calc_met(all_SE_sic, all_p_hats_sic, true_p)), 
                ("HQC", calc_met(all_SE_hqc, all_p_hats_hqc, true_p)), 
                ("OLS BMA (BIC-W)", calc_met(all_SE_ols_bma, None, true_p)),
                ("BVAR-RW (Tight tau=0.05)", calc_met(all_SE_minn_rw_t, None, true_p)),
                ("BVAR-RW (Std tau=0.20)", calc_met(all_SE_minn_rw_s, None, true_p)),
                ("BVAR-RW (Loose tau=0.50)", calc_met(all_SE_minn_rw_l, None, true_p)),
                ("BVAR-WN (Std tau=0.20)", calc_met(all_SE_minn_wn_s, None, true_p)),
                ("BVAR-BIC (Minn.)", calc_met(all_SE_bvar_bic, all_p_hats_sic, true_p)),
                ("BVAR-BMA (Minn.)", calc_met(all_SE_bvar_bma, None, true_p))
            ]
            
            for m_name, m_data in models:
                master_results_list.append({
                    "True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": m_name,
                    "Lag Detection Rate (%)": m_data[3] if not np.isnan(m_data[3]) else "N/A",
                    "Mean Evaluated Lag": m_data[4], "Geom Mean MSE Ratio": m_data[0],
                    "5th Percentile": m_data[1], "95th Percentile": m_data[2]
                })

    if len(master_results_list) > 0:
        final_df = pd.DataFrame(master_results_list)
        total_time = time.time() - global_start_time
        
        # Calculate the Prior Showdown Verdict
        rw_mean = final_df[final_df['Estimator'] == 'BVAR-RW (Std tau=0.20)']['Geom Mean MSE Ratio'].mean()
        wn_mean = final_df[final_df['Estimator'] == 'BVAR-WN (Std tau=0.20)']['Geom Mean MSE Ratio'].mean()
        
        print("\n" + "="*90)
        print(" PRIOR SHOWDOWN: RANDOM WALK vs. WHITE NOISE (tau=0.20)")
        print("="*90)
        print(f" Random Walk (delta=1) Average Relative MSE: {rw_mean:.4f}")
        print(f" White Noise (delta=0) Average Relative MSE: {wn_mean:.4f}")
        if rw_mean < wn_mean:
            print(" VERDICT: The Random Walk prior performs better across these DGPs.")
        else:
            print(" VERDICT: The White Noise prior performs better across these DGPs.")
        print("="*90)

        print("\n" + "="*90)
        print(" MASTER ASYMPTOTIC SIMULATION RESULTS: 11 Estimators")
        print("="*90)
        print(final_df.to_string(index=False))
        
        print("\n" + "-"*90)
        print(f"Total Master Execution Time: {round(total_time, 2)} Seconds")
        print("-" * 90)

        results_dir = os.path.join(km_base, 'Results')
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, f"Master_Final_SVAR_Comparison_iters{N_iterations}_draws{mc_draws}.csv")
        final_df.to_csv(save_path, index=False)
        print(f"\n[SUCCESS] Saved to: {save_path}")
    else:
        print("\n[ERROR] No data generated across any DGP. Check file paths and errors.")

if __name__ == '__main__':
    main()