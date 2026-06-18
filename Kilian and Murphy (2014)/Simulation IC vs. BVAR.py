import os
import multiprocessing
import numpy as np
import pandas as pd
from numba import njit
import time
from joblib import Parallel, delayed
import math
from alexandria import NormalWishartBayesianVar

# ---------------- Reproducibility & Environment ----------------
SEED = 12345
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

# Grid for the Tradeoff Plot (30 points from tight to loose)
TAU_GRID_PLOT = np.linspace(0.01, 1.5, 30)

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
# 2. FAST VAR ESTIMATORS (OLS & ALEXANDRIA BVAR)
# -------------------------------------------------------------------
def lsvarcSA2_silent(y, p, X_exo_aligned):
    t, K = y.shape
    y = y.T
    Y = y[:, p-1:t]

    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i : t-i]])

    # Extract the exact effective estimation period from the synced matrix
    X2 = X_exo_aligned[p:t, :] 

    # Combine Constant, Seasonals, and Lags
    X = np.vstack([np.ones((t - p, 1)).T, X2.T, Y[:, :t-p]])
    Y2 = y[:, p:t]

    B = np.linalg.lstsq(X.T, Y2.T, rcond=None)[0].T
    U = Y2 - B @ X
    SIGMA = np.ascontiguousarray((U @ U.T) / (t - p - p * K - 12))
    
    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    return A, SIGMA

def estimate_alexandria_bvar(y, p, X_exo=None, tau_val=0.2, prior_mean=None, optimize_tau=False):
    """
    Robust Wrapper mapping directly to Alexandria v3.0 / v4.0 Official API Structure.
    """
    K = y.shape[1]
    
    prior_arr = np.asarray(prior_mean) if prior_mean is not None else None
    if prior_arr is not None:
        if prior_arr.ndim == 1:
            ar_coeffs = prior_arr
        elif prior_arr.ndim == 2 and prior_arr.shape in ((K, 1), (1, K)):
            ar_coeffs = prior_arr.ravel()
        else:
            ar_coeffs = prior_arr
    else:
        ar_coeffs = 0.0

    bvar_model = NormalWishartBayesianVar(
        endogenous=y, 
        exogenous=X_exo if X_exo is not None else [], 
        lags=p,
        constant=True, # We pass seasonals in X_exo, but Alexandria must generate the intercept 
        ar_coefficients=ar_coeffs,
        pi1=tau_val, # pi1 is the overall tightness hyperparameter (tau)
        hyperparameter_optimization=optimize_tau,
        verbose=False
    )
    
    bvar_model.estimate()
    
    B_bar = bvar_model.B_bar
    SIGMA_est = bvar_model.Sigma_estimates
    
    # Because BMA weighting uses np.exp (Natural Log), we convert marginal_likelihood to base-e here
    mdd_log10 = bvar_model.marginal_likelihood()
    mdd_ln = mdd_log10 * np.log(10)
    
    opt_tau = bvar_model.pi1 if hasattr(bvar_model, 'pi1') else tau_val

    if hasattr(bvar_model, 'W') and bvar_model.W is not None:
        W_arr = np.asarray(bvar_model.W)
        if W_arr.ndim == 1:
            W_diag = W_arr
        elif W_arr.ndim == 2:
            W_diag = np.diag(W_arr)
        else:
            W_diag = W_arr.ravel()
        lag_indices = np.argsort(W_diag)[:K*p]
        lag_indices = np.sort(lag_indices) 
        A_lags = np.ascontiguousarray(B_bar[lag_indices, :].T)
    else:
        A_lags = np.ascontiguousarray(B_bar[:K*p, :].T)

    return A_lags, np.ascontiguousarray(SIGMA_est), mdd_ln, opt_tau

# -------------------------------------------------------------------
# 3. PURE SIGN RESTRICTION CORE
# -------------------------------------------------------------------
@njit
def compute_structural_irf_numba(A, B_tilde, h_max, K, p):
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            A_j = np.ascontiguousarray(A[:, (j-1)*K : j*K]) 
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
        Q = np.ascontiguousarray(Q)
        
        B_tilde = P @ Q
        B_tilde = np.ascontiguousarray(B_tilde)

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

        dynamic_match = True
        for h in range(3): 
            if irf_cumulative[h, 0, 0] >= 0:
                dynamic_match = False
                break
            if irf[h, 1, 0] >= 0:
                dynamic_match = False
                break
            if irf[h, 2, 0] <= 0:
                dynamic_match = False
                break
                
        if not dynamic_match:
            continue

        valid_IRFs[accepted] = irf_cumulative
        accepted += 1

    return valid_IRFs, attempts, accepted

def get_median_target_model(valid_irfs, accepted_count):
    if accepted_count == 0: raise ValueError("Empty set.")
    valid_irfs_sliced = valid_irfs[:accepted_count]
    pointwise_median_irf = np.median(valid_irfs_sliced, axis=0)

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
    N_SE = 15

    nan_array = np.full(true_IRF_target.shape, np.nan)

    def fail(msg):
        return (iter_idx,) + (None,) * N_SE + ((None,)*7, (None, None), None, None, msg)

    try:
        ols_cache = {}
        bic_scores = {}
        
        K = true_A.shape[0]
        simulated_data = simulate_var_dgp_fast(true_A, true_V, true_B_tilde, true_p, T_real, 120, iteration_seed)

        best_aic, best_aicc, best_sic, best_hqc = float('inf'), float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_aicc, p_hat_sic, p_hat_hqc = 1, 1, 1, 1
        N_eff_total = T_real - p_max

        delta_wn = np.zeros(K)  
        delta_rw = np.ones(K)   

        t_total = simulated_data.shape[0]
        x_base = np.vstack([np.eye(11), np.zeros((1, 11))])
        n_years = int(t_total // 12)
        remainder = int(t_total % 12)
        X_exo = np.tile(x_base, (n_years, 1)) if n_years > 0 else np.empty((0, 11))
        if remainder > 0:
            last_base = np.hstack([np.eye(remainder), np.zeros((remainder, 11 - remainder))])
            X_exo = np.vstack([X_exo, last_base])

        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            x_slice = np.ascontiguousarray(X_exo[p_max - p_test : , :])
            
            A_temp, SIGMA_temp = lsvarcSA2_silent(y_slice, p_test, x_slice)
            if not np.all(np.isfinite(SIGMA_temp)): continue
            ols_cache[p_test] = (A_temp.copy(), SIGMA_temp.copy())

            sign_det, logdet_ols = np.linalg.slogdet(SIGMA_temp)
            if sign_det > 0:
                df_correction = N_eff_total - (p_test * K) - 12
                logdet_ml = logdet_ols + K * np.log(df_correction / N_eff_total)

                num_params = K**2 * p_test
                aic_val = logdet_ml + (2.0 / N_eff_total) * num_params
                sic_val = logdet_ml + (np.log(N_eff_total) / N_eff_total) * num_params
                hqc_val = logdet_ml + (2.0 * np.log(np.log(N_eff_total)) / N_eff_total) * num_params

                k_eq = K * p_test + 12
                if N_eff_total - k_eq - 1 > 0:
                    aicc_val = aic_val + (2.0 * k_eq * (k_eq + 1)) / (N_eff_total - k_eq - 1)
                else:
                    aicc_val = float('inf')

                bic_scores[p_test] = sic_val

                if aic_val < best_aic: best_aic, p_hat_aic = aic_val, p_test
                if aicc_val < best_aicc: best_aicc, p_hat_aicc = aicc_val, p_test
                if sic_val < best_sic: best_sic, p_hat_sic = sic_val, p_test
                if hqc_val < best_hqc: best_hqc, p_hat_hqc = hqc_val, p_test

        min_bic = min(bic_scores.values())
        raw_weights = {p: np.exp(-0.5 * (score - min_bic)) for p, score in bic_scores.items()}
        weight_sum = sum(raw_weights.values())
        kept_lags_ols = [p for p, w in raw_weights.items() if (w / weight_sum) > 0.01]
        final_bic_weights = {p: raw_weights[p] / sum(raw_weights[p] for p in kept_lags_ols) for p in kept_lags_ols}
        exp_p_hybrid = sum(p * w for p, w in final_bic_weights.items())

        computed_ols_SEs = {}
        def get_ols_se_for_p(p_target):
            if p_target in computed_ols_SEs: return computed_ols_SEs[p_target]
            A_est, SIGMA_est = ols_cache[p_target]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8
                try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                except: return nan_array

            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_target, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + p_target)
            if acc == 0:
                se = nan_array
            else:
                se = (get_median_target_model(v_irfs, acc) - true_IRF_target)**2
            computed_ols_SEs[p_target] = se
            return se

        SE_aic = get_ols_se_for_p(p_hat_aic)
        SE_aicc = get_ols_se_for_p(p_hat_aicc)
        SE_sic = get_ols_se_for_p(p_hat_sic)
        SE_hqc = get_ols_se_for_p(p_hat_hqc)
        SE_p0 = get_ols_se_for_p(true_p)

        ols_bma_pool, ols_bma_accepted = [], 0
        for p_bma in kept_lags_ols:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError: SIGMA_est += np.eye(K) * 1e-8; P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 5000 + p_bma)
            if acc > 0: ols_bma_pool.append(v_irfs[:acc]); ols_bma_accepted += acc

        if ols_bma_accepted == 0:
            SE_ols_bma = nan_array
        else:
            SE_ols_bma = (get_median_target_model(np.vstack(ols_bma_pool), ols_bma_accepted) - true_IRF_target)**2

        def eval_minn_alex(tau_val, delta_array, seed_offset, p_tgt=p_max):
            y_slice_fixed = np.ascontiguousarray(simulated_data[p_max - p_tgt : , :])
            x_slice_fixed = np.ascontiguousarray(X_exo[p_max - p_tgt : , :])

            A_m, SIGMA_m, _, _ = estimate_alexandria_bvar(
                y_slice_fixed, p_tgt, X_exo=x_slice_fixed, tau_val=tau_val, prior_mean=delta_array, optimize_tau=False
            )
            try: P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            except np.linalg.LinAlgError: SIGMA_m += np.eye(K) * 1e-8; P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            v_m, att_m, acc_m = fast_draw_core(A_m, P_m, signs, p_tgt, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + seed_offset)
            if acc_m == 0: return nan_array
            return (get_median_target_model(v_m, acc_m) - true_IRF_target)**2

        SE_minn_wn_tight = eval_minn_alex(0.05, delta_wn, 999)
        SE_minn_wn_std   = eval_minn_alex(0.20, delta_wn, 1001)
        SE_minn_wn_loose = eval_minn_alex(0.50, delta_wn, 1002)
        SE_minn_rw_tight = eval_minn_alex(0.05, delta_rw, 998)
        SE_minn_rw_std   = eval_minn_alex(0.20, delta_rw, 1000)
        SE_minn_rw_loose = eval_minn_alex(0.50, delta_rw, 1003)
        
        SE_bvar_bic      = eval_minn_alex(0.20, delta_wn, 3000 + p_hat_sic, p_tgt=p_hat_sic)

        bvar_bma_pool, bvar_bma_acc = [], 0
        for p_bma in kept_lags_ols:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            
            y_slice_bma = np.ascontiguousarray(simulated_data[p_max - p_bma : , :])
            x_slice_bma = np.ascontiguousarray(X_exo[p_max - p_bma : , :])
            
            A_c, SIGMA_c, _, _ = estimate_alexandria_bvar(
                y_slice_bma, p_bma, X_exo=x_slice_bma, tau_val=0.20, prior_mean=delta_wn, optimize_tau=False
            )
            try: P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            except np.linalg.LinAlgError: SIGMA_c += np.eye(K) * 1e-8; P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            v_c, att_c, acc_c = fast_draw_core(A_c, P_c, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 4000 + p_bma)
            if acc_c > 0: bvar_bma_pool.append(v_c[:acc_c]); bvar_bma_acc += acc_c

        if bvar_bma_acc == 0:
            SE_bvar_bma = nan_array
        else:
            SE_bvar_bma = (get_median_target_model(np.vstack(bvar_bma_pool), bvar_bma_acc) - true_IRF_target)**2

        opt_taus, mdd_scores = {}, {}
        bvar_sota_cache = {}
        plausible_lags = list(set(kept_lags_ols + [true_p, p_hat_sic]))
        
        for p_test in plausible_lags:
            try:
                y_slice_sota = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
                x_slice_sota = np.ascontiguousarray(X_exo[p_max - p_test : , :])

                A_opt, SIGMA_opt, log_mdd, tau_opt = estimate_alexandria_bvar(
                    y_slice_sota, p_test, X_exo=x_slice_sota, prior_mean=delta_wn, optimize_tau=True
                )
                opt_taus[p_test] = tau_opt
                mdd_scores[p_test] = log_mdd
                bvar_sota_cache[p_test] = (A_opt.copy(), SIGMA_opt.copy())
            except Exception as e:
                opt_taus[p_test], mdd_scores[p_test] = 0.20, -float('inf')

        p_sota_bic = max(mdd_scores, key=mdd_scores.get)
        tau_sota_bic = opt_taus[p_sota_bic]
        
        A_sota, SIGMA_sota = bvar_sota_cache[p_sota_bic]
        try: P_sota = np.ascontiguousarray(np.linalg.cholesky(SIGMA_sota))
        except np.linalg.LinAlgError: SIGMA_sota += np.eye(K) * 1e-8; P_sota = np.ascontiguousarray(np.linalg.cholesky(SIGMA_sota))
        v_sota, att_sota, acc_sota = fast_draw_core(A_sota, P_sota, signs, p_sota_bic, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + 8000)
        
        if acc_sota == 0:
            SE_sota_bic = nan_array
        else:
            SE_sota_bic = (get_median_target_model(v_sota, acc_sota) - true_IRF_target)**2

        max_mdd = mdd_scores[p_sota_bic]
        raw_sota_weights = {p: np.exp(score - max_mdd) for p, score in mdd_scores.items()}
        
        posterior_probs = {
            p: w / sum(raw_sota_weights.values())
            for p, w in raw_sota_weights.items()
        }
        
        kept_sota = [p for p, prob in posterior_probs.items() if prob > 0.001]
        sota_weights = {p: posterior_probs[p] / sum(posterior_probs[k] for k in kept_sota) for p in kept_sota}
        exp_p_sota = sum(p * w for p, w in sota_weights.items())

        sota_bma_pool, sota_bma_acc = [], 0
        for p_bma in kept_sota:
            td = int(round(mc_draws * sota_weights[p_bma]))
            if td == 0: continue
            
            A_c, SIGMA_c = bvar_sota_cache[p_bma]
            try: P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            except np.linalg.LinAlgError: SIGMA_c += np.eye(K) * 1e-8; P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
            v_c, att_c, acc_c = fast_draw_core(A_c, P_c, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 9000 + p_bma)
            if acc_c > 0: sota_bma_pool.append(v_c[:acc_c]); sota_bma_acc += acc_c

        if sota_bma_acc == 0:
            SE_sota_bma = nan_array
        else:
            SE_sota_bma = (get_median_target_model(np.vstack(sota_bma_pool), sota_bma_acc) - true_IRF_target)**2

        tradeoff_mses = None
        mdd_surface = None

        if true_p == 4 and (T_real == 96 or T_real == 480):
            tradeoff_mses = np.zeros(len(TAU_GRID_PLOT))
            
            y_slice_plot = np.ascontiguousarray(simulated_data[p_max - p_max : , :])
            x_slice_plot = np.ascontiguousarray(X_exo[p_max - p_max : , :])

            for idx, t_val in enumerate(TAU_GRID_PLOT):
                try:
                    A_m, SIGMA_m, _, _ = estimate_alexandria_bvar(
                        y_slice_plot, p_max, X_exo=x_slice_plot, tau_val=t_val, prior_mean=delta_wn, optimize_tau=False
                    )
                    try: P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
                    except np.linalg.LinAlgError: SIGMA_m += np.eye(K) * 1e-8; P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
                    v_m, att_m, acc_m = fast_draw_core(A_m, P_m, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + 10000 + idx)
                    if acc_m > 0:
                        mse = (get_median_target_model(v_m, acc_m) - true_IRF_target)**2
                        tradeoff_mses[idx] = np.sum(mse)
                    else: tradeoff_mses[idx] = np.nan
                except: tradeoff_mses[idx] = np.nan
            
            if T_real == 480 and iter_idx == 0:
                mdd_surface = np.zeros(len(TAU_GRID_PLOT))
                for idx, t_val in enumerate(TAU_GRID_PLOT):
                    _, _, log_mdd_surf, _ = estimate_alexandria_bvar(
                        y_slice_plot, p_max, X_exo=x_slice_plot, tau_val=t_val, prior_mean=delta_wn, optimize_tau=False
                    )
                    mdd_surface[idx] = log_mdd_surf

        return (iter_idx, SE_aic, SE_aicc, SE_sic, SE_hqc, SE_ols_bma,
                SE_minn_wn_tight, SE_minn_wn_std, SE_minn_wn_loose, SE_minn_rw_std, 
                SE_bvar_bic, SE_bvar_bma, SE_sota_bic, SE_sota_bma,
                tau_sota_bic, SE_p0,
                (p_hat_aic, p_hat_aicc, p_hat_sic, p_hat_hqc, p_sota_bic, exp_p_hybrid, exp_p_sota),
                (final_bic_weights, sota_weights),
                tradeoff_mses, mdd_surface,
                "Success")

    except Exception as e:
        return fail(f"Error: {str(e)}")

# -------------------------------------------------------------------
# 7. PARALLEL ORCHESTRATION WITH NESTED LOOP
# -------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    km_base = os.path.join(script_dir, 'Kilian and Murphy (2014)') if os.path.basename(script_dir) != 'Kilian and Murphy (2014)' else script_dir

    Q_avg, h_max = 72.3, 24
    N_iterations, mc_draws, max_loops = 100, 50, 5000

    sign_matrix = np.ascontiguousarray(np.array([
        [-1,  1,  1, np.nan], [-1,  1, -1, np.nan],
        [ 1,  1,  1, np.nan], [np.nan, np.nan, 1, np.nan]
    ], dtype=np.float64))

    dgp_lag_orders, sample_sizes = [4, 6, 8, 10], [96, 144, 240, 480]
    master_results_list, bma_weights_list = [], []
    raw_taus_list, tradeoff_list, mdd_list = [], [], []
    iteration_mses_list = []
    
    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()

    print("\n" + "="*90)
    print(f" STARTING MASTER ASYMPTOTIC MONTE CARLO (State-Of-The-Art with Graph Exports)")
    print("="*90)

    for current_p0 in dgp_lag_orders:
        dgp_path = os.path.join(km_base, 'DGP files', f'true_dgp_parameters_{current_p0}_lags.npz')
        if not os.path.exists(dgp_path): continue
        dgp = np.load(dgp_path)
        true_A, true_V, true_B_tilde = np.ascontiguousarray(dgp['A_true']), np.ascontiguousarray(dgp['V_true']), np.ascontiguousarray(dgp['B_tilde_true'])
        true_IRF, true_p = np.ascontiguousarray(dgp['True_IRF']), int(dgp['p_true'])
        p_max = max(12, true_p)

        for current_T in sample_sizes:
            print(f"    Running Simulation for T = {current_T} ...", end="", flush=True)

            # --- 1. INITIALIZE ALL METRIC LISTS HERE (Before any parallel execution) ---
            all_SE_aic, all_SE_aicc, all_SE_sic, all_SE_hqc = [], [], [], []
            all_SE_ols_bma = []
            all_SE_minn_wn_t, all_SE_minn_wn_s, all_SE_minn_wn_l = [], [], []
            all_SE_minn_rw_s = []
            all_SE_bvar_bic, all_SE_bvar_bma = [], []
            all_SE_sota_bic, all_SE_sota_bma = [], []
            
            all_tau_sota, all_SE_p0 = [], []
            all_p_hats_aic, all_p_hats_aicc, all_p_hats_sic, all_p_hats_hqc, all_p_hats_sota = [], [], [], [], []
            all_exp_p_hybrid, all_exp_p_sota = [], []
            all_weights_hybrid, all_weights_sota = [], []
            all_tradeoffs_t = []

            # --- 2. EXECUTE PARALLEL TASKS ---
            tasks = [(i, SEED + i + (current_p0 * 10000) + (current_T * 100000), true_A, true_V, true_B_tilde, true_p, true_IRF, sign_matrix, Q_avg, h_max, mc_draws, max_loops, current_T, p_max) for i in range(N_iterations)]
            results = Parallel(n_jobs=n_cores, backend='loky')(delayed(single_monte_carlo_iteration)(task) for task in tasks)

            # --- 3. PROCESS RESULTS (Safely contained inside the loop) ---
            for res in results:
                if res[-1] != "Success": 
                    print(f"\n   -> [INTERNAL ERROR]: {res[-1]}")
                    continue
                
                # Append mappings perfectly match the return tuple of single_monte_carlo_iteration
                all_SE_aic.append(res[1])
                all_SE_aicc.append(res[2])
                all_SE_sic.append(res[3])
                all_SE_hqc.append(res[4])
                
                all_SE_ols_bma.append(res[5])
                
                all_SE_minn_wn_t.append(res[6])
                all_SE_minn_wn_s.append(res[7])
                all_SE_minn_wn_l.append(res[8])
                all_SE_minn_rw_s.append(res[9])
                
                all_SE_bvar_bic.append(res[10])
                all_SE_bvar_bma.append(res[11])
                all_SE_sota_bic.append(res[12])
                all_SE_sota_bma.append(res[13])
                
                all_tau_sota.append(res[14]) 
                all_SE_p0.append(res[15])
                
                p_tuple, weight_tuple = res[16], res[17]
                all_p_hats_aic.append(p_tuple[0])
                all_p_hats_aicc.append(p_tuple[1])
                all_p_hats_sic.append(p_tuple[2])
                all_p_hats_hqc.append(p_tuple[3])
                all_p_hats_sota.append(p_tuple[4])
                
                all_exp_p_hybrid.append(p_tuple[5])
                all_exp_p_sota.append(p_tuple[6])
                
                all_weights_hybrid.append(weight_tuple[0])
                all_weights_sota.append(weight_tuple[1])

                se_bic_iter = np.sum(res[10]) if not (np.isscalar(res[10]) and np.isnan(res[10])) else np.nan
                se_bma_iter = np.sum(res[13]) if not (np.isscalar(res[13]) and np.isnan(res[13])) else np.nan
                se_p0_iter = np.sum(res[15]) + 1e-12 if not (np.isscalar(res[15]) and np.isnan(res[15])) else np.nan
                
                iteration_mses_list.append({
                    'p0': current_p0, 'T': current_T, 'Iter': res[0],
                    'BIC_Rel_MSE': se_bic_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan,
                    'BMA_Rel_MSE': se_bma_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan
                })

                raw_taus_list.append({'p0': current_p0, 'T': current_T, 'iter': res[0], 'Opt_Tau': res[14]})
                if res[18] is not None: all_tradeoffs_t.append(res[18])
                if res[19] is not None:
                    for i, t_val in enumerate(TAU_GRID_PLOT):
                        mdd_list.append({'Tau': t_val, 'MDD': res[19][i]})

            if len(all_SE_aic) == 0:
                print(" [FAILED]"); continue
            print(f" [SUCCESS] ({len(all_SE_aic)}/{N_iterations})")

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                all_SE_p0_sums = []
                for se in all_SE_p0:
                    if np.isscalar(se) and np.isnan(se):
                        all_SE_p0_sums.append(np.nan)
                    elif np.isnan(se).all():
                        all_SE_p0_sums.append(np.nan)
                    else:
                        all_SE_p0_sums.append(np.sum(se))
                
                all_SE_p0_sums = np.array(all_SE_p0_sums)
                mean_p0_mse = np.nanmean(all_SE_p0_sums)
                
                MSE_p0_iter = np.copy(all_SE_p0_sums)
                MSE_p0_iter[np.isnan(MSE_p0_iter)] = mean_p0_mse
                MSE_p0_iter += 1e-12

            if len(all_tradeoffs_t) > 0:
                tradeoff_arr = np.array(all_tradeoffs_t) 
                ratio_matrix = tradeoff_arr / MSE_p0_iter[:, None]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    geom_mean_curve = np.exp(np.nanmean(np.log(ratio_matrix), axis=0))
                for i, t_val in enumerate(TAU_GRID_PLOT):
                    tradeoff_list.append({'T': current_T, 'Tau': t_val, 'Rel_MSE': geom_mean_curve[i]})

            avg_hybrid_dist = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_hybrid]) for p in range(1, p_max + 1)}
            avg_sota_dist = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_sota]) for p in range(1, p_max + 1)}
            
            h_row = {"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "Hybrid-BMA (OLS W)"}; h_row.update(avg_hybrid_dist)
            s_row = {"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "SOTA-BMA (MDD W)"}; s_row.update(avg_sota_dist)
            bma_weights_list.extend([h_row, s_row])

            def calc_met(se_list, p_list, true_val, tau_list=None):
                mse_sums = []
                for se in se_list:
                    if np.isscalar(se) and np.isnan(se): mse_sums.append(np.nan)
                    elif np.isnan(se).all(): mse_sums.append(np.nan)
                    else: mse_sums.append(np.sum(se))
                mse_sums = np.array(mse_sums)
                
                arr = mse_sums / MSE_p0_iter
                
                valid_mask = ~np.isnan(arr) & (arr > 0)
                valid_arr = arr[valid_mask]
                
                fail_pct = (1.0 - (np.sum(valid_mask) / len(arr))) * 100.0 if len(arr) > 0 else 100.0
                
                if len(valid_arr) > 0:
                    geom_mean = np.exp(np.mean(np.log(valid_arr)))
                    p5 = np.percentile(valid_arr, 5)
                    p95 = np.percentile(valid_arr, 95)
                else:
                    geom_mean, p5, p95 = np.nan, np.nan, np.nan
                    
                lg = (p_list.count(true_val) / len(p_list)) * 100 if isinstance(p_list, list) and not isinstance(p_list[0], float) else np.nan
                ml = np.mean([p for p in p_list if not np.isnan(p)]) if p_list else p_max
                avg_tau = round(np.mean([t for t in tau_list if not np.isnan(t)]), 4) if tau_list else "N/A"
                
                return geom_mean, p5, p95, round(lg, 2), round(ml, 2), avg_tau, round(fail_pct, 1)

            models = [
                ("AIC",                          calc_met(all_SE_aic,         all_p_hats_aic,   true_p)),
                ("AICc",                         calc_met(all_SE_aicc,        all_p_hats_aicc,  true_p)),
                ("SIC (BIC)",                    calc_met(all_SE_sic,         all_p_hats_sic,   true_p)),
                ("HQC",                          calc_met(all_SE_hqc,         all_p_hats_hqc,   true_p)),
                ("OLS BMA (BIC-W)",              calc_met(all_SE_ols_bma,     all_exp_p_hybrid, true_p)),
                ("BVAR-WN (Tight tau=0.05)",     calc_met(all_SE_minn_wn_t,   None,             true_p)),
                ("BVAR-WN (Std tau=0.20)",       calc_met(all_SE_minn_wn_s,   None,             true_p)),
                ("BVAR-WN (Loose tau=0.50)",     calc_met(all_SE_minn_wn_l,   None,             true_p)),
                ("BVAR-RW (Std tau=0.20)",       calc_met(all_SE_minn_rw_s,   None,             true_p)),
                ("Hybrid-BVAR (OLS p, Fix tau)", calc_met(all_SE_bvar_bic,    all_p_hats_sic,   true_p)),
                ("Hybrid-BMA (OLS W, Fix tau)",  calc_met(all_SE_bvar_bma,    all_exp_p_hybrid, true_p)),
                ("SOTA-BVAR (MDD p, Opt tau)",   calc_met(all_SE_sota_bic,    all_p_hats_sota,  true_p, all_tau_sota)),
                ("SOTA-BMA (MDD W, Opt tau)",    calc_met(all_SE_sota_bma,    all_exp_p_sota,   true_p, all_tau_sota)),
            ]

            for m_name, m_data in models:
                master_results_list.append({
                    "True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": m_name,
                    "Lag Detection Rate (%)": m_data[3] if not np.isnan(m_data[3]) else "N/A",
                    "Mean Evaluated Lag": m_data[4], 
                    "Avg Opt Tau": m_data[5],
                    "Fail Rate (%)": m_data[6],
                    "Geom Mean MSE Ratio": m_data[0], 
                    "5th Percentile": m_data[1], 
                    "95th Percentile": m_data[2]
                })

    if len(master_results_list) > 0:
        results_dir = os.path.join(km_base, 'Results')
        os.makedirs(results_dir, exist_ok=True)
        
        df_master = pd.DataFrame(master_results_list)
        df_master.to_csv(os.path.join(results_dir, f"Master_Final_SVAR_Comparison_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(bma_weights_list).to_csv(os.path.join(results_dir, f"Master_BMA_Weights_iters{N_iterations}.csv"), index=False)
        pd.DataFrame(raw_taus_list).to_csv(os.path.join(results_dir, "Master_Raw_Opt_Tau.csv"), index=False)
        pd.DataFrame(tradeoff_list).to_csv(os.path.join(results_dir, "Master_Tradeoff_Curve.csv"), index=False)
        pd.DataFrame(mdd_list).to_csv(os.path.join(results_dir, "Master_MDD_Surface.csv"), index=False)
        pd.DataFrame(iteration_mses_list).to_csv(os.path.join(results_dir, "Master_Iteration_MSEs.csv"), index=False)
        
        print("\n" + "="*90)
        print(" RESULTS PREVIEW:")
        print("="*90)
        print(df_master.to_string(index=False))

        print("\n" + "="*90)
        print(" SUCCESS: Simulation complete and 6 CSVs generated in 'Results'!")
        print("="*90)
    else:
        print("\n[ERROR] No data generated. Check file paths.")

if __name__ == '__main__':
    main()