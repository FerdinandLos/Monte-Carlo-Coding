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

# Discrete tau grid used for BVAR-WN fixed-tau models, MDD tau selection,
# AND the Tradeoff/Surface CSV exports.
TAU_DISCRETE = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
WN_SEED_OFFSETS = [999, 1001, 1008, 1002, 1009, 1010, 1011]

# -------------------------------------------------------------------
# 1. NUMBA-OPTIMIZED HYBRID DATA GENERATING PROCESS (DGP)
# -------------------------------------------------------------------
@njit
def simulate_var_dgp_hybrid(A, V, B_tilde, p, T_target, seed_val, empirical_data):
    """
    Hybrid DGP: 
    Initializes using a random contiguous block of empirical data.
    Iterates forward using purely parametric structural shocks.
    """
    np.random.seed(seed_val)
    K = A.shape[0]
    
    # No burn-in required because empirical data places us in a stationary state
    T_total = T_target 
    y_sim = np.zeros((K, T_total))

    # --- 1. EMPIRICAL WARM START ---
    T_emp = empirical_data.shape[1]
    max_start_idx = T_emp - p
    start_idx = np.random.randint(0, max_start_idx + 1)
    
    for t in range(p):
        y_sim[:, t] = empirical_data[:, start_idx + t]

    # --- 2. PARAMETRIC STRUCTURAL SHOCKS ---
    structural_shocks = np.random.randn(K, T_total)
    reduced_residuals = B_tilde @ structural_shocks

    # --- 3. AUTOREGRESSIVE LOOP ---
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

    return np.ascontiguousarray(y_sim.T)

# -------------------------------------------------------------------
# 2. FAST VAR ESTIMATORS (OLS & ALEXANDRIA BVAR)
# -------------------------------------------------------------------
def lsvarcSA2_silent(y, p, X_exo_aligned):
    t, K = y.shape
    y = y.T
    Y = y[:, p-1:t]

    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i : t-i]])

    X2 = X_exo_aligned[p:t, :]

    X = np.vstack([np.ones((t - p, 1)).T, X2.T, Y[:, :t-p]])
    Y2 = y[:, p:t]

    B = np.linalg.lstsq(X.T, Y2.T, rcond=None)[0].T
    U = Y2 - B @ X
    SIGMA = np.ascontiguousarray((U @ U.T) / (t - p - p * K - 12))

    A = np.ascontiguousarray(B[:, 12 : K*p + 12])
    return A, SIGMA

def estimate_alexandria_bvar(y, p, X_exo=None, tau_val=0.2, prior_mean=None, optimize_tau=False):
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
        constant=True,
        ar_coefficients=ar_coeffs,
        pi1=tau_val,
        hyperparameter_optimization=optimize_tau,
        verbose=False
    )

    bvar_model.estimate()

    B_bar = bvar_model.B_bar
    SIGMA_est = bvar_model.Sigma_estimates

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
    # Added empirical_data parameter extraction
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, Q_avg, h_max, mc_draws, max_loops, T_real, p_max, empirical_data = args
    N_SE = 26

    nan_array = np.full(true_IRF_target.shape, np.nan)

    def fail(msg):
        return (iter_idx,) + (None,) * N_SE + (None, None, (None,)*7, None, None, None, None, None, None, msg)

    try:
        ols_cache = {}
        bic_scores = {}

        K = true_A.shape[0]
        
        # Updated to call the hybrid DGP (burn_in removed)
        simulated_data = simulate_var_dgp_hybrid(true_A, true_V, true_B_tilde, true_p, T_real, iteration_seed, empirical_data)

        best_aic, best_sic, best_hqc = float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_sic, p_hat_hqc = 1, 1, 1
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

        bvar_mdd_grid = {}
        bvar_cache = {}

        # ---------------- GRID EVALUATION LOOP ----------------
        for p_test in range(1, p_max + 1):
            y_slice = np.ascontiguousarray(simulated_data[p_max - p_test : , :])
            x_slice = np.ascontiguousarray(X_exo[p_max - p_test : , :])

            # 1. OLS Evaluation
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

                bic_scores[p_test] = sic_val

                if aic_val < best_aic: best_aic, p_hat_aic = aic_val, p_test
                if sic_val < best_sic: best_sic, p_hat_sic = sic_val, p_test
                if hqc_val < best_hqc: best_hqc, p_hat_hqc = hqc_val, p_test

            # 2. BVAR Evaluation for Joint MDD Grid
            for tau_test, seed_off in zip(TAU_DISCRETE, WN_SEED_OFFSETS):
                A_c, SIGMA_c, mdd_ln, _ = estimate_alexandria_bvar(
                    y_slice, p_test, X_exo=x_slice, tau_val=tau_test, prior_mean=delta_wn, optimize_tau=False
                )
                bvar_mdd_grid[(p_test, tau_test)] = mdd_ln
                bvar_cache[(p_test, tau_test)] = (A_c.copy(), SIGMA_c.copy())

        # ---------------- BMA WEIGHT CALCULATIONS ----------------
        
        # 1a. OLS BIC Weights (Uniform Prior)
        min_bic = min(bic_scores.values())
        raw_weights_ols = {p: np.exp(-0.5 * N_eff_total * (score - min_bic)) for p, score in bic_scores.items()}
        weight_sum_ols = sum(raw_weights_ols.values())
        kept_lags_ols = [p for p, w in raw_weights_ols.items() if (w / weight_sum_ols) > 0.01]
        final_bic_weights = {p: raw_weights_ols[p] / sum(raw_weights_ols[p] for p in kept_lags_ols) for p in kept_lags_ols}
        exp_p_hybrid = sum(p * w for p, w in final_bic_weights.items())

        # 1b. OLS Geometric Weights (theta = 0.5)
        theta = 0.5
        raw_geom_ols = {p: np.exp(-0.5 * N_eff_total * (score - min_bic)) * (theta**p) for p, score in bic_scores.items()}
        sum_geom_ols = sum(raw_geom_ols.values())
        kept_geom_ols = {p: w for p, w in raw_geom_ols.items() if (w / sum_geom_ols) > 0.01}
        sum_kept_geom_ols = sum(kept_geom_ols.values())
        final_geom_ols_weights = {p: w / sum_kept_geom_ols for p, w in kept_geom_ols.items()}
        
        exp_p_geom_ols = sum(p * w for p, w in final_geom_ols_weights.items())
        geom_ols_p_dist = {p: final_geom_ols_weights.get(p, 0.0) for p in range(1, p_max+1)}

        # 2. Joint Grid BVAR Weights (MDD Uniform Prior)
        max_mdd = max(bvar_mdd_grid.values())
        raw_j = {k: np.exp(v - max_mdd) for k, v in bvar_mdd_grid.items()}
        sum_j = sum(raw_j.values())
        kept_j = {k: w for k, w in raw_j.items() if (w / sum_j) > 0.01}
        sum_kept_j = sum(kept_j.values())
        final_joint_weights = {k: w / sum_kept_j for k, w in kept_j.items()}
        
        exp_p_joint = sum(k[0] * w for k, w in final_joint_weights.items())
        joint_p_dist = {p: sum(w for (p_k, tau_k), w in final_joint_weights.items() if p_k == p) for p in range(1, p_max+1)}

        # 3. Geometric BVAR Weights (MDD with theta=0.5 prior)
        raw_g = {k: np.exp(v - max_mdd) * (theta**k[0]) for k, v in bvar_mdd_grid.items()}
        sum_g = sum(raw_g.values())
        kept_g = {k: w for k, w in raw_g.items() if (w / sum_g) > 0.01}
        sum_kept_g = sum(kept_g.values())
        final_geom_weights = {k: w / sum_kept_g for k, w in kept_g.items()}
        
        exp_p_geom = sum(k[0] * w for k, w in final_geom_weights.items())
        geom_p_dist = {p: sum(w for (p_k, tau_k), w in final_geom_weights.items() if p_k == p) for p in range(1, p_max+1)}

        # ---------------- OLS & INFORMATION CRITERION EVALUATIONS ----------------
        computed_ols_SEs = {}
        def get_ols_se_for_p(p_target):
            if p_target in computed_ols_SEs: return computed_ols_SEs[p_target]
            if p_target not in ols_cache: return nan_array
            A_est, SIGMA_est = ols_cache[p_target]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError:
                SIGMA_est += np.eye(K) * 1e-8
                try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
                except: return nan_array

            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_target, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + p_target)
            se = (get_median_target_model(v_irfs, acc) - true_IRF_target)**2 if acc > 0 else nan_array
            computed_ols_SEs[p_target] = se
            return se

        SE_aic = get_ols_se_for_p(p_hat_aic)
        SE_sic = get_ols_se_for_p(p_hat_sic)
        SE_hqc = get_ols_se_for_p(p_hat_hqc)
        SE_p0  = get_ols_se_for_p(true_p)

        # OLS BMA (Uniform) Evaluation
        ols_bma_pool, ols_bma_accepted = [], 0
        for p_bma in kept_lags_ols:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError: SIGMA_est += np.eye(K) * 1e-8; P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 5000 + p_bma)
            if acc > 0: ols_bma_pool.append(v_irfs[:acc]); ols_bma_accepted += acc

        SE_ols_bma = (get_median_target_model(np.vstack(ols_bma_pool), ols_bma_accepted) - true_IRF_target)**2 if ols_bma_accepted > 0 else nan_array

        # OLS BMA (Geometric) Evaluation
        ols_geom_pool, ols_geom_accepted = [], 0
        for p_bma in kept_geom_ols:
            td = int(round(mc_draws * final_geom_ols_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            except np.linalg.LinAlgError: SIGMA_est += np.eye(K) * 1e-8; P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
            
            v_irfs, att, acc = fast_draw_core(A_est, P_est, signs, p_bma, K, Q_avg, h_max, td, max_loops, iteration_seed + 9000 + p_bma)
            if acc > 0: ols_geom_pool.append(v_irfs[:acc]); ols_geom_accepted += acc

        SE_ols_geom = (get_median_target_model(np.vstack(ols_geom_pool), ols_geom_accepted) - true_IRF_target)**2 if ols_geom_accepted > 0 else nan_array

        # ---------------- FIXED P_MAX BVAR EVALUATIONS (Cached) ----------------
        bvar_wn_SEs  = {}
        bvar_wn_MDDs = {}
        for tau_wn, seed_off in zip(TAU_DISCRETE, WN_SEED_OFFSETS):
            A_wn, SIGMA_wn = bvar_cache[(p_max, tau_wn)]
            mdd_wn = bvar_mdd_grid[(p_max, tau_wn)]
            
            try: P_wn = np.ascontiguousarray(np.linalg.cholesky(SIGMA_wn))
            except np.linalg.LinAlgError: SIGMA_wn += np.eye(K) * 1e-8; P_wn = np.ascontiguousarray(np.linalg.cholesky(SIGMA_wn))
            v_wn, _, acc_wn = fast_draw_core(A_wn, P_wn, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + seed_off)
            
            bvar_wn_MDDs[tau_wn] = mdd_wn
            bvar_wn_SEs[tau_wn]  = (get_median_target_model(v_wn, acc_wn) - true_IRF_target)**2 if acc_wn > 0 else nan_array

        SE_minn_wn_005 = bvar_wn_SEs[0.05]
        SE_minn_wn_020 = bvar_wn_SEs[0.20]
        SE_minn_wn_035 = bvar_wn_SEs[0.35]
        SE_minn_wn_050 = bvar_wn_SEs[0.50]
        SE_minn_wn_065 = bvar_wn_SEs[0.65]
        SE_minn_wn_080 = bvar_wn_SEs[0.80]
        SE_minn_wn_095 = bvar_wn_SEs[0.95]

        best_mdd_tau   = max(bvar_wn_MDDs, key=bvar_wn_MDDs.get)
        SE_minn_wn_mdd = bvar_wn_SEs[best_mdd_tau]

        # ---------------- FIXED P_TRUE BVAR EVALUATIONS (Cached) ----------------
        bvar_p0_SEs  = {}
        bvar_p0_MDDs = {}
        for tau_wn, seed_off in zip(TAU_DISCRETE, WN_SEED_OFFSETS):
            # Extract from cache generated in the grid search loop
            A_p0, SIGMA_p0 = bvar_cache[(true_p, tau_wn)]
            mdd_p0 = bvar_mdd_grid[(true_p, tau_wn)]
            
            try: P_p0 = np.ascontiguousarray(np.linalg.cholesky(SIGMA_p0))
            except np.linalg.LinAlgError: SIGMA_p0 += np.eye(K) * 1e-8; P_p0 = np.ascontiguousarray(np.linalg.cholesky(SIGMA_p0))
            
            # Offset the seed strictly to avoid exact replication of draws from other evaluations
            v_p0, _, acc_p0 = fast_draw_core(A_p0, P_p0, signs, true_p, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + seed_off + 2000)
            
            bvar_p0_MDDs[tau_wn] = mdd_p0
            bvar_p0_SEs[tau_wn]  = (get_median_target_model(v_p0, acc_p0) - true_IRF_target)**2 if acc_p0 > 0 else nan_array

        SE_minn_wn_005_p0 = bvar_p0_SEs[0.05]
        SE_minn_wn_020_p0 = bvar_p0_SEs[0.20]
        SE_minn_wn_035_p0 = bvar_p0_SEs[0.35]
        SE_minn_wn_050_p0 = bvar_p0_SEs[0.50]
        SE_minn_wn_065_p0 = bvar_p0_SEs[0.65]
        SE_minn_wn_080_p0 = bvar_p0_SEs[0.80]
        SE_minn_wn_095_p0 = bvar_p0_SEs[0.95]

        best_mdd_tau_p0   = max(bvar_p0_MDDs, key=bvar_p0_MDDs.get)
        SE_minn_wn_mdd_p0 = bvar_p0_SEs[best_mdd_tau_p0]

        # BVAR RW tau=0.20 and tau=0.50 (Evaluated Fresh)
        def eval_minn_rw(tau_val, seed_offset):
            y_s = np.ascontiguousarray(simulated_data)
            x_s = np.ascontiguousarray(X_exo)
            A_m, SIGMA_m, _, _ = estimate_alexandria_bvar(y_s, p_max, X_exo=x_s, tau_val=tau_val, prior_mean=delta_rw, optimize_tau=False)
            try: P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            except np.linalg.LinAlgError: SIGMA_m += np.eye(K) * 1e-8; P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
            v_m, _, acc_m = fast_draw_core(A_m, P_m, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + seed_offset)
            return (get_median_target_model(v_m, acc_m) - true_IRF_target)**2 if acc_m > 0 else nan_array

        SE_minn_rw_020 = eval_minn_rw(0.20, 1000)
        SE_minn_rw_050 = eval_minn_rw(0.50, 1003)

        # ---------------- BVAR BMA EVALUATIONS (Cached) ----------------
        def pool_bma_draws(weights_dict, seed_offset):
            pool, acc_total = [], 0
            for (p_bma, tau_bma), weight in weights_dict.items():
                td = int(round(mc_draws * weight))
                if td == 0: continue
                
                A_c, SIGMA_c = bvar_cache[(p_bma, tau_bma)]
                try: P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
                except np.linalg.LinAlgError: SIGMA_c += np.eye(K) * 1e-8; P_c = np.ascontiguousarray(np.linalg.cholesky(SIGMA_c))
                
                unique_seed = iteration_seed + seed_offset + p_bma * 100 + int(tau_bma * 100)
                v_c, _, acc_c = fast_draw_core(A_c, P_c, signs, p_bma, K, Q_avg, h_max, td, max_loops, unique_seed)
                if acc_c > 0: pool.append(v_c[:acc_c]); acc_total += acc_c
            
            if acc_total > 0: return (get_median_target_model(np.vstack(pool), acc_total) - true_IRF_target)**2
            return nan_array

        SE_joint_bma = pool_bma_draws(final_joint_weights, 7000)
        SE_geom_bma  = pool_bma_draws(final_geom_weights, 8000)

        # ---------------- TRADEOFF & SURFACE EVALUATIONS ----------------
        tradeoff_mses = None
        mdd_surface   = None

        if true_p == 4 and (T_real == 96 or T_real == 480):
            tradeoff_mses = np.zeros(len(TAU_DISCRETE))
            y_plot = np.ascontiguousarray(simulated_data)
            x_plot = np.ascontiguousarray(X_exo)

            for idx, t_val in enumerate(TAU_DISCRETE):
                try:
                    A_m, SIGMA_m, _, _ = estimate_alexandria_bvar(y_plot, p_max, X_exo=x_plot, tau_val=t_val, prior_mean=delta_wn, optimize_tau=False)
                    try: P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
                    except np.linalg.LinAlgError: SIGMA_m += np.eye(K) * 1e-8; P_m = np.ascontiguousarray(np.linalg.cholesky(SIGMA_m))
                    v_m, _, acc_m = fast_draw_core(A_m, P_m, signs, p_max, K, Q_avg, h_max, mc_draws, max_loops, iteration_seed + 10000 + idx)
                    tradeoff_mses[idx] = np.sum((get_median_target_model(v_m, acc_m) - true_IRF_target)**2) if acc_m > 0 else np.nan
                except: tradeoff_mses[idx] = np.nan

            if T_real == 480 and iter_idx == 0:
                mdd_surface = np.zeros(len(TAU_DISCRETE))
                for idx, t_val in enumerate(TAU_DISCRETE):
                    _, _, log_mdd_surf, _ = estimate_alexandria_bvar(y_plot, p_max, X_exo=x_plot, tau_val=t_val, prior_mean=delta_wn, optimize_tau=False)
                    mdd_surface[idx] = log_mdd_surf

        return (iter_idx, SE_aic, SE_sic, SE_hqc, SE_ols_bma, SE_ols_geom,
                SE_minn_wn_005, SE_minn_wn_020, SE_minn_wn_035, SE_minn_wn_050,
                SE_minn_wn_065, SE_minn_wn_080, SE_minn_wn_095,
                SE_minn_rw_020, SE_minn_rw_050,
                SE_joint_bma, SE_geom_bma, SE_minn_wn_mdd,
                SE_p0,
                SE_minn_wn_005_p0, SE_minn_wn_020_p0, SE_minn_wn_035_p0, SE_minn_wn_050_p0,
                SE_minn_wn_065_p0, SE_minn_wn_080_p0, SE_minn_wn_095_p0, SE_minn_wn_mdd_p0,
                best_mdd_tau, best_mdd_tau_p0,
                (p_hat_aic, p_hat_sic, p_hat_hqc, exp_p_hybrid, exp_p_geom_ols, exp_p_joint, exp_p_geom),
                final_bic_weights, geom_ols_p_dist, joint_p_dist, geom_p_dist,
                tradeoff_mses, mdd_surface, "Success")

    except Exception as e:
        return fail(f"Error: {str(e)}")

# -------------------------------------------------------------------
# 7. PARALLEL ORCHESTRATION WITH NESTED LOOP
# -------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    km_base = os.path.join(script_dir, 'Kilian and Murphy (2014)') if os.path.basename(script_dir) != 'Kilian and Murphy (2014)' else script_dir

    Q_avg, h_max = 72.3, 24
    N_iterations, mc_draws, max_loops = 2, 3, 5000

    sign_matrix = np.ascontiguousarray(np.array([
        [-1,  1,  1, np.nan], [-1,  1, -1, np.nan],
        [ 1,  1,  1, np.nan], [np.nan, np.nan, 1, np.nan]
    ], dtype=np.float64))

    dgp_lag_orders, sample_sizes = [4, 6, 8, 10], [240, 300, 360, 480, 600]
    master_results_list, bma_weights_list = [], []
    raw_taus_list, raw_taus_p0_list = [], []
    tradeoff_list, mdd_list = [], []
    iteration_mses_list = []

    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()

    print("\n" + "="*90)
    print(f" STARTING MASTER HYBRID MONTE CARLO (Parametric Shocks + Empirical Initial Values)")
    print("="*90)

    for current_p0 in dgp_lag_orders:
        dgp_path = os.path.join(km_base, 'DGP files', f'true_dgp_parameters_{current_p0}_lags.npz')
        if not os.path.exists(dgp_path): continue
        dgp = np.load(dgp_path)
        true_A, true_V, true_B_tilde = np.ascontiguousarray(dgp['A_true']), np.ascontiguousarray(dgp['V_true']), np.ascontiguousarray(dgp['B_tilde_true'])
        true_IRF, true_p = np.ascontiguousarray(dgp['True_IRF']), int(dgp['p_true'])
        p_max = max(12, true_p)

        # --- EXTRACT EMPIRICAL DATA FOR THE WARM START ---
        try:
            empirical_data = np.ascontiguousarray(dgp['empirical_data']) 
        except KeyError:
            print(f"\n[ERROR] Missing 'empirical_data' in {current_p0}_lags.npz")
            print("Please ensure your DGP extraction script saved the empirical dataset.")
            continue

        for current_T in sample_sizes:
            print(f"    Running Simulation for T = {current_T} ...", end="", flush=True)

            all_SE_aic, all_SE_sic, all_SE_hqc = [], [], []
            all_SE_ols_bma, all_SE_ols_geom = [], []
            all_SE_minn_wn_005, all_SE_minn_wn_020, all_SE_minn_wn_035 = [], [], []
            all_SE_minn_wn_050, all_SE_minn_wn_065, all_SE_minn_wn_080, all_SE_minn_wn_095 = [], [], [], []
            all_SE_minn_rw_020, all_SE_minn_rw_050 = [], []
            all_SE_joint_bma, all_SE_geom_bma, all_SE_minn_wn_mdd = [], [], []
            all_SE_p0 = []
            
            # New p0 specific SE lists
            all_SE_minn_wn_005_p0, all_SE_minn_wn_020_p0, all_SE_minn_wn_035_p0 = [], [], []
            all_SE_minn_wn_050_p0, all_SE_minn_wn_065_p0, all_SE_minn_wn_080_p0, all_SE_minn_wn_095_p0 = [], [], [], []
            all_SE_minn_wn_mdd_p0 = []

            all_mdd_taus, all_mdd_taus_p0 = [], []
            
            all_p_hats_aic, all_p_hats_sic, all_p_hats_hqc = [], [], []
            all_exp_p_hybrid, all_exp_p_geom_ols, all_exp_p_joint, all_exp_p_geom = [], [], [], []
            all_weights_hybrid, all_weights_geom_ols, all_weights_joint, all_weights_geom = [], [], [], []
            all_tradeoffs_t = []

            # Passed empirical_data cleanly into the task tuples
            tasks = [(i, SEED + i + (current_p0 * 10000) + (current_T * 100000), 
                      true_A, true_V, true_B_tilde, true_p, true_IRF, sign_matrix, Q_avg, h_max, 
                      mc_draws, max_loops, current_T, p_max, empirical_data) for i in range(N_iterations)]
                      
            results = Parallel(n_jobs=n_cores, backend='loky')(delayed(single_monte_carlo_iteration)(task) for task in tasks)

            for res in results:
                if res[-1] != "Success":
                    print(f"\n   -> [INTERNAL ERROR]: {res[-1]}")
                    continue

                all_SE_aic.append(res[1])
                all_SE_sic.append(res[2])
                all_SE_hqc.append(res[3])
                all_SE_ols_bma.append(res[4])
                all_SE_ols_geom.append(res[5])
                all_SE_minn_wn_005.append(res[6])
                all_SE_minn_wn_020.append(res[7])
                all_SE_minn_wn_035.append(res[8])
                all_SE_minn_wn_050.append(res[9])
                all_SE_minn_wn_065.append(res[10])
                all_SE_minn_wn_080.append(res[11])
                all_SE_minn_wn_095.append(res[12])
                all_SE_minn_rw_020.append(res[13])
                all_SE_minn_rw_050.append(res[14])
                all_SE_joint_bma.append(res[15])
                all_SE_geom_bma.append(res[16])
                all_SE_minn_wn_mdd.append(res[17])
                all_SE_p0.append(res[18])
                
                # Unpack the new p0 SEs
                all_SE_minn_wn_005_p0.append(res[19])
                all_SE_minn_wn_020_p0.append(res[20])
                all_SE_minn_wn_035_p0.append(res[21])
                all_SE_minn_wn_050_p0.append(res[22])
                all_SE_minn_wn_065_p0.append(res[23])
                all_SE_minn_wn_080_p0.append(res[24])
                all_SE_minn_wn_095_p0.append(res[25])
                all_SE_minn_wn_mdd_p0.append(res[26])

                all_mdd_taus.append(res[27])
                all_mdd_taus_p0.append(res[28])

                p_tuple = res[29]
                all_p_hats_aic.append(p_tuple[0])
                all_p_hats_sic.append(p_tuple[1])
                all_p_hats_hqc.append(p_tuple[2])
                all_exp_p_hybrid.append(p_tuple[3])
                all_exp_p_geom_ols.append(p_tuple[4])
                all_exp_p_joint.append(p_tuple[5])
                all_exp_p_geom.append(p_tuple[6])
                
                all_weights_hybrid.append(res[30])
                all_weights_geom_ols.append(res[31])
                all_weights_joint.append(res[32])
                all_weights_geom.append(res[33])

                se_sic_iter = np.sum(res[2]) if not (np.isscalar(res[2]) and np.isnan(res[2])) else np.nan
                se_bma_iter = np.sum(res[15]) if not (np.isscalar(res[15]) and np.isnan(res[15])) else np.nan
                se_p0_iter  = np.sum(res[18]) + 1e-12 if not (np.isscalar(res[18]) and np.isnan(res[18])) else np.nan

                iteration_mses_list.append({
                    'p0': current_p0, 'T': current_T, 'Iter': res[0],
                    'BIC_Rel_MSE': se_sic_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan,
                    'Joint_BMA_Rel_MSE': se_bma_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan
                })

                raw_taus_list.append({'p0': current_p0, 'T': current_T, 'iter': res[0], 'MDD_Tau': res[27]})
                raw_taus_p0_list.append({'p0': current_p0, 'T': current_T, 'iter': res[0], 'MDD_Tau_p0': res[28]})
                
                if res[34] is not None: all_tradeoffs_t.append(res[34])
                if res[35] is not None:
                    for i, t_val in enumerate(TAU_DISCRETE):
                        mdd_list.append({'Tau': t_val, 'MDD': res[35][i]})

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
                mean_p0_mse    = np.nanmean(all_SE_p0_sums)
                MSE_p0_iter    = np.copy(all_SE_p0_sums)
                MSE_p0_iter[np.isnan(MSE_p0_iter)] = mean_p0_mse
                MSE_p0_iter += 1e-12

            if len(all_tradeoffs_t) > 0:
                tradeoff_arr = np.array(all_tradeoffs_t)
                ratio_matrix = tradeoff_arr / MSE_p0_iter[:, None]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    geom_mean_curve = np.exp(np.nanmean(np.log(ratio_matrix), axis=0))
                for i, t_val in enumerate(TAU_DISCRETE):
                    tradeoff_list.append({'T': current_T, 'Tau': t_val, 'Rel_MSE': geom_mean_curve[i]})

            # Compute Average Weights for CSVs
            avg_hybrid_dist   = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_hybrid]) for p in range(1, p_max + 1)}
            avg_geom_ols_dist = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_geom_ols]) for p in range(1, p_max + 1)}
            avg_joint_dist    = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_joint]) for p in range(1, p_max + 1)}
            avg_geom_dist     = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_geom]) for p in range(1, p_max + 1)}

            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-BMA", **avg_hybrid_dist})
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-Geom-BMA", **avg_geom_ols_dist})
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "Joint-BMA", **avg_joint_dist})
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "Geom-BMA", **avg_geom_dist})

            def calc_met(se_list, p_list, true_val, tau_list=None):
                mse_sums = []
                for se in se_list:
                    if np.isscalar(se) and np.isnan(se): mse_sums.append(np.nan)
                    elif np.isnan(se).all(): mse_sums.append(np.nan)
                    else: mse_sums.append(np.sum(se))
                mse_sums = np.array(mse_sums)

                arr = mse_sums / MSE_p0_iter
                valid_mask = ~np.isnan(arr) & (arr > 0)
                valid_arr  = arr[valid_mask]
                fail_pct   = (1.0 - (np.sum(valid_mask) / len(arr))) * 100.0 if len(arr) > 0 else 100.0

                if len(valid_arr) > 0:
                    geom_mean = np.exp(np.mean(np.log(valid_arr)))
                    p5  = np.percentile(valid_arr, 5)
                    p95 = np.percentile(valid_arr, 95)
                else:
                    geom_mean, p5, p95 = np.nan, np.nan, np.nan

                lg      = (p_list.count(true_val) / len(p_list)) * 100 if isinstance(p_list, list) and not isinstance(p_list[0], float) else np.nan
                ml      = np.mean([p for p in p_list if not np.isnan(p)]) if p_list else p_max
                avg_tau = round(np.mean([t for t in tau_list if not np.isnan(t)]), 4) if tau_list else "N/A"

                return geom_mean, p5, p95, round(lg, 2), round(ml, 2), avg_tau, round(fail_pct, 1)

            models = [
                ("AIC",                              calc_met(all_SE_aic,          all_p_hats_aic,     true_p)),
                ("SIC (BIC)",                        calc_met(all_SE_sic,          all_p_hats_sic,     true_p)),
                ("HQC",                              calc_met(all_SE_hqc,          all_p_hats_hqc,     true_p)),
                ("OLS BMA (BIC-W)",                  calc_met(all_SE_ols_bma,      all_exp_p_hybrid,   true_p)),
                ("OLS BMA (Geom-W, th=0.5)",         calc_met(all_SE_ols_geom,     all_exp_p_geom_ols, true_p)),
                ("BVAR-WN (tau=0.05, p_max)",        calc_met(all_SE_minn_wn_005,  None,               true_p)),
                ("BVAR-WN (tau=0.20, p_max)",        calc_met(all_SE_minn_wn_020,  None,               true_p)),
                ("BVAR-WN (tau=0.35, p_max)",        calc_met(all_SE_minn_wn_035,  None,               true_p)),
                ("BVAR-WN (tau=0.50, p_max)",        calc_met(all_SE_minn_wn_050,  None,               true_p)),
                ("BVAR-WN (tau=0.65, p_max)",        calc_met(all_SE_minn_wn_065,  None,               true_p)),
                ("BVAR-WN (tau=0.80, p_max)",        calc_met(all_SE_minn_wn_080,  None,               true_p)),
                ("BVAR-WN (tau=0.95, p_max)",        calc_met(all_SE_minn_wn_095,  None,               true_p)),
                ("BVAR-RW (tau=0.20, p_max)",        calc_met(all_SE_minn_rw_020,  None,               true_p)),
                ("BVAR-RW (tau=0.50, p_max)",        calc_met(all_SE_minn_rw_050,  None,               true_p)),
                ("Joint (p, tau) Grid BMA",          calc_met(all_SE_joint_bma,    all_exp_p_joint,    true_p)),
                ("Geom (p, tau) Grid BMA (th=0.5)",  calc_met(all_SE_geom_bma,     all_exp_p_geom,     true_p)),
                ("BVAR-WN (MDD tau, p_max)",         calc_met(all_SE_minn_wn_mdd,  None,               true_p, all_mdd_taus)),
                
                # --- New p0 Specific Additions ---
                ("BVAR-WN (tau=0.05, p0)",           calc_met(all_SE_minn_wn_005_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.20, p0)",           calc_met(all_SE_minn_wn_020_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.35, p0)",           calc_met(all_SE_minn_wn_035_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.50, p0)",           calc_met(all_SE_minn_wn_050_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.65, p0)",           calc_met(all_SE_minn_wn_065_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.80, p0)",           calc_met(all_SE_minn_wn_080_p0,  None,            true_p)),
                ("BVAR-WN (tau=0.95, p0)",           calc_met(all_SE_minn_wn_095_p0,  None,            true_p)),
                ("BVAR-WN (MDD tau, p0)",            calc_met(all_SE_minn_wn_mdd_p0,  None,            true_p, all_mdd_taus_p0)),
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
        pd.DataFrame(bma_weights_list).to_csv(os.path.join(results_dir, f"Master_BMA_Weights_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(raw_taus_list).to_csv(os.path.join(results_dir, f"Master_Raw_MDD_Tau_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(raw_taus_p0_list).to_csv(os.path.join(results_dir, f"Master_Raw_MDD_Tau_p0_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(tradeoff_list).to_csv(os.path.join(results_dir, f"Master_Tradeoff_Curve_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(mdd_list).to_csv(os.path.join(results_dir, f"Master_MDD_Surface_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(iteration_mses_list).to_csv(os.path.join(results_dir, f"Master_Iteration_MSEs_iters{N_iterations}_draws{mc_draws}.csv"), index=False)

        print("\n" + "="*90)
        print(" RESULTS PREVIEW:")
        print("="*90)
        print(df_master.to_string(index=False))

        print("\n" + "="*90)
        print(" SUCCESS: Simulation complete and CSVs generated in 'Results'!")
        print("="*90)
    else:
        print("\n[ERROR] No data generated. Check file paths.")

if __name__ == '__main__':
    main()