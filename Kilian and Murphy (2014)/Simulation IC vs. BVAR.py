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
TAU_DISCRETE = [0.20, 0.40, 0.60, 0.80]
WN_SEED_OFFSETS = [999, 1001, 1008, 1002]

# -------------------------------------------------------------------
# 1. NUMBA-OPTIMIZED HYBRID DATA GENERATING PROCESS (DGP)
# -------------------------------------------------------------------
@njit
def simulate_var_dgp_hybrid(A, V, B_tilde, p, T_target, seed_val, empirical_data):
    np.random.seed(seed_val)
    K = A.shape[0]
    
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
# 3. HIGH-PERFORMANCE PRECOMPUTATION & EARLY-EXIT REJECTION SAMPLING
# -------------------------------------------------------------------
@njit(fastmath=True)
def compute_cholesky_irf_numba(A, P, h_max, K, p):
    """
    Computes Theta_h (VMA * Cholesky P) outside the rejection loop.
    This saves massive computation time.
    """
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            A_j = np.ascontiguousarray(A[:, (j-1)*K : j*K])
            Phi[h] += A_j @ Phi[h-j]

    Theta = np.zeros((h_max, K, K))
    for h in range(h_max):
        Theta[h] = Phi[h] @ P
    return Theta

@njit(fastmath=True)
def fast_draw_core(Theta, signs, K, h_max, target_draws, max_loops, seed_val):
    """
    Highly optimized rejection sampler using early-exit checking 
    and vector dot products instead of full matrix math.
    """
    if target_draws <= 0: return np.zeros((1, h_max, K, K)), 0, 0

    np.random.seed(seed_val)
    valid_IRFs = np.zeros((target_draws, h_max, K, K))
    attempts, accepted = 0, 0

    P = Theta[0] # Because Phi_0 is Identity, Theta_0 = P

    while accepted < target_draws and attempts < max_loops:
        attempts += 1
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)

        for i in range(K):
            if R[i, i] < 0: Q[:, i] = -Q[:, i]
        Q = np.ascontiguousarray(Q)

        B_tilde = P @ Q

        # --- 1. EARLY EXIT: Check Signs at h=0 ---
        match = True
        for i in range(K):
            for j in range(K):
                if not np.isnan(signs[i, j]):
                    if np.sign(B_tilde[i, j]) != signs[i, j]:
                        match = False
                        break
            if not match: break
        if not match: continue

        # --- 2. EARLY EXIT: Sequential Dynamic Checks (h=1 and h=2) ---
        cum_var0_shock0 = B_tilde[0, 0]
        dynamic_match = True
        for h in range(1, 3):
            # var 1, shock 0 must be < 0
            if np.dot(Theta[h, 1, :], Q[:, 0]) >= 0:
                dynamic_match = False
                break
            
            # var 2, shock 0 must be > 0
            if np.dot(Theta[h, 2, :], Q[:, 0]) <= 0:
                dynamic_match = False
                break
            
            # cumulative var 0, shock 0 must be < 0
            cum_var0_shock0 += np.dot(Theta[h, 0, :], Q[:, 0])
            if cum_var0_shock0 >= 0:
                dynamic_match = False
                break

        if not dynamic_match:
            continue

        # --- 3. FINAL BUILD: Passed all tests, construct the full array ---
        irf_cumulative = np.zeros((h_max, K, K))
        irf_cumulative[0] = B_tilde
        for h in range(1, h_max):
            irf_h = Theta[h] @ Q
            for k in range(K):
                for j in range(K):
                    if k == 0 or k == 3: # Cumulative row targets
                        irf_cumulative[h, k, j] = irf_cumulative[h-1, k, j] + irf_h[k, j]
                    else:
                        irf_cumulative[h, k, j] = irf_h[k, j]

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
    iter_idx, iteration_seed, true_A, true_V, true_B_tilde, true_p, true_IRF_target, signs, h_max, mc_draws, max_loops, T_real, p_max, empirical_data = args
    
    K = true_A.shape[0]
    nan_array = np.full(true_IRF_target.shape, np.nan)

    def fail(msg):
        # Mapped perfectly to the 28 items returned upon success
        return (
            iter_idx,
            *[None]*15, # The 15 expected SE arrays (adjusted for 4 taus)
            None,       # best_mdd_tau
            (None,)*9,  # p_tuples
            *[None]*6,  # weight dicts
            None,       # tradeoff_mses
            None,       # mdd_surface_data
            None,       # expected_tau_dict
            msg         # "Success" slot
        )

    # Helper function to compute P, Theta, and execute the draw inside a secure block
    def get_valid_draws(A_est, SIGMA_est, p_val, target_draws, seed_offset):
        if target_draws == 0: return np.zeros((1, h_max, K, K)), 0, 0
        try: P_est = np.ascontiguousarray(np.linalg.cholesky(SIGMA_est))
        except np.linalg.LinAlgError:
            S_mod = SIGMA_est + np.eye(K) * 1e-8
            try: P_est = np.ascontiguousarray(np.linalg.cholesky(S_mod))
            except: return np.zeros((1, h_max, K, K)), 0, 0
            
        Theta_est = compute_cholesky_irf_numba(A_est, P_est, h_max, K, p_val)
        return fast_draw_core(Theta_est, signs, K, h_max, target_draws, max_loops, iteration_seed + seed_offset)

    try:
        ols_cache = {}
        bic_scores = {}
        aic_scores = {}
        
        simulated_data = simulate_var_dgp_hybrid(true_A, true_V, true_B_tilde, true_p, T_real, iteration_seed, empirical_data)

        best_aic, best_sic, best_hqc = float('inf'), float('inf'), float('inf')
        p_hat_aic, p_hat_sic, p_hat_hqc = 1, 1, 1
        N_eff_total = T_real - p_max

        delta_wn = np.zeros(K)

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
                aic_scores[p_test] = aic_val

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

        # 1c. OLS AIC Weights (Uniform Prior)
        min_aic = min(aic_scores.values())
        raw_weights_aic = {p: np.exp(-0.5 * N_eff_total * (score - min_aic)) for p, score in aic_scores.items()}
        weight_sum_aic = sum(raw_weights_aic.values())
        kept_lags_aic = [p for p, w in raw_weights_aic.items() if (w / weight_sum_aic) > 0.01]
        final_aic_weights = {p: raw_weights_aic[p] / sum(raw_weights_aic[p] for p in kept_lags_aic) for p in kept_lags_aic}
        exp_p_aic_bma = sum(p * w for p, w in final_aic_weights.items())
        aic_p_dist = {p: final_aic_weights.get(p, 0.0) for p in range(1, p_max+1)}

        # 1d. OLS AIC Geometric Weights (theta = 0.5)
        raw_geom_aic = {p: np.exp(-0.5 * N_eff_total * (score - min_aic)) * (theta**p) for p, score in aic_scores.items()}
        sum_geom_aic = sum(raw_geom_aic.values())
        kept_geom_aic = {p: w for p, w in raw_geom_aic.items() if (w / sum_geom_aic) > 0.01}
        sum_kept_geom_aic = sum(kept_geom_aic.values())
        final_geom_aic_weights = {p: w / sum_kept_geom_aic for p, w in kept_geom_aic.items()}
        exp_p_geom_aic = sum(p * w for p, w in final_geom_aic_weights.items())
        geom_aic_p_dist = {p: final_geom_aic_weights.get(p, 0.0) for p in range(1, p_max+1)}

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
            v_irfs, att, acc = get_valid_draws(A_est, SIGMA_est, p_target, mc_draws, p_target)
            
            se = (get_median_target_model(v_irfs, acc) - true_IRF_target)**2 if acc > 0 else nan_array
            computed_ols_SEs[p_target] = se
            return se

        SE_aic = get_ols_se_for_p(p_hat_aic)
        SE_sic = get_ols_se_for_p(p_hat_sic)
        SE_hqc = get_ols_se_for_p(p_hat_hqc)
        SE_p0  = get_ols_se_for_p(true_p) # Maintained strictly as an MSE baseline

        # OLS BMA (Uniform) Evaluation
        ols_bma_pool, ols_bma_accepted = [], 0
        for p_bma in kept_lags_ols:
            td = int(round(mc_draws * final_bic_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            v_irfs, att, acc = get_valid_draws(A_est, SIGMA_est, p_bma, td, 5000 + p_bma)
            if acc > 0: ols_bma_pool.append(v_irfs[:acc]); ols_bma_accepted += acc

        SE_ols_bma = (get_median_target_model(np.vstack(ols_bma_pool), ols_bma_accepted) - true_IRF_target)**2 if ols_bma_accepted > 0 else nan_array

        # OLS BMA (Geometric) Evaluation
        ols_geom_pool, ols_geom_accepted = [], 0
        for p_bma in kept_geom_ols:
            td = int(round(mc_draws * final_geom_ols_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            v_irfs, att, acc = get_valid_draws(A_est, SIGMA_est, p_bma, td, 9000 + p_bma)
            if acc > 0: ols_geom_pool.append(v_irfs[:acc]); ols_geom_accepted += acc

        SE_ols_geom = (get_median_target_model(np.vstack(ols_geom_pool), ols_geom_accepted) - true_IRF_target)**2 if ols_geom_accepted > 0 else nan_array

        # OLS BMA (AIC) Evaluation
        ols_aic_pool, ols_aic_accepted = [], 0
        for p_bma in kept_lags_aic:
            td = int(round(mc_draws * final_aic_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            v_irfs, att, acc = get_valid_draws(A_est, SIGMA_est, p_bma, td, 6000 + p_bma)
            if acc > 0: ols_aic_pool.append(v_irfs[:acc]); ols_aic_accepted += acc

        SE_ols_bma_aic = (get_median_target_model(np.vstack(ols_aic_pool), ols_aic_accepted) - true_IRF_target)**2 if ols_aic_accepted > 0 else nan_array

        # OLS BMA (AIC Geometric) Evaluation
        ols_geom_aic_pool, ols_geom_aic_accepted = [], 0
        for p_bma in kept_geom_aic:
            td = int(round(mc_draws * final_geom_aic_weights[p_bma]))
            if td == 0: continue
            A_est, SIGMA_est = ols_cache[p_bma]
            v_irfs, att, acc = get_valid_draws(A_est, SIGMA_est, p_bma, td, 11000 + p_bma)
            if acc > 0: ols_geom_aic_pool.append(v_irfs[:acc]); ols_geom_aic_accepted += acc

        SE_ols_geom_aic = (get_median_target_model(np.vstack(ols_geom_aic_pool), ols_geom_aic_accepted) - true_IRF_target)**2 if ols_geom_aic_accepted > 0 else nan_array


        # ---------------- FIXED P_MAX BVAR EVALUATIONS (Cached) ----------------
        bvar_wn_SEs  = {}
        bvar_wn_MDDs = {}
        for tau_wn, seed_off in zip(TAU_DISCRETE, WN_SEED_OFFSETS):
            A_wn, SIGMA_wn = bvar_cache[(p_max, tau_wn)]
            mdd_wn = bvar_mdd_grid[(p_max, tau_wn)]
            
            v_wn, _, acc_wn = get_valid_draws(A_wn, SIGMA_wn, p_max, mc_draws, seed_off)
            
            bvar_wn_MDDs[tau_wn] = mdd_wn
            bvar_wn_SEs[tau_wn]  = (get_median_target_model(v_wn, acc_wn) - true_IRF_target)**2 if acc_wn > 0 else nan_array

        SE_minn_wn_020 = bvar_wn_SEs[0.20]
        SE_minn_wn_040 = bvar_wn_SEs[0.40]
        SE_minn_wn_060 = bvar_wn_SEs[0.60]
        SE_minn_wn_080 = bvar_wn_SEs[0.80]

        best_mdd_tau   = max(bvar_wn_MDDs, key=bvar_wn_MDDs.get)
        SE_minn_wn_mdd = bvar_wn_SEs[best_mdd_tau]


        # ---------------- BVAR BMA EVALUATIONS (Cached) ----------------
        def pool_bma_draws(weights_dict, seed_offset):
            pool, acc_total = [], 0
            for (p_bma, tau_bma), weight in weights_dict.items():
                td = int(round(mc_draws * weight))
                if td == 0: continue
                
                A_c, SIGMA_c = bvar_cache[(p_bma, tau_bma)]
                unique_seed = seed_offset + p_bma * 100 + int(tau_bma * 100)
                
                v_c, _, acc_c = get_valid_draws(A_c, SIGMA_c, p_bma, td, unique_seed)
                if acc_c > 0: pool.append(v_c[:acc_c]); acc_total += acc_c
            
            if acc_total > 0: return (get_median_target_model(np.vstack(pool), acc_total) - true_IRF_target)**2
            return nan_array

        SE_joint_bma = pool_bma_draws(final_joint_weights, 7000)
        SE_geom_bma  = pool_bma_draws(final_geom_weights, 8000)

        # ---------------- EXPECTED TAU GIVEN P (JOINT & GEOM BMA) ----------------
        expected_tau_dict = {}
        for p_val in range(1, p_max + 1):
            w_joint = {k: w for k, w in final_joint_weights.items() if k[0] == p_val}
            sum_w_j = sum(w_joint.values())
            exp_tau_j = sum(k[1] * w for k, w in w_joint.items()) / sum_w_j if sum_w_j > 0 else np.nan
            
            w_geom = {k: w for k, w in final_geom_weights.items() if k[0] == p_val}
            sum_w_g = sum(w_geom.values())
            exp_tau_g = sum(k[1] * w for k, w in w_geom.items()) / sum_w_g if sum_w_g > 0 else np.nan
            
            expected_tau_dict[p_val] = {'Joint-BMA': exp_tau_j, 'Geom-BMA': exp_tau_g}

        # ---------------- TRADEOFF & SURFACE EVALUATIONS ----------------
        # Highly optimized: Only points to cached MDDs and Standard Errors.
        tradeoff_mses = np.zeros(len(TAU_DISCRETE))
        mdd_surface_data = [] 

        for idx, t_val in enumerate(TAU_DISCRETE):
            se_array = bvar_wn_SEs.get(t_val, nan_array)
            if np.isscalar(se_array) and np.isnan(se_array):
                tradeoff_mses[idx] = np.nan
            elif np.isnan(se_array).all():
                tradeoff_mses[idx] = np.nan
            else:
                tradeoff_mses[idx] = np.sum(se_array)

            cached_mdd = bvar_mdd_grid.get((p_max, t_val), np.nan)
            mdd_surface_data.append({'Tau': t_val, 'MDD': cached_mdd})

        return (iter_idx, 
                SE_aic, SE_sic, SE_hqc, SE_ols_bma, SE_ols_geom, 
                SE_ols_bma_aic, SE_ols_geom_aic, 
                SE_minn_wn_020, SE_minn_wn_040, SE_minn_wn_060, SE_minn_wn_080,
                SE_joint_bma, SE_geom_bma, SE_minn_wn_mdd,
                SE_p0,
                best_mdd_tau,
                (p_hat_aic, p_hat_sic, p_hat_hqc, exp_p_hybrid, exp_p_geom_ols, exp_p_aic_bma, exp_p_geom_aic, exp_p_joint, exp_p_geom),
                final_bic_weights, geom_ols_p_dist, aic_p_dist, geom_aic_p_dist, joint_p_dist, geom_p_dist, 
                tradeoff_mses, mdd_surface_data, expected_tau_dict, "Success")

    except Exception as e:
        return fail(f"Error: {str(e)}")

# -------------------------------------------------------------------
# 7. PARALLEL ORCHESTRATION WITH NESTED LOOP
# -------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    km_base = os.path.join(script_dir, 'Kilian and Murphy (2014)') if os.path.basename(script_dir) != 'Kilian and Murphy (2014)' else script_dir

    h_max = 24
    N_iterations, mc_draws, max_loops = 500, 1000, 5000000

    sign_matrix = np.ascontiguousarray(np.array([
        [-1,  1,  1, np.nan], [-1,  1, -1, np.nan],
        [ 1,  1,  1, np.nan], [np.nan, np.nan, 1, np.nan]
    ], dtype=np.float64))

    dgp_lag_orders, sample_sizes = [4, 10], [240, 600]
    master_results_list, bma_weights_list = [], []
    raw_taus_list = []
    tradeoff_list, mdd_list = [], []
    iteration_mses_list = []
    expected_tau_list = []

    global_start_time = time.time()
    n_cores = multiprocessing.cpu_count()

    print("\n" + "="*90)
    print(f" STARTING MASTER HYBRID MONTE CARLO (Optimized FastMath)")
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
            continue

        for current_T in sample_sizes:
            print(f"    Running Simulation for T = {current_T} ...", end="", flush=True)

            all_SE_aic, all_SE_sic, all_SE_hqc = [], [], []
            all_SE_ols_bma, all_SE_ols_geom = [], []
            all_SE_ols_bma_aic, all_SE_ols_geom_aic = [], [] 
            all_SE_minn_wn_020, all_SE_minn_wn_040 = [], []
            all_SE_minn_wn_060, all_SE_minn_wn_080 = [], []
            all_SE_joint_bma, all_SE_geom_bma, all_SE_minn_wn_mdd = [], [], []
            all_SE_p0 = []
            
            all_mdd_taus = []
            
            all_p_hats_aic, all_p_hats_sic, all_p_hats_hqc = [], [], []
            all_exp_p_hybrid, all_exp_p_geom_ols, all_exp_p_joint, all_exp_p_geom = [], [], [], []
            all_exp_p_aic_bma, all_exp_p_geom_aic = [], []
            all_weights_hybrid, all_weights_geom_ols, all_weights_joint, all_weights_geom = [], [], [], []
            all_weights_aic_bma, all_weights_geom_aic = [], [] 
            all_tradeoffs_t = []

            # Removed unused Q_avg parameter
            tasks = [(i, SEED + i + (current_p0 * 10000) + (current_T * 100000), 
                      true_A, true_V, true_B_tilde, true_p, true_IRF, sign_matrix, h_max, 
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
                all_SE_ols_bma_aic.append(res[6])    
                all_SE_ols_geom_aic.append(res[7])   
                all_SE_minn_wn_020.append(res[8])
                all_SE_minn_wn_040.append(res[9])
                all_SE_minn_wn_060.append(res[10])
                all_SE_minn_wn_080.append(res[11])
                all_SE_joint_bma.append(res[12])
                all_SE_geom_bma.append(res[13])
                all_SE_minn_wn_mdd.append(res[14])
                all_SE_p0.append(res[15])
                
                all_mdd_taus.append(res[16])

                p_tuple = res[17]
                all_p_hats_aic.append(p_tuple[0])
                all_p_hats_sic.append(p_tuple[1])
                all_p_hats_hqc.append(p_tuple[2])
                all_exp_p_hybrid.append(p_tuple[3])
                all_exp_p_geom_ols.append(p_tuple[4])
                all_exp_p_aic_bma.append(p_tuple[5])   
                all_exp_p_geom_aic.append(p_tuple[6])  
                all_exp_p_joint.append(p_tuple[7])
                all_exp_p_geom.append(p_tuple[8])
                
                all_weights_hybrid.append(res[18])
                all_weights_geom_ols.append(res[19])
                all_weights_aic_bma.append(res[20])    
                all_weights_geom_aic.append(res[21])   
                all_weights_joint.append(res[22])
                all_weights_geom.append(res[23])

                se_sic_iter = np.sum(res[2]) if not (np.isscalar(res[2]) and np.isnan(res[2])) else np.nan
                se_bma_iter = np.sum(res[12]) if not (np.isscalar(res[12]) and np.isnan(res[12])) else np.nan
                se_p0_iter  = np.sum(res[15]) + 1e-12 if not (np.isscalar(res[15]) and np.isnan(res[15])) else np.nan

                iteration_mses_list.append({
                    'p0': current_p0, 'T': current_T, 'Iter': res[0],
                    'BIC_Rel_MSE': se_sic_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan,
                    'Joint_BMA_Rel_MSE': se_bma_iter / se_p0_iter if pd.notna(se_p0_iter) else np.nan
                })

                raw_taus_list.append({'p0': current_p0, 'T': current_T, 'iter': res[0], 'MDD_Tau': res[16]})
                
                if res[24] is not None: all_tradeoffs_t.append(res[24])
                
                if res[25]: 
                    for item in res[25]:
                        mdd_list.append({
                            'p0': current_p0, 
                            'T': current_T, 
                            'Iter': res[0], 
                            'Tau': item['Tau'], 
                            'MDD': item['MDD']
                        })

                exp_tau_dict = res[26]
                for p_val, taus in exp_tau_dict.items():
                    if not np.isnan(taus['Joint-BMA']):
                        expected_tau_list.append({
                            'p0': current_p0, 'T': current_T, 'Iter': res[0], 
                            'p': p_val, 'Estimator': 'Joint-BMA', 'Expected_Tau': taus['Joint-BMA']
                        })
                    if not np.isnan(taus['Geom-BMA']):
                        expected_tau_list.append({
                            'p0': current_p0, 'T': current_T, 'Iter': res[0], 
                            'p': p_val, 'Estimator': 'Geom-BMA', 'Expected_Tau': taus['Geom-BMA']
                        })

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
            avg_aic_dist      = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_aic_bma]) for p in range(1, p_max + 1)} 
            avg_geom_aic_dist = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_geom_aic]) for p in range(1, p_max + 1)} 
            avg_joint_dist    = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_joint]) for p in range(1, p_max + 1)}
            avg_geom_dist     = {f"p={p}": np.mean([w.get(p, 0.0) for w in all_weights_geom]) for p in range(1, p_max + 1)}

            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-BMA", **avg_hybrid_dist})
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-Geom-BMA", **avg_geom_ols_dist})
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-BMA (AIC-W)", **avg_aic_dist}) 
            bma_weights_list.append({"True DGP (p0)": current_p0, "Sample Size (T)": current_T, "Estimator": "OLS-Geom-BMA (AIC-W)", **avg_geom_aic_dist}) 
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
                ("OLS BMA (AIC-W)",                  calc_met(all_SE_ols_bma_aic,  all_exp_p_aic_bma,  true_p)), 
                ("OLS BMA (Geom-AIC-W, th=0.5)",     calc_met(all_SE_ols_geom_aic, all_exp_p_geom_aic, true_p)), 
                ("BVAR-WN (tau=0.20, p_max)",        calc_met(all_SE_minn_wn_020,  None,               true_p)),
                ("BVAR-WN (tau=0.40, p_max)",        calc_met(all_SE_minn_wn_040,  None,               true_p)),
                ("BVAR-WN (tau=0.60, p_max)",        calc_met(all_SE_minn_wn_060,  None,               true_p)),
                ("BVAR-WN (tau=0.80, p_max)",        calc_met(all_SE_minn_wn_080,  None,               true_p)),
                ("Joint (p, tau) Grid BMA",          calc_met(all_SE_joint_bma,    all_exp_p_joint,    true_p)),
                ("Geom (p, tau) Grid BMA (th=0.5)",  calc_met(all_SE_geom_bma,     all_exp_p_geom,     true_p)),
                ("BVAR-WN (MDD tau, p_max)",         calc_met(all_SE_minn_wn_mdd,  None,               true_p, all_mdd_taus)),
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
        pd.DataFrame(tradeoff_list).to_csv(os.path.join(results_dir, f"Master_Tradeoff_Curve_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(mdd_list).to_csv(os.path.join(results_dir, f"Master_MDD_Surface_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(iteration_mses_list).to_csv(os.path.join(results_dir, f"Master_Iteration_MSEs_iters{N_iterations}_draws{mc_draws}.csv"), index=False)
        pd.DataFrame(expected_tau_list).to_csv(os.path.join(results_dir, f"Master_Expected_Tau_given_p_iters{N_iterations}_draws{mc_draws}.csv"), index=False)

        print("\n" + "="*90)
        print(" RESULTS PREVIEW:")
        print("="*90)
        print(df_master.to_string(index=False))

        print("\n" + "="*90)
        print(" SUCCESS: Simulation complete and CSVs generated in 'Results'!")
        print("="*90)

        print(f"Simulation took {time.time() - global_start_time} seconds")
    else:
        print("\n[ERROR] No data generated. Check file paths.")

if __name__ == '__main__':
    main()