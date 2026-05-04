# Load necessary libraries
# install.packages(c("vars", "MASS", "BVAR"))
library(vars)
library(MASS)
library(BVAR)

# ==============================================================================
# 1. SETUP & DGP LOADING
# ==============================================================================
set.seed(123) 

mc_iterations <- 100   # Set to 500-1000 for your final run
T_sample <- 150        # Sample size to test (e.g., matching Ramey's ~280 quarters)
h_max <- 20            # Maximum IRF horizon (Ramey uses 20 quarters)
p_max <- 8             # Maximum lag order to test for Information Criteria

# Load the force-fitted DGP parameters
# Brings B_hat, Sigma_hat, Y (historical data), p0 (true lag), K into environment
load("DGPs/ramey_2011_dgp_1.RData") 

# --- Extract Autoregressive Lags for the True IRF ---
# B_hat columns: 1:Intercept, 2:t, 3:t^2, 4 to (3+K):Lag1, etc.
A_lags <- list()
for (i in 1:p0) {
  start_col <- 4 + (i - 1) * K
  end_col <- start_col + K - 1
  A_lags[[i]] <- B_hat[, start_col:end_col]
}

# --- Function to calculate True Population IRF ---
get_true_irf <- function(A_lags, Sigma, h_max, K, p_true) {
  P <- t(chol(Sigma)) # Lower Cholesky factor
  IRF_array <- array(0, dim = c(K, K, h_max + 1))
  IRF_array[,,1] <- P # Impact matrix at h=0
  
  Phi <- list()
  Phi[[1]] <- diag(K)
  
  for(i in 1:h_max) {
    Phi_i <- matrix(0, K, K)
    for(j in 1:min(i, p_true)) {
      Phi_i <- Phi_i + A_lags[[j]] %*% Phi[[i - j + 1]]
    }
    Phi[[i + 1]] <- Phi_i
  }
  
  for(i in 1:(h_max + 1)) {
    IRF_array[,,i] <- Phi[[i]] %*% P
  }
  return(IRF_array)
}

# Calculate the locked-in absolute truth
true_irf <- get_true_irf(A_lags, Sigma_hat, h_max, K, p0)

# ==============================================================================
# 2. STORAGE FOR RESULTS
# ==============================================================================
methods <- c("AIC", "BIC", "HQ", "BVAR")
mse_results <- array(0, dim = c(K, K, h_max + 1, length(methods)), 
                     dimnames = list(paste0("Shock_", 1:K), 
                                     paste0("Resp_", 1:K), 
                                     paste0("H_", 0:h_max), 
                                     methods))

# ==============================================================================
# 3. MONTE CARLO LOOP
# ==============================================================================
cat("Starting Monte Carlo Simulation...\n")

for (iter in 1:mc_iterations) {
  if (iter %% 10 == 0) cat("Iteration:", iter, "/", mc_iterations, "\n")
  
  # --- A. Data Generation ---
  errors <- MASS::mvrnorm(n = T_sample, mu = rep(0, K), Sigma = Sigma_hat)
  Y_sim <- matrix(0, nrow = T_sample + p0, ncol = K)
  colnames(Y_sim) <- colnames(Y)
  
  # Draw initial block from historical data to start momentum
  rand_idx <- sample(1:(nrow(Y) - p0), 1)
  Y_sim[1:p0, ] <- Y[rand_idx:(rand_idx + p0 - 1), ]
  
  # Set up deterministic trends for the simulation period
  t_sim <- 1:(T_sample + p0)
  t2_sim <- t_sim^2
  
  # Step-by-step generation
  for (t in (p0 + 1):(T_sample + p0)) {
    # Combine deterministic terms and lagged values perfectly mirroring B_hat
    pred <- c(1, t_sim[t], t2_sim[t])
    for (l in 1:p0) {
      pred <- c(pred, Y_sim[t - l, ])
    }
    Y_sim[t, ] <- (B_hat %*% pred) + errors[t - p0, ]
  }
  
  # Trim initial values; keep only the generated sample
  Y_est <- ts(Y_sim[(p0 + 1):nrow(Y_sim), ])
  
  # Define Exogenous Variables (Trend and Trend Squared)
  exog_data <- matrix(c(t_sim[(p0 + 1):length(t_sim)], t2_sim[(p0 + 1):length(t2_sim)]), ncol=2)
  colnames(exog_data) <- c("t", "t2")
  
  # --- B. Information Criteria VARs ---
  var_select <- vars::VARselect(Y_est, lag.max = p_max, type = "const", exogen = exog_data)
  
  p_aic <- var_select$selection["AIC(n)"]
  p_bic <- var_select$selection["SC(n)"]
  p_hq  <- var_select$selection["HQ(n)"]
  
  # Estimate models
  var_aic <- vars::VAR(Y_est, p = p_aic, type = "const", exogen = exog_data)
  var_bic <- vars::VAR(Y_est, p = p_bic, type = "const", exogen = exog_data)
  var_hq  <- vars::VAR(Y_est, p = p_hq,  type = "const", exogen = exog_data)
  
  # Extract Orthogonalized IRFs explicitly using the 'vars' package
  irf_aic <- vars::irf(var_aic, n.ahead = h_max, ortho = TRUE, boot = FALSE)$irf
  irf_bic <- vars::irf(var_bic, n.ahead = h_max, ortho = TRUE, boot = FALSE)$irf
  irf_hq  <- vars::irf(var_hq,  n.ahead = h_max, ortho = TRUE, boot = FALSE)$irf
  
  # --- C. Bayesian VAR (Minnesota Prior) ---
  mn_settings <- BVAR::bv_minnesota(
    lambda = BVAR::bv_lambda(mode = 0.2, sd = 0.4), 
    alpha  = BVAR::bv_alpha(mode = 2), 
    var    = 1e07,
    psi    = BVAR::bv_psi(mode = diag(Sigma_hat)) 
  )
  
  bvar_priors <- BVAR::bv_priors(mn = mn_settings)
  
  bvar_success <- tryCatch({
    invisible(capture.output({
      bvar_model <- BVAR::bvar(Y_est, lags = p_max, x = exog_data,
                               priors = bvar_priors, 
                               n_draw = 2000, n_burn = 500, verbose = FALSE)
      
      bvar_irf_obj <- BVAR::irf(bvar_model, horizon = h_max + 2)
    }))
    
    # BVAR draws are array structured [draws, variables, shocks, horizon]
    # 'apply' preserves the [variables, shocks, horizon] dimensions
    bvar_irf_median <- apply(bvar_irf_obj$irf, c(2, 3, 4), median)
    
    TRUE # Returns TRUE if everything above worked perfectly
    
  }, error = function(e) {
    FALSE # Returns FALSE if the model crashed
  })
  
  # If the BVAR crashed on an explosive draw, skip the rest of this loop iteration
  if (!bvar_success) {
    cat("\nSkipping iteration", iter, "due to non-stationary simulated data.\n")
    next 
  }
  
  # --- D. Calculate Squared Errors vs True IRF ---
H_common <- min(
  h_max + 1,
  dim(bvar_irf_median)[1],
  nrow(irf_aic[[1]])
)

for (shock in 1:K) {
  for (resp in 1:K) {
    for (h in 1:H_common) {
      
      true_val <- true_irf[resp, shock, h]
      
      aic_val <- irf_aic[[shock]][h, resp]
      bic_val <- irf_bic[[shock]][h, resp]
      hq_val  <- irf_hq[[shock]][h, resp]
      bvar_val <- bvar_irf_median[h, resp, shock]
      
      mse_results[shock, resp, h, "AIC"]  <- mse_results[shock, resp, h, "AIC"]  + (aic_val - true_val)^2
      mse_results[shock, resp, h, "BIC"]  <- mse_results[shock, resp, h, "BIC"]  + (bic_val - true_val)^2
      mse_results[shock, resp, h, "HQ"]   <- mse_results[shock, resp, h, "HQ"]   + (hq_val - true_val)^2
      mse_results[shock, resp, h, "BVAR"] <- mse_results[shock, resp, h, "BVAR"] + (bvar_val - true_val)^2
    }
  }
}
    }
=============================================================================
# 4. COMPUTE & ANALYZE MSE
# ==============================================================================
mse_final <- mse_results / mc_iterations

cat("\nMonte Carlo Simulation Complete.\n")

# Example: Display the MSE specifically for how a "Defense News" shock (Variable 1) 
# impacts "Real GDP" (Variable 3) across all 20 horizons.
cat("\nMSE: Defense News Shock -> Real GDP Response (Horizon 0 to 20):\n")
print(mse_final[1, 3, , ])
