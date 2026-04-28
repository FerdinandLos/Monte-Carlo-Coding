# Monte Carlo Simulation Johansen Juselius



# Now calculate the true profile BEFORE the loop
true_pp <- calc_persistence_profile(alpha = alpha_hat, beta = beta, 
                                    gamma_list = list(gamma_hat), 
                                    Sigma = Sigma_hat, horizon = 24)


# 7. The Monte Carlo Simulation Loop
n_draws <- 5000
sample_size <- 80 # Example sample size T=80

# --- PRE-SIMULATION SETUP ---
p0 <- 2             # True lag order of the DGP
p_max <- 8          # Maximum lag order to test (Ivanov & Kilian use 8 for quarterly)
K <- 4              # Number of endogenous variables (m1, y, im, dp)
T_total <- 80       # Total sample size for this experiment

# Extract the exact coefficient matrix from your force-fitted DGP model
# Rows: Intercept, ECT1, ECT2, ECT3, dY_lag1(4 vars), D1_c, D2_c, D3_c
dgp_coefs <- coef(dgp_model) 

# Create storage for the selected lag orders across all 5,000 draws
selected_lags <- data.frame(AIC = numeric(n_draws), 
                            SIC = numeric(n_draws), 
                            HQC = numeric(n_draws))

# --- THE MONTE CARLO LOOP ---
for (draw in 1:n_draws) {
  
  # a. Draw synthetic Gaussian innovations
  sim_errors <- MASS::mvrnorm(n = T_total, mu = rep(0, K), Sigma = Sigma_hat)
  
  # b. Initialize the starting values
  # Ivanov & Kilian: "Initial values are obtained by randomly drawing blocks of 
  # data of length p0 with replacement from the original data set." 
  rand_idx <- sample(1:(nrow(Y) - p0), 1)
  Y_sim <- matrix(0, nrow = T_total + p0, ncol = K)
  Y_sim[1:p0, ] <- Y[rand_idx:(rand_idx + p0 - 1), ] 
  
  # Prepare centered seasonal dummies for the simulated sample
  quarter_seq <- rep(c(1, 2, 3, 4), length.out = T_total)
  D1_sim <- ifelse(quarter_seq == 1, 1, 0) - 0.25
  D2_sim <- ifelse(quarter_seq == 2, 1, 0) - 0.25
  D3_sim <- ifelse(quarter_seq == 3, 1, 0) - 0.25
  
  # c. Generate the synthetic time series
  for (t in 1:T_total) {
    t_sim <- t + p0 # Current time index in the Y_sim matrix
    
    # 1. Get lagged levels to compute the Error Correction Term
    Y_lag1 <- Y_sim[t_sim - 1, ]
    ECT_lag1_sim <- as.numeric(t(beta) %*% Y_lag1)
    
    # 2. Get lagged differences (short-run momentum)
    dY_lag1_sim <- Y_sim[t_sim - 1, ] - Y_sim[t_sim - 2, ]
    
    # 3. Get current seasonal dummies
    D_t <- c(D1_sim[t], D2_sim[t], D3_sim[t])
    
    # 4. Combine all predictors into a single vector 
    # Must perfectly match the order of variables in your lm() model!
    # Format: c(Intercept, ECT_lag1, dY_lag1, dummies)
    predictors <- c(1, ECT_lag1_sim, dY_lag1_sim, D_t)
    
    # 5. Generate current difference via matrix multiplication + random shock
    dY_t <- (predictors %*% dgp_coefs) + sim_errors[t, ]
    
    # 6. Update the current level of Y_sim
    Y_sim[t_sim, ] <- Y_sim[t_sim - 1, ] + dY_t
  }
  
  # d. Apply Lag Selection Criteria to Y_sim
  # We test lags p = 1 through p_max (which translates to 0 to p_max-1 difference lags in a VECM)
  ic_results <- data.frame(p = 1:p_max, AIC = NA, SIC = NA, HQC = NA)
  
  # Calculate effective sample size N (Total length minus max lags used for initialization)
  # Keeping N constant is strictly required to compare Information Criteria properly.
  N <- nrow(Y_sim) - p_max 
  
  for (p_test in 1:p_max) {
    
    # Create the dependent variable matrix (Delta Y_t) for the effective sample
    dY_eff <- diff(Y_sim)[(p_max):nrow(diff(Y_sim)), ]
    
    # Create the ECT matrix for the effective sample
    ECT_eff <- (Y_sim %*% beta)[(p_max):(nrow(Y_sim)-1), ]
    
    # Base formula string
    form_str <- "dY_eff ~ ECT_eff"
    
    # If testing lag p > 1, add lagged differences to the regression
    if (p_test > 1) {
      for (lag in 1:(p_test - 1)) {
        assign(paste0("dY_lag", lag), diff(Y_sim)[(p_max - lag):(nrow(diff(Y_sim)) - lag), ])
        form_str <- paste0(form_str, " + dY_lag", lag)
      }
    }
    
    # Fit the restricted VECM for this specific lag order
    test_model <- lm(as.formula(form_str))
    
    # Calculate Residual Covariance Matrix (Sigma_bar)
    Sigma_bar <- cov(residuals(test_model))
    det_Sigma <- det(Sigma_bar)
    
    # Count the number of freely estimated parameters in the system
    # K variables * (1 intercept + 3 ECTs + K*(p_test-1) short-run lags)
    num_params <- K * (1 + 3 + K*(p_test - 1)) 
    
    # Calculate Information Criteria based on Ivanov & Kilian formulas
    ic_results$AIC[p_test] <- log(det_Sigma) + (2 / N) * num_params
    ic_results$SIC[p_test] <- log(det_Sigma) + (log(N) / N) * num_params
    ic_results$HQC[p_test] <- log(det_Sigma) + (2 * log(log(N)) / N) * num_params
  }
  
  # Record the lag order that minimizes each criterion for this draw
  selected_lags$AIC[draw] <- which.min(ic_results$AIC)
  selected_lags$SIC[draw] <- which.min(ic_results$SIC)
  selected_lags$HQC[draw] <- which.min(ic_results$HQC)
  
  # (Optional): Estimate Persistence Profiles using the selected lag order
  # ...
}

# Showcase the accuracy of the lag selection by the different criteria
# Load necessary libraries (install them if you haven't: install.packages(c("dplyr", "tidyr", "ggplot2")))
library(dplyr)
library(tidyr)

# Calculate the frequency percentage of each selected lag for each criterion
summary_table <- selected_lags %>%
  pivot_longer(cols = everything(), names_to = "Criterion", values_to = "Lag") %>%
  group_by(Criterion, Lag) %>%
  summarise(Count = n(), .groups = 'drop') %>%
  mutate(Percentage = (Count / n_draws) * 100) %>%
  arrange(Criterion, Lag)

# Print the summary
print(summary_table)

# Visualize the distribution of selected lag orders for each criterion as a histogram
library(ggplot2)

# Reshape the data into a "long" format for ggplot
lags_long <- selected_lags %>%
  pivot_longer(cols = c(AIC, SIC, HQC), 
               names_to = "Criterion", 
               values_to = "Selected_Lag")

# Create the bar chart
ggplot(lags_long, aes(x = Selected_Lag, fill = Criterion)) +
  geom_bar(color = "black", alpha = 0.8) +
  geom_vline(xintercept = p0, color = "red", linetype = "dashed", size = 1) + # Adds a line for the TRUE lag
  scale_x_continuous(breaks = 1:p_max) +
  facet_wrap(~ Criterion, ncol = 1) + # Stacks the three charts vertically
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Set1") +
  theme(legend.position = "none") +
  labs(title = "Distribution of Selected Lag Orders (Monte Carlo Simulation)",
       subtitle = paste("True Lag Order (p0) =", p0, "| Sample Size (T) =", T_total),
       x = "Selected Lag Order",
       y = "Frequency (out of 5,000 draws)") +
  annotate("text", x = p0 + 0.5, y = n_draws * 0.5, label = "True Lag", color = "red", angle = 90)



# Persistence Profile Calculation Function



calc_persistence_profile <- function(alpha, beta, gamma_list, Sigma, horizon = 24) {
  K <- nrow(alpha)
  p <- length(gamma_list) + 1 # Total VAR lag order
  
  # 1. Convert VECM coefficients to VAR(p) coefficients (A_matrices)
  A <- list()
  I_k <- diag(K)
  
  if (p == 1) {
    A[[1]] <- I_k + alpha %*% t(beta)
  } else {
    A[[1]] <- I_k + alpha %*% t(beta) + gamma_list[[1]]
    if (p > 2) {
      for (i in 2:(p-1)) {
        A[[i]] <- gamma_list[[i]] - gamma_list[[i-1]]
      }
    }
    A[[p]] <- -gamma_list[[p-1]]
  }
  
  # 2. Calculate Moving Average matrices (Psi)
  Psi <- list()
  Psi[[1]] <- I_k # Psi_0
  
  for (h in 1:horizon) {
    Psi_h <- matrix(0, K, K)
    for (j in 1:min(h, p)) {
      Psi_h <- Psi_h + A[[j]] %*% Psi[[h - j + 1]]
    }
    Psi[[h + 1]] <- Psi_h
  }
  
  # 3. Calculate the Persistence Profile for the first cointegrating vector
  # (For Johansen & Juselius, beta[,1] is the m1 - y vector)
  b1 <- as.matrix(beta[, 1]) 
  denominator <- as.numeric(t(b1) %*% Sigma %*% b1)
  
  pp_values <- numeric(horizon + 1)
  for (h in 0:horizon) {
    numerator <- t(b1) %*% Psi[[h + 1]] %*% Sigma %*% t(Psi[[h + 1]]) %*% b1
    pp_values[h + 1] <- as.numeric(numerator / denominator)
  }
  
  return(pp_values)
}

# Calculate absolute truth PP before the loop starts
true_pp <- calc_persistence_profile(alpha = alpha_hat, beta = beta, 
                                    gamma_list = list(gamma_hat), # For p0=2
                                    Sigma = Sigma_hat, horizon = 24)

# Monte Carlo Loop with PP Calculation
# Run this BEFORE the Monte Carlo loop starts
horizon_len <- 24 + 1 # h=0 through h=24

sq_err_p0  <- matrix(NA, nrow = n_draws, ncol = horizon_len)
sq_err_AIC <- matrix(NA, nrow = n_draws, ncol = horizon_len)
sq_err_SIC <- matrix(NA, nrow = n_draws, ncol = horizon_len)
sq_err_HQC <- matrix(NA, nrow = n_draws, ncol = horizon_len)



# Helper function to estimate VECM for a given lag and return the Persistence Profile
get_estimated_pp <- function(p_target, Y_sim, beta) {
  
  # Set up data matrices for the effective sample (using constant N)
  # Assuming p_max = 8 from the previous code block
  p_max <- 8
  K <- 4
  
  dY_eff <- diff(Y_sim)[(p_max):nrow(diff(Y_sim)), ]
  ECT_eff <- (Y_sim %*% beta)[(p_max):(nrow(Y_sim)-1), ]
  
  # Build the regression formula
  form_str <- "dY_eff ~ ECT_eff"
  if (p_target > 1) {
    for (lag in 1:(p_target - 1)) {
      assign(paste0("dY_lag", lag), diff(Y_sim)[(p_max - lag):(nrow(diff(Y_sim)) - lag), ])
      form_str <- paste0(form_str, " + dY_lag", lag)
    }
  }
  
  # Estimate the model
  model_est <- lm(as.formula(form_str))
  coef_est <- coef(model_est)
  
  # Extract parameters
  alpha_est <- coef_est[2:4, ] # Assuming no intercept/dummies to simplify, or adjust index if included
  Sigma_est <- cov(residuals(model_est))
  
  # Extract short-run momentum matrices (Gamma)
  gamma_list_est <- list()
  if (p_target > 1) {
    for (lag in 1:(p_target - 1)) {
      # Calculate the column indices for this specific lag's coefficients
      # (Adjust indices based on the exact columns in your model_est)
      start_col <- 4 + K*(lag - 1) + 1 
      end_col <- start_col + (K - 1)
      gamma_list_est[[lag]] <- coef_est[start_col:end_col, ]
    }
  }
  
  # Calculate and return the persistence profile
  pp <- calc_persistence_profile(alpha = alpha_est, beta = beta, 
                                 gamma_list = gamma_list_est, 
                                 Sigma = Sigma_est, horizon = 24)
  return(pp)
}

# ... (End of lag selection code inside the loop) ...

  # 1. Retrieve the lag orders selected by each criterion
  p_aic <- selected_lags$AIC[draw]
  p_sic <- selected_lags$SIC[draw]
  p_hqc <- selected_lags$HQC[draw]
  
  # 2. Calculate the estimated Persistence Profiles
  # Note: true_pp is calculated ONCE before the Monte Carlo loop starts
  estimated_pp_p0  <- get_estimated_pp(p_target = p0, Y_sim, beta)
  estimated_pp_AIC <- get_estimated_pp(p_target = p_aic, Y_sim, beta)
  estimated_pp_SIC <- get_estimated_pp(p_target = p_sic, Y_sim, beta)
  estimated_pp_HQC <- get_estimated_pp(p_target = p_hqc, Y_sim, beta)
  
  # 3. Calculate the pointwise squared errors for this draw
  # (estimated value - true value)^2
  sq_err_p0[draw, ]  <- (estimated_pp_p0 - true_pp)^2
  sq_err_AIC[draw, ] <- (estimated_pp_AIC - true_pp)^2
  sq_err_SIC[draw, ] <- (estimated_pp_SIC - true_pp)^2
  sq_err_HQC[draw, ] <- (estimated_pp_HQC - true_pp)^2
