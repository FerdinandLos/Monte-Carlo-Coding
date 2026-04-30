# Monte Carlo Simulation Johansen Juselius

# Load the DGP parameters
load("Monte-Carlo-Coding/DGPs/Johansen Juselius (1990) DGP.RData")


# Now calculate the true profile BEFORE the loop
# NOTE: The 'calc_persistence_profile' function needs to be defined or sourced
# into your environment before this line can run.
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
# Reconstruct the full DGP coefficient matrix from the loaded parameters.
# The original 'dgp_model' object is not available, so we build the matrix
# that 'predictors %*% dgp_coefs' will use to generate dY_t.
# The order must match the 'predictors' vector: Intercept, ECTs, dY_lags, Dummies
dgp_coefs <- rbind(
  mu_hat,       # Intercept (1 x K)
  t(alpha_hat), # ECT coefs (r x K)
  t(gamma_hat), # Lagged diff coefs (K*(p-1) x K)
  t(phi_hat)    # Dummy coefs (n_dummies x K)
)

# Create storage for the selected lag orders across all 5,000 draws
selected_lags <- data.frame(AIC = numeric(n_draws), 
                            SIC = numeric(n_draws), 
                            HQC = numeric(n_draws))

# Create storage for the Mean Squared Errors of the persistence profiles
mse_results <- data.frame(AIC = numeric(n_draws),
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
  
  # Prepare seasonal dummies for the effective sample used in estimation
  sim_quarters_eff <- rep(c(1, 2, 3, 4), length.out = nrow(Y_sim))[(p_max + 1):nrow(Y_sim)]
  D1_eff <- ifelse(sim_quarters_eff == 1, 1, 0) - 0.25
  D2_eff <- ifelse(sim_quarters_eff == 2, 1, 0) - 0.25
  D3_eff <- ifelse(sim_quarters_eff == 3, 1, 0) - 0.25
  dummies_eff <- data.frame(D1_eff, D2_eff, D3_eff)
  
  for (p_test in 1:p_max) {
    
    # Create the dependent variable matrix (Delta Y_t) for the effective sample
    dY_eff <- diff(Y_sim)[(p_max):nrow(diff(Y_sim)), ]
    
    # Create the ECT matrix for the effective sample
    ECT_eff <- (Y_sim %*% beta)[(p_max):(nrow(Y_sim)-1), ]
    ECT_eff_df <- as.data.frame((Y_sim %*% beta)[(p_max):(nrow(Y_sim)-1), ])
    
    # Base formula string
    form_str <- "dY_eff ~ ECT_eff"
    # Base formula string (including dummies, which are part of the DGP)
    form_str <- "dY_eff ~ ."
    
    # If testing lag p > 1, add lagged differences to the regression
    if (p_test > 1) {
      for (lag in 1:(p_test - 1)) {
        assign(paste0("dY_lag", lag), diff(Y_sim)[(p_max - lag):(nrow(diff(Y_sim)) - lag), ])
        form_str <- paste0(form_str, " + dY_lag", lag)
      }
    }
    
    # Combine all regression data into one dataframe
    regression_data_test <- cbind(dY_eff, ECT_eff_df, dummies_eff, lag_data)
    
    # Fit the restricted VECM for this specific lag order
    test_model <- lm(as.formula(form_str))
    test_model <- lm(as.formula(form_str), data = regression_data_test)
    
    # Calculate Residual Covariance Matrix (Sigma_bar)
    Sigma_bar <- cov(residuals(test_model))
    det_Sigma <- det(Sigma_bar)
    
    # Count the number of freely estimated parameters in the system
    # K variables * (1 intercept + 3 ECTs + K*(p_test-1) short-run lags)
    num_params <- K * (1 + 3 + K*(p_test - 1)) 
    # K variables * (1 intercept + 3 ECTs + 3 dummies + K*(p_test-1) short-run lags)
    num_params <- K * (1 + ncol(beta) + 3 + K * (p_test - 1)) 
    
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
  # --- Estimate Persistence Profiles and MSE ---
  # This section re-estimates the model based on the selected lag order for each
  # criterion, calculates the persistence profile, and compares it to the true one.
  
  criteria <- c("AIC", "SIC", "HQC")
  
  for (crit in criteria) {
    p_selected <- selected_lags[[crit]][draw]
    
    # 1. Re-estimate the VECM with the selected lag order 'p_selected'
    # The data construction must match the loop above to ensure constant sample size.
    dY_eff <- diff(Y_sim)[(p_max):nrow(diff(Y_sim)), ]
    ECT_eff <- (Y_sim %*% beta)[(p_max):(nrow(Y_sim)-1), ]
    
    form_str_selected <- "dY_eff ~ ECT_eff"
    
    # Note: We are not including dummies here for simplicity, assuming their effect
    # on the estimated dynamic parameters (alpha, gamma) is secondary for this exercise.
    # For a more rigorous approach, they should be included.
    regression_data <- list(dY_eff = dY_eff, ECT_eff = ECT_eff)
    if (p_selected > 1) {
      for (lag in 1:(p_selected - 1)) {
        lag_name <- paste0("dY_lag", lag)
        regression_data[[lag_name]] <- diff(Y_sim)[(p_max - lag):(nrow(diff(Y_sim)) - lag), ]
        form_str_selected <- paste0(form_str_selected, " + ", lag_name)
      }
    }
    
    selected_model <- lm(as.formula(form_str_selected), data = as.data.frame(regression_data))
    
    # 2. Extract estimated parameters
    model_coefs <- t(coef(selected_model))
    
    # Coefs on ECTs (alpha)
    alpha_est <- model_coefs[, 2:(1 + ncol(beta))]
    
    # Coefs on lagged differences (Gammas)
    gamma_list_est <- list()
    if (p_selected > 1) {
      start_col <- 1 + ncol(beta) + 1 # After Intercept and ECTs
      for (i in 1:(p_selected - 1)) {
        end_col <- start_col + K - 1
        gamma_list_est[[i]] <- model_coefs[, start_col:end_col]
        start_col <- end_col + 1
      }
    }
    
    Sigma_est <- cov(residuals(selected_model))
    
    # 3. Calculate the persistence profile for this draw
    estimated_pp <- calc_persistence_profile(alpha = alpha_est, beta = beta, 
                                             gamma_list = gamma_list_est, Sigma = Sigma_est, horizon = 24)
    # 4. Calculate and store the Mean Squared Error (MSE)
    mse_results[[crit]][draw] <- mean((estimated_pp - true_pp)^2)
  }
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

# --- Summarize MSE Results ---
cat("\n--- Mean Squared Error of Persistence Profiles ---\n")
print(colMeans(mse_results))

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
