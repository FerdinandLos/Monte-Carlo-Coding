# Get dataset for Johansen Juselius (1990)
library(urca)
data("finland")
View(finland)

# 1. Load Data
library(urca)
data(finland)

# Extract the 4 variables
# Ordered as: lrm1 (log money), lny (log income), lnmr (marginal rate of interest), difp (inflation)
Y <- as.matrix(finland[, c("lrm1", "lny", "lnmr", "difp")])
T_total <- nrow(Y)

# 2. Impose the Theoretical Cointegrating Vectors (beta)
beta <- matrix(c( 1, -1,  0,  0,  # CV1: m1 - y
                  0,  0,  1,  0,  # CV2: im is stationary
                  0,  0,  0,  1), # CV3: dp is stationary
               nrow = 4, ncol = 3, byrow = FALSE)

# 3. Construct the Error Correction Terms (ECT)
# ECT_t = m1_t - y_t (how much money demand deviates from income, in long run they should be equal)
# ECT_t = Y_t %*% beta = residual of the long run relationship at time t
ECT <- Y %*% beta

# 4. Prepare the Differenced Data and Lags for p0 = 2
dY <- diff(Y) # Difference to get growth rates
dY_lag1 <- dY[1:(nrow(dY)-1), ] # Delta X_{t-1}
ECT_lag1 <- ECT[2:(nrow(ECT)-1), ] # ECT_{t-1}
dY_current <- dY[2:nrow(dY), ] # Delta X_t

# Determine seasonal dummies (quarterly data)
# Data starts in Q2, so Y_current starts in Q4
N_est <- nrow(dY_current) #no. obs.
quarter_seq <- rep(c(4, 1, 2, 3), length.out = N_est) # repeating quarters
Q1 <- ifelse(quarter_seq == 1, 1, 0) # three dummies
Q2 <- ifelse(quarter_seq == 2, 1, 0)
Q3 <- ifelse(quarter_seq == 3, 1, 0)
# By subtracting 1/4 (0.25) from each, the sum of any 4-quarter block is exactly 0.
# Important for interpretabilit of intercept as baseline level of growth.
# Average of dummies becomes 0 and not 0.25
Q1_c <- Q1 - 0.25
Q2_c <- Q2 - 0.25
Q3_c <- Q3 - 0.25
dummies <- cbind(Q1_c, Q2_c, Q3_c)

# 5. Force-Fit the VECM (The DGP Creation)
# Regress current differences on lagged ECT, lagged differences, and deterministic terms
dgp_model <- lm(dY_current ~ ECT_lag1 + dY_lag1 + dummies) # Add + dummies if constructed

# 6. Extract the Locked-in Parameters
# 6. Extract the Locked-in Parameters
# Add t() to transpose them into proper econometric column-vector format!
alpha_hat <- t(coef(dgp_model)[2:4, ])     # Now a 4x3 matrix
gamma_hat <- t(coef(dgp_model)[5:8, ])     # Now a 4x4 matrix
mu_hat <- as.numeric(coef(dgp_model)[1, ]) # Intercepts
Sigma_hat <- cov(residuals(dgp_model))     # 4x4 Innovation covariance matrix

save(mu_hat,
     alpha_hat, 
     beta, 
     gamma_hat, 
     Sigma_hat, 
     Y,        # Needed to draw the initial p0 values for each simulation
     file = "Monte-Carlo-Coding/DGPs/Johansen Juselius (1990) DGP.RData")
