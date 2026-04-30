# Get dataset for Ramey (2011) - Identifying Government Spending Shocks
library(readr)
library(dplyr)
ramey_data_raw <- read_csv("Datasets/Ramey-2011-Identify.csv")
View(ramey_data_raw)

library(dplyr)

# ---------------------------------------------------------
# Step 1: Extract the Raw Data
# ---------------------------------------------------------
ramey_data <- ramey_data_raw[, c("pdvmil", "ngdp...6", "rgov", "rgdp", "tb3", "amtbr", "tothours", "totpop")]
colnames(ramey_data)[2] <- "ngdp" # Rename for easier referencing

# ---------------------------------------------------------
# Step 2: Replicate Ramey's Data Transformations
# ---------------------------------------------------------
# 1. Scale the news variable by lagged nominal GDP
ramey_data$pdvmily <- ramey_data$pdvmil / dplyr::lag(ramey_data$ngdp)
ramey_data$pdvmily[1] <- ramey_data$pdvmil[1] / 89.7  # Ramey's hard-coded fallback for 1939 Q1

# 2. Convert to log per capita
ramey_data$lrgov <- log(ramey_data$rgov / ramey_data$totpop)
ramey_data$lrgdp <- log(ramey_data$rgdp / ramey_data$totpop)
ramey_data$ltothours <- log(ramey_data$tothours / ramey_data$totpop)

# 3. Create the final Y matrix in the exact order of her VAR
Y <- as.matrix(ramey_data[, c("pdvmily", "lrgov", "lrgdp", "tb3", "amtbr", "ltothours")])
T_total <- nrow(Y)
K <- ncol(Y)
p0 <- 4 # True lag order

# ---------------------------------------------------------
# Step 3: Construct the Deterministic Trends (t, t^2)
# ---------------------------------------------------------
t_seq <- 1:T_total
t2_seq <- t_seq^2

# ---------------------------------------------------------
# Step 4: Build the Predictor Matrix for the VAR(4)
# ---------------------------------------------------------
# Because of 4 lags, our effective sample starts at row 5
Y_current <- Y[5:T_total, ]

# Build the deterministic block (Intercept, t, t^2)
X_predictors <- cbind(Intercept = 1, 
                      t = t_seq[5:T_total], 
                      t2 = t2_seq[5:T_total])

# Bind the 4 lags of all 6 variables
for (lag in 1:p0) {
  Y_lagged <- Y[(5 - lag):(T_total - lag), ]
  colnames(Y_lagged) <- paste0(colnames(Y), "_lag", lag)
  X_predictors <- cbind(X_predictors, Y_lagged)
}

# ---------------------------------------------------------
# Step 5: Force-Fit the VAR and Extract DGP Parameters
# ---------------------------------------------------------
# Multivariate OLS regression (Current variables ~ Deterministic + Lags)
dgp_model <- lm(Y_current ~ X_predictors - 1)

# Extract and transpose the coefficients to standard econometric format
# Format: K x (3 deterministic + 24 lagged terms)
B_hat <- t(coef(dgp_model)) 

# Extract the residual covariance matrix
Sigma_hat <- cov(residuals(dgp_model))

# ---------------------------------------------------------
# Step 6: Save the DGP
# ---------------------------------------------------------
save(B_hat, 
     Sigma_hat, 
     Y,          # Needed to draw the initial p0 blocks for simulation
     p0, 
     K, 
     file = "DGPs/ramey_2011_dgp.RData")

print("Ramey DGP successfully fitted and saved!")

