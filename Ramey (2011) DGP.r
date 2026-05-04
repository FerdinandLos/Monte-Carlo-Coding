# Get dataset for Ramey (2011) - Identifying Government Spending Shocks
library(readr)
library(dplyr)
ramey_data_raw <- read_csv("Datasets/Ramey-2011-Identify.csv")
View(ramey_data_raw)

library(dplyr)

# ---------------------------------------------------------
# Step 1: Extract the Raw Data
# ---------------------------------------------------------
ramey_data <- ramey_data_raw[, c("pdvmil", "ngdp...6", "rgov", 
"rgdp", "tb3", "amtbr", "tothours", "totpop")]
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
     file = "DGPs/ramey_2011_dgp_1.RData")

print("Ramey DGP successfully fitted and saved!")


# ---------------------------------------------------------
# Step 1: Extract and Initialize the Data
# ---------------------------------------------------------
# Assume 'full_data' is loaded and filtered for quarter >= 1947
ramey_spf_data <- ramey_data_raw %>%
  select(quarter, rdef, rfed, rgov, rgdp, tb3, amtbr, tothours, totpop,
         spf_ndef1, spf_ndef0, spf_pgdp1, spf_pgdp0, 
         spf_rfed1, spf_rfed0) %>%
  arrange(quarter)

# Convert to numeric
ramey_spf_data <- ramey_spf_data %>%
  mutate(across(-quarter, as.numeric))
T_total <- nrow(ramey_spf_data)

# ---------------------------------------------------------
# Step 2: Replicate the SPF Shock Construction
# ---------------------------------------------------------
# 1. Calculate Implied Growth Forecasts
ramey_spf_data <- ramey_spf_data %>%
  mutate(
    fdlrdef1 = log(spf_ndef1 / spf_pgdp1) - log(spf_ndef0 / spf_pgdp0),
    fdlrfed1 = log(spf_rfed1 / spf_rfed0)
  )

# 2. Calculate Forecast Errors (The Shocks)
# We use lag() to pull the forecast made in the PREVIOUS quarter
ramey_spf_data <- ramey_spf_data %>%
  mutate(
    spfrdefshock1 = log(rdef / dplyr::lag(rdef)) - dplyr::lag(fdlrdef1),
    spfrfedshock1 = log(rfed / dplyr::lag(rfed)) - dplyr::lag(fdlrfed1)
  )

# 3. Splice them together to create spfshock1
ramey_spf_data <- ramey_spf_data %>%
  mutate(
    spfshock1 = ifelse(is.na(spfrdefshock1), spfrfedshock1, spfrdefshock1)
  )

# ---------------------------------------------------------
# Step 3: Replicate Ramey's Data Transformations
# ---------------------------------------------------------
ramey_spf_data <- ramey_spf_data %>%
  mutate(
    lrgov = log(rgov / totpop),
    lrgdp = log(rgdp / totpop),
    ltothours = log(tothours / totpop)
  )

# Create the final Y matrix in the exact order of her VAR
# We use drop_na() because the lag() operations created NAs in the first rows
Y_spf <- ramey_spf_data %>%
  select(spfshock1, lrgov, lrgdp, tb3, amtbr, ltothours) %>%
  tidyr::drop_na() %>%
  as.matrix()

T_eff <- nrow(Y_spf)
K <- ncol(Y_spf)
p0 <- 4 # True lag order

# ---------------------------------------------------------
# Step 4: Construct the Deterministic Trends
# ---------------------------------------------------------
t_seq <- 1:T_eff
t2_seq <- t_seq^2

# ---------------------------------------------------------
# Step 5: Build the Predictor Matrix for the VAR(4)
# ---------------------------------------------------------
Y_current <- Y_spf[5:T_eff, ]

X_predictors <- cbind(Intercept = 1, 
                      t = t_seq[5:T_eff], 
                      t2 = t2_seq[5:T_eff])

for (lag in 1:p0) {
  Y_lagged <- Y_spf[(5 - lag):(T_eff - lag), ]
  colnames(Y_lagged) <- paste0(colnames(Y_spf), "_lag", lag)
  X_predictors <- cbind(X_predictors, Y_lagged)
}

# ---------------------------------------------------------
# Step 6: Force-Fit the VAR and Extract DGP Parameters
# ---------------------------------------------------------
dgp_model_spf <- lm(Y_current ~ X_predictors - 1)

B_hat_spf <- t(coef(dgp_model_spf)) 
Sigma_hat_spf <- cov(residuals(dgp_model_spf))

# ---------------------------------------------------------
# Step 7: Save the DGP
# ---------------------------------------------------------
save(B_hat_spf, 
     Sigma_hat_spf, 
     Y_spf,      # Needed for initial p0 blocks
     p0, 
     K, 
     file = "DGPs/ramey_2011_spf_dgp.RData")

print("Ramey SPF-based DGP successfully fitted and saved!")

