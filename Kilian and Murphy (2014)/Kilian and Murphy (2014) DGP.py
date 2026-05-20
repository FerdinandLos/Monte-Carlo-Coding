# Simple file to execute the VAR fitting
import numpy as np
import pandas as pd
import os

# First define a function to fit a VAR with seasonal adjustment (SA) dummy variables
def lsvarcSA2(y, p):
    """
    Estimates a VAR(p) by Ordinary Least Squares (OLS).
    Includes seasonal adjustment (SA) dummy variables.
    *Modified version specifically for BAYESsign.m*
    
    Parameters:
    y : numpy.ndarray of shape (t, K) -> The time series data
    p : int -> The number of lags
    
    Returns:
    A : VAR slope coefficients
    B : Full coefficient matrix (including constants/dummies)
    X : Regressor matrix
    SIGMA : Covariance matrix of residuals
    U : Residuals
    V : Constant term AND Seasonal Dummies (First 12 columns of B)
    """
    # 1. Setup Regressors and Regressand
    t, K = y.shape
    
    # Transpose y to match MATLAB's (K, t) format internally
    y = y.T
    
    # MATLAB: Y = y(:, p:t). Python: p-1 is the start index, t is the exclusive end
    Y = y[:, p-1:t]
    
    # Stack lagged values vertically
    for i in range(1, p):
        Y = np.vstack([Y, y[:, p-1-i : t-i]])
        
    # 2. Create Seasonal Adjustment (SA) Dummies
    # x is a 12x11 matrix: an 11x11 identity matrix on top of a 1x11 row of zeros
    x = np.vstack([np.eye(11), np.zeros((1, 11))])
    
    n_years = int((t - p) // 12)
    remainder = int((t - p) % 12)
    
    # Replicate 'x' for the number of full years
    if n_years > 0:
        X2 = np.tile(x, (n_years, 1))
    else:
        X2 = np.empty((0, 11))
        
    # Add the remainder of the months
    if remainder > 0:
        last = np.hstack([np.eye(remainder), np.zeros((remainder, 11 - remainder))])
        X2 = np.vstack([X2, last])
        
    # Add a column of ones (constant) at the beginning
    X2 = np.hstack([np.ones((t - p, 1)), X2])
    
    # 3. Combine Dummies and Lags to form Regressor Matrix X
    X = np.vstack([X2.T, Y[:, :t-p]])
    Y2 = y[:, p:t]
    
    # 4. Run Least Squares (LS) Regression
    # Python matrix multiplication (@) and inversion
    B = Y2 @ X.T @ np.linalg.pinv(X @ X.T)
    U = Y2 - B @ X
    SIGMA = (U @ U.T) / (t - p - p * K - 12)
    
    # 5. Extract Coefficients
    # THIS IS THE KEY DIFFERENCE FROM lsvarcSA:
    # V now saves the constant AND the 11 seasonal dummies (12 columns total)
    V = B[:, 0:12] 
    
    # A extracts the VAR coefficients (ignoring the first 12 columns)
    A = B[:, 12 : K*p + 12]
    
    return A, B, X, SIGMA, U, V

# 1. Get the exact folder where this specific .py file lives
# This will always resolve to: D:\Monte Carlo Coding\Monte-Carlo-Coding\Kilian and Murphy (2014)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# 2. Join that directory with the subfolder and file names
km_data_path = os.path.join(script_dir, 'km-ascii-data', 'kmData.txt')
world_prod_path = os.path.join(script_dir, 'km-ascii-data', 'worldprod.txt')

# 3. Load the data
km_data_array = np.loadtxt(km_data_path)
world_prod_array = np.loadtxt(world_prod_path)

print("KM Data Array Shape:", km_data_array.shape)
print("World Prod Array Shape:", world_prod_array.shape)
print("Data loaded successfully!")

# Print the first 5 rows of the matrices
print("First 5 rows of KM Data:")
print(km_data_array[:5])
# Note:  km_data_array is in the following order, 
# 1.percent change in global oil production
# 2. real activity index from Kilian(AER 2009), the
# 3. log real price of oil, 
# 4. changes in OECD crude oil inventories
# world_prod_array is global world oil production (thousands of barrels per day)

# Estimate the VAR model using a certain lag order (e.g., p=2)
BETAnc, B, X, SIGMA, U, V = lsvarcSA2(km_data_array, 2)

# --- SUMMARY AND EXPLANATIONS OF ESTIMATED PARAMETERS ---

# Define the variable names in the exact order of the Kilian and Murphy (2014) dataset
var_names = ["Oil Production", "Real Activity", "Real Oil Price", "Inventories"]
p = 2 # Make sure this matches the lag order you passed to lsvarcSA2

print("\n" + "="*50)
print(" VAR MODEL ESTIMATION SUMMARY")
print("="*50)

# -------------------------------------------------------------------------
# 1. SIGMA: The Residual Covariance Matrix
# -------------------------------------------------------------------------
# What it signifies: 
# These are the variances (on the diagonal) and covariances (off-diagonal) 
# of your reduced-form residuals (the "unexplained shocks" U). 
# In SVAR analysis, this matrix is critical. You cannot observe structural 
# economic shocks directly, so you must decompose this SIGMA matrix 
# (using Cholesky decomposition or sign restrictions) to recover them.
sigma_df = pd.DataFrame(SIGMA, index=var_names, columns=var_names)

print("\n### 1. Residual Covariance Matrix (SIGMA) ###")
print("Dimensions:", SIGMA.shape)
print("-" * 50)
print(sigma_df.round(4)) # Rounding to 4 decimals for readability


# -------------------------------------------------------------------------
# 2. V: Deterministic Terms (Constant & Seasonal Dummies)
# -------------------------------------------------------------------------
# What it signifies:
# This captures the baseline levels of your variables that are NOT driven 
# by their past values. 
# - The 'Constant' is the intercept for each equation.
# - The 'Season_Dummy' columns capture repeating monthly fluctuations 
#   (e.g., oil demand might predictably spike every winter).
v_cols = ["Constant"] + [f"Month_{i}_Dummy" for i in range(1, 12)]
v_df = pd.DataFrame(V, index=var_names, columns=v_cols)

print("\n### 2. Deterministic Terms (V) ###")
print("Dimensions:", V.shape)
print("-" * 50)
# Showing just the Constant and first 3 seasonal dummies to save screen space
print(v_df.iloc[:, :4].round(4), "...\n(Showing first 4 of 12 columns)")


# -------------------------------------------------------------------------
# 3. A (or BETAnc): The VAR Lag Coefficients
# -------------------------------------------------------------------------
# What it signifies:
# These are the core dynamic multipliers. They tell you how a change in 
# ONE variable in the past affects ANOTHER variable today. 
# For example, the row "Real Oil Price" and column "L1_Real Activity" tells 
# you how last month's real economic activity impacts this month's oil price.

# Dynamically generate column names for however many lags (p) you used
lag_cols = []
for lag in range(1, p + 1):
    for var in var_names:
        # Abbreviating names slightly to fit on screen
        short_name = var.replace("Production", "Prod").replace("Activity", "Act")
        lag_cols.append(f"Lag{lag}_{short_name}")

a_df = pd.DataFrame(BETAnc, index=var_names, columns=lag_cols)

print("\n### 3. VAR Slope Coefficients (A / BETAnc) ###")
print("Dimensions:", BETAnc.shape)
print("-" * 50)
# Transposing (.T) the DataFrame here just so it prints vertically, 
# which is much easier to read in a terminal than a super wide matrix!
print(a_df.T.round(4)) 
print("="*50 + "\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. DEFINE THE KILIAN & MURPHY (2014) SIGN RESTRICTIONS
# -------------------------------------------------------------------
# Variables (Rows): [Oil Production, Real Activity, Real Oil Price, Inventories]
# Shocks (Columns): [Supply Shock, Flow Demand Shock, Speculative Demand Shock, Residual/Other]
# Note: 1 means > 0, -1 means < 0, np.nan means unrestricted.
# We are modeling a NEGATIVE supply shock (disruption) for easier reading.

sign_matrix = np.array([
    [-1,      1,      1,      np.nan],  # d_Oil_Production
    [-1,      1,     -1,      np.nan],  # Real_Activity
    [ 1,      1,      1,      np.nan],  # Real_Oil_Price
    [np.nan, np.nan,  1,      np.nan]   # d_Inventories
])


# -------------------------------------------------------------------
# 2. FUNCTION TO CALCULATE IMPULSE RESPONSES
# -------------------------------------------------------------------
def compute_structural_irf(A_var, A0, h_max, K, p):
    """
    Converts VAR coefficients into Vector Moving Average (VMA) IRFs.
    """
    # Phi holds the reduced-form VMA coefficients
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    
    # Calculate Phi_h = A_1*Phi_{h-1} + A_2*Phi_{h-2} ...
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            # Extract the j-th lag coefficient matrix from BETAnc
            # Remember BETAnc is shaped (K, K*p)
            A_j = A_var[:, (j-1)*K : j*K]
            Phi[h] += A_j @ Phi[h-j]
            
    # Multiply the VMA coefficients by the structural impact matrix A0
    IRF = np.zeros((h_max, K, K))
    for h in range(h_max):
        IRF[h] = Phi[h] @ A0
        
    return IRF


# -------------------------------------------------------------------
# 3. THE SIGN RESTRICTION ALGORITHM (UPDATED)
# -------------------------------------------------------------------
def draw_sign_restrictions(BETAnc, SIGMA, signs, p, K, h_max=24, n_draws=100):
    P = np.linalg.cholesky(SIGMA)
    
    valid_IRFs = []
    valid_A0s = []  # NEW: Array to store the valid A0 matrices
    attempts = 0
    
    print(f"Searching for {n_draws} valid models... This might take a moment.")
    
    while len(valid_IRFs) < n_draws:
        attempts += 1
        
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        A0 = P @ Q
        
        match = True
        for i in range(K):
            for j in range(K):
                if not np.isnan(signs[i, j]):
                    if np.sign(A0[i, j]) != signs[i, j]:
                        match = False
                        break
            if not match:
                break
                
        if match:
            irf = compute_structural_irf(BETAnc, A0, h_max, K, p)
            valid_IRFs.append(irf)
            valid_A0s.append(A0)  # NEW: Store the valid A0
            
    print(f"Success! Found {n_draws} valid models out of {attempts} random draws.")
    print(f"Acceptance rate: {(n_draws/attempts)*100:.2f}%\n")
    
    return np.array(valid_IRFs), np.array(valid_A0s) # NEW: Return both

# -------------------------------------------------------------------
# 4. EXECUTE AND FIND THE MEDIAN TARGET (A0_true)
# -------------------------------------------------------------------
h_max = 24  
n_draws = 1000 # Increased to 1000 for a more stable median target
K = SIGMA.shape[0]

# Run the algorithm (unpacking both returns)
accepted_irfs, accepted_A0s = draw_sign_restrictions(BETAnc, SIGMA, sign_matrix, p=2, K=K, h_max=h_max, n_draws=n_draws)

# Step 1: Calculate the pointwise median IRF benchmark
median_target_irf = np.median(accepted_irfs, axis=0)

# Step 2: Find the single model closest to the median target
# Sum of squared differences across horizons(1), variables(2), and shocks(3)
distances = np.sum((accepted_irfs - median_target_irf)**2, axis=(1, 2, 3))

# Step 3: Get the index of the model with the minimum distance
best_model_idx = np.argmin(distances)

# Step 4: Lock in your exact True DGP matrices
A0_true = accepted_A0s[best_model_idx]
True_IRF = accepted_irfs[best_model_idx]

print("=== DGP PARAMETERS ESTABLISHED ===")
print("True A0 Matrix Shape:", A0_true.shape)
print("True IRF Shape:", True_IRF.shape)
print("Index of the Median Target Model:", best_model_idx)

# --- EXPORT DGP PARAMETERS FOR MONTE CARLO ---
# Save the essential matrices into a compressed .npz file
export_path = os.path.join(script_dir, 'DGP files', 'true_dgp_parameters.npz')

np.savez(export_path, 
         B_true=BETAnc, 
         SIGMA_true=SIGMA, 
         A0_true=A0_true, 
         True_IRF=True_IRF,
         V_true=V,       # Save the deterministic terms too!
         p_true=p)

print(f"DGP parameters successfully saved to: {export_path}")