# Simple file to execute the VAR fitting
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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
    V = B[:, 0:12] 
    
    # A extracts the VAR coefficients (ignoring the first 12 columns)
    A = B[:, 12 : K*p + 12]
    
    return A, B, X, SIGMA, U, V

# 1. Get the exact folder where this specific .py file lives
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

# Estimate the VAR model using a certain lag order (e.g., p=2)
# NOTE: Swapped BETAnc to A
A, B, X, SIGMA, U, V = lsvarcSA2(km_data_array, 2)

# --- SUMMARY AND EXPLANATIONS OF ESTIMATED PARAMETERS ---
var_names = ["Oil Production", "Real Activity", "Real Oil Price", "Inventories"]
p = 2 

print("\n" + "="*50)
print(" VAR MODEL ESTIMATION SUMMARY")
print("="*50)

# 1. SIGMA
sigma_df = pd.DataFrame(SIGMA, index=var_names, columns=var_names)
print("\n### 1. Residual Covariance Matrix (SIGMA) ###")
print("Dimensions:", SIGMA.shape)
print("-" * 50)
print(sigma_df.round(4)) 

# 2. V
v_cols = ["Constant"] + [f"Month_{i}_Dummy" for i in range(1, 12)]
v_df = pd.DataFrame(V, index=var_names, columns=v_cols)
print("\n### 2. Deterministic Terms (V) ###")
print("Dimensions:", V.shape)
print("-" * 50)
print(v_df.iloc[:, :4].round(4), "...\n(Showing first 4 of 12 columns)")

# 3. A (The VAR Lag Coefficients)
lag_cols = []
for lag in range(1, p + 1):
    for var in var_names:
        short_name = var.replace("Production", "Prod").replace("Activity", "Act")
        lag_cols.append(f"Lag{lag}_{short_name}")

a_df = pd.DataFrame(A, index=var_names, columns=lag_cols)

print("\n### 3. VAR Slope Coefficients (A) ###")
print("Dimensions:", A.shape)
print("-" * 50)
print(a_df.T.round(4)) 
print("="*50 + "\n")

# -------------------------------------------------------------------
# 1. DEFINE THE KILIAN & MURPHY (2014) SIGN RESTRICTIONS
# -------------------------------------------------------------------
sign_matrix = np.array([
    [-1,      1,      1,      np.nan],  # d_Oil_Production
    [-1,      1,     -1,      np.nan],  # Real_Activity
    [ 1,      1,      1,      np.nan],  # Real_Oil_Price
    [np.nan, np.nan,  1,      np.nan]   # d_Inventories
])

# -------------------------------------------------------------------
# 2. FUNCTION TO CALCULATE IMPULSE RESPONSES
# -------------------------------------------------------------------
def compute_structural_irf(A, B_tilde, h_max, K, p):
    """
    Converts VAR coefficients into Vector Moving Average (VMA) IRFs.
    """
    Phi = np.zeros((h_max, K, K))
    Phi[0] = np.eye(K)
    
    for h in range(1, h_max):
        for j in range(1, min(h, p) + 1):
            A_j = A[:, (j-1)*K : j*K]
            Phi[h] += A_j @ Phi[h-j]
            
    IRF = np.zeros((h_max, K, K))
    for h in range(h_max):
        IRF[h] = Phi[h] @ B_tilde
        
    return IRF

# -------------------------------------------------------------------
# 3. THE SIGN RESTRICTION ALGORITHM (WITH ELASTICITY BOUNDS)
# -------------------------------------------------------------------
def draw_sign_restrictions(A, SIGMA, signs, p, K, Q_avg, h_max=24, n_draws=1000):
    P = np.linalg.cholesky(SIGMA)
    
    valid_IRFs = []
    valid_B_tildes = [] 
    attempts = 0
    
    print(f"Searching for {n_draws} valid models... This might take a moment.")
    
    while len(valid_IRFs) < n_draws:
        attempts += 1
        
        # 1. Draw and rotate
        W = np.random.randn(K, K)
        Q, R = np.linalg.qr(W)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        B_tilde = P @ Q
        
        # 2. Check Sign Restrictions
        match = True
        for i in range(K):
            for j in range(K):
                if not np.isnan(signs[i, j]):
                    if np.sign(B_tilde[i, j]) != signs[i, j]:
                        match = False
                        break
            if not match:
                break
                
        # 3. Check Elasticity Bounds (Only if signs matched!)
        if match:
            # --- A. Supply Elasticity ---
            # Ratio of Oil Prod response (row 0) to Price response (row 2) 
            # following a Flow Demand Shock (column 1)
            supply_elasticity = B_tilde[0, 1] / B_tilde[2, 1]
            
            # --- B. Demand Elasticity in Use ---
            # Calculated following a Supply Shock (column 0)
            # Formula from K&M (2014) Appendix:
            # ((Q * d_Prod / 100) - d_Inv) / (Q * d_Price / 100)
            numerator = (Q_avg * (B_tilde[0, 0] / 100)) - B_tilde[3, 0]
            denominator = Q_avg * (B_tilde[2, 0] / 100)
            demand_elasticity = numerator / denominator
            
            # The model is only valid if it satisfies BOTH bounds
            if (0 <= supply_elasticity <= 0.025) and (-0.8 <= demand_elasticity <= 0.0):
                irf = compute_structural_irf(A, B_tilde, h_max, K, p)
                
                # To plot permanent level shifts, we must cumulatively sum the 
                # differences for Oil Production (index 0) and Inventories (index 3)
                irf_cumulative = irf.copy()
                irf_cumulative[:, 0, :] = np.cumsum(irf[:, 0, :], axis=0)
                irf_cumulative[:, 3, :] = np.cumsum(irf[:, 3, :], axis=0)
                
                valid_IRFs.append(irf_cumulative)
                valid_B_tildes.append(B_tilde) 
                
                # Print progress tracker so you know it hasn't frozen!
                if len(valid_IRFs) % 100 == 0:
                    print(f"Found {len(valid_IRFs)} / {n_draws} valid models...")
            
    print(f"\nSuccess! Found {n_draws} valid models out of {attempts} random draws.")
    print(f"Acceptance rate: {(n_draws/attempts)*100:.4f}%")
    
    return np.array(valid_IRFs), np.array(valid_B_tildes)

# -------------------------------------------------------------------
# 4. EXECUTE AND FIND THE MEDIAN TARGET (B_tilde_true)
# -------------------------------------------------------------------
# IMPORTANT: Change p to 24 to capture long-term market dynamics!
p_true = 24 
h_max = 24  
n_draws = 1000 

# Re-estimate the OLS VAR using 24 lags instead of 2
print(f"\nRe-estimating VAR with {p_true} lags...")
A_24, B_24, X_24, SIGMA_24, U_24, V_24 = lsvarcSA2(km_data_array, p_true)
K = SIGMA_24.shape[0]

# Calculate historical average of global oil production for the elasticity formula
Q_avg = np.mean(world_prod_array)

# Run the algorithm using the 24-lag matrices and the Q_avg
accepted_irfs, accepted_B_tildes = draw_sign_restrictions(
    A=A_24, 
    SIGMA=SIGMA_24, 
    signs=sign_matrix, 
    p=p_true, 
    K=K, 
    Q_avg=Q_avg, 
    h_max=h_max, 
    n_draws=n_draws
)

# Step 1: Calculate the Pointwise Median (The invalid benchmark)
pointwise_median_irf = np.median(accepted_irfs, axis=0)

# Step 2: Find the distance of all real models to that benchmark
distances = np.sum((accepted_irfs - pointwise_median_irf)**2, axis=(1, 2, 3))
best_model_idx = np.argmin(distances)

# Step 3: Lock in your exact True DGP matrices (The Median Target Model)
B_tilde_true = accepted_B_tildes[best_model_idx]
True_IRF = accepted_irfs[best_model_idx]

print("\n=== DGP PARAMETERS ESTABLISHED ===")
print("True B_tilde Matrix Shape:", B_tilde_true.shape)
print("True IRF Shape:", True_IRF.shape)

# --- EXPORT DGP PARAMETERS FOR MONTE CARLO ---
export_folder = os.path.join(script_dir, 'DGP files')
os.makedirs(export_folder, exist_ok=True)
export_path = os.path.join(export_folder, 'true_dgp_parameters.npz')

np.savez(export_path, 
         A_true=A_24,            
         SIGMA_true=SIGMA_24,    
         B_tilde_true=B_tilde_true, 
         True_IRF=True_IRF,   
         V_true=V_24,            
         p_true=p_true)            

print(f"DGP parameters successfully saved to: {export_path}")



# --- VISUALIZE THE TRUE DGP PARAMETERS ---
export_folder_viz = os.path.join(script_dir, 'DGP files', 'Visualizations')
print("Generating visualizations...")

# 1. Plot the True Impulse Response Functions (IRFs)
fig_irf, axes_irf = plt.subplots(nrows=K, ncols=K, figsize=(15, 12))
shock_names = ["Supply Shock", "Flow Demand", "Spec. Demand", "Residual Shock"]

for i in range(K): # Responding variable (Rows)
    for j in range(K): # Shock (Columns)
        # Plot the line
        axes_irf[i, j].plot(True_IRF[:, i, j], color='darkblue', linewidth=2)
        # Add a zero line for reference
        axes_irf[i, j].axhline(0, color='black', linestyle='--', linewidth=1)
        
        # Add titles to the top row and labels to the left column
        if i == 0:
            axes_irf[i, j].set_title(f"Shock: {shock_names[j]}", fontweight='bold')
        if j == 0:
            axes_irf[i, j].set_ylabel(f"Response:\n{var_names[i]}", fontweight='bold')
            
        axes_irf[i, j].grid(alpha=0.3)

fig_irf.suptitle("True Structural Impulse Responses (Median Target Model)", fontsize=16, y=1.02)
plt.tight_layout()

# Save and show the IRF plot
irf_plot_path = os.path.join(export_folder_viz, 'True_IRF_plot.png')
plt.savefig(irf_plot_path, bbox_inches='tight', dpi=300)
plt.show()

# --- TEXT SUMMARY OF TRUE DGP PARAMETERS ---
print("\n" + "="*50)
print(" TRUE DGP PARAMETERS EXPORT SUMMARY")
print("="*50)

# 1. SIGMA_true
sigma_true_df = pd.DataFrame(SIGMA_true, index=var_names, columns=var_names)
print("\n### 1. True Residual Covariance Matrix (SIGMA_true) ###")
print("Dimensions:", SIGMA.shape)
print("-" * 50)
print(sigma_true_df.round(4)) 

# 2. V_true
v_cols = ["Constant"] + [f"Month_{i}_Dummy" for i in range(1, 12)]
v_true_df = pd.DataFrame(V, index=var_names, columns=v_cols)
print("\n### 2. True Deterministic Terms (V_true) ###")
print("Dimensions:", V.shape)
print("-" * 50)
print(v_true_df.iloc[:, :4].round(4), "...\n(Showing first 4 of 12 columns)")

# 3. A_true
lag_cols = []
for lag in range(1, p + 1):
    for var in var_names:
        short_name = var.replace("Production", "Prod").replace("Activity", "Act")
        lag_cols.append(f"Lag{lag}_{short_name}")

a_true_df = pd.DataFrame(A, index=var_names, columns=lag_cols)
print("\n### 3. True VAR Slope Coefficients (A_true) ###")
print("Dimensions:", A.shape)
print("-" * 50)
print(a_true_df.T.round(4)) 

# 4. B_tilde_true (Structural Impact Matrix)
shock_names = ["Supply Shock", "Flow Demand", "Spec. Demand", "Residual Shock"]
b_tilde_df = pd.DataFrame(B_tilde_true, index=var_names, columns=shock_names)

print("\n### 4. True Structural Impact Matrix (B_tilde_true) ###")
print("Dimensions:", B_tilde_true.shape)
print("-" * 50)
print(b_tilde_df.round(4))
print("="*50 + "\n")