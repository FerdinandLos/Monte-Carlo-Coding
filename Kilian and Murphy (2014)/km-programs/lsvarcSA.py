import numpy as np

def lsvarcSA(y, p):
    """
    Estimates a VAR(p) by Ordinary Least Squares (OLS).
    Includes seasonal adjustment (SA) dummy variables.
    
    Parameters:
    y : numpy.ndarray of shape (t, K) -> The time series data
    p : int -> The number of lags
    
    Returns:
    A : VAR slope coefficients
    B : Full coefficient matrix (including constants/dummies)
    X : Regressor matrix
    SIGMA : Covariance matrix of residuals
    U : Residuals
    V : Constant terms
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
    # Y[:, :t-p] matches MATLAB's Y(:, 1:t-p)
    X = np.vstack([X2.T, Y[:, :t-p]])
    
    # Redefine Y as the target variable (next step)
    Y_target = y[:, p:t]
    
    # 4. Run Least Squares (LS) Regression
    # MATLAB: B = (Y * X') / (X * X')
    B = Y_target @ X.T @ np.linalg.inv(X @ X.T)
    
    # Residuals
    U = Y_target - B @ X
    
    # Structural Innovation Variance Matrix
    SIGMA = (U @ U.T) / (t - p - p * K - 1)
    
    # Extract coefficients
    # V is the constant (first column). Slicing 0:1 keeps it as a 2D column vector
    V = B[:, 0:1] 
    
    # A extracts the VAR coefficients (ignoring the 1 constant + 11 dummies)
    A = B[:, 12 : K*p + 12]
    
    return A, B, X, SIGMA, U, V