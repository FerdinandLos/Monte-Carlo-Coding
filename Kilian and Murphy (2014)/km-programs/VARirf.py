import numpy as np

def VARirf(BETAnc, SIGMA, h):
    """
    Estimates VAR impulse response function using Cholesky decomposition.
    
    Parameters:
    BETAnc : numpy.ndarray -> Estimates of slope coefficients
    SIGMA : numpy.ndarray -> Structural innovation variance matrix
    h : int -> Horizon for IRF
    
    Returns:
    IRM : numpy.ndarray -> Orthogonalized Impulse Responses (K^2 x h+1)
    K : int -> Number of variables
    """
    K, n = BETAnc.shape
    p = n // K  # determine number of lags used in original estimation
    
    # A = [BETAnc; eye(K*(p-1),K*(p-1)), zeros(K*(p-1),K)]
    eye_block = np.eye(K * (p - 1))
    zeros_block = np.zeros((K * (p - 1), K))
    bottom_block = np.hstack([eye_block, zeros_block])
    A = np.vstack([BETAnc, bottom_block])
    
    # J = [eye(K,K) zeros(K,K*(p-1))]
    J = np.hstack([np.eye(K), np.zeros((K, K * (p - 1)))])
    
    # Python's np.linalg.cholesky returns a lower triangular matrix, 
    # which exactly matches MATLAB's chol(SIGMA)' (upper transposed).
    chol_SIGMA = np.linalg.cholesky(SIGMA)
    
    # Pre-allocate IRM array for speed (replaces MATLAB's eval/concatenation loop)
    IRM = np.zeros((K**2, h + 1))
    
    # --- Period 0 ---
    # J * A^0 * J' simplifies to the identity matrix for the KxK block
    Theta = J @ J.T @ chol_SIGMA
    IRM[:, 0] = Theta.flatten(order='F')
    
    # --- Periods 1 through h ---
    # Running multiplier for A^i. Starts at A^1.
    A_pow = A.copy()
    
    for i in range(1, h + 1):
        Theta = J @ A_pow @ J.T @ chol_SIGMA
        IRM[:, i] = Theta.flatten(order='F')
        
        # Update running matrix power for the next iteration
        A_pow = A_pow @ A 
        
    return IRM, K