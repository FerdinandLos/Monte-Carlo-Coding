import numpy as np

def VARdecomp(BETAnc, Btilda, h):
    """
    Estimates VAR Forecast Error Variance Decomposition (FEVD).
    
    Parameters:
    BETAnc : numpy.ndarray -> Estimates of slope coefficients
    Btilda : numpy.ndarray -> Structural impact matrix (identification matrix)
    h : int -> Horizon for Variance Decomposition
    
    Returns:
    VC : numpy.ndarray -> Variance decomposition in percentage terms (K x K).
                          Rows = variables, Columns = shocks.
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
    
    # Initial period (h=1, equivalent to MATLAB i=1/A^0)
    # J * A^0 * J' simplifies to the identity matrix for the KxK block
    TH1 = J @ J.T 
    TH = TH1 @ Btilda
    TH = TH.T
    TH2 = np.square(TH)  # Element-wise square (TH .* TH)
    TH3 = TH2.copy()
    
    # Running multiplier for A^(i-1). Starts at A^1.
    A_pow = A.copy()
    
    # Loop from 2 through h
    for i in range(2, h + 1):
        # TH = J * A^(i-1) * J' * Btilda
        TH = J @ A_pow @ J.T @ Btilda
        TH = TH.T
        TH2 = np.square(TH)
        TH3 += TH2
        
        # Update running matrix power for the next iteration
        A_pow = A_pow @ A 
        
    # TH4 = sum(TH3) -> sums down the columns (axis=0)
    TH4 = np.sum(TH3, axis=0)
    
    # NumPy broadcasting replaces the need for the j=1:K loop
    # This divides each row of TH3 by the corresponding column sums in TH4
    VC = TH3 / TH4
    
    # Display VDC in percentage terms at horizon h, K x K matrix.
    # Columns refer to shocks j=1,...,K that explain any given variable
    # Rows refer to variables whose variation is to be explained
    VC = VC.T * 100
    
    return VC, K