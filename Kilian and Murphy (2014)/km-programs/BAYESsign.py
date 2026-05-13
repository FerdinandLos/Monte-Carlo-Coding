import numpy as np
import scipy.io as sio

def lsvarcSA2(y, p):
    """
    STUB FUNCTION: You must replace this with your actual ported VAR function.
    It should return: BETAnc, B, X, SIGMA, U, V
    """
    # Dummy returns to satisfy the unpacker
    t, K = y.shape
    return (np.zeros((t, K)), np.zeros((K, K)), np.zeros((t, K)), 
            np.eye(K), np.zeros((t, K)), np.zeros((K, K)))

def vec(matrix):
    """Mimics MATLAB's vec() by flattening column-major and returning a column vector."""
    return matrix.flatten(order='F')[:, np.newaxis]

def BAYESsign(y, h, q, p, n1, n2, ProdMBPM, OECDCrudeDif):
    
    t, K = y.shape # q and K are the same based on MATLAB comments
    IRMposs_list = [] # We use a list to append successful draws (faster than dynamic array expansion)
    
    BETAnc, B, X, SIGMA, U, V = lsvarcSA2(y, 24)
    
    # J = [eye(K,K) zeros(K,K*(p-1))]
    J = np.hstack([np.eye(K), np.zeros((K, K * (p - 1)))])
    
    pXX = np.linalg.inv((X @ X.T) / (t - p))
    
    index = 0 # 0-based indexing in Python
    
    for r in range(n1):
        
        # MATLAB: chol(inv(SIGMA)) -> Upper triangular
        # Python: np.linalg.cholesky returns lower, so we transpose (.T)
        inv_SIGMA = np.linalg.inv(SIGMA)
        chol_inv_SIGMA = np.linalg.cholesky(inv_SIGMA).T
        
        RANTR = np.random.randn(t - p, q) / np.sqrt(t - p) @ chol_inv_SIGMA
        SIGMAr = np.linalg.inv(RANTR.T @ RANTR)
        
        Bvec = np.vstack([vec(V), vec(BETAnc[:, :q])])
        for l in range(1, p): # MATLAB: 1:p-1 -> Python: 1 to p-1
            Bvec = np.vstack([Bvec, vec(BETAnc[:, q*l : q*(l+1)])])
            
        kron_mat = np.kron(pXX / (t - p), SIGMAr)
        chol_kron = np.linalg.cholesky(kron_mat).T # Upper triangular
        
        # vecAr = Bvec + (chol(kron(...)))' * randn(...)
        vecAr = Bvec + chol_kron.T @ np.random.randn(q * 12 + q * q * p, 1)
        
        # Ar construction
        top_block_vec = vecAr[12*q : 12*q + (q**2)*p]
        top_block = np.reshape(top_block_vec, (q, q * p), order='F')
        
        bottom_left = np.eye(q*p - q)
        bottom_right = np.zeros((q*p - q, q))
        bottom_block = np.hstack([bottom_left, bottom_right])
        
        Ar = np.vstack([top_block, bottom_block])
        
        # Compute sign IRFs
        for i in range(n2):
            
            newmatrix = np.random.normal(0, 1, (q, q))
            Q, R = np.linalg.qr(newmatrix)
            for ii in range(q):
                if R[ii, ii] < 0:
                    Q[:, ii] = -Q[:, ii]
                    
            eigval, eigvec = np.linalg.eig(SIGMAr)
            # MATLAB eig returns diagonal matrix for values, Python returns 1D array.
            # P = eigvec * sqrt(diag(eigval))
            P = eigvec @ np.diag(np.sqrt(eigval))
            
            # compute impulse response
            IRM = np.zeros((K**2, h + 1))
            
            # Ar^0 is identity matrix
            IRM[:, 0] = (J @ np.eye(q * p) @ J.T @ P @ Q).flatten(order='F')
            
            for j_idx in range(1, h + 1):
                Ar_j = np.linalg.matrix_power(Ar, j_idx)
                IRM[:, j_idx] = (J @ Ar_j @ J.T @ P @ Q).flatten(order='F')
            
            # ---------------------------------------------------------
            # THE SIGN RESTRICTION PERMUTATION LOGIC
            # Note: Python uses 0-based indexing (MATLAB 1 -> Python 0)
            # ---------------------------------------------------------
            IRMint = None
            
            if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] <= 0: # 1st col is supply shock
                if IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col is Flow Dem
                    if IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0: 
                        IRMint = IRM
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[0:4, :], IRM[4:8, :], IRM[12:16, :], IRM[8:12, :]])
                    else: continue
                elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                    if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[0:4, :], IRM[8:12, :], IRM[4:8, :], IRM[12:16, :]])
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[0:4, :], IRM[8:12, :], IRM[12:16, :], IRM[4:8, :]])
                    else: continue
                elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                    if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[0:4, :], IRM[12:16, :], IRM[4:8, :], IRM[8:12, :]])
                    elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                        IRMint = np.vstack([IRM[0:4, :], IRM[12:16, :], IRM[8:12, :], IRM[4:8, :]])
                    else: continue
                else: continue
                    
            elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] <= 0: # 2nd col is supply shock
                if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                    if IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[0:4, :], IRM[8:12, :], IRM[12:16, :]])
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[0:4, :], IRM[12:16, :], IRM[8:12, :]])
                    else: continue
                elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[8:12, :], IRM[0:4, :], IRM[12:16, :]])
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[8:12, :], IRM[12:16, :], IRM[0:4, :]])
                    else: continue
                elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[12:16, :], IRM[0:4, :], IRM[8:12, :]])
                    elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                        IRMint = np.vstack([IRM[4:8, :], IRM[12:16, :], IRM[8:12, :], IRM[0:4, :]])
                    else: continue
                else: continue
                    
            elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] <= 0: # 3rd col is supply shock
                if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                    if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[0:4, :], IRM[4:8, :], IRM[12:16, :]])
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[0:4, :], IRM[12:16, :], IRM[4:8, :]])
                    else: continue
                elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[4:8, :], IRM[0:4, :], IRM[12:16, :]])
                    elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[4:8, :], IRM[12:16, :], IRM[0:4, :]])
                    else: continue
                elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[12:16, :], IRM[0:4, :], IRM[4:8, :]])
                    elif IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[8:12, :], IRM[12:16, :], IRM[4:8, :], IRM[0:4, :]])
                    else: continue
                else: continue
                    
            elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] <= 0: # 4th col is supply shock
                if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                    if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[0:4, :], IRM[4:8, :], IRM[8:12, :]])
                    elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[0:4, :], IRM[8:12, :], IRM[4:8, :]])
                    else: continue
                elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[4:8, :], IRM[0:4, :], IRM[8:12, :]])
                    elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[4:8, :], IRM[8:12, :], IRM[0:4, :]])
                    else: continue
                elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                    if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[8:12, :], IRM[0:4, :], IRM[4:8, :]])
                    elif IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                        IRMint = np.vstack([IRM[12:16, :], IRM[8:12, :], IRM[4:8, :], IRM[0:4, :]])
                    else: continue
                else: continue
            else:
                continue
            
            # --- Elasticity checks ---
            if IRMint is not None:
                IRprod = IRMint[0, 0]
                IRprice = IRMint[2, 0]
                IRinv = IRMint[3, 0]
                
                FlowNew = ProdMBPM * (1 + IRprod / 100) - np.mean(OECDCrudeDif) - IRinv
                Flow = ProdMBPM - np.mean(OECDCrudeDif)
                PctChange = 100 * (FlowNew - Flow) / Flow
                ElasUse = PctChange / IRprice
                elasuse = np.mean(ElasUse)
                
                SupplyelasAD = IRMint[4, 0] / IRMint[6, 0] 
                SupplyelasPD = IRMint[8, 0] / IRMint[10, 0] 
                
                # Check bounds and horizon restrictions
                # 0:12 in Python targets indices 0 through 11 (the first 12 elements, matching MATLAB 1:12)
                if (SupplyelasAD < 0.1 and SupplyelasPD < 0.1 and 
                    elasuse <= 0 and elasuse > -0.8 and 
                    np.min(np.cumsum(IRMint[0, 0:12])) >= 0 and 
                    np.min(IRMint[1, 0:12]) >= 0 and 
                    np.max(IRMint[2, 0:12]) <= 0):
                    
                    IRMposs_list.append(IRMint)
                    index += 1
        
        print(f"r: {r + 1}")
        print(f"index: {index}")
        
        # In MATLAB this continuously overwrote the file. We will stack the list here just to save progress.
        if len(IRMposs_list) > 0:
            IRMposs_stacked = np.stack(IRMposs_list, axis=2)
            sio.savemat('BayesUpdate.mat', {'IRMposs': IRMposs_stacked, 'r': r + 1, 'index': index})

    # Final Stack and Return
    if len(IRMposs_list) > 0:
        return np.stack(IRMposs_list, axis=2)
    else:
        return np.zeros((K**2, h + 1, 0)) # Return empty 3D array if no matches