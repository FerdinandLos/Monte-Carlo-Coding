import numpy as np

def IRFsign(BETAnc, SIGMA, h, jmax):
    K, n = BETAnc.shape
    p = n // K  # determine number of lags used in original estimation (integer division)
    
    # Pre-allocate the list to hold our valid IRM arrays
    IRMposs_list = []
    
    # A = [BETAnc; eye(K*(p-1),K*(p-1)), zeros(K*(p-1),K)]
    eye_block = np.eye(K * (p - 1))
    zeros_block = np.zeros((K * (p - 1), K))
    bottom_block = np.hstack([eye_block, zeros_block])
    A = np.vstack([BETAnc, bottom_block])
    
    # J = [eye(K,K) zeros(K,K*(p-1))]
    J = np.hstack([np.eye(K), np.zeros((K, K * (p - 1)))])
    
    index = 0
    
    for j in range(jmax):
        # Generate random normal matrix
        newmatrix = np.random.normal(0, 1, (K, K))
        Q, R = np.linalg.qr(newmatrix)
        
        # Rubio-Ramirez QR normalization
        for i in range(K):
            if R[i, i] < 0:
                Q[:, i] = -Q[:, i]
                
        Q = Q.T
        
        # Eigendecomposition
        eigval, eigvec = np.linalg.eig(SIGMA)
        # MATLAB eig() returns a diagonal matrix for eigenvalues; Python returns a 1D array.
        # We must explicitly convert it back to a diagonal matrix for the matrix multiplication.
        P = eigvec @ np.diag(np.sqrt(eigval))
        
        # Pre-allocate the IRM array for speed instead of concatenating in a loop
        IRM = np.zeros((K**2, h + 1))
        
        # Period 0 (A^0 is the identity matrix)
        IRM[:, 0] = (J @ J.T @ P @ Q).flatten(order='F')
        
        # Periods 1 through h
        A_pow = A.copy()
        for i in range(1, h + 1):
            IRM[:, i] = (J @ A_pow @ J.T @ P @ Q).flatten(order='F')
            A_pow = A_pow @ A  # Iterative multiplication is faster than np.linalg.matrix_power
            
        # ---------------------------------------------------------
        # SIGN RESTRICTION PERMUTATION LOGIC
        # Python uses 0-based indexing (MATLAB 1 -> Python 0)
        # Note: min(cumsum(IRM[scalar])) is mathematically just the scalar, 
        # so it simplifies to checking IRM[idx, 0] >= 0.
        # ---------------------------------------------------------
        
        if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] <= 0: # 1st col is supply shock
            if IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col is Flow Dem
                if IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0: 
                    IRMposs_list.append(IRM)
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[0:4, :], IRM[4:8, :], IRM[12:16, :], IRM[8:12, :]]))
                    index += 1
                else: continue
            elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[0:4, :], IRM[8:12, :], IRM[4:8, :], IRM[12:16, :]]))
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[0:4, :], IRM[8:12, :], IRM[12:16, :], IRM[4:8, :]]))
                    index += 1
                else: continue
            elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[0:4, :], IRM[12:16, :], IRM[4:8, :], IRM[8:12, :]]))
                    index += 1
                elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[0:4, :], IRM[12:16, :], IRM[8:12, :], IRM[4:8, :]]))
                    index += 1
                else: continue
            else: continue
                
        elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] <= 0: # 2nd col is supply shock
            if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                if IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[0:4, :], IRM[8:12, :], IRM[12:16, :]]))
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[0:4, :], IRM[12:16, :], IRM[8:12, :]]))
                    index += 1
                else: continue
            elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[8:12, :], IRM[0:4, :], IRM[12:16, :]]))
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[8:12, :], IRM[12:16, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[12:16, :], IRM[0:4, :], IRM[8:12, :]]))
                    index += 1
                elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[4:8, :], IRM[12:16, :], IRM[8:12, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            else: continue
                
        elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] <= 0: # 3rd col is supply shock
            if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[0:4, :], IRM[4:8, :], IRM[12:16, :]]))
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[0:4, :], IRM[12:16, :], IRM[4:8, :]]))
                    index += 1
                else: continue
            elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[4:8, :], IRM[0:4, :], IRM[12:16, :]]))
                    index += 1
                elif IRM[12, 0] >= 0 and IRM[13, 0] <= 0 and IRM[14, 0] >= 0 and IRM[15, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[4:8, :], IRM[12:16, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] >= 0: # 4th col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[12:16, :], IRM[0:4, :], IRM[4:8, :]]))
                    index += 1
                elif IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[8:12, :], IRM[12:16, :], IRM[4:8, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            else: continue
                
        elif IRM[12, 0] >= 0 and IRM[13, 0] >= 0 and IRM[14, 0] <= 0: # 4th col is supply shock
            if IRM[0, 0] >= 0 and IRM[1, 0] >= 0 and IRM[2, 0] >= 0: # 1st col Flow Dem
                if IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[0:4, :], IRM[4:8, :], IRM[8:12, :]]))
                    index += 1
                elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[0:4, :], IRM[8:12, :], IRM[4:8, :]]))
                    index += 1
                else: continue
            elif IRM[4, 0] >= 0 and IRM[5, 0] >= 0 and IRM[6, 0] >= 0: # 2nd col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[4:8, :], IRM[0:4, :], IRM[8:12, :]]))
                    index += 1
                elif IRM[8, 0] >= 0 and IRM[9, 0] <= 0 and IRM[10, 0] >= 0 and IRM[11, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[4:8, :], IRM[8:12, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            elif IRM[8, 0] >= 0 and IRM[9, 0] >= 0 and IRM[10, 0] >= 0: # 3rd col Flow Dem
                if IRM[0, 0] >= 0 and IRM[1, 0] <= 0 and IRM[2, 0] >= 0 and IRM[3, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[8:12, :], IRM[0:4, :], IRM[4:8, :]]))
                    index += 1
                elif IRM[4, 0] >= 0 and IRM[5, 0] <= 0 and IRM[6, 0] >= 0 and IRM[7, 0] >= 0:
                    IRMposs_list.append(np.vstack([IRM[12:16, :], IRM[8:12, :], IRM[4:8, :], IRM[0:4, :]]))
                    index += 1
                else: continue
            else: continue
        else:
            continue
            
    # Final step: Stack the list into a 3D NumPy array
    if len(IRMposs_list) > 0:
        # Stack along the 3rd dimension (axis=2) to perfectly match MATLAB's output format
        return np.stack(IRMposs_list, axis=2)
    else:
        # If no valid rotations were found, return an empty array with correct dimensions
        return np.zeros((K**2, h + 1, 0))