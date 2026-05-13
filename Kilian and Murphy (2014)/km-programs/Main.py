import numpy as np

import scipy.io as sio

# --- Load your custom ported functions ---
# This assumes lsvarcSA.py, VARirf.py, IRFsign.py, and VARdecomp.py are in the same folder
from lsvarcSA import lsvarcSA
from VARirf import VARirf
from IRFsign import IRFsign
from VARdecomp import VARdecomp

if __name__ == "__main__":
    
    # 1. Initialization and Data Loading
    print("Loading data and estimating VAR...")
    mat_km = sio.loadmat('kmData.mat')
    kmData = mat_km['kmData']
    
    # Estimate the VAR model
    BETAnc, B, X, SIGMA, U, V = lsvarcSA(kmData, 24)
    
    xmax = 17
    jmax = 5000000
    
    # Set random seed (MATLAB: randn('state', 316))
    np.random.seed(316)
    
    # 2. Impulse Responses
    print("Computing Cholesky IRFs...")
    IRFaer, K = VARirf(BETAnc, SIGMA, xmax)
    
    print(f"Computing Sign-Restricted IRFs (up to {jmax} rotations)...")
    # Note: Depending on jmax, this step will take some time
    IRFposs = IRFsign(BETAnc, SIGMA, xmax, jmax)
    
    j_dim, k_dim, l_dim = IRFposs.shape
    
    mat_world = sio.loadmat('worldprod.mat')
    worldprod = mat_world['worldprod'].flatten()
    
    # MATLAB 2:end -> Python 1:
    ProdMBPM = worldprod[1:] * 30 / 1000
    OECDCrudeDif = kmData[:, 3] # Column 4 in MATLAB is index 3 in Python
    
    # 3. Imposing Additional Restrictions
    print("Filtering admissible IRFs based on elasticity constraints...")
    IRFelas_list = []
    elasuse_list = []
    
    # Elasticity vectors across the 3rd dimension (l_dim)
    # MATLAB: 9 -> 8, 11 -> 10, 5 -> 4, 7 -> 6
    elasticity = IRFposs[8, 0, :] / IRFposs[10, 0, :] 
    ADelas = IRFposs[4, 0, :] / IRFposs[6, 0, :]
    
    mean_OECDCrudeDif = np.mean(OECDCrudeDif)
    
    for i in range(l_dim):
        # Elasticity in use
        IRprod = IRFposs[0, 0, i]
        IRprice = IRFposs[2, 0, i]
        IRinv = IRFposs[3, 0, i]
        
        FlowNew = ProdMBPM * (1 + IRprod / 100) - mean_OECDCrudeDif - IRinv
        Flow = ProdMBPM - mean_OECDCrudeDif
        PctChange = 100 * (FlowNew - Flow) / Flow
        ElasUseSeries = PctChange / IRprice
        
        mean_elas = np.mean(ElasUseSeries)
        
        # Check constraints (Indices shifted by -1 for Python)
        if (elasticity[i] <= 0.0258 and 
            ADelas[i] <= 0.0258 and 
            mean_elas <= 0 and 
            np.min(np.cumsum(IRFposs[0, 0:12, i])) >= 0 and 
            np.min(IRFposs[1, 0:12, i]) >= 0 and 
            np.max(IRFposs[2, 0:12, i]) <= 0):
            
            IRFelas_list.append(IRFposs[:, :, i])
            elasuse_list.append(mean_elas)
            
    # Stack into final 3D array
    if len(IRFelas_list) > 0:
        IRFelas = np.stack(IRFelas_list, axis=2)
        elasuse = np.array(elasuse_list)
    else:
        IRFelas = np.zeros((4**2, xmax + 1, 0))
        elasuse = np.array([])
        print("Warning: No admissible IRFs found. Check your random seed or jmax.")

    # 4. Find closest elasticity to median
    print("Finding median elasticity...")
    mat_med = sio.loadmat('medelasuse.mat')
    medelasuse = mat_med['medelasuse'][0,0] # Extract scalar
    
    # distance = abs(elasuse - medelasuse)
    distance = np.abs(elasuse - medelasuse)
    mindist = np.min(distance)
    findex = np.argmin(distance)
    
    print(f"Minimum distance: {mindist:.4f}")
    print(f"Index of closest IRF: {findex}")
    
    # ==============================================================================
    # 5. Execute Figure Scripts
    # We use exec() to run the script files in the current global workspace, 
    # perfectly mimicking MATLAB's behavior of sharing variables across scripts.
    # Make sure you have saved the plotting codes as Figure1.py, Figure2.py, etc.
    # ==============================================================================
    print("Generating Figures...")
    try:
        exec(open('Figure1.py').read(), globals())
        exec(open('Figure2.py').read(), globals())
        exec(open('Figures3to7.py').read(), globals())
    except FileNotFoundError as e:
        print(f"Skipping figures: Could not find script file - {e}")
        
    # ==============================================================================
    # 6. Table 2: Variance Decomposition
    # ==============================================================================
    print("Calculating Variance Decompositions (Table 2)...")
    
    # Recovering identification matrix (flattened with Fortran ordering)
    Btilda = np.reshape(IRFelas[:, 0, findex], (4, 4), order='F')
    
    VDC = np.zeros((15, 4))
    VDCrpoil = np.zeros((15, 4))
    
    # Calculate for horizons 1 to 15
    for h_idx in range(15):
        h = h_idx + 1 # MATLAB loops 1 to 15
        VC, K_decomp = VARdecomp(BETAnc, Btilda, h)
        
        # Inventory change is 4th variable (Python index 3)
        # Real price of oil is 3rd variable (Python index 2)
        VDC[h_idx, :] = VC[3, :]
        VDCrpoil[h_idx, :] = VC[2, :]
        
    # Calculate for infinite horizon (600)
    VC_inf, K_inf = VARdecomp(BETAnc, Btilda, 600)
    VDCinf = VC_inf[3, :]
    VDCinfrpoil = VC_inf[2, :]
    
    print("\n--- Table 2 Data ---")
    print(f"VDCinf (Inventories):\n{VDCinf}")
    print(f"VDCinfrpoil (Real Price of Oil):\n{VDCinfrpoil}")