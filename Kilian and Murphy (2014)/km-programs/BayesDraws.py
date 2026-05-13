import numpy as np
import scipy.io as sio
import time

# 1. Load Data
# scipy.io.loadmat reads .mat files into Python dictionaries. 
# We extract the specific variables using their dictionary keys.
mat_km = sio.loadmat('kmData.mat')
kmData = mat_km['kmData']

mat_world = sio.loadmat('worldprod.mat')
# flatten() ensures it's a 1D array, which prevents broadcasting issues later
worldprod = mat_world['worldprod'].flatten() 

# 2. Variable Initialization & Slicing
# Note: Python uses 0-based indexing. MATLAB's (2:end) becomes [1:]
ProdMBPM = worldprod[1:] * 30 / 1000

# MATLAB column 4 becomes Python column 3
OECDCrudeDif = kmData[:, 3]

xmax = 17       # horizon
jmax = 5000000  # number of draws for sign restrictions
rdraws = 50     # posterior draws

# Equivalent to randn('state', 1112)
np.random.seed(1112)

# 3. Execution and Timing
start_time = time.time()

# --- EXTERNAL FUNCTION CALL ---
# Note: You must have a Python version of your BAYESsign function available.
# IRFposs = BAYESsign(kmData, xmax, 4, 24, rdraws, jmax, ProdMBPM, OECDCrudeDif)
# ------------------------------

# For demonstration purposes, assuming IRFposs is defined here as a numpy array.
end_time = time.time()
print(f"Elapsed time: {end_time - start_time:.4f} seconds")

# 4. Save to .mat file
IRMposs = IRFposs # Assuming IRFposs is defined from your function
sio.savemat('BayesPosterior.mat', {'IRMposs': IRMposs})

# 5. Extract Dimensions and Vectorized Operations
# We don't need 'j' or 'k' or pre-allocated zero arrays anymore.
# Extract the entire 3rd dimension at once. These will be 1D arrays of length 'l'.
IRprod = IRFposs[0, 0, :]  
IRprice = IRFposs[2, 0, :] 
IRinv = IRFposs[3, 0, :]   

mean_OECDCrudeDif = np.mean(OECDCrudeDif)

# Flow is constant across 'l', shape is (T,) where T is the length of ProdMBPM
Flow = ProdMBPM - mean_OECDCrudeDif

# -- Broadcasting Magic --
# By using [:, np.newaxis], we reshape 1D arrays of length 'l' into 2D arrays of shape (l, 1).
# When we multiply this by ProdMBPM (shape T,), NumPy automatically "broadcasts" 
# the math to create a full 2D matrix of shape (l, T) instantly.

FlowNew = ProdMBPM * (1 + IRprod[:, np.newaxis] / 100) - mean_OECDCrudeDif - IRinv[:, np.newaxis]

# PctChange broadcasting: (l, T) matrix minus (T,) array, divided by (T,) array
PctChange = 100 * (FlowNew - Flow) / Flow

# ElasUseSeries is shape (l, T). We divide by our (l, 1) price array.
ElasUseSeries = PctChange / IRprice[:, np.newaxis]

# 6. Final Outputs
# We want the mean across the time series dimension (axis 1), resulting in an array of shape (l,)
elasuse = np.mean(ElasUseSeries, axis=1)

# Simple element-wise array division for elasprod (Shape: l,)
elasprod = IRprod / IRprice

# 7. Statistics and Saving
medelasuse = np.median(elasuse)
print(f"medelasuse:\n{medelasuse}")
sio.savemat('medelasuse.mat', {'medelasuse': medelasuse})

# prctile in MATLAB is equivalent to np.percentile in Python
elasusepctile = np.percentile(elasuse, [16, 50, 84])
elasprodpctile = np.percentile(elasprod, [16, 50, 84])

print(f"elasusepctile:\n{elasusepctile}")
print(f"elasprodpctile:\n{elasprodpctile}")

# IMPORTANT: MATLAB calculates standard deviation using N-1 (sample std) by default.
# Python (Numpy) uses N (population std) by default. 
# You MUST add ddof=1 in Python to get the exact same numbers as MATLAB.
print(f"Std elasusepctile: {np.std(elasusepctile, ddof=1)}")
print(f"Std elasprodpctile: {np.std(elasprodpctile, ddof=1)}")