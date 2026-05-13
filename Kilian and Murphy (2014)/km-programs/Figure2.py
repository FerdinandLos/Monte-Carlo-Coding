import numpy as np
import matplotlib.pyplot as plt

# --- 1. Initialization and Setup ---
# Assuming IRFelas, findex, U, kmData, and BETAnc are already defined

# MATLAB defaults to column-major (Fortran) ordering for reshaping
IdentMat = np.reshape(IRFelas[:, 0, findex], (4, 4), order='F')
Uhat = U
p = 24
t = len(kmData)  # Overrides the static t=439 assignment in your script
K, q = IdentMat.shape

# --- 2. Compute Structural Multipliers ---
# A = [BETAnc; eye(K*(p-1),K*(p-1)), zeros(K*(p-1),K)]
eye_block = np.eye(K * (p - 1))
zeros_block = np.zeros((K * (p - 1), K))
bottom_block = np.hstack([eye_block, zeros_block])
A = np.vstack([BETAnc, bottom_block])

# J = [eye(K,K) zeros(K,K*(p-1))]
J = np.hstack([np.eye(K), np.zeros((K, K * (p - 1)))])

# Pre-allocate IRF array for speed: Shape (K^2, t - p)
IRF = np.zeros((K**2, t - p))

# Initialize A_pow for the loop (A^0 is the identity matrix)
A_pow = np.eye(A.shape[0])

# We loop from 0 to t-p-1
for i in range(t - p):
    # Flatten with order='F' to match MATLAB's column-major (K^2, 1) vector
    IRF[:, i] = (J @ A_pow @ J.T @ IdentMat).flatten(order='F')
    # Multiply A_pow by A for the next iteration (much faster than matrix_power)
    A_pow = A_pow @ A

# --- 3. Compute Structural Shocks (Ehat) ---
# Assuming Uhat is shape (K, T). MATLAB's Uhat(1:q,:) grabs the first q rows.
Ehat = np.linalg.inv(IdentMat) @ Uhat[:q, :]

# --- 4. Cross-Multiply Weights (Historical Decomposition) ---
yhat1 = np.zeros(t - p)
yhat2 = np.zeros(t - p)
yhat3 = np.zeros(t - p)
yhat4 = np.zeros(t - p)

for i in range(t - p):
    # Python 0-based index adjustments:
    # MATLAB row 3 -> Python row 2
    # MATLAB row 7 -> Python row 6
    # MATLAB row 11 -> Python row 10
    # MATLAB row 15 -> Python row 14
    
    # Ehat[idx, i::-1] steps backwards from i down to 0
    yhat1[i] = np.dot(IRF[2, :i+1], Ehat[0, i::-1])
    yhat2[i] = np.dot(IRF[6, :i+1], Ehat[1, i::-1])
    yhat3[i] = np.dot(IRF[10, :i+1], Ehat[2, i::-1])
    yhat4[i] = np.dot(IRF[14, :i+1], Ehat[3, i::-1])

cumshock = yhat1 + yhat2 + yhat3 + yhat4

# --- 5. Time Vector Definition ---
# Starts at 1973 + 2/12 + 24/12 (which is 1975.1666...)
start_time = 1973 + 2/12 + p/12
# Creates an exact array of length t-p, stepping by 1/12
time = start_time + np.arange(t - p) / 12

# --- 6. Plotting ---
# Create 3 rows, 1 column of subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

events = [1990 + 7/12, 1978 + 9/12, 1980 + 9/12, 2002 + 11/12, 1985 + 12/12]
titles = [
    'Cumulative Effect of Flow Supply Shock on Real Price of Crude Oil',
    'Cumulative Effect of Flow Demand Shock on Real Price of Crude Oil',
    'Cumulative Effect of Speculative Demand Shock on Real Price of Crude Oil'
]
yhats = [yhat1, yhat2, yhat3]

for j, ax in enumerate(axes):
    ax.plot(time, yhats[j], 'b-', linewidth=2)
    ax.set_title(titles[j])
    
    # Apply axis limits
    ax.set_xlim([1978 + 6/12, 2009 + 8/12])
    ax.set_ylim([-100, 100])
    
    # Draw vertical lines for events
    for event in events:
        # vlines is the exact equivalent to MATLAB's line([x x], [ymin ymax])
        ax.vlines(x=event, ymin=-100, ymax=100, color='black', linewidth=2)
        
    ax.grid(True)

plt.tight_layout()
plt.show()