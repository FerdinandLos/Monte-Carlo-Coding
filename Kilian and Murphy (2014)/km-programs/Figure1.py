import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --- 1. Data Setup ---
# Assuming IRFelas, findex, and xmax are defined in your active environment
# IRF = IRFelas[:, :, findex]

# Load posterior draws
mat_post = sio.loadmat('BayesPosterior.mat')
IRMposs = mat_post['IRMposs']

time = np.arange(0, xmax + 1)

# --- 2. Calculate Confidence Intervals (Percentiles) ---
# Note: np.percentile puts the percentiles on the first axis (axis 0).
# Shape of CI will be (2, num_vars, time_horizon)
CI = np.percentile(IRMposs, [16, 84], axis=2)

# Calculate cumulative sum across the time dimension (axis 1)
IRMposs_cumsum = np.cumsum(IRMposs, axis=1)
CI1458912 = np.percentile(IRMposs_cumsum, [16, 84], axis=2)

# Indices to replace (MATLAB: 1, 4, 5, 8, 9, 12 -> Python: 0, 3, 4, 7, 8, 11)
idx_cumsum = [0, 3, 4, 7, 8, 11]

# Replace specific rows with their cumulative equivalents
CI[:, idx_cumsum, :] = CI1458912[:, idx_cumsum, :]

# Repeat for 95% bands (if you intend to plot them later)
CI5 = np.percentile(IRMposs, [2.5, 97.5], axis=2)
CI5_1458912 = np.percentile(IRMposs_cumsum, [2.5, 97.5], axis=2)
CI5[:, idx_cumsum, :] = CI5_1458912[:, idx_cumsum, :]

# --- 3. Plotting ---
# Create a 3x4 grid of subplots. 
# figsize=(16, 12) defines the width and height in inches.
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.canvas.manager.set_window_title('Figure 1')

# Helper variable to make indexing axes easier (flattening the 3x4 grid into a 1D array of 12 subplots)
ax = axes.flatten()

# Note: In the CI array, CI[0] is the 16th percentile, CI[1] is the 84th percentile.

# Subplot 1
ax[0].plot(time, -np.cumsum(IRF[0, :]), 'r', linewidth=2)
ax[0].plot(time, -CI[0, 0, :], 'b--', linewidth=2)
ax[0].plot(time, -CI[1, 0, :], 'b--', linewidth=2)
ax[0].set_title('Flow supply shock')
ax[0].set_ylabel('Oil production')
ax[0].axhline(0, color='black', linewidth=2)
ax[0].set_xlim([0, xmax])
ax[0].set_ylim([-2, 1])

# Subplot 2
ax[1].plot(time, -IRF[1, :], 'r', linewidth=2)
ax[1].plot(time, -CI[0, 1, :], 'b--', linewidth=2)
ax[1].plot(time, -CI[1, 1, :], 'b--', linewidth=2)
ax[1].set_title('Flow supply shock')
ax[1].set_ylabel('Real activity')
ax[1].axhline(0, color='black', linewidth=2)
ax[1].set_xlim([0, xmax])
ax[1].set_ylim([-5, 10])

# Subplot 3
ax[2].plot(time, -IRF[2, :], 'r', linewidth=2)
ax[2].plot(time, -CI[0, 2, :], 'b--', linewidth=2)
ax[2].plot(time, -CI[1, 2, :], 'b--', linewidth=2)
ax[2].set_title('Flow supply shock')
ax[2].set_ylabel('Real price of oil')
ax[2].axhline(0, color='black', linewidth=2)
ax[2].set_xlim([0, xmax])
ax[2].set_ylim([-5, 10])

# Subplot 4
ax[3].plot(time, -np.cumsum(IRF[3, :]), 'r', linewidth=2)
ax[3].plot(time, -CI[0, 3, :], 'b--', linewidth=2)
ax[3].plot(time, -CI[1, 3, :], 'b--', linewidth=2)
ax[3].set_title('Flow supply shock')
ax[3].set_ylabel('Inventories')
ax[3].axhline(0, color='black', linewidth=2)
ax[3].set_xlim([0, xmax])
ax[3].set_ylim([-20, 20])

# Subplot 5
ax[4].plot(time, np.cumsum(IRF[4, :]), 'r', linewidth=2)
ax[4].plot(time, CI[0, 4, :], 'b--', linewidth=2)
ax[4].plot(time, CI[1, 4, :], 'b--', linewidth=2)
ax[4].set_title('Flow demand shock')
ax[4].set_ylabel('Oil production')
ax[4].axhline(0, color='black', linewidth=2)
ax[4].set_xlim([0, xmax])
ax[4].set_ylim([-1, 2])

# Subplot 6
ax[5].plot(time, IRF[5, :], 'r', linewidth=2)
ax[5].plot(time, CI[0, 5, :], 'b--', linewidth=2)
ax[5].plot(time, CI[1, 5, :], 'b--', linewidth=2)
ax[5].set_title('Flow demand shock')
ax[5].set_ylabel('Real activity')
ax[5].axhline(0, color='black', linewidth=2)
ax[5].set_xlim([0, xmax])
ax[5].set_ylim([-5, 10])

# Subplot 7
ax[6].plot(time, IRF[6, :], 'r', linewidth=2)
ax[6].plot(time, CI[0, 6, :], 'b--', linewidth=2)
ax[6].plot(time, CI[1, 6, :], 'b--', linewidth=2)
ax[6].set_title('Flow demand shock')
ax[6].set_ylabel('Real price of oil')
ax[6].axhline(0, color='black', linewidth=2)
ax[6].set_xlim([0, xmax])
ax[6].set_ylim([-5, 10])

# Subplot 8
ax[7].plot(time, np.cumsum(IRF[7, :]), 'r', linewidth=2)
ax[7].plot(time, CI[0, 7, :], 'b--', linewidth=2)
ax[7].plot(time, CI[1, 7, :], 'b--', linewidth=2)
ax[7].set_title('Flow demand shock')
ax[7].set_ylabel('Inventories')
ax[7].axhline(0, color='black', linewidth=2)
ax[7].set_xlim([0, xmax])
ax[7].set_ylim([-20, 20])

# Subplot 9
ax[8].plot(time, np.cumsum(IRF[8, :]), 'r', linewidth=2)
ax[8].plot(time, CI[0, 8, :], 'b--', linewidth=2)
ax[8].plot(time, CI[1, 8, :], 'b--', linewidth=2)
ax[8].set_title('Speculative demand shock')
ax[8].set_ylabel('Oil production')
ax[8].set_xlabel('Months')
ax[8].axhline(0, color='black', linewidth=2)
ax[8].set_xlim([0, xmax])
ax[8].set_ylim([-1, 2])

# Subplot 10
ax[9].plot(time, IRF[9, :], 'r', linewidth=2)
ax[9].plot(time, CI[0, 9, :], 'b--', linewidth=2)
ax[9].plot(time, CI[1, 9, :], 'b--', linewidth=2)
ax[9].set_title('Speculative demand shock')
ax[9].set_ylabel('Real activity')
ax[9].set_xlabel('Months')
ax[9].axhline(0, color='black', linewidth=2)
ax[9].set_xlim([0, xmax])
ax[9].set_ylim([-5, 10])

# Subplot 11
ax[10].plot(time, IRF[10, :], 'r', linewidth=2)
ax[10].plot(time, CI[0, 10, :], 'b--', linewidth=2)
ax[10].plot(time, CI[1, 10, :], 'b--', linewidth=2)
ax[10].set_title('Speculative demand shock')
ax[10].set_ylabel('Real price of oil')
ax[10].set_xlabel('Months')
ax[10].axhline(0, color='black', linewidth=2)
ax[10].set_xlim([0, xmax])
ax[10].set_ylim([-5, 10])

# Subplot 12
ax[11].plot(time, np.cumsum(IRF[11, :]), 'r', linewidth=2)
ax[11].plot(time, CI[0, 11, :], 'b--', linewidth=2)
ax[11].plot(time, CI[1, 11, :], 'b--', linewidth=2)
ax[11].set_title('Speculative demand shock')
ax[11].set_ylabel('Inventories')
ax[11].set_xlabel('Months')
ax[11].axhline(0, color='black', linewidth=2)
ax[11].set_xlim([0, xmax])
ax[11].set_ylim([-20, 20])

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()
plt.show()