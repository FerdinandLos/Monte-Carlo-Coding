import numpy as np
import matplotlib.pyplot as plt

# --- 1. Initialization and Matrix Setup ---
# Assuming IRFelas, findex, U, and BETAnc are already loaded in your environment

# Reshape with order='F' (Fortran) to match MATLAB's column-major flattening
IdentMat = np.reshape(IRFelas[:, 0, findex], (4, 4), order='F')
Uhat = U
p = 24
t = 439  # Or len(kmData)
K, q = IdentMat.shape

# Compute structural multipliers
# A = [BETAnc; eye(K*(p-1),K*(p-1)), zeros(K*(p-1),K)]
eye_block = np.eye(K * (p - 1))
zeros_block = np.zeros((K * (p - 1), K))
bottom_block = np.hstack([eye_block, zeros_block])
A = np.vstack([BETAnc, bottom_block])

# J = [eye(K,K) zeros(K,K*(p-1))]
J = np.hstack([np.eye(K), np.zeros((K, K * (p - 1)))])

# Pre-allocate IRF array for speed: Shape (K^2, t - p)
IRF = np.zeros((K**2, t - p))
A_pow = np.eye(A.shape[0])

# Loop to calculate multipliers
for i in range(t - p):
    IRF[:, i] = (J @ A_pow @ J.T @ IdentMat).flatten(order='F')
    A_pow = A_pow @ A

# --- 2. Compute Structural Shocks ---
Ehat = np.linalg.inv(IdentMat) @ Uhat[:q, :]

# --- 3. Cross-Multiply Weights (Decompositions) ---
# Pre-allocate arrays
yhat1 = np.zeros(t - p); yhat2 = np.zeros(t - p)
yhat3 = np.zeros(t - p); yhat4 = np.zeros(t - p)

Phat1 = np.zeros(t - p); Phat2 = np.zeros(t - p)
Phat3 = np.zeros(t - p); Phat4 = np.zeros(t - p)

for i in range(t - p):
    # Inventories (MATLAB 4,8,12,16 -> Python 3,7,11,15)
    yhat1[i] = np.dot(IRF[3, :i+1], Ehat[0, i::-1])
    yhat2[i] = np.dot(IRF[7, :i+1], Ehat[1, i::-1])
    yhat3[i] = np.dot(IRF[11, :i+1], Ehat[2, i::-1])
    yhat4[i] = np.dot(IRF[15, :i+1], Ehat[3, i::-1])
    
    # Prices (MATLAB 3,7,11,15 -> Python 2,6,10,14)
    Phat1[i] = np.dot(IRF[2, :i+1], Ehat[0, i::-1])
    Phat2[i] = np.dot(IRF[6, :i+1], Ehat[1, i::-1])
    Phat3[i] = np.dot(IRF[10, :i+1], Ehat[2, i::-1])
    Phat4[i] = np.dot(IRF[14, :i+1], Ehat[3, i::-1])

# Time Vector Generation
time = (1973 + 2/12 + p/12) + np.arange(t - p) / 12

# =====================================================================
# --- 4. Plotting Historical Episodes ---
# =====================================================================

# Helper variables to make styling consistent
line_k_dash = {'color': 'black', 'linestyle': '--', 'linewidth': 1.5}
line_k_solid = {'color': 'black', 'linestyle': '-', 'linewidth': 1.5}

# ---- Figure 3: Gulf War ----
xlabels_gulf = ['1990.7', '1990.8', '1990.9', '1990.10', '1990.11', '1990.12', '1991.1', '1991.2']
xticks_gulf = [1990+7/12, 1990+8/12, 1990+9/12, 1990+10/12, 1990+11/12, 1990+12/12, 1991+1/12, 1991+2/12]

fig_gulf, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(time, Phat1, **line_k_dash, label='Cumulative effect of flow supply shock')
ax1.plot(time, Phat3, **line_k_solid, label='Cumulative effect of speculative demand shock')
ax1.axhline(0, color='black', linewidth=2)
ax1.legend()
ax1.set_title('Real price of oil (Gulf War)')
ax1.set_xlim([1990 + 7/12, 1991 + 2/12])
ax1.set_ylim([-35, 35])
ax1.set_xticks(xticks_gulf)
ax1.set_xticklabels(xlabels_gulf)
ax1.grid(True)

ax2.plot(time, yhat1, **line_k_dash)
ax2.plot(time, yhat3, **line_k_solid)
ax2.axhline(0, color='black', linewidth=2)
ax2.set_title('Change in oil inventories')
ax2.set_xlim([1990 + 7/12, 1991 + 2/12])
ax2.set_ylim([-50, 20])
ax2.set_xticks(xticks_gulf)
ax2.set_xticklabels(xlabels_gulf)
ax2.grid(True)
plt.tight_layout()

# ---- Figure 4: Iranian Revolution ----
xlabels_iran = ['1978.10', '1979.1', '1979.4', '1979.7', '1980.1']
xticks_iran = [1978+10/12, 1979+1/12, 1979+4/12, 1979+7/12, 1980+1/12]

fig_iran, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(time, Phat1, **line_k_dash, label='Cumulative effect of flow supply shock')
ax1.plot(time, Phat3, **line_k_solid, label='Cumulative effect of speculative demand shock')
ax1.axhline(0, color='black', linewidth=2)
ax1.legend()
ax1.set_title('Real price of oil (Iranian Revolution)')
ax1.set_xlim([1978 + 10/12, 1980 + 2/12])
ax1.set_ylim([-30, 50])
ax1.set_xticks(xticks_iran)
ax1.set_xticklabels(xlabels_iran)
ax1.grid(True)

ax2.plot(time, yhat1, **line_k_dash)
ax2.plot(time, yhat3, **line_k_solid)
ax2.axhline(0, color='black', linewidth=2)
ax2.set_title('Change in oil inventories')
ax2.set_xlim([1978 + 10/12, 1980 + 2/12])
ax2.set_ylim([-30, 50])
ax2.set_xticks(xticks_iran)
ax2.set_xticklabels(xlabels_iran)
ax2.grid(True)
plt.tight_layout()

# ---- Figure 5: Iran-Iraq War ----
xlabels_war = ['1980.9', '1980.12', '1981.3', '1981.6']
xticks_war = [1980+9/12, 1980+12/12, 1981+3/12, 1981+6/12]

fig_war, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(time, Phat1, **line_k_dash, label='Cumulative effect of flow supply shock')
ax1.plot(time, Phat3, **line_k_solid, label='Cumulative effect of speculative demand shock')
ax1.axhline(0, color='black', linewidth=2)
ax1.legend()
ax1.set_title('Real price of oil (Iran-Iraq War)')
ax1.set_xlim([1980 + 8/12, 1981 + 6/12])
ax1.set_ylim([-30, 50])
ax1.set_xticks(xticks_war)
ax1.set_xticklabels(xlabels_war)
ax1.grid(True)

ax2.plot(time, yhat1, **line_k_dash)
ax2.plot(time, yhat3, **line_k_solid)
ax2.axhline(0, color='black', linewidth=2)
ax2.set_title('Change in oil inventories')
ax2.set_xlim([1980 + 8/12, 1981 + 6/12])
ax2.set_ylim([-30, 50])
ax2.set_xticks(xticks_war)
ax2.set_xticklabels(xlabels_war)
ax2.grid(True)
plt.tight_layout()

# ---- Figure 6: OPEC Collapse ----
xlabels_opec = ['1986.1', '1986.3', '1986.5']
xticks_opec = [1986+1/12, 1986+3/12, 1986+5/12]

fig_opec, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(time, Phat1, **line_k_dash, label='Cumulative effect of flow supply shock')
ax1.plot(time, Phat3, **line_k_solid, label='Cumulative effect of speculative demand shock')
ax1.axhline(0, color='black', linewidth=2)
ax1.legend()
ax1.set_title('Real price of oil (OPEC Collapse)')
ax1.set_xlim([1985 + 12/12, 1986 + 6/12])
ax1.set_ylim([-40, 40])
ax1.set_xticks(xticks_opec)
ax1.set_xticklabels(xlabels_opec)
ax1.grid(True)

ax2.plot(time, yhat1, **line_k_dash)
ax2.plot(time, yhat3, **line_k_solid)
ax2.axhline(0, color='black', linewidth=2)
ax2.set_title('Change in oil inventories')
ax2.set_xlim([1985 + 12/12, 1986 + 6/12])
ax2.set_ylim([-40, 40])
ax2.set_xticks(xticks_opec)
ax2.set_xticklabels(xlabels_opec)
ax2.grid(True)
plt.tight_layout()

# ---- Figure 7: Venezuela ----
xlabels_ven = ['2002.11', '2003.1', '2003.4']
xticks_ven = [2002+11/12, 2003+1/12, 2003+4/12]

fig_ven, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(time, Phat1, **line_k_dash, label='Cumulative effect of flow supply shock')
ax1.plot(time, Phat3, **line_k_solid, label='Cumulative effect of speculative demand shock')
ax1.axhline(0, color='black', linewidth=2)
ax1.legend()
ax1.set_title('Real price of oil (Venezuela)')
ax1.set_xlim([2002 + 11/12, 2003 + 5/12])
ax1.set_ylim([-40, 40])
ax1.set_xticks(xticks_ven)
ax1.set_xticklabels(xlabels_ven)
ax1.grid(True)

ax2.plot(time, yhat1, **line_k_dash)
ax2.plot(time, yhat3, **line_k_solid)
ax2.axhline(0, color='black', linewidth=2)
ax2.set_title('Change in oil inventories')
ax2.set_xlim([2002 + 11/12, 2003 + 5/12])
ax2.set_ylim([-40, 40])
ax2.set_xticks(xticks_ven)
ax2.set_xticklabels(xlabels_ven)
ax2.grid(True)
plt.tight_layout()

plt.show()