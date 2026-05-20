import numpy as np
import os

# 1. Setup the path
script_dir = os.path.dirname(os.path.abspath(__file__))
dgp_path = os.path.join(script_dir, 'Kilian and Murphy (2014)', 'true_dgp_parameters.npz')

# 2. Load the parameters
dgp = np.load(dgp_path)

B_true = dgp['B_true']
SIGMA_true = dgp['SIGMA_true']
A0_true = dgp['A0_true']
True_IRF = dgp['True_IRF']
V_true = dgp['V_true']
p_true = dgp['p_true']

print("Loaded True DGP Parameters!")
print("True IRF Shape:", True_IRF.shape)

# ... Start your Monte Carlo loop down here ...