import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 1. Load your raw degrees
print(">> Loading dihedrals...")
raw_dihs = np.loadtxt("../hp35.dihs")
n_frames, n_cols = raw_dihs.shape

# Define the strides (the tau values) we want to test
# From 'instantaneous' (stride 1) to 'long-term' (stride 500)
strides = [1]
window = 100 # Fixed smoothing window

# Pre-calculate radians to save time
data_rad = np.radians(raw_dihs)

# Setup the 5x7 grid (covering residues 1-35)
fig, axes = plt.subplots(5, 7, figsize=(20, 15), sharex=True, sharey=True)
axes = axes.flatten()

# We loop through each dihedral pair (assuming data is [phi, psi, phi, psi...])
# To get per-residue info, we can average phi/psi R-values
for res_idx in range(n_cols // 2):
    ax = axes[res_idx]
    res_num = res_idx + 2  # Adjusting for your HP35 indexing
    
    avg_rs = []
    for s in strides:
        # Subsample data by stride
        sub_rad = data_rad[::s, :]
        
        # Compute R for this stride
        cos_data = np.cos(sub_rad)
        sin_data = np.sin(sub_rad)
        
        # Sliding mean over the subsampled data
        m_cos = uniform_filter1d(cos_data, size=window, axis=0)
        m_sin = uniform_filter1d(sin_data, size=window, axis=0)
        r_vals = np.sqrt(m_cos**2 + m_sin**2)
        
        # Mean R for this residue (averaging phi and psi columns)
        mean_r = np.mean(r_vals[:, [2*res_idx, 2*res_idx+1]])
        avg_rs.append(mean_r)
    
    # Plotting the decorrelation curve
    ax.plot(strides, avg_rs, marker='o', color='teal', linewidth=2)
    ax.set_title(f"Res {res_num}", fontsize=10)
    ax.set_ylim(0, 1.0)
    
    if res_idx >= 28: ax.set_xlabel("Stride (Tau)")
    if res_idx % 7 == 0: ax.set_ylabel("Avg R")

plt.tight_layout()
plt.suptitle(f"Dihedral Decorrelation (Stability vs. Stride)", fontsize=22, y=1.02)
plt.savefig("Residue_Decorrelation_Grid.png", dpi=300)
plt.show()
