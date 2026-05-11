import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 1. Faster Loading
print(">> Loading dihedrals (this is the slow part, hang tight)...")
# Use usecols or cast to float32 if memory is tight
data = np.loadtxt("../hp35.dihs", dtype=np.float32)
n_frames, n_cols = data.shape

# 2. Pre-calculate Trig once
print(">> Pre-calculating Sin/Cos components...")
data_rad = np.radians(data)
cos_all = np.cos(data_rad)
sin_all = np.sin(data_rad)

# Cleanup to free memory
del data_rad
del data

strides = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
window = 100 

# Result storage: [num_residues, num_strides]
n_res = n_cols // 2
res_decay_data = np.zeros((n_res, len(strides)))

print(">> Starting Stride Sweep...")
for s_idx, s in enumerate(strides):
    print(f"   Processing Stride {s}...")
    
    # Subsample the pre-calculated trig data
    c_sub = cos_all[::s, :]
    s_sub = sin_all[::s, :]
    
    # Apply sliding window mean along the frame axis
    # uniform_filter1d is fast, but window size should scale or stay fixed?
    # Usually, for decorrelation, we keep the window fixed in 'subsampled space'
    m_cos = uniform_filter1d(c_sub, size=window, axis=0, mode='nearest')
    m_sin = uniform_filter1d(s_sub, size=window, axis=0, mode='nearest')
    
    # Calculate R-matrix for all residues at this stride
    r_matrix = np.sqrt(m_cos**2 + m_sin**2)
    
    # Average across the frames, then combine Phi/Psi for each residue
    mean_r_per_col = np.mean(r_matrix, axis=0)
    for r in range(n_res):
        # Average the Phi (2r) and Psi (2r+1) stability
        res_decay_data[r, s_idx] = (mean_r_per_col[2*r] + mean_r_per_col[2*r+1]) / 2

# 3. Plotting the 5x7 Grid
fig, axes = plt.subplots(5, 7, figsize=(20, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(n_res):
    ax = axes[i]
    ax.plot(strides, res_decay_data[i, :], marker='s', markersize=4, color='crimson')
    ax.set_title(f"Res {i+2}")
    ax.set_ylim(0, 1)
    if i >= 28: ax.set_xlabel("Stride")
    if i % 7 == 0: ax.set_ylabel("Avg R")

plt.tight_layout()
plt.savefig("Fast_Decorrelation_Grid.png")
plt.show()
