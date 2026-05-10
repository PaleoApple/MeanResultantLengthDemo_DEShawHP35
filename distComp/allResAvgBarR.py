import numpy as np
import matplotlib.pyplot as plt

r_matrix = np.load("../r_features_sliding.npy") # [frames, residues]
coords = np.load("../vamp_projection.npy")     # [frames, ICs]

fig, axes = plt.subplots(5, 7, figsize=(20, 15), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(68):
    ax = axes[i]
    # Color the VAMP plot by the R-value of the specific residue
#    sc = ax.scatter(coords[:, 0], coords[:, 1], c=r_matrix[:, i], 
#                    cmap='plasma', s=1, alpha=0.3)
    ax.hexbin(coords[:, 0], coords[:, 1], C=r_matrix[:, i], 
         gridsize=30, cmap='plasma', reduce_C_function=np.mean)
    ax.set_title(f"Residue {i+1}")
#    if i == 34: # Add a colorbar to the last one
#        plt.colorbar(sc, ax=ax, label='Coherence (R)')

plt.tight_layout()
plt.savefig("hp35_stability_grid.png")
#plt.show()
