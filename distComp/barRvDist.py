import numpy as np
import matplotlib.pyplot as plt

# Load text-based distances
dists = np.loadtxt("hp35.mindists")
r_feat = np.load("../r_features_sliding.npy")

# Mapping your indices
indices = {"Res 11 Phi": 18, "Res 11 Psi": 19,
           "Res 33 Phi": 62, "Res 33 Psi": 63} # Adjusted for Res 33

fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.flatten()

for i, (label, idx) in enumerate(indices.items()):
    ax = axes[i]
    
    # hexbin shows density + average R-value per spatial bin
    hb = ax.hexbin(dists[:, 11], r_feat[:, idx], 
                   gridsize=30, cmap='plasma', mincnt=1)
    
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.set_ylabel("Stability (R-value)")
    if i >= 2: ax.set_xlabel("Min Distance (Res 12 region)")

# Add colorbar to show density of frames
cb = fig.colorbar(hb, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.8)
cb.set_label('Count (Density of States)')

plt.tight_layout()
plt.savefig("hexbin_stability_vs_dist2b.png", dpi=200)
plt.show()
