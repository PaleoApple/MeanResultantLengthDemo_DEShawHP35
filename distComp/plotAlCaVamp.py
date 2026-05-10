import numpy as np
import matplotlib.pyplot as plt

# 1. Load your data
coords = np.load("../vamp_projection.npy")
# Let's use the first distance column as a test (e.g., a core contact)
distances = np.loadtxt("hp35.mindists")


# Assuming distances is (frames, 53)
# and coords is (frames, 2)
num_dists = distances.shape[1]
cols = 6
rows = (num_dists // cols) + 1

fig, axes = plt.subplots(rows, cols, figsize=(24, 20), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(num_dists):
    ax = axes[i]
    # Hexbin is often faster than binned_statistic_2d for large grids
    hb = ax.hexbin(coords[:, 0], coords[:, 1], C=distances[:, i],
                   gridsize=40, cmap='RdYlBu_r', reduce_C_function=np.mean)
    
    ax.set_title(f"Contact {i}", fontsize=10)
    
    # Hide axes for cleaner look, only show on edges
    if i % cols != 0: ax.set_ylabel("")
    if i < (num_dists - cols): ax.set_xlabel("")


# Clean up empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("Evolution of 53 Ca Contacts on VAMP Landscape", fontsize=24, y=1.02)

plt.savefig("Global_Contact_Evolution.png", dpi=300, bbox_inches='tight')
plt.show()
