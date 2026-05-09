import numpy as np
import matplotlib.pyplot as plt

# 1. Load your VAMP coordinates and the mindists
coords = np.load("../vamp_projection.npy") # [frames, 3]
distances = np.loadtxt("hp35.mindists") # [frames, n_distances]

# 2. Pick a specific distance to "Color" the VAMP space
# Let's say column 5 in mindists is the critical contact for the core
target_dist = distances[:, 6] 

# 3. Plot the VAMP projection color-coded by distance
plt.figure(figsize=(8, 6))
sc = plt.scatter(coords[:, 0], coords[:, 1], c=target_dist, cmap='viridis_r', s=2, alpha=0.5)
plt.colorbar(sc, label='Distance [nm]')
plt.xlabel('IC1 (Structural Transition)')
plt.ylabel('IC2 (Stability/Backtracking)')
plt.title('VAMP Projection Color-mapped by Mindist Contact')
plt.show()
