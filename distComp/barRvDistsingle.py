import numpy as np
import matplotlib.pyplot as plt

# Load text-based distances
dists = np.loadtxt("hp35.mindists")
r_feat = np.load("../r_features_sliding.npy")

# Quick projection script
plt.figure(figsize=(10, 8))
hb = plt.hexbin(dists[:, 1], r_feat[:, 2], 
                gridsize=40, cmap='viridis', 
                reduce_C_function=np.mean, mincnt=1)

plt.xlabel("Helix 1-3 Distance (Index 25) [nm]")
plt.ylabel("Dihedral 9 (Phe6 Phi) Stability [R]")
plt.title("Coupling of Helix 1 Stability to Global Packing")
plt.colorbar(hb, label='Mean Resultant Length (R)')
plt.show()
