import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

# 1. Load your data
coords = np.load("../vamp_projection.npy")
# Let's use the first distance column as a test (e.g., a core contact)
distances = np.loadtxt("hp35.mindists")
target_dist = distances[:, 3] 

# 2. Create the Binned Map
# This calculates the MEAN distance in every x,y bin of VAMP space
bin_means, x_edges, y_edges, _ = binned_statistic_2d(
    coords[:, 0], coords[:, 1], target_dist, 
    statistic='mean', bins=100
)

# 3. Plot the Heatmap
plt.figure(figsize=(10, 8))
plt.imshow(
    bin_means.T, origin='lower', 
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    aspect='auto', cmap='RdYlBu_r' # Red = Far/Disordered, Blue = Close/Native
)

plt.colorbar(label='Average Contact Distance [nm]')
plt.xlabel('VAMP IC1 (Folding Progress)')
plt.ylabel('VAMP IC2 (Conformational Path)')
plt.title('Binned Mean Distance on VAMP Landscape')
plt.savefig('DistFeatVampProj.png', dpi=500)
plt.show()
