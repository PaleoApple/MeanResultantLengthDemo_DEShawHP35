import numpy as np

# Load your data
dists = np.loadtxt("hp35.mindists")[:, 25] # Using the dist you identified
r_feat = np.load("../r_features_sliding.npy")

# Define distance bins (e.g., from 0.2nm to 1.5nm)
bins = np.linspace(0.2, 1.5, 50)
bin_centers = (bins[:-1] + bins[1:]) / 2

results = []

# Iterate through all 66 dihedrals
for i in range(r_feat.shape[1]):
    # Bin the R-values based on the distance
    # np.digitize returns which bin each distance belongs to
    binned_indices = np.digitize(dists, bins)
    
    avg_r_per_bin = []
    for b in range(1, len(bins)):
        mask = binned_indices == b
        if np.any(mask):
            avg_r_per_bin.append(np.mean(r_feat[mask, i]))
        else:
            avg_r_per_bin.append(0)
    
    # Find the distance at which average R is maximized
    max_idx = np.argmax(avg_r_per_bin)
    optimal_dist = bin_centers[max_idx]
    max_r_val = avg_r_per_bin[max_idx]
    
    results.append((i, optimal_dist, max_r_val))

# Print the "Blueprint"
print("Dihedral Index | Optimal Dist (nm) | Max Stability (R)")
for res in results:
    print(f"{res[0]:14} | {res[1]:17.3f} | {res[2]:.3f}")
