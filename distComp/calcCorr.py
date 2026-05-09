import numpy as np

# Load your coordinates and distances
coords = np.load("../vamp_projection.npy")
ic1 = coords[:, 0]
distances = np.loadtxt("hp35.mindists")

correlations = []

# Loop through all 52 distance columns
for i in range(distances.shape[1]):
    # Pearson correlation between IC1 and distance_i
    corr = np.corrcoef(ic1, distances[:, i])[0, 1]
    correlations.append((i, corr))

# Sort by absolute correlation (highest impact first)
ranked_features = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

print("Top 5 'Folding Drivers' (Distance Index : Pearson R)")
for idx, val in ranked_features[:5]:
    print(f"Index {idx} : {val:.4f}")
