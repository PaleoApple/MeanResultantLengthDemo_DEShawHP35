import numpy as np
import matplotlib.pyplot as plt

def map_driver_residues(proj_path, r_feat_path):
    # 1. Load data
    coords = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    ic1 = coords[:, 0]
    
    n_frames, n_angles = r_feats.shape
    n_residues = n_angles // 2 # Two angles (phi/psi) per residue
    
    # 2. Calculate Correlation for every angle
    correlations = np.zeros(n_angles)
    for i in range(n_angles):
        correlations[i] = np.corrcoef(ic1, r_feats[:, i])[0, 1]
    
    # 3. Aggregate by Residue (Average of phi and psi correlation)
    res_corrs = []
    for i in range(n_residues):
        avg_corr = (correlations[i*2] + correlations[i*2 + 1]) / 2
        res_corrs.append(avg_corr)
        
    res_indices = np.arange(2, 2 + n_residues) # Residues 2 to 34

    # --- Plotting the "Driver Map" ---
    plt.figure(figsize=(12, 5))
    
    # Use absolute value to show magnitude of influence
    colors = plt.cm.RdBu(np.array(res_corrs) / 2 + 0.5)
    plt.bar(res_indices, res_corrs, color=colors, edgecolor='black', alpha=0.8)
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Per-Residue Kinetic Driver Map (Correlation with IC1)")
    plt.xlabel("Residue Number")
    plt.ylabel("Pearson Correlation ($r$)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Highlight the top drivers
    top_driver_idx = np.argmax(np.abs(res_corrs))
    plt.annotate(f'Primary Driver: Res {res_indices[top_driver_idx]}', 
                 xy=(res_indices[top_driver_idx], res_corrs[top_driver_idx]),
                 xytext=(res_indices[top_driver_idx]+2, res_corrs[top_driver_idx]*0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig("residue_driver_map.png", dpi=300)
    print(f">> Map saved. Residue {res_indices[top_driver_idx]} is the strongest driver.")
    return res_corrs

if __name__ == "__main__":
    corrs = map_driver_residues("vamp_projection.npy", "r_features_sliding.npy")
