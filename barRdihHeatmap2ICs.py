import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def plot_dual_vamp_heatmaps(proj_path, r_feat_path, n_bins=100):
    # 1. Load your data
    # proj[:, 0] is IC1, proj[:, 1] is IC2
    proj = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    # Pre-process: Average phi/psi R-values per residue
    n_res = r_feats.shape[1] // 2
    r_res = np.zeros((r_feats.shape[0], n_res))
    for i in range(n_res):
        r_res[:, i] = (r_feats[:, i*2] + r_feats[:, i*2+1]) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
    
    # Helices for reference (HP35 standard)
    helices = [4, 11, 15, 18, 22, 32]
    
    for i, (ax, label) in enumerate(zip([ax1, ax2], ["IC1 (Main Folding)", "IC2 (Secondary Mode)"])):
        ic_data = proj[:, i]
        bin_means = np.zeros((n_res, n_bins))
        
        for res in range(n_res):
            means, edges, _ = binned_statistic(ic_data, r_res[:, res], statistic='mean', bins=n_bins)
            bin_means[res, :] = means

        # Plotting
        extent = [ic_data.min(), ic_data.max(), 2 + n_res, 2]
        im = ax.imshow(bin_means, aspect='auto', extent=extent, cmap='magma', interpolation='nearest')
        
        ax.set_title(f"Residue Order vs. {label}")
        ax.set_xlabel(f"Coordinate Value")
        ax.set_ylabel("Residue Number")
        plt.colorbar(im, ax=ax, label='Mean Resultant Length (R)')
        
        # Mark helix boundaries
        for h in helices:
            ax.axhline(h, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("double_vamp_heatmap.png")
    print(">> Double heatmap saved. Check for the 'Helix 1 Dip' in both!")

plot_dual_vamp_heatmaps("vamp_projection.npy", "r_features_sliding.npy")
