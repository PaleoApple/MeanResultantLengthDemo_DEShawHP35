import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def plot_triple_kinetic_heatmap(proj_path, r_feat_path, n_bins=80):
    # 1. Load Data (frames x 3 for proj, frames x n_feats for r_feats)
    proj = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    # Average phi/psi per residue
    n_res = r_feats.shape[1] // 2
    r_res = np.zeros((r_feats.shape[0], n_res))
    for i in range(n_res):
        r_res[:, i] = (r_feats[:, i*2] + r_feats[:, i*2+1]) / 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharey=True)
    helices = [4, 11, 15, 18, 22, 32]
    
    for i in range(3):
        ic_data = proj[:, i]
        bin_means = np.zeros((n_res, n_bins))
        
        for res in range(n_res):
            # Calculate mean R-value for each IC bin
            means, _, _ = binned_statistic(ic_data, r_res[:, res], statistic='mean', bins=n_bins)
            bin_means[res, :] = means

        extent = [ic_data.min(), ic_data.max(), 2 + n_res, 2]
        im = axes[i].imshow(bin_means, aspect='auto', extent=extent, cmap='magma', interpolation='nearest')
        
        axes[i].set_title(f"IC{i+1}: Kinetic Mode {i+1}")
        axes[i].set_xlabel(f"VAMP IC{i+1} Value")
        axes[i].set_ylabel("Residue Number")
        plt.colorbar(im, ax=axes[i], label='Order (R)')
        
        for h in helices:
            axes[i].axhline(h, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("triple_kinetic_heatmap.png")
    plt.show()

plot_triple_kinetic_heatmap("vamp_projection.npy", "r_features_sliding.npy")
