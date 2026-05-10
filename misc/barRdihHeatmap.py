import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

def plot_kinetic_heatmap(proj_path, r_feat_path, n_bins=100):
    # 1. Load your features
    ic1 = np.load(proj_path)[:, 0]
    r_feats = np.load(r_feat_path)
    
    # Average phi/psi R-values for each residue to get one 'stiffness' value per res
    # r_feats shape is (frames, n_dihedrals)
    n_res = r_feats.shape[1] // 2
    r_res = np.zeros((r_feats.shape[0], n_res))
    for i in range(n_res):
        r_res[:, i] = (r_feats[:, i*2] + r_feats[:, i*2+1]) / 2

    # 2. Setup Binning
    # We bin the R-values based on the IC1 coordinate
    bin_means = np.zeros((n_res, n_bins))
    
    for res in range(n_res):
        # binned_statistic calculates the mean R for each slice of IC1
        means, bin_edges, _ = binned_statistic(ic1, r_res[:, res], 
                                               statistic='mean', bins=n_bins)
        bin_means[res, :] = means

    # 3. Plotting the "Ghost Structure"
    plt.figure(figsize=(12, 8))
    # Res num starts from ~2
    extent = [ic1.min(), ic1.max(), 2 + n_res, 2] 
    
    im = plt.imshow(bin_means, aspect='auto', extent=extent, 
                    cmap='magma', interpolation='nearest')
    
    plt.colorbar(im, label='Backbone Stiffness (R)')
    plt.title("The Kinetic Heatmap: Structural 'Flicker' vs. Folding Progress")
    plt.xlabel("VAMP IC1 (Tail $\\rightarrow$ Head)")
    plt.ylabel("Residue Number")
    
    # Draw horizontal lines where the literature says helices should be
    for h in [4, 11, 15, 18, 22, 32]:
        plt.axhline(h, color='white', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_kinetic_heatmap("vamp_projection.npy", "r_features_sliding.npy")
