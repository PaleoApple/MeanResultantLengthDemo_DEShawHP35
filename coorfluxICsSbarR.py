import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

def analyze_ic_correlation(proj_path, r_feat_path, bins=160):
    # 1. Load the data
    # coords: (frames, 2) | r_feats: (frames, n_residues_angles)
    coords = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    ic1 = coords[:, 0]
    ic2 = coords[:, 1]
    
    # Calculate Sum of R per frame (Total ordering signature)
    # Higher Sum(R) = More Folded/Ordered
    sum_r = np.sum(r_feats, axis=1)
    
    # 2. Binning IC1/IC2 to find Avg ΣR
    # This shows the "Order Map" across the kinetic landscape
    ret_r = binned_statistic_2d(ic1, ic2, sum_r, statistic='mean', bins=bins)
    
    # 3. Binning IC1/IC2 to find "Fluctuation" (STD of IC1)
    # This acts as a kinetic proxy for RMSF in VAMP space
    ret_std = binned_statistic_2d(ic1, ic2, ic1, statistic='std', bins=bins)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: Average Order (ΣR)
    im1 = ax1.imshow(ret_r.statistic.T, origin='lower', 
                     extent=[ret_r.x_edge[0], ret_r.x_edge[-1], ret_r.y_edge[0], ret_r.y_edge[-1]],
                     cmap='viridis_r', aspect='auto')
    ax1.set_title("Structural Order Map (Avg $\sum R$)")
    ax1.set_xlabel("VAMP IC 1")
    ax1.set_ylabel("VAMP IC 2")
    fig.colorbar(im1, ax=ax1, label="$\sum R$ (Order)")

    # Plot B: Kinetic Fluctuation (RMSF proxy)
    im2 = ax2.imshow(ret_std.statistic.T, origin='lower', 
                     extent=[ret_std.x_edge[0], ret_std.x_edge[-1], ret_std.y_edge[0], ret_std.y_edge[-1]],
                     cmap='magma', aspect='auto')
    ax2.set_title("Kinetic Fluctuation (STD of IC1)")
    ax2.set_xlabel("VAMP IC 1")
    ax2.set_ylabel("VAMP IC 2")
    fig.colorbar(im2, ax=ax2, label="IC1 Std Dev")

    plt.tight_layout()
    plt.savefig("order_vs_fluctuation.png", dpi=300)
    
    # 4. Global Correlation Printout
    correlation = np.corrcoef(sum_r, ic1)[0, 1]
    print(f">> Pearson Correlation between IC1 and $\sum R$: {correlation:.3f}")

if __name__ == "__main__":
    analyze_ic_correlation("vamp_projection.npy", "r_features_sliding.npy")
