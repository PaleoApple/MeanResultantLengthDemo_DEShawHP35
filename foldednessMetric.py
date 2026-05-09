import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

def plot_fes_vs_order(proj_path, r_feat_path, bins=120, output_name="fes_vs_order.png"):
    # 1. Load Data
    coords = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    ic1 = coords[:, 0]
    ic2 = coords[:, 1]
    sum_r = np.sum(r_feats, axis=1)

    # 2. Compute FES: G = -ln(P)
    # We use a histogram to find probability density P
    counts, x_edges, y_edges = np.histogram2d(ic1, ic2, bins=bins, density=True)
    free_energy = -np.log(counts.T + 1e-8)
    free_energy -= free_energy.min() # Relative to global minimum
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    # 3. Compute Structural Order Map: Mean ΣR per bin
    ret_r = binned_statistic_2d(ic1, ic2, sum_r, statistic='mean', bins=bins)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Free Energy Surface
    # Use 'magma' or 'RdBu' to highlight basins
    im1 = ax1.imshow(free_energy, extent=extent, origin='lower', 
                     cmap='magma', aspect='auto', interpolation='gaussian')
    ax1.set_title("Thermodynamic Landscape (FES)")
    ax1.set_xlabel("VAMP IC 1")
    ax1.set_ylabel("VAMP IC 2")
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Free Energy ($k_BT$)', rotation=270, labelpad=15)

    # Plot 2: Structural Order Map
    # Use 'viridis' where yellow = most ordered
    im2 = ax2.imshow(ret_r.statistic.T, extent=extent, origin='lower', 
                     cmap='magma_r', aspect='auto')
    ax2.set_title("Structural Order Map (Avg $\sum R$)")
    ax2.set_xlabel("VAMP IC 1")
    ax2.set_ylabel("VAMP IC 2")
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('$\sum R$ (Ordering)', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f">> Visualization saved to {output_name}")

if __name__ == "__main__":
    plot_fes_vs_order("vamp_projection.npy", "r_features_sliding.npy")
