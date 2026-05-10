import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm, PowerNorm

def plot_enhanced_funnel(proj_path, r_feat_path, bins=160):
    # 1. Load Data
    coords = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    ic1, ic2 = coords[:, 0], coords[:, 1]
    sum_r = np.sum(r_feats, axis=1)

    # 2. Compute FES and Mean Order
    counts, x_edges, y_edges = np.histogram2d(ic1, ic2, bins=bins)
    # Mask zero-count bins to avoid log(0)
    counts = np.where(counts == 0, np.nan, counts)
    
    # Structural Order Map (Mean ΣR)
#    ret_r = binned_statistic_2d(ic1, ic2, sum_r, statistic='mean', bins=bins)
# Change the statistic to find the 'Peak Order' in each bin
    ret_r_max = binned_statistic_2d(ic1, ic2, sum_r, statistic=lambda x: np.percentile(x, 95), bins=bins)

# Now, plot ret_r_max.statistic.T
    order_map = ret_r_max.statistic.T
    
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Plot A: Log-Scaled Free Energy / Density ---
    # LogNorm allows us to see the 'comet tail' which has much lower density than the basin
    im1 = ax1.imshow(counts.T, extent=extent, origin='lower', 
                     norm=LogNorm(vmin=1, vmax=np.nanmax(counts)),
                     cmap='magma', aspect='auto')
    ax1.set_title("Log-Scaled Population Density (The Comet Shape)")
    ax1.set_xlabel("VAMP IC 1")
    ax1.set_ylabel("VAMP IC 2")
    fig.colorbar(im1, ax=ax1, label='Log(Count)')

    # --- Plot B: Non-Linear Structural Order Map ---
    # Using PowerNorm or a customized vmin/vmax to highlight the gradient in the funnel
    # This prevents the 'folded' basin from saturating the scale.
    v_min = np.nanpercentile(sum_r, 5)  # Focus on the 5th to 95th percentile
    v_max = np.nanmax(sum_r)
    
    im2 = ax2.imshow(order_map, extent=extent, origin='lower', 
                     norm=PowerNorm(gamma=0.5, vmin=v_min, vmax=v_max),
                     cmap='viridis', aspect='auto')
    ax2.set_title(f"Structural Order Funnel (Gamma-Corrected $\sum R$)")
    ax2.set_xlabel("VAMP IC 1")
    ax1.set_ylabel("VAMP IC 2")
    fig.colorbar(im2, ax=ax2, label='$\sum R$ (Stretched Scale)')

    plt.tight_layout()
    plt.savefig("enhanced_funnel_viz.png", dpi=300)
    print(">> Enhanced funnel visualization saved.")

if __name__ == "__main__":
    plot_enhanced_funnel("vamp_projection.npy", "r_features_sliding.npy")
