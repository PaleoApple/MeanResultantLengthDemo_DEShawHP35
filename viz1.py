import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_vamp_visualization(projection_path, output_name="vamp_viz.png"):
    # 1. Load the coordinates
    # Projection shape: (n_frames, 2)
    coords = np.load(projection_path)
    x, y = coords[:, 0], coords[:, 1]
    n_frames = len(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot 1: Free Energy Surface (FES) ---
    z, x_edges, y_edges = np.histogram2d(x, y, bins=100, density=True)
    # G = -ln(P), shift min to 0
    free_energy = -np.log(z.T + 1e-8)
    free_energy -= free_energy.min()
    
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    
    # Use 'nipy_spectral' or 'magma' for high contrast
    im1 = ax1.imshow(free_energy, extent=extent, origin='lower', 
                     cmap='viridis', aspect='auto', interpolation='gaussian')
    ax1.contour(free_energy, levels=15, linewidths=0.5, colors='black', 
                extent=extent, alpha=0.3)
    
    fig.colorbar(im1, ax=ax1, label='Free Energy ($k_BT$)')
    ax1.set_title("Free Energy Surface (VAMP IC1/IC2)")
    ax1.set_xlabel("VAMP IC 1")
    ax1.set_ylabel("VAMP IC 2")

    # --- Plot 2: Trajectory Colored by Time ---
    # This helps see if the protein "hops" between basins over time
    cmap = plt.get_cmap('plasma')
    norm = Normalize(vmin=0, vmax=n_frames)
    
    # We plot with a small alpha and size to see density
    sc = ax2.scatter(x, y, c=np.arange(n_frames), cmap=cmap, s=2, alpha=0.5)
    
    fig.colorbar(sc, ax=ax2, label='Frame Index (Time)')
    ax2.set_title("Trajectory Path (Temporal Mapping)")
    ax2.set_xlabel("VAMP IC 1")
    ax2.set_ylabel("VAMP IC 2")

    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    print(f">> Visualization saved to {output_name}")
    plt.show()

if __name__ == "__main__":
    # Ensure your projection file exists from the previous step
    PROJ_FILE = "vamp_projection.npy"
    plot_vamp_visualization(PROJ_FILE)
