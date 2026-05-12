import numpy as np
import matplotlib.pyplot as plt

def plot_vamp_fes(projection, bins=100):
    """
    Plots 1D and 2D Free Energy Surfaces from VAMP coordinates.
    F(x) = -kT * ln(P(x))
    """
    # Assuming projection shape is [n_frames, n_dims]
    v1 = projection[:, 0]
    v2 = projection[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1D FES (First VAMP Coordinate) ---
    counts, bin_edges = np.histogram(v1, bins=bins, density=True)
    # Avoid log(0) by adding a tiny epsilon
    fes_1d = -np.log(counts + 1e-10) 
    fes_1d -= np.min(fes_1d) # Normalize so global minimum is 0
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[0].plot(bin_centers, fes_1d, color='black', lw=2)
    ax[0].set_title("1D FES: VAMP 1")
    ax[0].set_xlabel("VAMP 1")
    ax[0].set_ylabel("Free Energy ($k_BT$)")

    # --- 2D FES (VAMP 1 vs VAMP 2) ---
    counts_2d, x_edges, y_edges = np.histogram2d(v1, v2, bins=bins, density=True)
    fes_2d = -np.log(counts_2d.T + 1e-10)
    fes_2d -= np.min(fes_2d)

    im = ax[1].imshow(
        fes_2d, 
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower', 
        cmap='viridis_r', 
        aspect='auto'
    )
    fig.colorbar(im, ax=ax[1], label="Free Energy ($k_BT$)")
    ax[1].set_title("2D FES: VAMP 1 vs VAMP 2")
    ax[1].set_xlabel("VAMP 1")
    ax[1].set_ylabel("VAMP 2")

    plt.tight_layout()
    plt.show()

# Usage:
projection = np.load("vamp_projection.npy")
plot_vamp_fes(projection)
