import numpy as np
import matplotlib.pyplot as plt

# Load data
coords = np.load("../vamp_projection.npy")
r_matrix = np.load("../r_features_sliding.npy") # Now (frames, 66)

def plot_refined_grid(data_slice, angle_name):
    fig, axes = plt.subplots(5, 7, figsize=(22, 14), sharex=True, sharey=True)
    fig.suptitle(f"HP35 {angle_name} Stability (Residues 2-34)", fontsize=22, y=0.98)
    axes = axes.flatten()

    for i in range(33):
        ax = axes[i]
        # hexbin to see the density of the 'smudge'
        hb = ax.hexbin(coords[::5, 0], coords[::5, 1], 
                       C=data_slice[::5, i], 
                       gridsize=25, cmap='plasma', 
                       reduce_C_function=np.mean)
        
        # Label as Residue i+2 since we truncated Res 1
        ax.set_title(f"Res {i+2}", fontsize=12, fontweight='bold')
        
        if i >= 28: ax.set_xlabel("IC1 (Folding Progress)")
        if i % 7 == 0: ax.set_ylabel("IC2 (Dynamics)")

    # Clean up the 2 empty subplots
    axes[-1].axis('off')
    axes[-2].axis('off')
    
    # Add a global colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(hb, cax=cbar_ax, label='Mean Resultant Length (R)')
    
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(f"HP35_{angle_name}_Grid.png", dpi=200)
    plt.show()

# Slicing the 66 columns
phi_data = r_matrix[:, 0::2]
psi_data = r_matrix[:, 1::2]

plot_refined_grid(phi_data, "Phi")
plot_refined_grid(psi_data, "Psi")
