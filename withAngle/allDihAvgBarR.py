import numpy as np
import matplotlib.pyplot as plt

# Load data
coords = np.load("vamp_projection.npy")
r_matrix = np.load("r_features_sliding.npy")
angle_matrix = np.load("mean_angles_sliding.npy") 

def plot_refined_grid(data_slice, angle_name, metric_type="R"):
    fig, axes = plt.subplots(5, 7, figsize=(22, 14), sharex=True, sharey=True)
    fig.suptitle(f"HP35 {angle_name} {metric_type} (Residues 2-34)", fontsize=22, y=0.98)
    axes = axes.flatten()

    # Use 'plasma' for Stability (R), 'twilight_shifted' or 'hsv' for Angles
    cmap = 'plasma' if metric_type == "R" else 'twilight_shifted'
    label = 'Mean Resultant Length (R)' if metric_type == "R" else 'Mean Angle (Degrees)'

    for i in range(33):
        ax = axes[i]
        hb = ax.hexbin(coords[::5, 0], coords[::5, 1], 
                       C=data_slice[::5, i], # Fixed from angle_slice to data_slice
                       gridsize=25, cmap=cmap, 
                       reduce_C_function=np.mean)
        
        ax.set_title(f"Res {i+2}", fontsize=12, fontweight='bold')
        if i >= 28: ax.set_xlabel("VAMP 1")
        if i % 7 == 0: ax.set_ylabel("VAMP 2")

    for i in range(33, 35): axes[i].axis('off')
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(hb, cax=cbar_ax, label=label)
    
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    plt.savefig(f"HP35_{angle_name}_{metric_type}_Grid.png", dpi=200)
    plt.show()

# 1. Plot Stability (R)
plot_refined_grid(r_matrix[:, 0::2], "Phi", metric_type="R")
plot_refined_grid(r_matrix[:, 1::2], "Psi", metric_type="R")

# 2. Plot Orientation (Angles)
plot_refined_grid(angle_matrix[:, 0::2], "Phi", metric_type="Angle")
plot_refined_grid(angle_matrix[:, 1::2], "Psi", metric_type="Angle")
