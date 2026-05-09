import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_pulses(proj_path, r_feat_path, n_pulses=5, window=150):
    # 1. Load data
    ic1 = np.load(proj_path)[:, 0]
    r_feats = np.load(r_feat_path)
    
    # 2. Find the biggest transitions (pulses)
    # Using a rolling window of the derivative to find sharpest jumps
    diff = np.abs(np.diff(ic1))
    
    # Find indices of the N largest jumps, ensuring they aren't right next to each other
    pulse_indices = []
    temp_diff = diff.copy()
    for _ in range(n_pulses):
        idx = np.argmax(temp_diff)
        pulse_indices.append(idx)
        # Zero out the surrounding area so we don't pick the same pulse twice
        temp_diff[max(0, idx-window):min(len(diff), idx+window)] = 0
    
    # 3. Setup Plotting
    fig, axes = plt.subplots(n_pulses, 1, figsize=(12, 3*n_pulses), sharex=False)
    if n_pulses == 1: axes = [axes]
    
    # Selected Driver Residues
    drivers = {"Res 11": (11-2)*2, "Res 14": (14-2)*2, "Res 15": (15-2)*2, "Res 34": (34-2)*2}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#ffff00']

    for i, p_idx in enumerate(pulse_indices):
        start, end = max(0, p_idx-window), min(len(ic1), p_idx+window)
        time_axis = np.arange(start, end)
        
        # Plot IC1 on a twin axis for scale
        ax_ic = axes[i].twinx()
        ax_ic.plot(time_axis, ic1[start:end], color='black', alpha=0.2, label='IC1')
        ax_ic.set_ylabel("Kinetic Coord")
        
        # Plot Residue R-values
        for (name, feat_idx), color in zip(drivers.items(), colors):
            r_val = (r_feats[start:end, feat_idx] + r_feats[start:end, feat_idx+1]) / 2
            axes[i].plot(time_axis, r_val, label=name, color=color, linewidth=2)
            
        axes[i].set_title(f"Pulse Event at Frame {p_idx}")
        if i == 0: axes[i].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("multi_pulse_analysis.png")
    print(f">> Analyzed {n_pulses} pulses.")
if __name__ == "__main__":
    plot_multiple_pulses("vamp_projection.npy", "r_features_sliding.npy")
