import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

def run_entropy_to_npy(diheds_path, npy_projection_path, output_name="state_entropy_profiles.npy"):
    # 1. Load Data
    print("Loading datasets...")
    projection = np.load(npy_projection_path)
    # Load shifted dihedrals: cols are phi2, psi2, phi3, psi3... phi34, psi34
    diheds = np.loadtxt(diheds_path) 
    
    # 2. Clustering
    print("Clustering VAMP projection...")
    bandwidth = estimate_bandwidth(projection, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    cluster_labels = ms.fit_predict(projection)
    
    unique_states = np.sort(np.unique(cluster_labels))
    n_residues = 33  # Residues 2 to 34 inclusive
    
    # Initialize storage: (Number of States, Number of Residues)
    entropy_map = np.zeros((len(unique_states), n_residues))

    # 3. Calculate Entropy per State
    print(f"Processing {len(unique_states)} states...")
    for i, state in enumerate(unique_states):
        state_mask = (cluster_labels == state)
        state_diheds = diheds[state_mask]
        
        if len(state_diheds) < 2:
            continue
            
        for res_offset in range(n_residues):
            # Column mapping: 0,1 for Res 2; 2,3 for Res 3...
            col_phi = res_offset * 2
            col_psi = col_phi + 1
            
            # Convert shifted degrees to radians
            phi_rad = np.radians(state_diheds[:, col_phi])
            psi_rad = np.radians(state_diheds[:, col_psi])
            
            # Calculate R (Mean Resultant Length)
            r_phi = np.sqrt(np.mean(np.cos(phi_rad))**2 + np.mean(np.sin(phi_rad))**2)
            r_psi = np.sqrt(np.mean(np.cos(psi_rad))**2 + np.mean(np.sin(psi_rad))**2)
            
            # Structural Entropy Metric (1 - R)
            # 0 = Perfectly rigid; 1 = Perfectly disordered
            entropy_map[i, res_offset] = 1.0 - ((r_phi + r_psi) / 2.0)

    # 4. Save the result
    np.save(output_name, entropy_map)
    print(f"Success! Saved entropy profiles of shape {entropy_map.shape} to {output_name}")
    return entropy_map

if __name__ == "__main__":
    DIHEDS = "hp35.dihs.shifted"
    NPY_PROJ = "vamp_projection_W100_L600.npy"
    entropy_data = run_entropy_to_npy(DIHEDS, NPY_PROJ)
