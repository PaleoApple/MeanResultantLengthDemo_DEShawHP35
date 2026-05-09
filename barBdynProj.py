import numpy as np
import matplotlib.pyplot as plt
from deeptime.decomposition import VAMP
from scipy.ndimage import uniform_filter1d

def compute_r_sliding_manual(data, window=100):
    """
    Computes sliding Mean Resultant Length from shifted degrees.
    Data expected in degrees: [frames, cols] where cols are phi2, psi2...
    """
    n_frames, n_cols = data.shape
    # Convert all degrees to radians at once
    data_rad = np.radians(data)
    
    r_matrix = np.zeros((n_frames, n_cols), dtype=np.float32)
    
    # Compute Cos and Sin components
    cos_data = np.cos(data_rad)
    sin_data = np.sin(data_rad)
    
    # Apply sliding window mean to all columns
    print(f">> Applying sliding window (size={window})...")
    m_cos = uniform_filter1d(cos_data, size=window, axis=0, mode='nearest')
    m_sin = uniform_filter1d(sin_data, size=window, axis=0, mode='nearest')
    
    # R = sqrt(<cos>^2 + <sin>^2)
    r_matrix = np.sqrt(m_cos**2 + m_sin**2)
    return r_matrix

def run_vamp_r_pipeline(dihs_path, window=100, lag=600):
    # 1. Load the pre-processed dihedrals
    print(f">> Loading {dihs_path}...")
    # columns: phi2, psi2, phi3, psi3 ... phi34, psi34
    raw_dihs = np.loadtxt(dihs_path)
    
    # 2. Compute R-matrix (The "processed" features)
    r_feats = compute_r_sliding_manual(raw_dihs, window=window)
    
    # 3. VAMP Projection
    print(f">> Projecting into VAMP space (Lag={lag})...")
    vamp_model = VAMP(lagtime=lag, dim=3).fit(r_feats).fetch_model()
    projection = vamp_model.transform(r_feats)
    
    # 4. SAVE DATA for rapid viz
    print(">> Saving processed data...")
    # Save the R-matrix (ordering over time)
    np.save("r_features_sliding.npy", r_feats)
    # Save the VAMP coordinates (2D projection)
    np.save("vamp_projection.npy", projection)
    
    # 5. Plotting (using your plot_fes function)
    # plot_fes(projection)
    print(">> Done. Files saved: r_features_sliding.npy, vamp_projection.npy")
    return r_feats, projection

if __name__ == "__main__":
    # Path to your shifted degrees file
    DIHEDS = "hp35.dihs" 
    W = 100
    L = 600
    r_matrix, coords = run_vamp_r_pipeline(DIHEDS, window=W, lag=L)
