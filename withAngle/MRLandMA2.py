import numpy as np
from deeptime.decomposition import VAMP
from scipy.ndimage import uniform_filter1d

def compute_vamp_trig_features(data, window=100):
    """
    Computes sliding R and Mean Angle components (Cos/Sin).
    Representing angles as vectors (cos, sin) prevents periodicity artifacts in VAMP.
    """
    # Convert degrees to radians
    data_rad = np.radians(data)
    
    # Raw components
    cos_data = np.cos(data_rad)
    sin_data = np.sin(data_rad)
    
    m_cos = uniform_filter1d(cos_data, size=window, axis=0, mode='nearest')
    m_sin = uniform_filter1d(sin_data, size=window, axis=0, mode='nearest')
    
    # 1. R (Mean Resultant Length) - measure of stability/dispersion
    r_matrix = np.sqrt(m_cos**2 + m_sin**2)
    
    # 2. Normalized Directional Components (Cos/Sin of the Mean Angle)
    eps = 1e-8
    mean_cos = m_cos / (r_matrix + eps)
    mean_sin = m_sin / (r_matrix + eps)
    
    # 3. Explicit Mean Angle in degrees for later plotting
    mean_angle_deg = np.degrees(np.arctan2(m_sin, m_cos))
    
    return r_matrix, mean_cos, mean_sin, mean_angle_deg

def run_vamp_trig_pipeline(dihs_path, window=100, lag=450, stride=75):
    """
    Full pipeline: Load -> Trig Featurize -> Stride -> VAMP -> Save
    """
    print(f">> Loading {dihs_path}...")
    raw_dihs = np.loadtxt(dihs_path)
    
    # 1. Compute features
    # Note: We compute on the full trajectory to keep the sliding window smooth,
    # then we stride the results.
    r_mat, cos_mat, sin_mat, a_deg = compute_vamp_trig_features(raw_dihs, window=window)
    
    # 2. Apply Stride
    print(f">> Applying stride of {stride} frames...")
    r_strided = r_mat[::stride]
    cos_strided = cos_mat[::stride]
    sin_strided = sin_mat[::stride]
    angle_strided = a_deg[::stride]
    
    # 3. Stack features for VAMP: [R, Cos(theta), Sin(theta)]
    # This captures stability and orientation without 180/-180 jumps
    vamp_input = np.hstack([r_strided, cos_strided, sin_strided])
    
    print(f">> Fitting VAMP2 (Lag={lag}, Stride={stride}, Features={vamp_input.shape[1]})...")
    vamp_model = VAMP(lagtime=lag, dim=3).fit(vamp_input).fetch_model()
    projection = vamp_model.transform(vamp_input)
    
    # 4. Save results for the visualization script
    print(">> Saving data...")
    np.save("vamp_projection.npy", projection)
    np.save("r_features_sliding.npy", r_strided)
    np.save("mean_angles_sliding.npy", angle_strided)
    
    print(">> Done. Files ready for allDihAvgBarR.py")
    return r_strided, angle_strided, projection

if __name__ == "__main__":
    PATH = "../hp35.dihs"
    WINDOW = 75
    LAG = 450
    STRIDE = 1
    
    r, angles, coords = run_vamp_trig_pipeline(PATH, window=WINDOW, lag=LAG, stride=STRIDE)
    print(f">> Final Projection Shape: {coords.shape}")
