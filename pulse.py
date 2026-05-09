import numpy as np
import matplotlib.pyplot as plt



def check_pulsing(proj_path, r_feat_path):
    coords = np.load(proj_path)
    r_feats = np.load(r_feat_path)
    
    ic1 = coords[:, 0]
    sum_r = np.sum(r_feats, axis=1)
    
    # Normalize for overlay
    ic1_norm = (ic1 - ic1.min()) / (ic1.max() - ic1.min())
    sum_r_norm = (sum_r - sum_r.min()) / (sum_r.max() - sum_r.min())

    plt.figure(figsize=(15, 5))
    plt.plot(ic1_norm, label="Kinetic Pulse (IC1)", alpha=0.7, color='blue')
    plt.plot(sum_r_norm, label="Structural Pulse ($\sum R$)", alpha=0.7, color='orange')
    
    plt.title("The 'Pulse' Synchronicity")
    plt.xlabel("Frame (Time)")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_pulsing("vamp_projection.npy", "r_features_sliding.npy")
