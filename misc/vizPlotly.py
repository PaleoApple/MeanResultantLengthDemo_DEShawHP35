import numpy as np
import plotly.graph_objects as go

def generate_interactive_vamp_plot(projection_path, stride=20, output_name="vamp_3d_traj.html"):
    # 1. Load data
    coords = np.load(projection_path)
    
    # 2. Apply stride to downsample for performance
    # We take every 20th frame
    indices = np.arange(len(coords))
    strided_indices = indices[::stride]
    strided_coords = coords[::stride]
    
    x = strided_coords[:, 0]
    y = strided_coords[:, 1]
    z = strided_indices  # Time / Frame number
    
    # 3. Create the 3D Scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+lines',
        marker=dict(
            size=3,
            color=z,                # Color by frame number
            colorscale='Viridis',   # High contrast time mapping
            opacity=0.8,
            colorbar=dict(title="Frame Number")
        ),
        line=dict(
            color='rgba(100,100,100,0.2)', # Subtle lines connecting frames
            width=1
        ),
        text=[f"Frame: {idx}" for idx in strided_indices], # Hover text
        hoverinfo='text+x+y'
    )])

    # 4. Update layout for a "Free Energy-like" perspective
    fig.update_layout(
        title=f"VAMP Projection: IC1 vs IC2 vs Time (Stride {stride})",
        scene=dict(
            xaxis_title='VAMP IC 1',
            yaxis_title='VAMP IC 2',
            zaxis_title='Frame Number (Time)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)) # Initial view angle
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark" # Dark mode makes the colors pop
    )

    # 5. Save as interactive HTML
    fig.write_html(output_name)
    print(f">> Interactive plot saved to {output_name}")

if __name__ == "__main__":
    PROJ_FILE = "vamp_projection.npy"
    generate_interactive_vamp_plot(PROJ_FILE, stride=20)
