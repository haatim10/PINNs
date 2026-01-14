import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from src.pifmi.models.mlp import PIFMI_Net

def generate_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "outputs/checkpoints/pifmi_alpha05.pth"
    output_dir = "outputs/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Load Model
    model = PIFMI_Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Grid
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    x_test = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_test = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy().reshape(100, 100)
    u_exact = (T**0.5) * np.sin(np.pi * X)

    # --- FIGURE 1: 3D Surface Comparison ---
    fig = plt.figure(figsize=(15, 5))
    
    # Exact
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, T, u_exact, cmap='viridis')
    ax1.set_title("Exact Solution")
    
    # Prediction
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, T, u_pred, cmap='magma')
    ax2.set_title("PI-fMI Prediction")

    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, T, np.abs(u_exact - u_pred), cmap='inferno')
    ax3.set_title("Absolute Error")
    
    plt.savefig(f"{output_dir}/surface_comparison.png")
    print(f"Saved: {output_dir}/surface_comparison.png")

    # --- FIGURE 2: Singularity Check at x=0.5 ---
    plt.figure(figsize=(8, 5))
    mid_idx = 50
    plt.plot(t, u_exact[:, mid_idx], 'k-', label='Exact', linewidth=2)
    plt.plot(t, u_pred[:, mid_idx], 'r--', label='PI-fMI', linewidth=2)
    plt.title("Time-History at x=0.5 (Singularity at t=0)")
    plt.xlabel("t") ; plt.ylabel("u(0.5, t)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{output_dir}/singularity_slice.png")
    print(f"Saved: {output_dir}/singularity_slice.png")

if __name__ == "__main__":
    generate_plots()