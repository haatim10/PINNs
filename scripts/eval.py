import torch
import numpy as np
import math
import os
from src.pifmi.models.mlp import PIFMI_Net

def calculate_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "outputs/checkpoints/pifmi_alpha05.pth"
    
    # Load Model
    model = PIFMI_Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate Test Grid
    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)
    X, T = np.meshgrid(x, t)
    
    x_test = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(device)
    t_test = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(device)

    # Predictions
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy().reshape(100, 100)

    # Exact Solution: u = t^0.5 * sin(pi * x)
    u_exact = (T**0.5) * np.sin(np.pi * X)

    # Error Calculations
    l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    linf_error = np.max(np.abs(u_exact - u_pred))

    print("\n" + "="*30)
    print("      EVALUATION RESULTS")
    print("="*30)
    print(f"Relative L2 Error:   {l2_error:.6e}")
    print(f"Max (L-inf) Error:   {linf_error:.6e}")
    print("="*30 + "\n")

if __name__ == "__main__":
    calculate_metrics()