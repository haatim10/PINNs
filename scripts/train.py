import torch
import math
from src.pifmi.models.mlp import PIFMI_Net
from src.pifmi.physics.fractional import generate_graded_mesh, compute_l1_coefficients
from src.pifmi.pinn.loss import compute_total_loss

# Define the problem-specific source term
def source_term(x, t, alpha):
    # f(x,t) = (Gamma(alpha+1) + pi^2) * sin(pi*x)
    return (math.gamma(alpha + 1) + math.pi**2) * torch.sin(math.pi * x)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alpha, beta = 0.5, 2.0
N_t, N_x = 20, 100 # Resolution

# Initialize
model = PIFMI_Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Precompute Mesh and Coeffs
t_nodes = generate_graded_mesh(N_t, beta).to(device)
coeffs = [c.to(device) for c in compute_l1_coefficients(t_nodes.cpu(), alpha)]
x_colloc = torch.linspace(0, 1, N_x).view(-1, 1).to(device)

print(f"Training PI-fMI on {device}...")
for epoch in range(20001):
    optimizer.zero_grad()
    loss = compute_total_loss(model, x_colloc, t_nodes, coeffs, alpha, source_term)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e}")

# Corrected Typo: state_dict() instead of state_with()
torch.save(model.state_dict(), "outputs/checkpoints/pifmi_alpha05.pth")
print("Training Complete. Model Saved.")