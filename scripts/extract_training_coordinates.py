"""
Extract and save all training coordinates used in the model.
"""

import sys
sys.path.append('/workspace/f20220519/PINNs')

import torch
import yaml
import numpy as np
from src.mesh import GradedMesh, L1Coefficients
from src.dataset import CollocationDataset

def main():
    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility (same as training)
    seed = config['seed']
    torch.manual_seed(seed)
    
    # Setup
    device = 'cpu'  # Use CPU for extraction
    alpha = config['problem']['alpha']
    beta = config['problem']['beta']
    N_t = config['discretization']['N_t']
    N_x = config['discretization']['N_x']
    N_collocation = config['discretization']['N_collocation']
    N_boundary = config['discretization']['N_boundary']
    N_initial = config['discretization']['N_initial']
    
    # Create mesh
    mesh = GradedMesh(N=N_t, beta=beta, t_max=1.0, device=device)
    
    # Create dataset
    dataset = CollocationDataset(
        mesh=mesh,
        N_x=N_x,
        N_collocation=N_collocation,
        N_boundary=N_boundary,
        N_initial=N_initial,
        x_min=config['problem']['x_min'],
        x_max=config['problem']['x_max'],
        device=device,
        seed=seed
    )
    
    # Sample training data (one batch - what was used in training)
    data = dataset.get_training_data(resample_collocation=True)
    
    # Convert to numpy for saving
    coords = {
        'collocation_points': {
            'x': data.x_coll.cpu().numpy(),
            't': data.t_coll.cpu().numpy(),
            'n': data.n_coll.cpu().numpy(),
            'count': len(data.x_coll)
        },
        'boundary_points': {
            'x': data.x_bc.cpu().numpy(),
            't': data.t_bc.cpu().numpy(),
            'u_target': data.u_bc.cpu().numpy(),
            'count': len(data.x_bc)
        },
        'initial_points': {
            'x': data.x_ic.cpu().numpy(),
            't': data.t_ic.cpu().numpy(),
            'u_target': data.u_ic.cpu().numpy(),
            'count': len(data.x_ic)
        },
        'mesh_info': {
            't_nodes': mesh.get_nodes().cpu().numpy(),
            'alpha': alpha,
            'beta': beta,
            'N_t': N_t,
            'N_x': N_x
        },
        'config': config
    }
    
    # Save as numpy file
    output_file = 'outputs/training_coordinates.npz'
    np.savez(output_file, **{k: v for k, v in coords.items() if k != 'config'})
    print(f"Saved coordinates to {output_file}")
    
    # Save as readable text file
    text_file = 'outputs/training_coordinates.txt'
    with open(text_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING COORDINATES USED IN PINN MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  α (fractional order): {alpha}\n")
        f.write(f"  β (mesh grading):     {beta}\n")
        f.write(f"  N_x (spatial points): {N_x}\n")
        f.write(f"  N_t (time steps):     {N_t}\n")
        f.write(f"  Random seed:          {seed}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"COLLOCATION POINTS (Interior PDE Residual): {coords['collocation_points']['count']} points\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'x':<15} {'t':<15} {'n (time step)':<15}\n")
        for i in range(min(20, len(data.x_coll))):
            f.write(f"{i:<8} {data.x_coll[i].item():<15.8f} {data.t_coll[i].item():<15.8f} {int(data.n_coll[i].item()):<15}\n")
        if len(data.x_coll) > 20:
            f.write(f"... ({len(data.x_coll) - 20} more points)\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"BOUNDARY POINTS (x=0 and x=1): {coords['boundary_points']['count']} points\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'x':<15} {'t':<15} {'u_target':<15}\n")
        for i in range(min(20, len(data.x_bc))):
            f.write(f"{i:<8} {data.x_bc[i].item():<15.8f} {data.t_bc[i].item():<15.8f} {data.u_bc[i].item():<15.8f}\n")
        if len(data.x_bc) > 20:
            f.write(f"... ({len(data.x_bc) - 20} more points)\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"INITIAL CONDITION POINTS (t=0): {coords['initial_points']['count']} points\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'x':<15} {'t':<15} {'u_target':<15}\n")
        for i in range(min(20, len(data.x_ic))):
            f.write(f"{i:<8} {data.x_ic[i].item():<15.8f} {data.t_ic[i].item():<15.8f} {data.u_ic[i].item():<15.8f}\n")
        if len(data.x_ic) > 20:
            f.write(f"... ({len(data.x_ic) - 20} more points)\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"GRADED MESH TIME NODES: {len(mesh.get_nodes())} nodes\n")
        f.write("-" * 80 + "\n")
        f.write(f"t_n = (n/N)^β where N={N_t}, β={beta}\n\n")
        f.write(f"{'n':<8} {'t_n':<15}\n")
        t_nodes = mesh.get_nodes().cpu().numpy()
        for i in range(min(20, len(t_nodes))):
            f.write(f"{i:<8} {t_nodes[i]:<15.8f}\n")
        if len(t_nodes) > 20:
            f.write(f"... ({len(t_nodes) - 20} more nodes)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Total training points per epoch: {coords['collocation_points']['count'] + coords['boundary_points']['count'] + coords['initial_points']['count']}\n")
        f.write("Note: Collocation points are resampled each epoch during training.\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved readable format to {text_file}")
    
    # Also save full coordinates as CSV for easy loading
    csv_file = 'outputs/training_coordinates_full.csv'
    with open(csv_file, 'w') as f:
        f.write("# COLLOCATION POINTS\n")
        f.write("index,x,t,n\n")
        for i in range(len(data.x_coll)):
            f.write(f"{i},{data.x_coll[i].item()},{data.t_coll[i].item()},{int(data.n_coll[i].item())}\n")
        
        f.write("\n# BOUNDARY POINTS\n")
        f.write("index,x,t,u_target\n")
        for i in range(len(data.x_bc)):
            f.write(f"{i},{data.x_bc[i].item()},{data.t_bc[i].item()},{data.u_bc[i].item()}\n")
        
        f.write("\n# INITIAL POINTS\n")
        f.write("index,x,t,u_target\n")
        for i in range(len(data.x_ic)):
            f.write(f"{i},{data.x_ic[i].item()},{data.t_ic[i].item()},{data.u_ic[i].item()}\n")
    
    print(f"Saved CSV format to {csv_file}")
    print(f"\nSummary:")
    print(f"  Collocation points: {coords['collocation_points']['count']}")
    print(f"  Boundary points:    {coords['boundary_points']['count']}")
    print(f"  Initial points:     {coords['initial_points']['count']}")
    print(f"  Total:              {coords['collocation_points']['count'] + coords['boundary_points']['count'] + coords['initial_points']['count']}")

if __name__ == "__main__":
    main()
