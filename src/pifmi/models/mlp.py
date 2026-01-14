import torch
import torch.nn as nn

class PIFMI_Net(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=6, neurons=50):
        super(PIFMI_Net, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))