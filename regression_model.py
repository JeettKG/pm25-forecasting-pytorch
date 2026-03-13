import torch
import torch.nn as nn

class PM_Model(nn.Module):
    def __init__(self, input_dim):
        super(PM_Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    # Example usage
    model = PM_Model(input_dim=10)  # Replace 10 with your actual feature count
    x = torch.randn(4, 10)
    y = model(x)
    print(y)