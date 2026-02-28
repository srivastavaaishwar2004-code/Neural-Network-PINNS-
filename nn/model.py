import torch
import torch.nn as nn

class RayleighNet(nn.Module):
    
    def __init__(self):
        super(RayleighNet, self).__init__()
        
        # convolutional layer
        # in_channels=2  because input has 2 channels [freq, Vr]
        # out_channels=16 — number of patterns to detect
        # kernel_size=3  — look at 3 consecutive points at a time
        self.conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        
        # after conv the output is (batch, 16, 80)
        # flatten to (batch, 16*80) = (batch, 1280) before dense layers
        self.flatten = nn.Flatten()
        
        self.dense = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU(),
            nn.Linear(128,  64),  nn.ReLU(),
            nn.Linear(64,   32),  nn.ReLU(),
            nn.Linear(32,   5),
        )
        
    def forward(self, x):
        # x shape: (batch, 80, 2)
        
        # transpose to (batch, 2, 80) for Conv1D
        x = x.transpose(1, 2)
        
        # conv + relu
        x = torch.relu(self.conv(x))
        
        # flatten to (batch, 1280)
        x = self.flatten(x)
        
        # dense layers
        x = self.dense(x)
        
        return x
