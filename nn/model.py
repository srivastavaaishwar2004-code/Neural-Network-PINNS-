import torch 
import torch.nn as nn 

class RayleighNet(nn.module):

    def __init__(self):
        super(RayleighNet, self).__init__()

        self.conv = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(1280, 128),nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU()
            nn.Linear(32, 5)
        )

    def forward(self, x):

        x = x.transpose(1,2)

        x = torch.relu(self.conv(x))

        x = self.flatten(x)

        x = self.dense(x)

        return x
    