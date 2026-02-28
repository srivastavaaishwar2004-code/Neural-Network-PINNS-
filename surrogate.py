import torch
import torch.nn as nn


class SurrogateNet(nn.Module):

    def __init__(self):
        super(SurrogateNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(6,   64),  nn.ReLU(),
            nn.Linear(64, 128),  nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128,  64), nn.ReLU(),
            nn.Linear(64,    1)
        )

    def forward(self, x):

        return self.net(x)


if __name__ == '__main__':

    net = SurrogateNet()

    total = sum(p.numel() for p in net.parameters())
    print(f"Surrogate parameters: {total:,}")

    # test forward pass
    fake_input = torch.rand(32, 6)
    output = net(fake_input)

    print(f"Input shape:  {fake_input.shape}")
    print(f"Output shape: {output.shape}")