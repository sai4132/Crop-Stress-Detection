import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(
            self,
            input_channels: int = 13
    ):
        super().__init__()
        self.C = input_channels

        # input layer
        self.conv1 = nn.Conv2d(in_channels=self.C, out_channels=6, kernel_size=(3, 3), stride =1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ffnn = nn.Sequential(
            nn.Linear(16, 32, True),
            nn.ReLU(),
            nn.Linear(32, 16, True),
            nn.ReLU(),
            nn.Linear(16, 1, True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.ffnn(x)
        return x