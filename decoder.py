import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 256, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            # nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.final = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        # x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        # x = torch.cat([x, skip1], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        x = self.up5(x)
        x = self.final(x)
        return x