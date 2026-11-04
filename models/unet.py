"""Small binary U-Net trained only on the committed training IDs."""
import torch
from torch import nn


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.bottleneck = block(3, 32), block(32, 64), block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.up1 = nn.ConvTranspose2d(128, 64, 2, 2), nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2, self.dec1 = block(128, 64), block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.bottleneck(self.pool(b))
        d = self.dec2(torch.cat((self.up2(c), b), dim=1))
        return self.head(self.dec1(torch.cat((self.up1(d), a), dim=1)))
