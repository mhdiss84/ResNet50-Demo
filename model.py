import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # projection shortcut if dimensions differ

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 reduce
        out = self.relu(self.bn2(self.conv2(out))) # 3x3 convolve
        out = self.bn3(self.conv3(out))            # 1x1 expand

        out += identity  # skip connection — adds original input back (the residual)
        return self.relu(out)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem — takes raw 3-channel image, outputs 64 channels at 1/4 spatial size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 layer groups — channel sizes and block counts come from the paper
        self.layer1 = self._make_layer(64,   64,  256, blocks=3)
        self.layer2 = self._make_layer(256, 128,  512, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, 1024, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, out_channels, blocks, stride=1):
        # First block always needs a downsample because channels change
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        layers = [BottleneckBlock(in_channels, mid_channels, out_channels, stride=stride, downsample=downsample)]
        # Remaining blocks have matching dimensions so no downsample needed
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)
