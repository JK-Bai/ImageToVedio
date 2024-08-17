import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # 根据打印的输出，修改全连接层的输入大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((15, 20))

        self.fc1 = nn.Linear(512 * 15 * 20, 1)  # 展平后的维度为 256 * 15 * 20

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
        x = self.bn4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x

