import torch
from torch import Tensor
import torch.nn as nn
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)),
        transforms.Resize((28, 28))])

class ResBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()

        if downsample:  # если размер входного и получившегося тензора разный
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            self.shortcut = nn.Sequential()

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = (self.conv1(input))
        input = (self.conv2(input))
        input = input + shortcut
        return nn.ReLU()(input)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU())

        self.layer2 = nn.Sequential(
            ResBlocks(64, 64, downsample = False),
            ResBlocks(64, 64, downsample = False)
        )

        self.layer3 = nn.Sequential(
            ResBlocks(64, 128, downsample = True),
            ResBlocks(128, 128, downsample = False)
        )

        self.pool = nn.AvgPool2d(1)
        self.fc = nn.Linear(25088, num_classes)
        self.probs = nn.Softmax(1)

    def forward(self, x):
      x = self.layer1(x)
      x = self.layer2(x)
      x = self.layer3(x)
      x = self.pool(x)
      x = x.reshape(x.size(0), -1)
      out = self.fc(x)
      return torch.round(self.probs(out), decimals=2)



path = 'my_model.pth'
device = torch.device('cpu')
model = ResNet18(10)
model.load_state_dict(torch.load(path, map_location=device))
model.eval()


