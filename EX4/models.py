import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)


        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        # nhwc -> nchw
        x = x.permute(0, 3, 1, 2)

        # crop to snake's head in center [3x3]
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.

        x = F.relu(self.bn1(self.conv1(center)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(x.view(-1, 32)))
        x = self.fc2(x)
        return x

class BiggerModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(BiggerModel, self).__init__()
        self.conv1 = nn.Conv2d(10, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        # nhwc -> nchw
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.fc1(x.view(-1, 32)))
        x = self.fc2(x)
        return x

class SmallModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=3)
        self.flatten = nn.Flatten()


    def forward(self, x):
        """
        :param x: state of the game -  1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        """
        x = x.permute(0, 3, 1, 2)   # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = F.relu(self.conv1(center))
        center = self.flatten(center)
        center = F.relu(self.fc1(center))
        center = self.fc2(center)
        return center.squeeze()

class VanilaModel(nn.Module):

    def __init__(self):
        super(VanilaModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=int((3 - 1) / 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=int((3 - 1) / 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(in_features=32, out_features=3)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        x = x[:, :, 3:6, 3:6]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.softmax(self.fc1(x))
        return x.squeeze()



class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # nhwc -> nchw
        x = x.permute(0, 3, 1, 2)

        # crop to snake's head in center [3x3]
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.

        features = self.conv(center)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = torch.squeeze(self.advantage_stream(features))
        q_vals = values + (advantages - advantages.mean())

        return q_vals.squeeze()
