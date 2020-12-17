import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    # A simple MLP that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=3)
        self.flatten = nn.Flatten()


    def forward(self, x):
        """
        :param x: state of the game -  1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :return:
        """


        x = x.permute(0, 3, 1, 2)   # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = F.relu(self.conv1(center))
        center = self.flatten(center)
        center = F.relu(self.fc1(center))
        center = F.relu(self.fc2(center))
        return center.squeeze()







    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])


