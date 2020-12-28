from EX5.models import GeneratorForMnistGLO
from EX5.datasets import MNIST
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, train_loader, log_dir):
        # TODO change values
        self.glo = GeneratorForMnistGLO(code_dim=100, out_channels=1)
        self.train_loader = train_loader
        self.optimizer = Adam(self.glo.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

    def train(self):
        pass







if __name__ == '__main__':
    train_dataset = MNIST(num_samples=200)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=32,
                                              num_workers=2)

