import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from EX5.datasets import MNIST
from EX5.models import GeneratorForMnistGLO

NUM_CLASSES = 10


class Trainer:
    def __init__(self, train_dataset, lr_glo, lr_content, lr_class, noise_std, batch_size, loss_fraction=0.5, log_dir='runs'):
        self.glo = GeneratorForMnistGLO(code_dim=256)

        self.train_loader =  torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              num_workers=1,
                                              shuffle=True)

        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # Class Embedding (for each class there is one embedding vector)
        # TODO Change this!
        self.class_embedding = nn.Embedding(NUM_CLASSES, 128)

        # For each image we learn the content embedding
        self.content_embedding = nn.Embedding(len(train_dataset), 128)

        # Hyper Parameters
        self.lr_glo = lr_glo
        self.lr_content = lr_content
        self.lr_class = lr_class
        self.noise_std = noise_std
        self.batch_size = batch_size

    def train(self, num_epochs=50,  plot_net_error=False):

        content_i = torch.zeros(self.batch_size, 128)
        class_i = torch.zeros(self.batch_size, 128)

        # requires_grad for optimize the latent vector for the class label and the images content
        content_i = content_i.clone().detach().requires_grad_(True)
        class_i = class_i.clone().detach().requires_grad_(True)

        optimizer = Adam([
            {'params': self.glo.parameters(), 'lr': self.lr_glo},
            {'params': content_i, 'lr': self.lr_content},
            {'params': class_i, 'lr': self.lr_class},
        ])

        # Xi_val, _, idx_val = next(iter(val_loader))

        for epoch in range(num_epochs):

            for i, (images, class_label, idx) in enumerate(self.train_loader):

                # Take the according embedding vectors of the classes and make it a tensor

                # class_embedding = self.class_embedding[class_label.numpy()]


                # class_embedding = self.class_embedding[class_label.numpy()]
                class_i.data = self.class_embedding(torch.LongTensor(class_label))

                # Take the according embedding vector of the images (per batch) and make it a tensor
                content_embedding_batch = self.content_embedding(torch.LongTensor(idx))
                noise_matrix = torch.empty(content_embedding_batch.shape).normal_(mean=0,std=self.noise_std)

                # Add noise to the content
                content_i.data  = content_embedding_batch + noise_matrix

                # Concatenate class labels + content vectors
                code = torch.cat((class_i, content_i), dim=1)

                # Start Optimizing
                optimizer.zero_grad()

                generated_images = self.glo(code)
                loss = self.l1_loss(images, generated_images)

                self.writer.add_scalar('training/loss', loss.item(), epoch)
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

                loss.backward()
                optimizer.step()

                # Update the embedding vector for each class and content vectors
                self.content_embedding[idx.numpy()] = content_i.data.numpy()
                self.class_embedding[class_label.numpy()] = class_i.data.numpy()

                # Visualize Reconstructions
                self.writer.add_image('generated_images',make_grid(generated_images) , i)
                self.writer.add_image('real_images', make_grid(images), i)


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_glo', type=float, default=1e-3)
    p.add_argument('--lr_content', type=float, default=1e-3)
    p.add_argument('--lr_class', type=float, default=1e-3)
    p.add_argument('--noise_std', type=float, default=0.3)

    args = p.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_glo_{}_lr_content_{}_lr_class_{}_std_noise_{}'\
        .format(args.lr_glo, args.lr_content, args.lr_class, args.noise_std)

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    # Train the model on MNIST data set
    train_dataset = MNIST(num_samples=256)

    glo_trainer = Trainer(train_dataset, **args.__dict__)
    glo_trainer.train()

