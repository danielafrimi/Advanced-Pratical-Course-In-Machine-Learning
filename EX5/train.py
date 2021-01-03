import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import EX5.percptual_loss as vgg_metric
from EX5.datasets import MNIST
from EX5.models import GeneratorForMnistGLO

NUM_CLASSES = 10


class Trainer:
    def __init__(self, train_dataset, lr_glo, lr_content, lr_class, noise_std, batch_size, loss_fraction=0.5, log_dir='runs'):
        self.glo: nn.Module = GeneratorForMnistGLO(code_dim=100)

        self.train_loader =  torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              num_workers=1,
                                              shuffle=False)

        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # Class Embedding (for each class there is one embedding vector)
        self.class_embedding1 = torch.rand(NUM_CLASSES, 50).requires_grad_()

        # For each image we learn the content embedding
        self.content_embedding1 = torch.rand(len(train_dataset), 50).requires_grad_()

        # TODO delete?

        # Class Embedding (for each class there is one embedding vector)
        # requires_grad for optimize the latent vector for the class label and the images content
        self.class_embedding = nn.Embedding(NUM_CLASSES, 50).requires_grad_(requires_grad=True)

        # For each image we learn the content embedding
        self.content_embedding = nn.Embedding(len(train_dataset), 50).requires_grad_(requires_grad=True)

        # Hyper-Parameters
        self.lr_glo = lr_glo
        self.lr_content = lr_content
        self.lr_class = lr_class
        self.noise_std = noise_std
        self.batch_size = batch_size
        self.loss_fraction = loss_fraction

    def train(self, num_epochs=50):

        optimizer = Adam([
            {'params': self.glo.parameters(), 'lr': self.lr_glo},
            {'params': self.class_embedding.parameters(), 'lr': self.lr_class},
            {'params': self.content_embedding.parameters(), 'lr': self.lr_content},
        ])

        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, (images, class_label, idx) in enumerate(self.train_loader):
                # Take the according embedding vectors of the classes and make it a tensor todo float tensor or what?
                class_embedding = self.class_embedding1[class_label]

                # Take the according embedding vector of the images (per batch) and make it a tensor
                content_embedding_batch = self.content_embedding1[idx]

                if self.noise_std > 0:
                    noise = torch.normal(0, self.noise_std, size=content_embedding_batch.shape)
                else:
                    noise = torch.zeros(content_embedding_batch.shape)

                # Add noise to the content
                content_embedding_noise = content_embedding_batch + noise

                # Concatenate class labels + content vectors
                code = torch.cat((class_embedding, content_embedding_noise), dim=1)

                # TODO Using Embedding Layers

                # # Take the according embedding vectors of the classes and make it a tensor
                # class_batch = self.class_embedding(torch.LongTensor(class_label))
                #
                # # Take the according embedding vector of the images (per batch) and make it a tensor
                # content_embedding_batch = self.content_embedding(torch.LongTensor(idx))
                #
                # noise = torch.normal(mean=0, std=self.noise_std,
                #                      size=content_embedding_batch.shape) if self.noise_std > 0 \
                #     else torch.zeros(content_embedding_batch.shape)
                #
                #
                # # Concatenate class labels + content vectors
                # code = torch.cat((class_batch, content_embedding_batch + noise), dim=1)

                # Start Optimizing
                optimizer.zero_grad()

                generated_images = self.glo(code)

                loss = self.l1_loss(images, generated_images) +  self.l2_loss(images, generated_images)

                loss.backward()
                optimizer.step()

                # Visualize Reconstructions
                self.writer.add_image('generated_images',make_grid(generated_images), epoch)
                self.writer.add_image('real_images', make_grid(images), epoch)

                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                running_loss += loss

            self.writer.add_scalar('training/loss', running_loss, epoch)

        self.glo.save('weights.ckpt')
        self.glo.generate_digit(digit_embedding=self.class_embedding[7])


def parse_args():
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--log_dir', type=str, default='runs', help='directory for tensorboard logs (common to many runs)')

    # opt
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr_glo', type=float, default=0.001)
    p.add_argument('--lr_content', type=float, default=0.01)
    p.add_argument('--lr_class', type=float, default=0.01)
    p.add_argument('--noise_std', type=float, default=0.3)
    p.add_argument('--loss', type=str, default='combined')

    args = p.parse_args()
    return args


def distance_metric(sz, force_l2=False):
    """

    :param sz:
    :param force_l2:
    :return:
    """

    if force_l2:
        return nn.L1Loss().cuda()
    if sz == 16:
        return vgg_metric._VGGDistance(2)
    elif sz == 32:
        return vgg_metric._VGGDistance(3)
    elif sz == 64:
        return vgg_metric._VGGDistance(4)
    elif sz > 64:
        return vgg_metric._VGGMSDistance()



if __name__ == '__main__':
    args = parse_args()
    run_name = 'lr_glo_{}_lr_content_{}_lr_class_{}_std_noise_{}_loss_{}'\
        .format(args.lr_glo, args.lr_content, args.lr_class, args.noise_std, args.loss)

    # Create a directory with log name
    args.log_dir = os.path.join(args.log_dir, run_name)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    del args.__dict__['loss']

    # Train the model on MNIST data set
    train_dataset = MNIST(num_samples=256)

    glo_trainer = Trainer(train_dataset, **args.__dict__)
    glo_trainer.train()

