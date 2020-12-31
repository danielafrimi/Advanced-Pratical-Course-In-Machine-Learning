from EX5.models import GeneratorForMnistGLO
from EX5.datasets import MNIST
import torch
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import tqdm
from torchvision.utils import make_grid
from PIL import Image


NUM_CLASSES = 10


class Trainer:
    def __init__(self, train_dataset, batch_size=32, log_dir='runs'):
        self.glo = GeneratorForMnistGLO(code_dim=256)

        self.train_loader =  torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              num_workers=1,
                                              shuffle=True)

        self.writer: SummaryWriter = SummaryWriter(log_dir=log_dir)

        self.loss = nn.L1Loss()

        # Class Embedding (for each class there is 1 embedding vector)
        self.class_embedding = np.random.rand(NUM_CLASSES, 128)

        # For each image we learn the content embedding
        self.content_embedding = np.random.rand(len(train_dataset), 128)

    # TODO delete batch_size from here after it works
    def train(self, num_epochs=50, batch_size=32, plot_net_error=False):

        content_i = torch.zeros(batch_size, 128)
        class_i = torch.zeros(batch_size, 128)

        # requires_grad for optimize the latent vector for the class label and the images content
        content_i = content_i.clone().detach().requires_grad_(True)
        class_i = class_i.clone().detach().requires_grad_(True)

        optimizer = Adam([
            {'params': self.glo.parameters(), 'lr': 0.001},
            {'params': content_i, 'lr': 0.001},
            {'params': class_i, 'lr': 0.001},
        ])

        # Xi_val, _, idx_val = next(iter(val_loader))

        for epoch in range(num_epochs):
            losses = []

            for i, (imgs, class_label, idx) in enumerate(self.train_loader):

                # Take the according embedding vectors of the classes and make it a tensor todo float tensor or what?
                class_embedding = self.class_embedding[class_label.numpy()]
                class_i.data = torch.FloatTensor(class_embedding)

                # Take the according embedding vector of the images (per batch) and make it a tensor
                content_embedding_batch = self.content_embedding[idx.numpy()]
                noise_vector = np.random.normal(0, 0.1, 128)
                # Add noise to the content
                content_embedding_noise = content_embedding_batch + noise_vector
                # TODO maybe not float Tensor ?!
                content_i.data = torch.FloatTensor(content_embedding_noise)

                # Concatenate class labels + content vectors
                # code = np.concatenate((class_i, content_i), axis=1)
                code = torch.cat((class_i, content_i), dim=1)

                # Start Optimizing
                optimizer.zero_grad()

                generated_images = self.glo(code)
                loss = self.loss(imgs, generated_images)

                self.writer.add_scalar('training/loss', loss.item(), epoch)
                loss.backward()
                optimizer.step()


                # Update the embedding vector for each class and content vectors
                self.content_embedding[idx.numpy()] = content_i.data.numpy()
                self.class_embedding[class_label.numpy()] = class_i.data.numpy()

                losses.append(loss.item())
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

                grid = make_grid(generated_images)
                self.writer.add_image('generated_images', grid, i)

            # Visualize Reconstructions
            # TODO nneds to cocat and than ?!?!!?!?!?
            # content_batch = self.content_embedding[idx.numpy()]
            # rec = self.glo(torch.tensor(torch.FloatTensor(Z[idx_val.numpy()])))


            # TODO
            # imsave('%s_rec_epoch_%03d.png' % ('da', epoch),
            #        make_grid(reconstcut_img.data / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))




def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)




if __name__ == '__main__':
    train_dataset = MNIST(num_samples=256)

    glo_trainer = Trainer(train_dataset)
    glo_trainer.train()

