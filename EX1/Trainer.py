import torch
import torch.nn as nn
import Visualizer

from models import SimpleModel


class Trainer():

    def __init__(self, num_epochs, batch_size, trainloader, lr):
        self.net = SimpleModel()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.trainloader = trainloader
        self.lr = lr


    def train(self, plot_net_error=False):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return:
        """

        # Define the Opitmizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        net_loss_per_batch = list()

        # Train the model
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):

                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.batch_size == self.batch_size - 1:  # print every batch
                    net_loss_per_batch.append(( running_loss / self.batch_size))
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / self.batch_size))
                    running_loss = 0.0

        if plot_net_error:
            Visualizer.plot_net_error(net_loss_per_batch)

        print('Finished Training')

        # Save the weights of the trained model
        self.net.save('./203865837.ckpt')

