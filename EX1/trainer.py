import torch
import torch.nn as nn
import visualizer
import torch.nn.init as init
import numpy as np

from models import SimpleModel

#The weights_init function takes an initialized model as input and reinitializes all convolutional,
# convolutional-transpose, and batch normalization layers to meet this criteria. This function is
# applied to the models immediately after initialization
def weights_init(m): # m is a model
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Emb') != -1:
        init.normal_(m.weight, mean=0, std=0.01)

class Trainer():

    def __init__(self, trainloader):
        self.net = SimpleModel()
        self.net.apply(weights_init)
        self.trainloader = trainloader


    def train(self, num_epochs=30, batch_size=32, lr=0.001, plot_net_error=False):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return:
        """
        print("Start Training with lr={}, batch_size= {}".format(lr, batch_size))

        # Define the Opitmizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        net_loss_per_batch = list()

        # Train the model
        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(self.trainloader, 0):

                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                # Calculating loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % batch_size == batch_size - 1:  # print every batch
                    net_loss_per_batch.append(( running_loss / batch_size))
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / batch_size))
                    running_loss = 0.0

        if plot_net_error:
            print(" THIS ARE THE ERRORS {} ".format(net_loss_per_batch))
            visualizer.plot_net_error(net_loss_per_batch, lr, save_img=True)

        print('Finished Training')

        # Save the weights of the trained model
        self.net.save('./203865837.ckpt')

