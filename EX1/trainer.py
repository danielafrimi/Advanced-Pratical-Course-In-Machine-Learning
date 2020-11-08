import torch
import torch.nn as nn

import visualizer
from models import SimpleModel
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from dataset import MyDataset, get_dataset_as_array, change_cats_label_in_dataset, transform_data


NUM_EPOCHS = 60
BATCH_SIZE = 32


class Trainer():

    def __init__(self, trainloader):
        self.net = SimpleModel()
        self.trainloader = trainloader


    def train(self, num_epochs=100, batch_size=32, lr=0.0001, plot_net_error=False):
        """
        Train the model on the data in the data loader and save the weights of the model
        :return: None
        """
        print("Start Training with lr={}, batch_size= {}".format(lr, batch_size))

        # Define the Opitmizer and the loss function for the model
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.00001)
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


def get_weighted_random_sampler(train_dataset):
    """
    Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    :param train_dataset: the train data set
    :return: WeightedRandomSampler, which sample the data in a custom distribution
    """

    # Calculate How many samples there are in each class
    labels = [image_data[1] for image_data in train_dataset]
    samples_per_class = list()
    for i in range(3):
        samples_per_class.append(labels.count(i))

    samples_per_class = np.array(samples_per_class)

    # Calculate the weight if the each class
    weight = 1. / samples_per_class

    # For each label replace it with the according weight label
    samples_weight = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    # Create a WeightedRandomSampler which load images during the training according to the defined weights
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def main():
    # Create the trainloader using the train data set
    train_dataset_arr = get_dataset_as_array('data/train.pickle')

    # Change the wrong labels of 'cat' class
    change_cats_label_in_dataset(train_dataset_arr)

    augmented_data = transform_data(train_dataset_arr)

    # Create DataSet object with transformation functions
    train_dataset = MyDataset(augmented_data + train_dataset_arr, transform=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            transforms.RandomHorizontalFlip(p=0.3),
                                                            transforms.ToTensor(),
                                                            ]))

    # Create DataSet object without transformation functions
    train_dataset = MyDataset(augmented_data + train_dataset_arr)
    # TODO Change
    train_dataset = MyDataset(train_dataset_arr)

    weighted_random_sampler = get_weighted_random_sampler(train_dataset_arr)

    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=BATCH_SIZE,
                                              sampler=weighted_random_sampler,
                                              num_workers=2)

    # Train the model
    trainer = Trainer(trainloader)
    trainer.train(NUM_EPOCHS, BATCH_SIZE, lr=0.001 , plot_net_error=True)


if __name__ == '__main__':
    main()
