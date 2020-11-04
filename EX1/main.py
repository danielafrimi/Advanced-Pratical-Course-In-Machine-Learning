import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from Trainer import Trainer
from dataset import MyDataset, get_dataset_as_array
from models import SimpleModel

NUM_EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.01

def get_weighted_random_sampler(train_dataset):
    """
    Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    :param train_dataset:
    :return:
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

    # Create the trainloader using the train dataset
    train_dataset_arr = get_dataset_as_array('data/train.pickle')
    train_dataset = MyDataset(train_dataset_arr)

    weighted_random_sampler = get_weighted_random_sampler(train_dataset_arr)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              sampler=weighted_random_sampler,
                                              num_workers=2)

    # Train the model
    trainer = Trainer(NUM_EPOCHS, BATCH_SIZE, trainloader, LEARNING_RATE)
    trainer.train(plot_net_error=True)

if __name__ == '__main__':
    main()