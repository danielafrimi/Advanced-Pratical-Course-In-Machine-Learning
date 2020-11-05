import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from trainer import Trainer
import dataset
from dataset import MyDataset, get_dataset_as_array, change_cats_label_in_dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import visualizer


NUM_EPOCHS = 30
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

def train_net_different_learning_rates(lr_arr, trainloader):
    """

    :param lr_arr:
    :param trainloader:
    :return:
    """

    for i in range(len(lr_arr)):

        # Train the model
        trainer = Trainer(trainloader)
        trainer.train(NUM_EPOCHS, BATCH_SIZE, lr_arr[i], plot_net_error=True)

def visualize_dataset(dataset_visulize):
    """

    :param dataset:
    :return:
    """
    # TODO delete
    size = 10
    for i in range(int(len(dataset_visulize)/(size**2))):
        curr_figure = plt.figure(figsize=(32, 32))
        for j in range(1, (size**2) + 1):
            index = (size**2)*i + (j-1)
            image, label = dataset_visulize[index]
            img = dataset.un_normalize_image(image)
            sub_plot = curr_figure.add_subplot(size, size, j)
            label_name = dataset.label_names()[label]
            sub_plot.set_title(label_name + ' index: %d' % index)
            plt.imshow(img)
            plt.axis('off')


        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Results/')
        sample_file_name = "data.png"

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + sample_file_name)


def main():
    # Create the trainloader using the train data set
    train_dataset_arr = get_dataset_as_array('data/train.pickle')
    visualize_dataset(train_dataset_arr)

    # Change the wrong labels of 'cat' class
    change_cats_label_in_dataset(train_dataset_arr)
    # train_dataset = MyDataset(train_dataset_arr, transform=transforms.Compose([
    #                                             # Transform which randomly adjusts brightness, contrast and saturation
    #                                             # in a random order.
    #                                            transforms.ColorJitter(brightness=30, contrast=50),
    #                                            transforms.RandomAffine(degrees=15),
    #                                        ]))
    train_dataset = MyDataset(train_dataset_arr)

    # TODO del
    # visualizer.vizualize_transform(train_dataset[4][0])


    weighted_random_sampler = get_weighted_random_sampler(train_dataset_arr)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                              sampler=weighted_random_sampler,
                                              num_workers=2)

    # Train the model
    trainer = Trainer(trainloader)
    trainer.train(NUM_EPOCHS, BATCH_SIZE,  plot_net_error=True)



if __name__ == '__main__':
    main()