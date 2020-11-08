import matplotlib.pyplot as plt
import torch
import torchvision

import dataset
from torchvision import transforms
import numpy as np
from typing import List
import os
from math import sqrt
from matplotlib.pyplot import imshow


def plot_net_error(net_error: List, learning_rate, save_img=False):
    """
    PLot the loss error graph
    :param net_error: the error of the net along the training
    :return:
    """
    plt.plot(net_error)
    plt.ylabel('Loss')
    plt.xlabel('Number of Mini-Batches')
    plt.suptitle('Model Running Loss With ' + str(learning_rate) + ' learning rate')
    plt.ylim(0.9, 1.3)
    if save_img:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Results/')
        sample_file_name = "net_error_lr={}.png".format(learning_rate)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + sample_file_name)
    else:
        plt.show()

def plot_class_imbalance(train_dataset, save_img=False):
    """
    Plot the imbalce classes in the train set

    :param train_dataset: the train data set
    :return: plotting the differents
    """
    classes = ['car', 'truck', 'cat']

    labels = [image_data[1] for image_data in train_dataset]
    class_size = list()
    for i in range(3):
        # Count how many times there is each label
        class_size.append(labels.count(i))

    plt.figure(figsize=(9, 3))
    plt.subplot(132)
    plt.bar(classes, class_size)

    plt.suptitle('Number of samples per class')
    if save_img:
        script_dir = os.path.dirname(__file__)
        results_dir = os.path.join(script_dir, 'Results/')
        sample_file_name = "imbalance_data.png"

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(results_dir + sample_file_name)
    else:
        plt.show()

def vizualize_transform(sample, builtin_transforms=True, transform_list=None):
    """

    :param sample: Image as tensor
    :param builtin_transforms: bool indicator for getting transforms from the user
    :param transform_list: transforms that we want to operate on the image
    :return:
    """

    random_affine = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=15, fillcolor=0),
        transforms.ToTensor(),
    ])


    composed_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    random_horizontal = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
    ])

    transform_list = [random_horizontal, random_affine, composed_transforms, ]

    if builtin_transforms:
        transform_list = transform_list


    # Apply each of the above transforms on sample.
    for i, tsfrm in enumerate(transform_list):
        transformed_sample = tsfrm(dataset.un_normalize_image(sample))
        imshow(np.asarray(transformed_sample))



train_dataset = dataset.get_dataset_as_torch_dataset('./data/train.pickle')
train_dataset_arr = dataset.get_dataset_as_array('./data/train.pickle')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=1)

def main():
    vizualize_transform(train_dataset_arr[4][0])


if __name__ == '__main__':
    main()