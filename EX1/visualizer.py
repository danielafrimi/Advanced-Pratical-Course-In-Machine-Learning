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

    :param net_error:
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

def matplotlib_imshow(img, one_channel=False):
    """
    Helper function to show an image
    :param img:
    :param one_channel:
    :return:
    """
    npimg = unormalize_img(img, one_channel)
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def unormalize_img(img, one_channel=False):
    """
    Helper function to show an image
    :param img:
    :param one_channel:
    :return:
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    if one_channel:
        return npimg
    return np.transpose(npimg, (1, 2, 0))


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

    :param sample:
    :param builtin_transforms:
    :param transform_list:
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
        # matplotlib_imshow(transformed_sample)


train_dataset = dataset.get_dataset_as_torch_dataset('./data/train.pickle')
train_dataset_arr = dataset.get_dataset_as_array('./data/train.pickle')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=1)

def main():
    vizualize_transform(train_dataset_arr[4][0])


if __name__ == '__main__':
    main()