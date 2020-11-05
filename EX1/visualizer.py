import matplotlib.pyplot as plt
import torch
import torchvision

import dataset
from torchvision import transforms
import numpy as np
from typing import List
import os

def plot_net_error(net_error: List, learning_rate, save_img=False):
    """

    :param net_error:
    :return:
    """
    plt.plot(net_error)
    plt.ylabel('Loss')
    plt.xlabel('Number of Mini-Batches')
    plt.suptitle('Model Running Loss With ' + str(learning_rate) + ' learning rate')
    plt.ylim(0, 3)
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
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
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

def visualize_dataset(dataset):
    """

    :param dataset:
    :return:
    """
    # TODO delete
    size = 10
    for i in range(int(len(dataset)/(size**2))):
        curr_figure = plt.figure(figsize=(32, 32))
        for j in range(1, (size**2) + 1):
            index = (size**2)*i + (j-1)
            image, label = dataset[index]
            img = dataset.un_normalize_image(image)
            sub_plot = curr_figure.add_subplot(size, size, j)
            label_name = dataset.label_names()[label]
            sub_plot.set_title(label_name + ' index: %d' % index)
            plt.imshow(img)
            plt.axis('off')
            print(index)
        plt.show()

def vizualize_transform(sample):

    color_changer = transforms.ColorJitter(saturation=20)
    random_affine = transforms.RandomAffine(degrees=15)
    composed = transforms.Compose([transforms.ColorJitter(brightness=30, contrast=50),   ])

    # Apply each of the above transforms on sample.
    for i, tsfrm in enumerate([color_changer, random_affine]):
        # TODO Fix it
        transformed_sample = tsfrm(transforms.ToPILImage()(sample))

        plt.imshow(transformed_sample)

    plt.show()


# train_dataset = dataset.get_dataset_as_torch_dataset('/data/train.pickle')
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
#                                           shuffle=True, num_workers=2)
# image, label = train_dataset[5000]
# img = dataset.un_normalize_image(image)
# plt.imshow(img)
# # #