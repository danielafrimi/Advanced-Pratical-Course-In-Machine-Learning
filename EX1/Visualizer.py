import matplotlib.pyplot as plt
import torch
import torchvision

import dataset

import numpy as np
from typing import List

def plot_net_error(net_error: List):
    """

    :param net_error:
    :return:
    """
    plt.plot(net_error)
    plt.ylabel('Loss')
    plt.ylabel('Number of Mini-Batches')
    plt.suptitle('Model Running Loss')
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

def plot_class_imbalance(self):
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
    plt.show()

classes = ('car','truck', 'cat')

train_dataset = dataset.get_dataset_as_torch_dataset('data/train.pickle')
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# print images
matplotlib_imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % dataset.label_names().get(labels[j]) for j in range(4)))
