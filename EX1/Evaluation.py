import torch

import dataset
import numpy as np
from models import Net
import matplotlib.pyplot as plt

classes = ['car','truck', 'cat']

# Load the weights of the net
net = Net()
# net.load('./model_weights')
net.load('./model_weights')

train_dataset = dataset.get_dataset_as_array('data/dev.pickle')

test_dataset = dataset.get_dataset_as_torch_dataset('data/dev.pickle')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                          shuffle=True, num_workers=2)

def total_accuracy():
    """

    :return:
    """

    # Pass on the test dataset and evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def accuracy_per_class():
    """

    :return:
    """
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(1):
                label = labels[i]
                class_correct[label] += c[i].item()

                class_total[label] += 1

    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def class_imbalance_plot(train_dataset):
    """

    :param train_dataset:
    :return:
    """

    labels = [image_data[1] for image_data in train_dataset]
    class_size = list()
    for i in range(3):
        class_size.append(labels.count(i))
# i
#     for i in range(0, len(train_dataset), 32):
#         labels = [image_data[1] for image_data in train_dataset[i: i + 32]]

    plt.figure(figsize=(9, 3))
    plt.subplot(132)
    plt.bar(classes, class_size)

    plt.suptitle('Number of samples per class')
    plt.show()

def main():
    accuracy_per_class()
    total_accuracy()
    class_imbalance_plot(train_dataset)

if __name__ == "__main__":
    main()

