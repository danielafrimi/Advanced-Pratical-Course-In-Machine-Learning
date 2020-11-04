import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dataset
from models import SimpleModel

classes = ['car','truck', 'cat']

train_dataset = dataset.get_dataset_as_array('data/dev.pickle')

test_dataset = dataset.get_dataset_as_torch_dataset('data/dev.pickle')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True, num_workers=1)
class Evaluation():

    def __init__(self, model_weights_path='./weights.ckpt', batch_size=32):
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size

        # Load the weights of the net
        self.net = SimpleModel()

        # net.load('./model_weights')
        self.net.load(model_weights_path)

    def total_accuracy(self):
        # Pass on the test dataset and evaluate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))


    def accuracy_per_class(self):
        """

        :return:
        """
        class_correct = list(0. for i in range(3))
        class_total = list(0. for i in range(3))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    if c.ndimension() == 0:
                        class_correct[label] += c.item()
                    else:
                        class_correct[label] += c[i].item()
                    class_total[label] += 1



        for i in range(3):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

    def plot_class_imbalance(self):
        """
        Plot the imbalce classes in the train set

        :param train_dataset: the train data set
        :return: plotting the differents
        """

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



def main():
    evaluater = Evaluation('data/pre_trained.ckpt')
    evaluater.accuracy_per_class()


if __name__ == "__main__":
    main()

