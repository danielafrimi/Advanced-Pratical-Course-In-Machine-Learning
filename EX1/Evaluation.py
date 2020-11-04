import matplotlib.pyplot as plt
import torch

import dataset
from models import SimpleModel

train_dataset = dataset.get_dataset_as_array('data/dev.pickle')

test_dataset = dataset.get_dataset_as_torch_dataset('data/dev.pickle')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True, num_workers=1)
class Evaluation():

    def __init__(self, model_weights_path='./203865837.ckpt', batch_size=32):
        self.model_weights_path = model_weights_path
        self.batch_size = batch_size

        # Load the weights of the net
        self.net = SimpleModel()

        # net.load('./model_weights')
        self.net.load(model_weights_path)

    def total_accuracy(self):
        """
        Calculate total accuracy of the net on the train set (dev.pickle)
        :return: The accuracy of the model
        """

        # Pass on the test dataset and evaluate the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                # Calculate the error of the prediction output
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


    def accuracy_per_class(self):
        """
        Calculate accuracy per class of the net on the train set (dev.pickle)
        :return: The accuracy per class
        """

        class_correct = list(0. for i in range(3))
        class_total = list(0. for i in range(3))

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()

                # For each class check if the predicted output is equal to the ground truth label
                for i in range(len(labels)):
                    label = labels[i]
                    if c.ndimension() == 0:
                        class_correct[label] += c.item()
                    else:
                        class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(3):
            print('Accuracy of %5s : %2d %%' % (dataset.label_names().get(i),
                                                100 * class_correct[i] / class_total[i]))


def main():
    evaluater = Evaluation('data/pre_trained.ckpt')
    # evaluater = Evaluation('data/203865837.ckpt')
    evaluater.accuracy_per_class()


if __name__ == "__main__":
    main()

