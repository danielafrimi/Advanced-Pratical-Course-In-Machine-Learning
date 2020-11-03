import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dataset
from models import SimpleModel

classes = ['car','truck', 'cat']
BATCH_SIZE = 32

# Load the weights of the net
net = SimpleModel()
# net.load('./model_weights')
net.load('./weights')

train_dataset = dataset.get_dataset_as_array('data/dev.pickle')

test_dataset = dataset.get_dataset_as_torch_dataset('data/dev.pickle')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                          shuffle=True, num_workers=1)

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

def class_imbalance_plot(train_dataset):
    """

    :param train_dataset:
    :return:
    """

    labels = [image_data[1] for image_data in train_dataset]
    class_size = list()
    for i in range(3):
        class_size.append(labels.count(i))

    plt.figure(figsize=(9, 3))
    plt.subplot(132)
    plt.bar(classes, class_size)

    plt.suptitle('Number of samples per class')
    plt.show()

def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM attack code
    :param image:
    :param epsilon:
    :param data_grad:
    :return:
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test_generative_example(epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    net.eval()

    # Loop over all examples in test set
    for data, target in testloader:

        # Set requires_grad attribute of tensor.
        # Important for Attack because we want to calculate the gradient according to the image
        data.requires_grad = True

        # Forward pass the data through the model
        output = net(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if not (torch.all(init_pred.eq(target))):
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        net.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = net(perturbed_data)

        # Check for success
        final_prediction = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if (torch.all(init_pred.eq(target))):
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred, final_prediction, adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred, final_prediction, adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(testloader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(testloader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def generative_example():
    """

    :return:
    """
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    accuracies = []
    examples = []

    # Run test for each epsilon
    for epsilon in epsilons:
        acc, example = test_generative_example(epsilon)
        accuracies.append(acc)
        examples.append(example)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    fig = plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            fig = plt.figure() # todo
            for i in range(4):
                sub = fig.add_subplot(4, 1, i + 1)
                sub.imshow(ex[i, 0, :, :])
            # plt.imshow(ex, cmap="gray")
            plt.show()
    plt.tight_layout()
    plt.show()

def main():
    # generative_example()
    accuracy_per_class()
    # total_accuracy()
    # class_imbalance_plot(train_dataset)

if __name__ == "__main__":
    main()

