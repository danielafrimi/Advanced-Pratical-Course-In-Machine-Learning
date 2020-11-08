from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dataset
import visualizer
from models import SimpleModel

test_dataset = dataset.get_dataset_as_torch_dataset('data/dev.pickle')

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=1)

# Load the weights of the net
net = SimpleModel()

net.load('./203865837.ckpt')

net.eval()

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    noise = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*noise
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image, dataset.un_normalize_image(noise.squeeze())

def check_test( model, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data, noise_signal = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 3:
                adv_ex = perturbed_data.squeeze().detach()
                original_img = data.squeeze().detach()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, noise_signal, original_img))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []
epsilons = [0, .05, .1, .15, .2, .25, .3]
# Run test for each epsilon
for eps in epsilons:
    acc, distorted_image = check_test(net, testloader, eps)
    accuracies.append(acc)
    examples.append(distorted_image)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), 3 ,cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        original_label, final_prediction, distorted_image, noise_signal, original_img = examples[i][j]
        # Get the label's names
        original_label, final_prediction = dataset.label_names().get(int(original_label)), \
                                           dataset.label_names().get(int(final_prediction))
        plt.title("{} -> {}".format(original_label, final_prediction))
        plt.imsave('noise_signal_{}_epsilon{}.jpg'.format(j, epsilons[i]), noise_signal)
        plt.imsave('distorted_image_{}_epsilon{}.jpg'.format(j, epsilons[i]), dataset.un_normalize_image(distorted_image))
        plt.imsave('original_img_{}_epsilon{}.jpg'.format(j, epsilons[i]), dataset.un_normalize_image(original_img))

plt.tight_layout()
plt.show()