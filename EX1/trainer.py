import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler

from dataset import MyDataset, get_dataset_as_array
from models import Net

NUM_EPOCHS = 10
BATCH_SIZE = 32

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
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


model = Net()

# Define the Opitmizer and the loss function for the model training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_dataset_arr = get_dataset_as_array('data/train.pickle')
train_dataset = MyDataset(train_dataset_arr)

weighted_random_sampler = get_weighted_random_sampler(train_dataset_arr)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          sampler=weighted_random_sampler,
                                          num_workers=2)


# Train the model
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % BATCH_SIZE == 31:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / BATCH_SIZE))
            running_loss = 0.0

print('Finished Training')


# Save the weights of the trained model
model.save('./model_weights')

