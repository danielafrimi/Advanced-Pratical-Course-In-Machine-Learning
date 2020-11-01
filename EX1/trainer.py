from dataset import MyDataset, get_dataset_as_array
import torch
import matplotlib.pyplot as plt
from models import Net
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


num_epochs = 10
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()

# Define the Opitmizer and the loss function for the model training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



train_dataset = get_dataset_as_array('data/train.pickle')
train_dataset = MyDataset(train_dataset, transform=transforms.Compose([transforms.RandomCrop(32),
                                                                       transforms.ToTensor()]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=2)


# Train the model
for epoch in range(num_epochs):
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
        if i % 32 == 31:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 32))
            running_loss = 0.0

print('Finished Training')


# Save the weights of the trained model
model.save('./model_weights')

