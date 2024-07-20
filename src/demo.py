#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import PIL
import copy

# for reproducibility
# Ref: https://pytorch.org/docs/stable/notes/randomness.html
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=1e-3)
        m.bias.data.fill_(0.0)

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.0)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

num_training= 49000
num_validation =1000
batch_size = 128

norm_layer = None
data_aug_transforms = []

norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='../data/exercise-2/',
                                           train=True,
                                           transform=norm_transform,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../data/exercise-2/',
                                          train=False,
                                          transform=test_transform,
                                          download=True
                                          )
#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

input_size = 3072
num_classes = 10
hidden_size = [50]
drop_prob = 0.2
num_classes = 10

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, drop_prob):
        super(MultiLayerPerceptron, self).__init__()
        
        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._num_classes = num_classes
        self._activation = nn.ReLU()
        self._drop_prob = drop_prob

        self.build_model()        

    def build_model(self):

        layers = []

        layers.append(nn.Linear(self._input_size, self._hidden_layers[0]))
        layers.append(self._activation)
        if self._drop_prob > 0:
            layers.append(nn.Dropout(self._drop_prob))
        
        for i in range(1, len(self._hidden_layers)):
            layers.append(nn.Linear(self._hidden_layers[i-1], self._hidden_layers[i]))
            layers.append(self._activation)
            if self._drop_prob > 0:
                layers.append(nn.Dropout(self._drop_prob))
            
        layers.append(nn.Linear(self._hidden_layers[-1], self._num_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        return x
        
model = MultiLayerPerceptron(input_size, hidden_size, num_classes, drop_prob).to(device)
model.apply(weights_init)
# Print the model
print(model)

# Loss and optimizer
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 5e-4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
num_epochs = 10

# for early stopping and plotting
best_acc = 0.0
loss_store = []
val_acc_store =[]
counter = 0
best_model_wts = None

for epoch in range(num_epochs):
    
    model.train()

    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        counter += 1
        running_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            loss_store.append([counter, running_loss / 100])
            running_loss = 0.0

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        val_acc_store.append([epoch+1, (100 * correct / total)])

        best_model_wts = copy.deepcopy(model.state_dict())

    model.train()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
model.load_state_dict(best_model_wts)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

torch.save(model.state_dict(), 'model.ckpt')