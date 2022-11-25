import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import resnet

import matplotlib.pyplot as plt
import numpy as np
import copy

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128
testset = torchvision.datasets.CIFAR10(root='/home/lixin/Classes/ECE411/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

correct = 0
total = 0
best_correct = 0
best_total = 0
correct_predictions = {}
all_predictions = {}

net = resnet.resnet20().load_state_dict(torch.load('./saved_models/coarse20_orig0005.pth'))

for cls in classes:
    correct_predictions[cls] = 0
    all_predictions[cls] = 0
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    best_total += labels.size(0)
    best_correct += (predicted == labels.flatten()).sum().item()
    for i, cls in enumerate(labels.flatten()):
        all_predictions[classes[cls]] += 1
        if predicted[i] == cls:
            correct_predictions[classes[cls]] += 1
print("latest: ", total, correct, correct/total)
print("best validation: ", best_total, best_correct, best_correct/best_total)

plt.figure()
print(all_predictions)
print(correct_predictions)
plt.bar(list(correct_predictions.keys()), correct_predictions.values())
plt.title("Class accuracy")
plt.savefig('class_pred.svg')
plt.show()