import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import resnet
import pruning
import matplotlib.pyplot as plt
import numpy as np
import copy

torch.manual_seed(0)
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
correct_predictions = {}
all_predictions = {}

#net = resnet.resnet20().cuda()
# coarse train (change Prune = 0 in resnet.py)
# 90.75% accuracy
#net.load_state_dict(torch.load('./saved_models/coarse20_orig0005.pth'))

# modified alpha equation (change Prune = 2 in resnet.py)
# 42.2317% reduction, 89.03% accuracy
#net.load_state_dict(torch.load('./saved_models/fine20_noAlphaChange_2_orig0005.pth'))

# paper alpha equation (change Prune = 1 in resnet.py)
# 46.2791% reduction, 88.31% accuracy
#net.load_state_dict(torch.load('./saved_models/fine20_orig0005.pth'))

# loss backward propagation every batch during sampling (Prune = 1)
# 44.6282% reduction, 87.87% accuracy
#net.load_state_dict(torch.load('./saved_models/fine20_freqUpdateWithAlpha_orig0005.pth'))

net = resnet.resnet56().cuda()
# coarse train (change Prune = 0 in resnet.py)
# 92.31% accuracy
#net.load_state_dict(torch.load('./saved_models/coarse56_orig0005.pth'))

# paper alpha equation (change Prune = 1 in resnet.py)
# 42.8446% reduction, 91.62% accuracy
#net.load_state_dict(torch.load('./saved_models/fine56_orig0005.pth'))

# modified alpha equation (change Prune = 2 in resnet.py)
# 44.2761% reduction, 91.44% accuracy
net.load_state_dict(torch.load('./saved_models/fine56_modifiedAlpha_orig0005.pth'))

pruning.check_net(net)
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
    for i, cls in enumerate(labels.flatten()):
        all_predictions[classes[cls]] += 1
        if predicted[i] == cls:
            correct_predictions[classes[cls]] += 1
print("latest: ", total, correct, correct/total)

plt.figure()
#print(all_predictions)
#print(correct_predictions)
plt.bar(list(correct_predictions.keys()), correct_predictions.values())
plt.title("Class accuracy")
#plt.savefig('class_pred.svg')
plt.show()