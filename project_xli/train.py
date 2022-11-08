import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

import resnet
import reducedresnet

import matplotlib.pyplot as plt
import numpy as np
import copy
def alphaL2(net: nn.Module):
    sumofAlphaSquared = 0
    for name, param in net.named_parameters():
        if 'alpha' in name:
            sumofAlphaSquared += param.norm()**2
    #print(sumofAlphaSquared)
    return torch.sqrt(sumofAlphaSquared)

def unweightedFilter(param, alphas):
    new_param = torch.empty(param.size())
    #print(alphas.size(), param.size())
    for i, filter in enumerate(param):
        alpha = alphas[i]
        new_param[i] = filter/alpha
    return new_param

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

dataset = torchvision.datasets.CIFAR10(root='/home/xing/Classes/ECE411/data', train=True,
                                        download=True, transform=transform)
train_size = int(.8 * len(dataset))
valid_size = len(dataset) - train_size
print("train data: ", train_size, " valid data: ", valid_size)
trainset, validset = torch.utils.data.random_split(dataset,[train_size, valid_size])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle = True, num_workers = 2)
testset = torchvision.datasets.CIFAR10(root='/home/xing/Classes/ECE411/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# settings
Coarse_Epoch = 100

# target model
net = resnet.resnet20()
if torch.cuda.is_available():
    net.cuda()
    print(torch.cuda.get_device_name(0))

criterion = nn.CrossEntropyLoss().cuda()
# SGD
optimizer = optim.SGD(net.parameters(),
                        lr = .1,
                        momentum=.9,
                        weight_decay=.0005)
# cosine scheduler for lr
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Coarse_Epoch)
print(optimizer)


# train & valid
best_model = resnet.resnet20()
best_val = 0
best_epoch = 0
iteration = []
train_acc = []
valid_acc = []

for epoch in range(Coarse_Epoch):
    # generate resnet (via conv weights and channels)
    #target_alpha = copy.deepcopy(net.alpha)
    #target_filters = []
    #layer_transfer_network = 0
    #for name, param in net.named_parameters():
    #    if 'conv' in name:
    #        #print(name, param.size())
    #        filterparam = unweightedFilter(param, target_alpha[layer_transfer_network])
    #        target_filters.append(filterparam)
    #        layer_transfer_network += 1
    #red_net = reducedresnet.resnet20(target_alpha, target_filters, 0)
    # check structure
    #for name, param in red_net.named_parameters():
    #    if 'layer' in name and 'conv' in name:
    #        print(name, param.size())

    iteration.append(epoch)
    running_loss = 0
    print(epoch)
    train_total = 0
    train_correct = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = net(inputs)
        coarse_loss_regu = .1 * alphaL2(net)
        loss = criterion(outputs, labels) + coarse_loss_regu
        loss.backward()
        optimizer.step()
        #print(optimizer)
        
        _, predicted = torch.max(outputs.data, dim=1)
        train_total += labels.size(0)
        train_correct += (predicted == labels.flatten()).sum().item()
        if i%100 == 0:
            print(i,loss.data)
    print("training: ", train_total, train_correct, train_correct/train_total)
    train_acc.append(train_correct/train_total)


    # validation
    valid_total = 0
    valid_correct = 0
    for i, data in enumerate(validloader,0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        valid_total += labels.size(0)
        valid_correct += (predicted == labels.flatten()).sum().item()
    print("validation: ", valid_total, valid_correct, valid_correct/valid_total)
    valid_acc.append(valid_correct/valid_total)
    if valid_correct/valid_total > best_val:
        best_val = valid_correct/valid_total
        best_epoch = epoch
        best_model = copy.deepcopy(net)
    print("best model (epoch, accuracy): ", best_epoch, best_val)

    scheduler.step()
    #print(optimizer)

# test

correct = 0
total = 0
best_correct = 0
best_total = 0
correct_predictions = {}
all_predictions = {}
for cls in classes:
    correct_predictions[cls] = 0
    all_predictions[cls] = 0
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = net(inputs)
    best_outputs = best_model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    _, best_predicted = torch.max(best_outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    best_total += labels.size(0)
    best_correct += (best_predicted == labels.flatten()).sum().item()
    for i, cls in enumerate(labels.flatten()):
        all_predictions[classes[cls]] += 1
        if best_predicted[i] == cls:
            correct_predictions[classes[cls]] += 1
print("latest: ", total, correct, correct/total)
print("best validation: ", best_total, best_correct, best_correct/best_total)
plt.figure()
plt.plot(iteration, train_acc)
plt.plot(iteration, valid_acc)
plt.figure()
print(all_predictions)
print(correct_predictions)
plt.bar(list(correct_predictions.keys()), correct_predictions.values())
plt.show()
    