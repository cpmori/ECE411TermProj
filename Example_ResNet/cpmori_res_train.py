import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import resnet as r
import numpy as np
from matplotlib import pyplot as plt
import time

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def resnet_validation(testloader, classes, net, device):
    

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            _,predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()
    return 100 * correct // total
    #print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    """ # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %') """


def resnet_train(trainloader, net, device, PATH, num_epochs,testloader,classes):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    accuracy = []
    start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        accuracy.append(resnet_validation(testloader,classes,net,device))
        epoch_end = time.time()
        epoch_time = epoch_start - epoch_end
        print("Epoch ", epoch, " finish in ", epoch_time)
    end = time.time()
    total_time = end - start
    print("Finished Training in ", total_time)
    
    torch.save(net.state_dict(), PATH)
    return accuracy


def main():

    

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_epochs = 3


    net = r.resnet20()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)


    PATH = './cifar_net.pth'
    path = PATH
    accuracy = resnet_train(trainloader, net, device, PATH, num_epochs,testloader,classes)


    net = r.resnet20()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)

    net.load_state_dict(torch.load(PATH))

    

    goodness = resnet_validation(testloader, classes, net, device)
    print("Model is ", goodness , "%")

    plt.plot(range(num_epochs), accuracy)
    plt.title("Accurcay vs Number of Epochs")
    plt.xlabel("Num Epochs")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    main()