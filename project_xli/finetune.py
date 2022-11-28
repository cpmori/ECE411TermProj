import copy
import resnet
import pruning
import thresnet as tnet
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torchvision

if __name__ == '__main__':
    batch_size = 128
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='/home/lixin/Classes/ECE411/data', train=True,
                                        download=True, transform=transform)
    train_size = int(.8 * len(dataset))
    valid_size = len(dataset) - train_size


    trainset, validset = torch.utils.data.random_split(dataset,[train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle = True, num_workers = 4)

    net = resnet.resnet20().cuda()
    thres_net = tnet.ThresNet().cuda()
    net.load_state_dict(torch.load('./saved_models/sampled20_orig0005.pth'))
    thres_net.load_state_dict(torch.load('./saved_models/thres20_orig0005.pth'))


    pruning.prune_net(net, thres_net)

    Fine_Epoch = 100

    criterion = nn.CrossEntropyLoss().cuda()
    # SGD
    optimizer = optim.SGD(net.parameters(),
                        lr = .01,
                        momentum=.9,
                        weight_decay=.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Fine_Epoch)
    
    train_acc = []
    saved_path = './saved_models/fine20_orig0005.pth'

    for epoch in range(Fine_Epoch):
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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.flatten()).sum().item()
            if i%100 == 0:
                print('batch: ',i,' loss: ',loss.data.item())
            #print(net.get_parameter("layer1.0.alpha1").flatten())
            #print(F.softmax(net.get_parameter("layer1.0.alpha1").clone().flatten(),dim=0))
        scheduler.step()
        print("training: ", train_total, train_correct, train_correct/train_total)
        train_acc.append(train_correct/train_total)
        pruning.check_net(net)
        valid_total = 0
        valid_correct = 0
        with torch.no_grad():
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
    pruning.update_net(net)
    if True:
        torch.save(net.state_dict(),saved_path)