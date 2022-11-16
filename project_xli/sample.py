import pruning
import thresnet as tnet
import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy 

Train_Epochs = 40
Valid_Epochs = 5
Num_SubNets = 30

def changeConvFilter(target_net:nn.Module, sampled_net:nn.Module):
    with torch.no_grad():
        for name, param in target_net.named_parameters():
            print(name)
            print(target_net.get_parameter(name))

if __name__ == "__main__":
    batch_size = 128
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='/home/xing/Classes/ECE411/data', train=True,
                                        download=True, transform=transform)
    train_size = int(.8 * len(dataset))
    valid_size = len(dataset) - train_size
    print("train data: ", train_size, " valid data: ", valid_size)
    trainset, validset = torch.utils.data.random_split(dataset,[train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle = True, num_workers = 2)
    
    
    target_net = resnet.resnet20()
    target_net.load_state_dict(torch.load('./saved_models/coarse.pth'))
    thres_net = tnet.ThresNet().cuda()
    print(type(target_net))

    #resnet.test(target_net)
    for subnet_count in range(Num_SubNets):
        if subnet_count > 20:
            Train_Epochs = 30
            learnrate = 0.01
        else:
            Train_Epochs = 40
            learnrate = 0.05
        print(f"sampled net{subnet_count}")
        sampled_net = copy.deepcopy(target_net).cuda()
        pruning.prune_net(sampled_net, thres_net)
        #pruning.update_net(sampled_net)
        #resnet.test(sampled_net)
        
        criterion = nn.CrossEntropyLoss().cuda()
# SGD
        optimizer = optim.SGD(sampled_net.parameters(),
                        lr = learnrate,
                        momentum=.9,
                        weight_decay=.0005)
        # train sample
        net_loss = 0

        #  
        for epoch in range(Train_Epochs):
            min_train_loss = 1  # used to sample best conv filters
            min_conv_filter_net = sampled_net.state_dict() # saved network with best conv filters (place holder)
            for i, data in enumerate(trainloader,0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = sampled_net(inputs)
                loss = criterion(outputs, labels)
                if (loss < min_train_loss):
                    print(f'minloss at batch:{i}. minloss:{loss}')
                    min_train_loss = loss
                    min_conv_filter_net = sampled_net.state_dict()
                
                if i%100 == 0:
                    print(f'train batch {i}. loss: {loss}. minloss: {min_train_loss}')
                    #pruning.check_net(sampled_net)
                    #print(resnet.test(sampled_net))
            print(f'train epoch {epoch}. minloss: {min_train_loss}')
            min_train_loss.backward()
            optimizer.step()
        print(min_train_loss)
        pruning.update_net(sampled_net)
        optimizer.param_groups[0]['lr'] = 0.001
        min_valid_loss = 0 # used to sample best alpha terms
        min_alpha_net = sampled_net.state_dict() # saved network with best alpha terms (place holder)
        for epoch in range(Valid_Epochs):
            print('valid epoch')
            break
        
        #resnet.test(sampled_net)
        
