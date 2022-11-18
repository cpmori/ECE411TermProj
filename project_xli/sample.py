import pruning
import thresnet as tnet
import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import copy 
import numpy as np
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
    target_net.load_state_dict(torch.load('./saved_models/coarse20_randAlpha.pth'))
    thres_net = tnet.ThresNet().cuda()
    print(type(target_net))

    #resnet.test(target_net)
    for subnet_count in range(Num_SubNets):
        print("-----------New SubNet------------")
        if subnet_count > 20:
            Train_Epochs = 30
            learnrate = 0.01
        else:
            Train_Epochs = 40
            learnrate = 0.05
        print(f"sampled net{subnet_count}")
        sampled_net = target_net.cuda()
        log_probs = pruning.prune_net(sampled_net, thres_net)
        print(len(log_probs))
        sum_log_prob = 0
        for log_prob in log_probs:
            sum_log_prob += log_prob
        print(sum_log_prob)

        #pruning.update_net(sampled_net)
        #resnet.test(sampled_net)
        
        criterion = nn.CrossEntropyLoss().cuda()
        # SGD
        optimizer = optim.SGD(sampled_net.parameters(),
                        lr = learnrate,
                        momentum=.9,
                        weight_decay=.0005)
        # train sample
        net_loss = []

        #  TRAIN LOOP
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
                net_loss.append(loss.item())
                if (loss < min_train_loss):
                    print(f'\t\tminloss at batch:{i}. minloss:{loss:.3f}')
                    min_train_loss = loss
                    min_conv_filter_net = sampled_net.state_dict()
                    #print(len(net_loss), net_loss[i], np.mean(net_loss))
                    tnet.calc_episode_reward(sampled_net,net_loss)
                if i%100 == 0:
                    print(f'\ttrain batch {i}. loss: {loss:.3f}. minloss: {min_train_loss:.3f}')
                    #pruning.check_net(sampled_net)
                    #print(resnet.test(sampled_net))
            print(f'train epoch {epoch}. minloss: {min_train_loss:.3f}')
            min_train_loss.backward()
            optimizer.step()
        print("---------end of training sampled net---------")

        #pruning.update_net(sampled_net)


        # VALID LOOP
        optimizer.param_groups[0]['lr'] = 0.001
        for epoch in range(Valid_Epochs):
            min_valid_loss = 1  # used to sample best conv filters
            min_alpha_net = sampled_net.state_dict() # saved network with best conv filters (place holder)
            for i, data in enumerate(validloader,0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = sampled_net(inputs)
                loss = criterion(outputs, labels)
                net_loss.append(loss.item())
                if (loss < min_valid_loss):
                    print(f'\t\tminloss at batch:{i}. minloss:{loss}')
                    min_valid_loss = loss
                    min_alpha_net = sampled_net.state_dict()
                
                if i%100 == 0:
                    print(f'\tvalid batch {i}. loss: {loss}. minloss: {min_valid_loss}')
                    #pruning.check_net(sampled_net)
                    #print(resnet.test(sampled_net))
            print(f'valid epoch {epoch}. minloss: {min_valid_loss}')
            min_valid_loss.backward()
            optimizer.step()
        print("---------end of validation sampled net---------")
        

        # update thresnet
        tnet.calc_episode_reward(sampled_net,net_loss)

        # update targetnet
        
        #resnet.test(sampled_net)
        
