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
import matplotlib.pyplot as plt
from collections import OrderedDict
Num_SubNets = 30

def changeConv(target_net:nn.Module, conv_net:OrderedDict):
    with torch.no_grad():
        for name, param in target_net.named_parameters():
            if 'conv' in name:
                #print(name)
                param.copy_(conv_net.get(name))
                setattr(target_net,name,conv_net.get(name))

def changeAlpha(target_net:nn.Module, alpha_net:OrderedDict):
    with torch.no_grad():
        for name, param in target_net.named_parameters():
            if 'alpha' in name:
                #print(name)
                
                new_alpha = param.clone().cuda()
                #print(new_alpha.flatten())
                #print((alpha_net.get(name+'_mask')*alpha_net.get(name+'_orig')).flatten())
                non_zero_idx = torch.nonzero(alpha_net.get(name+'_mask'))[:,0]
                new_alpha[non_zero_idx] = alpha_net.get(name+'_orig')[non_zero_idx]
                #print(new_alpha.flatten())
                #print(param.flatten())
                param.copy_(new_alpha)
                #print(param.flatten())
                #print(alpha_net.get(name+'_orig').flatten())
                #print(alpha_net.get(name+'_mask').size())
            #print(target_net.get_parameter(name))

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
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
    print("train data: ", train_size, " valid data: ", valid_size)
    trainset, validset = torch.utils.data.random_split(dataset,[train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle = True, num_workers = 2)
    
    
    target_net = resnet.resnet20().cuda()
    target_net.load_state_dict(torch.load('./saved_models/coarse20_orig0005.pth'))
    thres_net = tnet.ThresNet().cuda()
    print(type(target_net))

    # test variables
    #resnet.test(target_net)

    # start sampling
    rewards = []
    
    for subnet_count in range(Num_SubNets):
        print("-----------New SubNet------------")
        if subnet_count > 20:
            Train_Epochs = 30
            Valid_Epochs = 5
            learnrate = 0.01
        else:
            Train_Epochs = 40
            Valid_Epochs = 5
            learnrate = 0.05
        print(f"sampled net{subnet_count}")
        sampled_net = copy.deepcopy(target_net).cuda()
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
        tnet_optimizer = optim.Adam(thres_net.parameters(), lr=1e-3)
        # train sample
        net_loss = []

        #  TRAIN LOOP
        Train_Loss_History = []
        for epoch in range(Train_Epochs):
            min_train_loss = 10000  # used to sample best conv filters
            min_conv_filter_net = sampled_net.state_dict() # saved network with best conv filters (place holder)
            for i, data in enumerate(trainloader,0):
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = sampled_net(inputs)
                loss = criterion(outputs, labels)
                #print(loss)
                net_loss.append(loss.item())
                if (loss < min_train_loss):
                    print(f'\t\tminloss at batch:{i}. minloss:{loss:.3f}')
                    min_train_loss = loss
                    min_conv_filter_net = sampled_net.state_dict()
                    #print(len(net_loss), net_loss[i], np.mean(net_loss))
                #if i%100 == 0:
                #    print(f'\ttrain batch {i}. loss: {loss:.3f}. minloss: {min_train_loss:.3f}')
                #    pruning.check_net(sampled_net)
                    #print(resnet.test(sampled_net))
            print(f'train epoch {epoch}. minloss: {min_train_loss:.3f}')
            min_train_loss.backward()
            pruning.check_net(sampled_net)
            optimizer.step()
            Train_Loss_History.append(min_train_loss.item())
        print("---------end of training sampled net---------")
        #pruning.check_net(sampled_net)
        #pruning.update_net(sampled_net)
        plt.subplot(2,1,1)
        plt.plot(Train_Loss_History)
        #plt.legend([learnrate])
        #plt.show()

        # VALID LOOP
        optimizer.param_groups[0]['lr'] = 0.001
        for epoch in range(Valid_Epochs):
            min_valid_loss = 10000  # used to sample best conv filters
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
                    
                #if i%100 == 0:
                #    print(f'\tvalid batch {i}. loss: {loss}. minloss: {min_valid_loss}')
                    #pruning.check_net(sampled_net)
                    #print(resnet.test(sampled_net))
            print(f'valid epoch {epoch}. minloss: {min_valid_loss}')
            min_valid_loss.backward()
            optimizer.step()
        print("---------end of validation sampled net---------")
        print(min_alpha_net is min_conv_filter_net)

        # update thresnet
        # idk if this is working properly
        reward = tnet.calc_episode_reward(sampled_net,net_loss)
        tnet.update_policy(tnet_optimizer,reward,log_probs)
        rewards.append(reward.item())

        # update targetnet
        #resnet.test(target_net)
        changeConv(target_net, min_conv_filter_net)
        changeAlpha(target_net, min_alpha_net)
        #resnet.test(target_net)

        #resnet.test(sampled_net)
    plt.subplot(2,1,2)
    plt.plot(rewards)
    plt.show()
