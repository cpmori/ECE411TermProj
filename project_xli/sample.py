import pruning
import thresnet as tnet
import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import copy 

Train_Batches = 30
Valid_Batches = 40
Num_SubNets = 5

if __name__ == "__main__":
    target_net = resnet.resnet20()
    target_net.load_state_dict(torch.load('./saved_models/coarse.pth'))
    thres_net = tnet.ThresNet()
    print(type(target_net))




    resnet.test(target_net)
    for subnet_count in range(Num_SubNets):
        sampled_net = copy.deepcopy(target_net)
        pruning.prune_net(sampled_net, thres_net)
        resnet.test(sampled_net)
        criterion = nn.CrossEntropyLoss().cuda()
# SGD
        optimizer = optim.SGD(sampled_net.parameters(),
                        lr = .05,
                        momentum=.9,
                        weight_decay=.0005)
        # train sample
        net_loss = 0
        min_train_loss = 0  # used to sample best conv filters
        min_conv_filter_net = copy.deepcopy(sampled_net) # network with best conv filters (place holder)
        #  
        for batch in range(Train_Batches):
            current_train_los = 0
            print('train batch')
            break

        min_valid_loss = 0 # used to sample best alpha terms
        min_alpha_net = copy.deepcopy(sampled_net) # network with best alpha terms (place holder)
        for batch in range(Valid_Batches):
            print('valid batch')
            break
        pruning.update_net(sampled_net)
        resnet.test(sampled_net)
        
