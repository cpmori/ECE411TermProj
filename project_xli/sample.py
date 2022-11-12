import pruning
import thresnet as tnet
import resnet
import torch
import copy 

Train_Batches = 30
Valid_Batches = 40
Num_SubNets = 5

if __name__ == "__main__":
    target_net = resnet.resnet20()
    target_net.load_state_dict(torch.load('./saved_models/coarse.pth'))
    thres_net = tnet.ThresNet()
    print(type(target_net))
    for subnet_count in range(Num_SubNets):
        pruning.prune_net(copy.deepcopy(target_net), thres_net)
        # train sample
        train_loss = 0
        for batch in range(Train_Batches):
            print('train batch')
            break
        valid_loss = 0
        for batch in range(Valid_Batches):
            print('valid batch')
            break
        
