import pruning
import thresnet as tnet
import torch

Train_Batches = 30
Valid_Batches = 40
Num_SubNets = 5

if __name__ == "__main__":
    target_net = torch.load('./saved_models/coarse.pth').cuda()
    thres_net = tnet.ThresNet().cuda()
    print('Begin Sampling And Train')
    for subnet_count in range(Num_SubNets):
        pruning.prune_net(target_net, thres_net)
        # train sample
        train_loss = 0
        for batch in range(Train_Batches):
            print('train batch')
        valid_loss = 0
        for batch in range(Valid_Batches):
            print('valid batch')
        
