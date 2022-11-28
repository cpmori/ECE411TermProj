import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn.init as init
import logging
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

# assume each episode is finished at the end of subnetwork (line 12 of algorithm 2)
# only 1 reward, store the sum of all log_probs
# gradient is then R * sum(log_probs)

class ThresNet(nn.Module):
    def __init__(self):
        super(ThresNet, self).__init__()

        self.in16_1 = nn.Linear(16,32)
        self.in16_2 = nn.Linear(32,64)
        self.in16_3 = nn.Linear(64,128)
        self.in32_1 = nn.Linear(32,64)
        self.in32_2 = nn.Linear(64,128)
        self.in64 = nn.Linear(64,128)

        self.shrink1 = nn.Linear(128,64)
        self.shrink2 = nn.Linear(64,16)
        self.shrink3 = nn.Linear(16,5)

        self.apply(_weights_init)
        
    def forward(self, x):
        out = x.flatten().clone() # idk why clone is needed but it breaks without
        #print(x.size())
        #print(x.size())
        if x.size()[0] == 16:
            #print(x.size()[0])
            out = F.relu(self.in16_1(out))
            out = F.relu(self.in16_2(out))
            out = F.relu(self.in16_3(out))
        elif x.size()[0] == 32:
            #print(x.size()[0])
            out = F.relu(self.in32_1(out))
            out = F.relu(self.in32_2(out))
        else:
            
            out = F.relu(self.in64(out))
        out = F.relu(self.shrink1(out))
        out = F.relu(self.shrink2(out))
        out = F.softmax(self.shrink3(out),dim=0)
        #print(out)
        return out

def get_threshold(net, alpha):
    selections = net(alpha)
    m = Categorical(selections)
    selection = m.sample()
    log_prob = m.log_prob(selection)
    return selection, log_prob

def update_policy(optimizer:torch.optim.Adam, rewards, probs, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'Updating Thresnet Policy. R:{rewards}. log_prob{torch.sum(torch.stack(probs,dim=0))}')
    optimizer.zero_grad()
    loss = torch.mean(rewards * torch.sum(torch.stack(probs,dim=0)))
    logging.info(f'ThresNet Loss: {loss}')
    loss.backward()
    optimizer.step()
    return

def calc_episode_reward(subnet:nn.Module, losses, log_file, hyperparam = 2):
    import numpy as np
    l_avg = np.mean(losses)
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info(f'Calculating Thresnet Episode Reward. # of losses: {len(losses)} avg: {l_avg}')
    param_count = 0
    reduced_count = 0
    for name, param in subnet.named_modules():
        if "alpha1" in list(dir(param)) or "alpha2" in list(dir(param)):
            conv1_size = param.conv1.weight.size()
            conv2_size = param.conv2.weight.size()
            # # of params in each filter of the corresponding alpha
            filter1_params = np.prod(list(conv1_size)[1:])
            filter2_params = np.prod(list(conv2_size)[1:])
            # # of zero alphas
            num_zeros1 = conv1_size[0] - torch.count_nonzero(param.alpha1)
            num_zeros2 = conv2_size[0] - torch.count_nonzero(param.alpha2)
            # # of params zeroed out
            reduced_count += num_zeros1 * filter1_params
            reduced_count += num_zeros2 * filter2_params
    # total params in the network
    for x in filter(lambda p: p.requires_grad, subnet.parameters()):
        param_count += np.prod(x.data.cpu().numpy().shape)
    # theoretical params
    pruned_param_count = param_count - reduced_count
    logging.info(f'Original count: {param_count}, After Prune count: {pruned_param_count}, % Reduced: {1-pruned_param_count/param_count}')
    #reward = -(l_avg + hyperparam * pruned_param_count)
    reward = -(l_avg + hyperparam * pruned_param_count/param_count)
    logging.info(f'Reward: {reward}')
    return reward, pruned_param_count/param_count

def model_info(net:nn.Module):
    #print(net)
    for name, param in net.named_modules():
        if 'in' in name:
            print(name, param.weight.requires_grad)

if __name__ == "__main__":
    net = ThresNet()
    model_info(net)