import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

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

    def forward(self, x):
        x = x.flatten()
        print(x.size())
        if x.size()[0] == 16:
            print(x.size()[0])
            out = F.relu(self.in16_1(x))
            out = F.relu(self.in16_2(out))
            out = F.relu(self.in16_3(out))
        elif x.size()[0] == 32:
            print(x.size()[0])
            out = F.relu(self.in32_1(x))
            out = F.relu(self.in32_2(out))
        else:
            out = F.relu(self.in64(x))
        out = F.relu(self.shrink1(out))
        out = F.relu(self.shrink2(out))
        out = F.softmax(self.shrink3(out),dim=0)
        return out

def get_threshold(net, alpha):
    selections = net(alpha)
    m = Categorical(selections)
    selection = m.sample()
    log_prob = m.log_prob(selection)
    return selection, log_prob

def update_policy(optimizer:torch.optim.RMSprop, rewards, probs):
    print(f'Updating Thresnet Policy. R:{rewards}. log_prob{probs}')
    optimizer.zero_grad()
    loss = torch.mean(rewards * torch.sum(probs))
    loss.backward()
    optimizer.step()
    return

def calc_episode_reward(subnet, losses, hyperparam = 2):
    print(f'Calculating Thresnet Episode Reward. losses{losses}')
    import numpy as np
    l_avg = torch.mean(losses)
    param_count = 0
    for x in filter(lambda p: p.requires_grad, subnet.parameters()):
        param_count += np.prod(x.data.numpy().shape)
    print(f'Param count: {param_count}')
    reward = -(l_avg + hyperparam * param_count)
    return reward

def model_info(net:nn.Module):
    #print(net)
    for name, param in net.named_modules():
        if 'in' in name:
            print(name, param.weight.requires_grad)

if __name__ == "__main__":
    net = ThresNet()
    model_info(net)