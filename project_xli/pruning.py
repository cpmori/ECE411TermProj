import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.nn.functional as F

import thresnet as tnet
import resnet
# constant thres value
class ConstantPrune(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"
    def __init__(self, thres):
        self.thres = thres

    def compute_mask(self, alphas, default_mask):
        # log(log(uniform)) noise
        noise_alpha = alphas - torch.log(-torch.log(torch.rand(alphas.size())))
        # uniform noise
        #noise_alpha = alpha - torch.log(-torch.log(torch.rand(alpha.size())))
        noise_alpha = F.softmax(noise_alpha, dim=0)
        c = 1
        thresh = self.thres
        sum_of_noise_alpha = 0
        while sum_of_noise_alpha < thresh:
            selected_vals, selected_idx = torch.topk(noise_alpha,c,dim=0)
            sum_of_noise_alpha = selected_vals.sum()
            c += 1
        _, pruned_alpha_idx = torch.topk(alphas, c)
        mask = torch.zeros(alphas.size())
        mask[pruned_alpha_idx] = 1

        return mask

# adaptive thres selection via thresnet
class PaperPrune(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"
    def __init__(self, thresnet, thres_array):
        self.thresnet : tnet.ThresNet = thresnet
        self.thres_array = thres_array
        self.log_prob = torch.tensor()

    def compute_mask(self, alphas, default_mask):
        # log(log(uniform)) noise
        noise_alpha = alphas - torch.log(-torch.log(torch.rand(alphas.size())))
        # uniform noise
        #noise_alpha = alpha - torch.log(-torch.log(torch.rand(alpha.size())))
        noise_alpha = F.softmax(noise_alpha, dim=0)
        c = 1

        thresh_select, log_prob = tnet.get_threshold(self.thresnet, alphas)
        self.log_prob = log_prob
        thresh = self.thres_array[thresh_select]

        sum_of_noise_alpha = 0
        while sum_of_noise_alpha < thresh:
            selected_vals, selected_idx = torch.topk(noise_alpha,c,dim=0)
            sum_of_noise_alpha = selected_vals.sum()
            c += 1
        _, pruned_alpha_idx = torch.topk(alphas, c)
        mask = torch.zeros(alphas.size())
        mask[pruned_alpha_idx] = 1

        return mask

    def get_thresnet_returns(self):
        return self.log_prob

def prune_layer(module, name, thresnet):
    PaperPrune.apply(module, name, thresnet)

def prune_net(net, thresnet):
    for name, param in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            prune_layer(param, 'alpha1',thresnet)
            prune_layer(param, 'alpha2',thresnet)
            
def update_layer(module, name):
    old_weights = torch.clone(getattr(module, name+"_orig"))
    old_mask = torch.clone(getattr(module, name+'_mask'))
    prune.remove(module, name)
    with torch.no_grad():
        layer = getattr(module, name)
        layer[old_mask] = old_weights[old_mask]

def update_net(net):
    for name, param in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            update_layer(param, 'alpha1')
            update_layer(param, 'alpha2')
