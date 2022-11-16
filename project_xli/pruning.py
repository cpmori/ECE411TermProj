import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
        print(self)
        print(alphas.size(), noise_alpha.size())
        while sum_of_noise_alpha < thresh:
            selected_vals, selected_idx = torch.topk(noise_alpha,c,dim=0)
            sum_of_noise_alpha = selected_vals.sum()
            c += 1
            print(c)
        print(alphas.size(), c)
        _, pruned_alpha_idx = torch.topk(alphas, c, dim=0)
        mask = torch.zeros(alphas.size())
        mask[pruned_alpha_idx] = 1
        print(mask.flatten())
        return mask

# adaptive thres selection via thresnet
class PaperPrune(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"
    def __init__(self, thresnet, thres_array):
        self.thresnet : tnet.ThresNet = thresnet
        self.thres_array = thres_array
        self.log_prob = torch.Tensor()

    def compute_mask(self, alphas, default_mask):
        # log(log(uniform)) noise
        noise_alpha = alphas - torch.log(-torch.log(torch.rand(alphas.size()))).cuda()
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
        _, pruned_alpha_idx = torch.topk(alphas, c, dim=0)
        mask = torch.zeros(alphas.size()).cuda()
        mask[pruned_alpha_idx] = 1
        return mask

    def get_thresnet_returns(self):
        return self.log_prob

def prune_layer(module, name, thresnet):
    #ConstantPrune.apply(module, name, 0.7)
    PaperPrune.apply(module, name, thresnet, [0.5,0.6,0.7,0.8,0.9])

def prune_net(net, thresnet):
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            #print('----------')
            #print(name)
            prune_layer(module, 'alpha1',thresnet)
            prune_layer(module, 'alpha2',thresnet)
            
def update_layer(module, name):
    print(module, name)
    old_weights = torch.clone(getattr(module, name+"_orig"))
    old_mask = torch.clone(getattr(module, name+'_mask'))
    print(name)
    
    prune.remove(module, name)

    # I dont think this is necessary? (layer is the same before and after)
    
    with torch.no_grad():
        layer = getattr(module, name)
        prev_layer = torch.clone(layer)
        #print(layer.flatten())
        layer[old_mask.bool()] = old_weights[old_mask.bool()]
        print(layer.flatten())
        print(torch.equal(prev_layer, layer))

def update_net(net):
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            update_layer(module, 'alpha1')
            update_layer(module, 'alpha2')

def check_net(net):
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name:
            if ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
                #print(dir(module))
                print(getattr(module, 'alpha1').size())
                print(torch.count_nonzero(getattr(module, 'alpha1')))
                #print(getattr(module, 'alpha1_orig').size())
                #check_layer(module, 'alpha1')
                #check_layer(module, 'alpha2')
            elif 'conv' in name:
                print(module, name)
                print(module.weight.size()) # weight of the conv net