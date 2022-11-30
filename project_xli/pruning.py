import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

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
    def __init__(self, thresnet, thres_array, log_prob):
        self.thresnet : tnet.ThresNet = thresnet
        self.thres_array = thres_array
        self.log_prob = log_prob

    def compute_mask(self, alphas, default_mask):
        # log(log(uniform)) noise (c approx to 1/4 to 1/6 of alpha size)
        #noise_alpha = alphas - torch.log(-torch.log(torch.rand(alphas.size()))).cuda()
        
        # log uniform noise (similar to loglog)
        #noise_alpha = alphas - torch.log(torch.rand(alphas.size())).cuda()
        
        # uniform noise (c approx to half of alpha size)
        noise_alpha = alphas - torch.rand(alphas.size()).cuda()

        noise_alpha = F.softmax(noise_alpha, dim=0)
        c = 1
        thresh_select, log_prob = tnet.get_threshold(self.thresnet, alphas)
        self.log_prob.append(log_prob)
        #print(len(self.log_prob))
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
        print(self.log_prob)
        return self.log_prob
    def clear_thresnet(self):
        self.log_prob = []

def prune_layer(module, name, thresnet, pruner:PaperPrune):
    #ConstantPrune.apply(module, name, 0.7)
    pruner.apply(module, name, pruner.thresnet, pruner.thres_array, pruner.log_prob)

def prune_net(net, thresnet):
    log_prob = []
    pruner = PaperPrune(thresnet, [0.55,0.6,0.65,0.7,0.75], log_prob)
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            #print('----------')
            #print(name)
            prune_layer(module, 'alpha1',thresnet, pruner)
            prune_layer(module, 'alpha2',thresnet, pruner)
    return log_prob
            
def update_layer(module, name):
    #print(module, name)
    old_weights = torch.clone(getattr(module, name+"_orig"))
    old_mask = torch.clone(getattr(module, name+'_mask'))
    #print(name)
    
    prune.remove(module, name)

    # I dont think this is necessary? (layer is the same before and after)
    
    #with torch.no_grad():
    #    layer = getattr(module, name)
    #    prev_layer = torch.clone(layer)
    #    #print(layer.flatten())
    #    layer[old_mask.bool()] = old_weights[old_mask.bool()]
    #    print(layer.flatten())
    #    print(torch.equal(prev_layer, layer))

def update_net(net):
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            update_layer(module, 'alpha1')
            update_layer(module, 'alpha2')

def check_net(net):
    reduced_count = 0
    param_count = 0
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        param_count += np.prod(x.data.cpu().numpy().shape)
    for name, param in net.named_modules():
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
    pruned_param_count = param_count - reduced_count
    print(f'Original count: {param_count}, After Prune count: {pruned_param_count}, % Reduced: {1-pruned_param_count/param_count}')