'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
# https://github.com/akamaster/pytorch_resnet_cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import numpy as np

#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet164','resnet1202']
__all__ = ['resnet20']
class AlphaTerm():
    a = []
    weights = []
    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        #print(classname)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Conv2d):
                print(m.weight.size())
                in_planes = m.weight.size()[1]
                alpha = torch.from_numpy(np.random.uniform(0,1,size=in_planes))
                alpha = F.softmax(alpha, dim = 0)
                print(alpha)
                AlphaTerm.a.append(alpha)
                weights_copy = m.weight.clone().detach()
                AlphaTerm.weights.append(weights_copy)
                for i in range(in_planes):
                    weights_copy[i] *= alpha[i]
                #print(weights_copy.is_leaf)
                #print(weights_copy.requires_grad_(True).is_leaf)
                #print(weights_copy[0][0][0][0],m.weight[0][0][0][0],alpha[0])
                m.weight = nn.Parameter(weights_copy.requires_grad_(True))
                #a.append(alpha)
    

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ReducedBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, inter_planes, planes, alphas, weights,stride=1, option='A'):
        super(ReducedBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes-in_planes)//2, (planes-in_planes)//2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
                self.sc_weights = torch.from_numpy(np.random.uniform(0,1,size=in_planes))
                self.sc_weights = F.softmax(self.sc_weights, dim = 0)
                print('shortcut_a: ',self.sc_weights.size())
                for i in range(in_planes):
                    self.shortcut.weight[i] *= self.sc_weights[i]
        alpha_1 = alphas[0]
        alpha_2 = alphas[1]
        weight_1 = weights[0]
        weight_2 = weights[1]
        #print('layer1_a: ',self.layer_weights1.size())
        #self.layer_weights2 = torch.from_numpy(np.random.uniform(0,1,size=planes))
        #self.layer_weights2 = F.softmax(self.layer_weights2, dim = 0)
        #print('layer2_a: ',self.layer_weights2.size())
        for i in range(in_planes):
            weight_1[i] *= alpha_1[i]
        self.conv1.weight = nn.Parameter(weight_1.requires_grad_(True))
        for i in range(inter_planes):
            weight_2[i] *= alpha_2[i]
        self.conv2.weight = nn.Parameter(weight_2.requires_grad_(True))
        #for i in range(planes):
        #    self.conv2.weight[i] *= self.layer_weights2[i]
        #print(self.conv1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TargetResNet(nn.Module):
    def __init__(self, block, num_blocks, pruned_alpha, pruned_weights, num_classes=10):
        super(TargetResNet, self).__init__()
        
        # filters and their weights for the input layer 
        input2first_layer_a = pruned_alpha[0]
        input2first_layer_w = pruned_weights[0]
        first_layer_a = pruned_alpha[0:num_blocks[0]*2]
        second_layer_a = pruned_alpha[2*num_blocks[0]: 2*(sum(num_blocks[:2]))]
        third_layer_a = pruned_alpha[2*(sum(num_blocks[:2])): 2*(sum(num_blocks))]
        
        first_layer_w = pruned_weights[0:num_blocks[0]*2]
        second_layer_w = pruned_weights[2*num_blocks[0]: 2*(sum(num_blocks[:2]))]
        third_layer_w = pruned_weights[2*(sum(num_blocks[:2])): 2*(sum(num_blocks))]
        
        
        self.conv1 = nn.Conv2d(3, input2first_layer_a.size()[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input2first_layer_a.size()[0])
        self.layer1 = self._make_layer(block, first_layer_a, first_layer_w, second_layer_a[0].size()[0], 1)
        self.layer2 = self._make_layer(block, second_layer_a, second_layer_w, third_layer_a[0].size()[0], 2)
        self.layer3 = self._make_layer(block, third_layer_a, third_layer_w, 64, 2)
        self.linear = nn.Linear(64, num_classes)
        #alpha_apply = AlphaTerm()

        #self.apply(alpha_apply._weights_init)
        #self.alpha = alpha_apply.a
        #self.weights = alpha_apply.weights
        #print(alpha_apply.a[1].size())
        #print(alpha_apply.weights[1].size())

    def _make_layer(self, block, alphas, weights, output_layers, stride = 1):
        num_blocks = len(alphas)//2

        strides = [stride] + [1]*(num_blocks-1)
        print(strides)
        layers = []
        for i in range(num_blocks):            
            in_planes = alphas[2*i].size()[0]
            inter_planes = alphas[2*i+1].size()[0]
            if 2*i+2 < len(alphas):
                out_planes = alphas[2*i+2].size()[0]
            else:
                out_planes = output_layers
            print(in_planes, inter_planes, out_planes)
            block_weights = weights[2*i:2*i+2]
            block_alpha = alphas[2*i:2*i+2]
            layers.append(block(in_planes, inter_planes, out_planes, block_alpha, block_weights, strides[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def set_layer_weights(self, l1, l2, l3):
        return

#
def resnet20():
    return TargetResNet(ReducedBlock, [3, 3, 3])


def resnet32():
    return TargetResNet(ReducedBlock, [5, 5, 5])


def resnet44():
    return TargetResNet(ReducedBlock, [7, 7, 7])

#
def resnet56(alpha, filter):
    return TargetResNet(ReducedBlock, [9, 9, 9], alpha, filter)

#
def resnet110():
    return TargetResNet(ReducedBlock, [18, 18, 18])

#
def resnet164():
    return TargetResNet(ReducedBlock, [27, 27, 27])

def resnet1202():
    return TargetResNet(ReducedBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    #print(net)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":

    layer_weights1 = torch.from_numpy(np.random.uniform(0,1,size=16))
    layer_weights1 = F.softmax(layer_weights1, dim = 0)
    print('layer1_a: ',layer_weights1.size())
    
    for net_name in __all__:
        if net_name.startswith('resnet'):
            current_net = globals()[net_name]()
            print(net_name)
            test(current_net)
            for name, param in current_net.named_parameters():
                print(name,'\t', param.size())
                #for i, filter in enumerate(param):
                #    a = layer_weights1[i]
                #    aW = a * filter
                #    W = aW / filter
                    #print(a)
                #print(current_net.get_parameter(name))
            for layer, module in current_net.named_children():
                for layer_2, module_2 in module.named_children():
                    print(layer_2,':::\n', list(module_2.named_children()))
                print(layer,'::\n', len(list(module.named_children())))
            print()
    
    
