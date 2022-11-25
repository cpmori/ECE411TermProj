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

apply_alpha = False
#__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet164','resnet1202']
__all__ = ['resnet20']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class TargetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(TargetBlock, self).__init__()
        self.alpha1 = nn.Parameter(torch.rand([planes,1,1,1], requires_grad=True))
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.alpha2 = nn.Parameter(torch.rand([planes,1,1,1], requires_grad=True))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.alpha_sc = nn.Parameter(torch.rand([self.expansion * planes,1,1,1], requires_grad=True))
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        Prune = True
        # in sampling
        # softalpha: p_hat
        # alpha: p
        if Prune is True:
            non_masked_alpha1_idx = torch.nonzero(self.alpha1)[:,0]
            non_masked_alpha2_idx = torch.nonzero(self.alpha2)[:,0]
            
            self.softalpha1 = torch.clone(self.alpha1)
            self.softalpha2 = torch.clone(self.alpha2)
            # only compute non_masked alphas (remove masked 0s from softmax)
            self.softalpha1[non_masked_alpha1_idx] = F.softmax(self.alpha1[non_masked_alpha1_idx], dim=0)
            self.softalpha2[non_masked_alpha2_idx] = F.softmax(self.alpha2[non_masked_alpha2_idx], dim=0)
            #if (self.alpha1.size()[0] == 16):
            #    print(non_masked_alpha1_idx)
            #    print(self.softalpha1.flatten())
            #    print(self.conv1.weight.size())
            #    print(torch.nonzero(self.conv1.weight.mul(F.softmax(self.alpha1, dim=0))).size())
            #    print(torch.nonzero(self.conv1.weight.mul(self.softalpha1)).size())
            #    print(F.softmax(self.alpha1, dim=0).flatten())
            out = F.conv2d(x,self.conv1.weight.mul(self.softalpha1),\
                stride=self.conv1.stride, padding=self.conv1.padding)
            out = F.relu(self.bn1(out))
            out = F.conv2d(out,self.conv2.weight.mul(self.softalpha2),\
                stride=self.conv2.stride, padding=self.conv2.padding)
            #if (self.alpha1.size()[0] == 16):          # check if parameters are truly zeroed out
                #print(torch.nonzero(out).size())
            #    print(len(self.alpha1),len(non_masked_alpha1_idx))
        else:
        # in normal training
            #print(self.alpha1.size(), self.conv1.weight.size(), x.size())
            out = F.conv2d(x,self.conv1.weight.mul(F.softmax(self.alpha1, dim=0)),\
                stride=self.conv1.stride, padding=self.conv1.padding)
            out = F.relu(self.bn1(out))
            out = F.conv2d(out,self.conv2.weight.mul(F.softmax(self.alpha2, dim=0)),\
                stride=self.conv2.stride, padding=self.conv2.padding)
            #if (self.alpha1.size()[0] == 16):          # count for unpruned resnet
            #    print(torch.nonzero(out).size())
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TargetResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TargetResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

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
    def alphaL2(self):
        l2norm = 0
        
        return l2norm
#
def resnet20():
    return TargetResNet(TargetBlock, [3, 3, 3])


def resnet32():
    return TargetResNet(TargetBlock, [5, 5, 5])


def resnet44():
    return TargetResNet(TargetBlock, [7, 7, 7])

#
def resnet56():
    return TargetResNet(TargetBlock, [9, 9, 9])

#
def resnet110():
    return TargetResNet(TargetBlock, [18, 18, 18])

#
def resnet164():
    return TargetResNet(TargetBlock, [27, 27, 27])

def resnet1202():
    return TargetResNet(TargetBlock, [200, 200, 200])


def test(net:nn.Module):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(torch.clone(x).cpu().data.numpy().shape)
    print("Total number of params", total_params)
    #print(net)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

    for name, param in net.named_parameters():
        if 'alpha' in name:
            print(name, param.flatten())
        #if 'conv' in name:
        #    print(name, param.flatten())
        #if 'alpha' in name:
        #    print(name, param.size())
    #for name, module in net.named_modules():
    #    print(list(module.named_buffers()))


if __name__ == "__main__":

    net = resnet20()
    all_filters = []
    for name, param in net.named_parameters():
        if 'conv' in name:
            print(name, param.size())
            #print(net[name])
        if 'alpha' in name:
            print(name, param.size())
    for name, module in net.named_modules():
        if 'layer' in name and '.' in name and ('conv' not in name and 'bn' not in name and 'shortcut' not in name):
            print('----------')
            print(name, module)
    #for net_name in __all__:
    #    if net_name.startswith('resnet'):
    #        current_net = globals()[net_name]()
    #        print(net_name)
    #        test(current_net)
    #        
    #        #for name, param in current_net.named_parameters():
    #        #    print(name,'\t', param.size())
    #        #    #for i, filter in enumerate(param):
    #        #    #    a = layer_weights1[i]
    #        #    #    aW = a * filter
    #        #    #    W = aW / filter
    #        #        #print(a)
    #        #    #print(current_net.get_parameter(name))
    #        #for layer, module in current_net.named_children():
    #        #    for layer_2, module_2 in module.named_children():
    #        #        print(layer_2,':::\n', list(module_2.named_children()))
    #        #    print(layer,'::\n', len(list(module.named_children())))
    #        print()
    #
    
