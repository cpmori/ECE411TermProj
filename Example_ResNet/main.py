import resnet
import torch

target_net = resnet.resnet20()
#for i, c in target_net.layer1.named_children():
#    print(i, ' (c): ', c)
print('---modules---')
total_layer = 0
for name, m in target_net.named_modules():
    print(type(name), type(m), len(name))
    if 'conv' in name or 'linear' in name:# or 'bn' in name:# or 'shortcut' in i:
        print(name, ' is sublayer')
        print(name, ' (m): ', m)
        total_layer += 1
        
        #if 'bn' in name:
            #print("weight: ",type(m.weight), m.weight)
            #print("bias: ", type(m.bias), m.bias)
        for param in m.parameters():
            print(param.size()) # out_channel, in_channel, kernel_row, kernel_col
            #if 'conv' in name:
                #print(m.weight.size())
                #for input_filter in param:
                    #print(input_filter.requires_grad)
                #    for kernel in out_c:
                #        print(torch.sum(kernel))
    elif len(name) > 0:
        print(name, ' (m): ', m)
    else:
        continue
    print(total_layer, '\n')
x=torch.rand(5,3)
print(x)
print(torch.cuda.is_available())