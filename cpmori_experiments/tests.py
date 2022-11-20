import numpy as np
import torch.nn.functional as F
import torch
alphas = torch.rand(10)



noise = torch.log(torch.rand(alphas.size()))  
noise_hat = F.softmax(noise,dim=0)   #Softmax to get the probability of the noises
    
print(noise_hat)