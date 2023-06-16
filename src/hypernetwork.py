# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn

class HyperNet(nn.Module):
    def __init__(self, latent_size : int, output_size : int, args):
        super(HyperNet,self).__init__()
        if args.bias_hypernet:
            print("Bias in the hypernetwork")
        if args.hypernetwork == "linear":
            self.net = nn.Sequential(
                nn.Linear(latent_size,output_size,bias=args.bias_hypernet), # Linear combination for now
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_size,latent_size,bias=args.bias_hypernet),
                nn.ReLU(),
                nn.Linear(latent_size,latent_size,bias=args.bias_hypernet),
                nn.ReLU(),
                nn.Linear(latent_size,output_size,bias=args.bias_hypernet),
            )
        
    def forward(self, x : torch.Tensor):
        out = self.net(x)
        return out

    
class ParametrizedNet(nn.Module):
    def __init__(self,equivariant_size : int, latent_size : int, args):
        super(ParametrizedNet,self).__init__()
        if args.predictor == "":
            archi_str = str(equivariant_size) + "-" + str(equivariant_size)
        else:
            archi_str = str(equivariant_size) + "-"+ args.predictor +"-" + str(equivariant_size)
        print("Predictor architecture: ", archi_str)
        self.predictor = [int(x) for x in archi_str.split("-")]
        self.args = args
        
        self.num_weights_each = [ self.predictor[i]*self.predictor[i+1] for i in range(len(self.predictor)-1)]

        if self.args.bias_pred:
            self.num_biases_each = [self.predictor[i+1] for i in range(len(self.predictor)-1)]
            self.num_params_each = [self.num_weights_each[i] + self.num_biases_each[i] for i in range(len(self.num_biases_each))]
        else:
            self.num_params_each = self.num_weights_each
        print(self.num_params_each)
        self.cum_params = [0] + list(np.cumsum(self.num_params_each))        
        self.hypernet = HyperNet(latent_size, self.cum_params[-1], self.args)
        self.activation = nn.ReLU() if args.predictor_relu else nn.Identity()
        
    def forward(self, x : torch.Tensor, z : torch.Tensor):
        """
         x must be (batch_size, 1, size)
        
         Since F.linear(x,A,b) = x @ A.T + b (to have A (out_dim,in_dim) and be coherent with nn.linear)
         and  torch.bmm(x,A)_i = x_i @ A_i
         to emulate the same behaviour, we transpose A along the last two axes before bmm
        """
        weights = self.hypernet(z)
        out=x
        for i in range(len(self.predictor)-1):
            w = weights[...,self.cum_params[i]:self.cum_params[i] + self.num_weights_each[i]].view(-1,self.predictor[i+1],self.predictor[i])
            out = torch.bmm(out,torch.transpose(w,-2,-1))
            if self.args.bias_pred:
                b = weights[...,self.cum_params[i+1] - self.num_biases_each[i]:self.cum_params[i+1]].unsqueeze(1)
                out = out + b
            if i < len(self.predictor)-2:
                out = self.activation(out)
        
        return out.squeeze()