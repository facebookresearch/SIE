# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class MLPPredictor(nn.Module):
    def __init__(self,repr_dim=512,latent_dim=4,n_layers=2, output_activation=nn.Identity()):
        super(MLPPredictor, self).__init__()
        
        self.repr_dim = repr_dim
        self.latent_dim = latent_dim
        self.first_proj = [nn.Linear(self.repr_dim+self.latent_dim,self.repr_dim)] if n_layers == 1 else [nn.Linear(self.repr_dim+self.latent_dim,self.repr_dim), nn.ReLU()]
        self.layers = []
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(self.repr_dim,self.repr_dim))
            if i < n_layers-2:
                self.layers.append(nn.ReLU())
        self.pred = nn.Sequential(*(self.first_proj+self.layers))
        print(self.pred)
        self.output_activation = output_activation
    
    def forward(self, representation,latent=None):
        if latent is not None:
            out = torch.concat((latent,representation),dim=1)
        else:
            out = representation
        out = self.pred(out)
        out = self.output_activation(out)
        return out