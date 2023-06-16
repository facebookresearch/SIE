# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import transforms

import os.path
from copy import deepcopy
from pathlib import Path
import json
from scipy.spatial.transform import Rotation as R
import copy

import argparse
import src.resnet as resnet
import errno

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--equi-dims-reprs",type=int,default=512)
parser.add_argument("--projector-mlp",type=str,default="1024-1024-1024")
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--no-norm",  action="store_true")
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)



args = parser.parse_args()
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import scipy.linalg


class HyperNet(nn.Module):
    def __init__(self, latent_size : int, output_size : int):
        super(HyperNet,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size,output_size,bias=False), # Linear combination for now
        )
    
        
    def forward(self, x : torch.Tensor):
        out = self.net(x)
        return out


class ParametrizedNet(nn.Module):
    def __init__(self,equivariant_size : int, latent_size : int):
        super(ParametrizedNet,self).__init__()
        archi_str = str(equivariant_size) + "-" + str(equivariant_size)
        print("Predictor architecture: ", archi_str)
        self.predictor = [int(x) for x in archi_str.split("-")]
        
        self.num_weights_each = [ self.predictor[i]*self.predictor[i+1] for i in range(len(self.predictor)-1)]

        self.num_params_each = self.num_weights_each
        print(self.num_params_each)
        self.cum_params = [0] + list(np.cumsum(self.num_params_each))        
        self.hypernet = HyperNet(latent_size, self.cum_params[-1])
        self.activation =  nn.Identity()
        
        self.mat=None
        
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
            self.mat = w.detach().cpu()
            out = torch.bmm(out,torch.transpose(w,-2,-1))
            if i < len(self.predictor)-2:
                out = self.activation(out)
        
        return out.squeeze()


class EvalDataset(Dataset):

    def __init__(self, embs, latents):
        self.embeddings = embs
        self.latents = latents

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        start = idx//50 * 50
        end = idx

        rot_start = R.from_euler("xyz",self.latents[start])
        rot_end = R.from_euler("xyz",self.latents[end])
        target = rot_end.as_quat().astype(np.float32) 
        angle = (rot_start.inv()*rot_end).as_quat().astype(np.float32) 

        embedding = self.embeddings[start]

        return embedding,torch.Tensor(angle),torch.Tensor(target),start,end

class Dataset3DIEBench(Dataset):
    def __init__(self, dataset_root, samples, size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = samples
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, i):
        # Latent vector creation
        with open(self.dataset_root + self.samples[i],"rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)    

normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )

def Projector(embedding, mlp, last_relu=False):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    if last_relu :
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)
 

normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )



def load_from_state_dict(model, state_dict, prefix, new_suffix):
        state_dict = copy.deepcopy(state_dict)
        state_dict = {
            k.replace(prefix, new_suffix): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print(
                    'key "{}" could not be found in provided state dict'.format(k)
                )
            elif state_dict[k].shape != v.shape:
                print(
                    'key "{}" is of different shape in model and provided state dict {} vs {}'.format(
                        k, v.shape, state_dict[k].shape
                    )
                )
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))

def create_dir(dir):
    try:
        os.mkdir(dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            pass

## Data initialisation

imgs = np.load("./data/train_images.npy")
new = []
for img in imgs:
    new += [img + f"/image_{view}.jpg" for view in range(50)]
    
files_train = np.array(new)

imgs = np.load("./data/val_images.npy")
new = []
for img in imgs:
    new += [img + f"/image_{view}.jpg" for view in range(50)]
    
files_val = np.array(new)

ds_train = Dataset3DIEBench(args.dataset_root,
                            files_train,
                            transform=transforms.Compose([transforms.ToTensor(),normalize]))
ds_val = Dataset3DIEBench(args.dataset_root,
                          files_val,
                          transform=transforms.Compose([transforms.ToTensor(),normalize]))

loader_train = DataLoader(ds_train, batch_size=256, shuffle=False, num_workers=10)
loader_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=10)

### Model loading

emb_dim = int(args.projector_mlp.split("-")[-1])



ckpt  = torch.load(args.exp_dir / "model.pth", map_location="cpu")

net,_ = resnet.__dict__["resnet18"]()
net = net.to("cuda:0")

proj_equi = Projector(args.equi_dims_reprs,args.projector_mlp).to("cuda:0")

predictor = ParametrizedNet(emb_dim,4).to("cuda:0")

load_from_state_dict(net,ckpt["model"],prefix="module.backbone.",new_suffix="")
load_from_state_dict(proj_equi,ckpt["model"],prefix="module.projector_equi.",new_suffix="")
load_from_state_dict(predictor,ckpt["model"],prefix="module.predictor.",new_suffix="")

## Feature extraction

create_dir(args.exp_dir / "pred_eval/")

for loader,name in [(loader_train,"train"),
                    (loader_val,"val")] :
    if os.path.exists(args.exp_dir / f"pred_eval/representations_{name}.npy") and os.path.exists(args.exp_dir / f"pred_eval/embeddings_{name}.npy"):
        print(f"Feature extraction for the {name} set already done, skipping")
        continue
    print(f"Extracting features for the {name} set ....")

    all_reprs = []
    all_embs = []
    net.eval()
    proj_equi.eval()
    with torch.no_grad():
        for i, inputs in enumerate(tqdm(loader)):
            inputs = inputs.to("cuda:0")
            # forward + backward + optimize
            outputs = net(inputs)[:,-args.equi_dims_reprs:]
            all_reprs.append(outputs.cpu().numpy())
            outputs = proj_equi(outputs)
            all_embs.append(outputs.cpu().numpy())

    representations = np.concatenate(all_reprs)
    np.save(args.exp_dir / f"pred_eval/representations_{name}.npy",representations)
    embeddings = np.concatenate(all_embs)
    np.save(args.exp_dir / f"pred_eval/embeddings_{name}.npy",embeddings)


## Predictor evaluations

# Train-Train, val-val, val-all

if not args.no_norm:
    log_file = "log-classical"
else:
    log_file = "log-classical-unnorm"


for source,target in [("train","train"),("val","val")]:
    print(f"Evaluating {source}-{target}")
   
    embeddings_target = np.load(args.exp_dir / f"pred_eval/embeddings_{target}.npy")
    latents_target = np.load(f"./data/all_latents_{target}.npy")[:,:3]

    embeddings_source = np.load(args.exp_dir / f"pred_eval/embeddings_{source}.npy")
    if not args.no_norm:
        embeddings_target = embeddings_target/(np.linalg.norm(embeddings_target,axis=1)+1e-8).reshape(-1,1)
    equi = torch.Tensor(embeddings_target).to("cuda:0")

    latents_source = np.load(f"./data/all_latents_{source}.npy")[:,:3]

    dataset = EvalDataset(torch.Tensor(embeddings_source),latents_source)
    dataloader = DataLoader(dataset,batch_size=128,num_workers=10,shuffle=True)

    dot_products = []
    correct_ranks = []

    predictor.eval()
    with torch.no_grad():
        for idcs, (embs,angles,targets,start,end) in enumerate(tqdm(dataloader)):
            
            embs = embs.to("cuda:0")
            angles = angles.to("cuda:0")
            targets = targets
            
            output = predictor(embs.unsqueeze(1),angles)
            if not args.no_norm:
                output = output/torch.linalg.norm(output,axis=1).view(-1,1)

            similarities = []
            for i,out in enumerate(output):

                sims = out@equi[start[i]:start[i]+50].T
                target_idx  = end[i] - start[i]
                nns = torch.argsort(-sims).cpu()
                correct_rank = torch.argwhere(nns == target_idx)[0][0]+1
                correct_ranks.append(correct_rank)
    correct_ranks = torch.stack(correct_ranks)
    MRR = torch.mean(1/correct_ranks)
    H_at_1 = (correct_ranks <= 1).sum()/correct_ranks.shape[0]
    H_at_2 = (correct_ranks <= 2).sum()/correct_ranks.shape[0]
    H_at_5 = (correct_ranks <= 5).sum()/correct_ranks.shape[0]
    H_at_10 = (correct_ranks <= 10).sum()/correct_ranks.shape[0]


    with open(args.exp_dir / f"pred_eval/{log_file}", 'a+') as fd:
        fd.write(json.dumps({
            'mode':f"{source}-{target}",
            "MRR":MRR.item(),
            "H@1":H_at_1.item(),
            "H@2":H_at_2.item(),
            "H@5":H_at_5.item(),
            "H@10":H_at_10.item()
        }) + '\n')
        fd.flush()

