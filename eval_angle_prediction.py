# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import  transforms

import os.path
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from pathlib import Path
import json
import sys
import math
from scipy.spatial.transform import Rotation as R
import copy

import argparse
import src.resnet as resnet

parser = argparse.ArgumentParser()

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--experience", type=str, choices=["quat","euler"],default="quat",help="Whether to use Euler angles or quaternions for the targets.")
parser.add_argument("--deep-end", action="store_true",help="If used, uses a MLP instead of linear head")
parser.add_argument("--equi-dims",type=int,default=512,help="Number of equivariant dimensions (to evaluate). Put the full size to evaluate the whole representation.")
parser.add_argument("--inv-part",action="store_true",help="Whether or not to evaluate the invariant part")
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)

# Experience loading
parser.add_argument("--weights-file", type=str, default="./resnet50.pth")
parser.add_argument("--supervised",action="store_true")

# Optim
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0)

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--log-freq-time", type=int, default=10)

# Running
parser.add_argument("--num-workers", type=int, default=10)

args = parser.parse_args()


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone,_ = resnet.__dict__[args.arch](zero_init_residual=True)
        self.equi_dims = args.equi_dims
        self.out_dim = 3
        self.inv = args.inv_part
        if args.experience == "quat":
            self.out_dim = 4
        
        self.in_dims = self.equi_dims if not self.inv else 512-self.equi_dims

        if args.deep_end:
            self.head = nn.Sequential(
                nn.Linear(self.in_dims*2,1024),
                nn.ReLU(),
                nn.Linear(1024,1024),
                nn.ReLU(),
                nn.Linear(1024, self.out_dim),
            )
        else:
            self.head = nn.Linear(self.in_dims*2, self.out_dim)


    def forward(self, x, y):
        if not self.inv:
            out_x = self.backbone(x)[...,-self.equi_dims:]
            out_y = self.backbone(y)[...,-self.equi_dims:]
        else:
            out_x = self.backbone(x)[...,:-self.equi_dims]
            out_y = self.backbone(y)[...,:-self.equi_dims]
        concat = torch.cat([out_x,out_y],axis=1)
        #concat = concat.squeeze(2).squeeze(2)
        out = self.head(concat)
        return out

class Dataset3DIEBench(Dataset):
    def __init__(self,dataset_root, img_file,experience, size_dataset=-1, transform=None):
        self.dataset_root = dataset_root
        self.samples = np.load(img_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.rng = np.random.RandomState()    
        self.experience = experience    

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        # Latent vector creation
        views = self.rng.choice(50,2, replace=False)
        img_1 = self.get_img(self.dataset_root + self.samples[i]+ f"/image_{views[0]}.jpg")
        img_2 = self.get_img(self.dataset_root + self.samples[i]+ f"/image_{views[1]}.jpg")         
    
        angles_1 =np.load(self.dataset_root + self.samples[i]+ f"/latent_{views[0]}.npy")[:3].astype(np.float32)
        angles_2 =np.load(self.dataset_root + self.samples[i]+ f"/latent_{views[1]}.npy")[:3].astype(np.float32)
        rot_1 = R.from_euler("xyz",angles_1)
        rot_2 = R.from_euler("xyz",angles_2)
        rot_1_to_2 = rot_1.inv()*rot_2
        if self.experience == "quat":
            angles = rot_1_to_2.as_quat().astype(np.float32)
        else:
            angles = rot_1_to_2.as_euler("xyz").astype(np.float32)

        return img_1, img_2, torch.FloatTensor(angles)

    def __len__(self):
        return len(self.samples)
 

normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1

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

### INIT STUFF
dict_args = deepcopy(vars(args))
for key,value in dict_args.items():
    if isinstance(value,Path):
        dict_args[key] = str(value)
with open(args.exp_dir / "params.json", 'w') as f:
    json.dump(dict_args, f)

if str(args.exp_dir)[-1] == "/":
    exp_name = str(args.exp_dir)[:-1].split("/")[-1]
else:
    exp_name = str(args.exp_dir).split("/")[-1]
logdir = args.root_log_dir / exp_name
writer = SummaryWriter(log_dir=logdir)

args.exp_dir.mkdir(parents=True, exist_ok=True)
args.root_log_dir.mkdir(parents=True, exist_ok=True)
print(" ".join(sys.argv))


### DATA

ds_train = Dataset3DIEBench(args.dataset_root,
                            "./data/train_images.npy",args.experience,
                               transform=transforms.Compose([transforms.ToTensor(),normalize]))
ds_val = Dataset3DIEBench(args.dataset_root,
                            "./data/val_images.npy",args.experience,
                             transform=transforms.Compose([transforms.ToTensor(),normalize]))

train_loader = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


## MODEL AND OPTIM

net = Model(args)
# Change number of output dimensions to match our problem
net = net.to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

epochs = args.epochs

# Load and freeze the model
ckpt  = torch.load(args.weights_file, map_location="cpu")
if args.supervised:
     load_from_state_dict(net.backbone,ckpt,prefix="backbone.", new_suffix="")
     for param in net.backbone.parameters():
        param.requires_grad = False
else:
    if "final_weights" in args.weights_file:
        load_from_state_dict(net.backbone,ckpt,prefix="", new_suffix="")
    else:
        load_from_state_dict(net.backbone,ckpt["model"],prefix="module.backbone.", new_suffix="")
    for param in net.backbone.parameters():
        param.requires_grad = False

start_epoch = 0
## LOOP

for epoch in range(start_epoch,epochs):
    net.train()
    for step, (inputs_1,inputs_2, latents) in enumerate(train_loader,start=epoch * len(train_loader)):
        inputs_1 = inputs_1.to(args.device)
        inputs_2 = inputs_2.to(args.device)
        latents = latents.to(args.device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs_1,inputs_2)
        loss = F.mse_loss(outputs, latents)
        r2 = r2_score(outputs,latents)
        if step%args.log_freq_time == 0:
            writer.add_scalar('Loss/loss', loss.item(), step)
            writer.add_scalar('Metrics/train_R2', r2.item(), step)
            writer.add_scalar('General/lr', args.lr, step)
            writer.flush()

        loss.backward()
        optimizer.step()
        if step%50 == 0 :
            print(f"[Epoch {epoch}, step : {step}]: Loss: {loss.item():.6f}, R2 score: {r2.item():.3f}")
    net.eval()
    with torch.no_grad():
        avg_mse = 0
        len_ds = len(ds_val)
        for i, (inputs_1,inputs_2, latents) in enumerate(val_loader):
            inputs_1 = inputs_1.to(args.device)
            inputs_2 = inputs_2.to(args.device)
            latents = latents.to(args.device)
            
            outputs = net(inputs_1,inputs_2)
            mse = F.mse_loss(outputs,latents)
            avg_mse += inputs_1.shape[0]*mse.item()/len_ds
            if i == 0:
                total_latents = latents.cpu()
                total_preds = outputs.cpu()
            else:
                total_latents = torch.cat((total_latents,latents.cpu()),axis=0)
                total_preds = torch.cat((total_preds,outputs.cpu()),axis=0)
        r2 = r2_score(total_preds,total_latents)
        writer.add_scalar('Metrics/val_MSE', avg_mse, step)
        writer.add_scalar('Metrics/val_R2', r2.item(), step)
        writer.flush()
        print(f"[Epoch {epoch}, validation]: MSE: {avg_mse:.6f}, R2 score: {r2.item():.3f}")
    
    ## CHECKPOINT
    state = dict(
                epoch=epoch + 1,
                model=net.state_dict(),
                optimizer=optimizer.state_dict(),
            )
    torch.save(state, args.exp_dir / "model.pth")

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass