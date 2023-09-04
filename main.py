# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from torchvision import transforms

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

#import augmentations as aug

import src.dataset as ds
import src.models as m
from copy import deepcopy

parser = argparse.ArgumentParser()

# Model
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--equi", type=int, default=256)
parser.add_argument("--experience", type=str, choices=["SIENoVar","SIE","SIEOnlyEqui","VICReg","SimCLR","VICRegPartInv",
                                                        "SimCLROnlyEqui","SIERotColor","SimCLRAugSelf","SimCLRAugSelfRotColor",
                                                        "SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod","VICRegEquiMod"],
                                                        default="SIE")
parser.add_argument("--hypernetwork", type=str, choices=["linear","deep"],default="linear")
# Only for when using an expander
parser.add_argument("--mlp", default="2048-2048-2048")
#Predictor architecture, in format "intermediate1-intermediate2-..."
parser.add_argument("--predictor", default="")
parser.add_argument("--pred-size-in",type=int, default=10)
parser.add_argument("--predictor-relu",  action="store_true")

# Predictor
parser.add_argument("--predictor-type",type=str,choices=["hypernetwork","mlp"],default="hypernetwork")
parser.add_argument("--bias-pred", action="store_true")
parser.add_argument("--bias-hypernet", action="store_true")
parser.add_argument("--simclr-temp",type=float,default=0.1)
parser.add_argument("--ec-weight",type=float,default=1)
parser.add_argument("--tf-num-layers",type=int,default=1)



# Optim
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--base-lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-6)

parser.add_argument("--warmup-start",type=int, default=0)
parser.add_argument("--warmup-length",type=int, default=0)


# Data
parser.add_argument("--dataset-root", type=Path, default="DATA_FOLDER", required=True)
parser.add_argument("--images-file", type=Path, default="./data/train_images.npy", required=True)
parser.add_argument("--labels-file", type=Path, default="./data/val_images.npy", required=True)
parser.add_argument("--resolution", type=int, default=256)

# Checkpoints
parser.add_argument("--exp-dir", type=Path, default="")
parser.add_argument("--root-log-dir", type=Path,default="EXP_DIR/logs/")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--eval-freq", type=int, default=10)
parser.add_argument("--log-freq-time", type=int, default=30)

# Loss
parser.add_argument("--sim-coeff", type=float, default=10.0)
parser.add_argument("--equi-factor", type=float, default=4.5)
parser.add_argument("--std-coeff", type=float, default=10.0)
parser.add_argument("--cov-coeff", type=float, default=1.0)

# Running
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--no-amp", action="store_true")
parser.add_argument("--port", type=int, default=52472)



def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv("SLURM_NODEID")) * args.ngpus_per_node
        args.world_size = int(os.getenv("SLURM_NNODES")) * args.ngpus_per_node
        args.dist_url = f"tcp://{host_name}:{args.port}"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = f"tcp://localhost:{args.port}"
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)



def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    # Config dump
    if args.rank == 0:
        dict_args = deepcopy(vars(args))
        for key,value in dict_args.items():
            if isinstance(value,Path):
                dict_args[key] = str(value)
        with open(args.exp_dir / "params.json", 'w') as f:
            json.dump(dict_args, f)

    # Tensorboard setup
    if args.rank == 0:
        if str(args.exp_dir)[-1] == "/":
            exp_name = str(args.exp_dir)[:-1].split("/")[-1]
        else:
            exp_name = str(args.exp_dir).split("/")[-1]
        logdir = args.root_log_dir / exp_name
        writer = SummaryWriter(log_dir=logdir)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        args.root_log_dir.mkdir(parents=True, exist_ok=True)
        print(" ".join(sys.argv))

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
       mean=[0.5016, 0.5037, 0.5060], std=[0.1030, 0.0999, 0.0969]
    )
    if args.experience in ["SIERotColor","SimCLRAugSelfRotColor","SimCLROnlyEquiRotColor","SimCLREquiModRotColor"]:
        dataset = ds.Dataset3DIEBenchRotColor(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
    else:
        dataset = ds.Dataset3DIEBench(args.dataset_root,args.images_file, args.labels_file,transform=transforms.Compose([ transforms.Resize((args.resolution,args.resolution)),transforms.ToTensor(),normalize]))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    print("per_device_batch_size",per_device_batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = m.__dict__[args.experience](args).cuda(gpu)
    if args.experience in ["SimCLR","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
        model.gpu = gpu
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=False)

    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.wd
    )

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (x, y, z, labels) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            z = z.cuda(gpu, non_blocking=True)
            labels = labels.cuda(gpu, non_blocking=True)

            
            optimizer.zero_grad()

            # MAIN TRAINING PART
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss, classif_loss, stats, stats_eval = model.forward(x, y, z,labels)
                total_loss = loss + classif_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                # General logs
                writer.add_scalar('General/epoch', epoch, step)
                writer.add_scalar('General/time_elapsed', int(current_time - start_time), step)
                #writer.add_scalar('General/lr', lr, step)
                writer.add_scalar('General/lr', args.base_lr, step)
                writer.add_scalar('General/Current GPU memory', torch.cuda.memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)
                writer.add_scalar('General/Max GPU memory', torch.cuda.max_memory_allocated(torch.cuda.device('cuda:0'))/1e9, step)

                # Loss related logs
                writer.add_scalar('Loss/Total loss', stats["loss"].item(), step)
                if args.experience in ["SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor"]:
                    writer.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                if not args.experience in ["SimCLR","SimCLRAugSelf","SimCLRAugSelfFull","SimCLRAugSelfRotColor","SimCLROnlyEqui","SimCLROnlyEquiRotColor","SimCLREquiModRotColor","SimCLREquiMod"]:
                    writer.add_scalar('Loss/Invariance loss', stats["repr_loss_inv"].item(), step)
                    writer.add_scalar('Loss/Std loss', stats["std_loss"].item(), step)
                    writer.add_scalar('Loss/Covariance loss', stats["cov_loss"].item(), step)
                if not args.experience in ["VICReg","VICRegNoCov","VICRegCos","VICRegL1","VICRegL1repr","FullEqui","VICRegPartInv","SimCLR","VICRegPartInv2Exps","SimCLROnlyEqui","SIERotColor","SimCLROnlyEquiRotColor"] :
                    writer.add_scalar('Loss/Equivariance loss', stats["repr_loss_equi"].item(), step)
                if args.experience in ["SIEOnlyEqui","SIE","SIEAll","SIERotColor"]:
                    writer.add_scalar('Loss/Pred Std loss', stats["pred_std_loss"].item(), step)
                # Representations/embeddings stats
                writer.add_scalar('Stats/Corr. representations view1', stats["coremb_view1"].item(), step)
                writer.add_scalar('Stats/Corr. representations view2', stats["coremb_view2"].item(), step)
                writer.add_scalar('Stats/Std representations view1', stats["stdemb_view1"].item(), step)
                writer.add_scalar('Stats/Std representations view2', stats["stdemb_view2"].item(), step)
                writer.add_scalar('Stats/Corr. embeddings view1', stats["corhead_view1"].item(), step)
                writer.add_scalar('Stats/Corr. embeddings view2', stats["corhead_view2"].item(), step)
                writer.add_scalar('Stats/Std embeddings view1', stats["stdhead_view1"].item(), step)
                writer.add_scalar('Stats/Std embeddings view2', stats["stdhead_view2"].item(), step)
                if "stdemb_pred" in stats.keys():
                    writer.add_scalar('Stats/Corr. predictor output', stats["coremb_pred"].item(), step)
                    writer.add_scalar('Stats/Std predictor output', stats["stdemb_pred"].item(), step)

                
                # Online evaluation logs
                for key,value in stats_eval.items():
                    if "representations" in key:
                        writer.add_scalar(f'Online eval reprs/{key}', value, step)
                    elif "embeddings" in key:
                        writer.add_scalar(f'Online eval embs/{key}', value, step)
                for key,value in stats.items():
                    if "Latent/" in key:
                        writer.add_scalar(key, value, step)
                writer.flush()
                print("Logged, step :",step)
                last_logging = current_time
        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model.pth")
    if args.rank == 0:
        writer.close()
        torch.save(model.module.backbone.state_dict(), args.exp_dir / "final_weights.pth")


def exclude_bias_and_norm(p):
    return p.ndim == 1

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    main()
