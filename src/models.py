# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

import predictors
import resnet
import hypernetwork


#-------------------- Online evaluator -------------------

class OnlineEvaluator(nn.Module):
    def __init__(self,inv_repr_size,equi_repr_size,inv_emb_size,equi_emb_size,num_classes=55):
        super().__init__()
        self.inv_repr_size = inv_repr_size
        self.equi_repr_size = equi_repr_size
        self.inv_emb_size = inv_emb_size
        self.equi_emb_size = equi_emb_size

        self.repr_size = self.inv_repr_size+self.equi_repr_size
        self.emb_size = self.inv_emb_size+self.equi_emb_size
         # Classifiers/predictors on representations
        self.classifier_repr = nn.Linear(self.repr_size, num_classes)
        if self.inv_repr_size > 0:
            self.classifier_inv_repr = nn.Linear(self.inv_repr_size, num_classes)

        self.predictor_angles_repr = nn.Sequential(
                nn.Linear(2*self.repr_size,2*self.repr_size),
                nn.ReLU(),
                nn.Linear(2*self.repr_size,2*self.repr_size),
                nn.ReLU(),
                nn.Linear(2*self.repr_size, 4),
            )
        if self.equi_repr_size > 0:
            self.predictor_angles_equi_repr = nn.Sequential(
                    nn.Linear(2*self.equi_repr_size,2*self.equi_repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.equi_repr_size,2*self.equi_repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.equi_repr_size, 4),
                )
        # Classifiers/predictors on embeddings
        self.classifier_emb = nn.Linear(self.emb_size, num_classes)
        if self.inv_emb_size > 0:
            self.classifier_inv_emb = nn.Linear(self.inv_emb_size, num_classes)

        self.predictor_angles_emb = nn.Sequential(
                nn.Linear(2*self.emb_size,2*self.emb_size),
                nn.ReLU(),
                nn.Linear(2*self.emb_size,2*self.emb_size),
                nn.ReLU(),
                nn.Linear(2*self.emb_size, 4),
            )
        if self.equi_emb_size > 0:
            self.predictor_angles_equi_emb = nn.Sequential(
                    nn.Linear(2*self.equi_emb_size,2*self.equi_emb_size),
                    nn.ReLU(),
                    nn.Linear(2*self.equi_emb_size,2*self.equi_emb_size),
                    nn.ReLU(),
                    nn.Linear(2*self.equi_emb_size, 4),
                )

    def forward(self,reprs,embs,labels,angles):
        # reprs = [x,y]
        # embs = [x,y]
        reprs = [repr.detach() for repr in reprs]
        embs = [emb.detach() for emb in embs]
        reprs_inv = [repr[...,:self.inv_repr_size] for repr in reprs]
        reprs_equi = [repr[...,self.inv_repr_size:] for repr in reprs]
        embs_inv = [emb[...,:self.inv_emb_size] for emb in embs]
        embs_equi = [emb[...,self.inv_emb_size:] for emb in embs]

        labels = torch.concat([labels,labels],dim=0)

        classifier_repr_out = self.classifier_repr(torch.concat(reprs,dim=0))
        if self.inv_repr_size > 0:
            classifier_inv_repr_out = self.classifier_inv_repr(torch.concat(reprs_inv,dim=0))
        predictor_angles_repr_out = self.predictor_angles_repr(torch.concat(reprs,dim=1))
        if self.equi_repr_size > 0:
            predictor_angles_equi_repr_out = self.predictor_angles_equi_repr(torch.concat(reprs_equi,dim=1))

        classifier_emb_out = self.classifier_emb(torch.concat(embs,dim=0))
        if self.inv_emb_size > 0:
            classifier_inv_emb_out = self.classifier_inv_emb(torch.concat(embs_inv,dim=0))
        predictor_angles_emb_out = self.predictor_angles_emb(torch.concat(embs,dim=1))
        if self.equi_emb_size > 0:
            predictor_angles_equi_emb_out = self.predictor_angles_equi_emb(torch.concat(embs_equi,dim=1))

        stats = {}
        total_loss = 0
        # Reprs
        loss = F.cross_entropy(classifier_repr_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_repr_out, labels, topk=(1, 5))
        stats["CE representations"] = loss.item()
        stats["Top-1 representations"] = acc1.item()
        stats["Top-5 representations"] = acc5.item()
        if self.inv_repr_size > 0:
            loss = F.cross_entropy(classifier_inv_repr_out, labels)
            total_loss += loss
            acc1, acc5 = accuracy(classifier_inv_repr_out, labels, topk=(1, 5))
            stats["CE representations-inv"] = loss.item()
            stats["Top-1 representations-inv"] = acc1.item()
            stats["Top-5 representations-inv"] = acc5.item()

        loss = F.mse_loss(predictor_angles_repr_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_repr_out,angles)
        stats["MSE representations"] = loss.item()
        stats["R2 representations"] = r2.item()
        if self.equi_repr_size > 0:
            loss = F.mse_loss(predictor_angles_equi_repr_out,angles)
            total_loss += loss
            r2 = r2_score(predictor_angles_equi_repr_out,angles)
            stats["MSE representations-equi"] = loss.item()
            stats["R2 representations-equi"] = r2.item()
        
        # Embs
        loss = F.cross_entropy(classifier_emb_out, labels)
        total_loss += loss
        acc1, acc5 = accuracy(classifier_emb_out, labels, topk=(1, 5))
        stats["CE embeddings"] = loss.item()
        stats["Top-1 embeddings"] = acc1.item()
        stats["Top-5 embeddings"] = acc5.item()
        if self.inv_emb_size > 0:
            loss = F.cross_entropy(classifier_inv_emb_out, labels)
            total_loss += loss
            acc1, acc5 = accuracy(classifier_inv_emb_out, labels, topk=(1, 5))
            stats["CE embeddings-inv"] = loss.item()
            stats["Top-1 embeddings-inv"] = acc1.item()
            stats["Top-5 embeddings-inv"] = acc5.item()

        loss = F.mse_loss(predictor_angles_emb_out,angles)
        total_loss += loss
        r2 = r2_score(predictor_angles_emb_out,angles)
        stats["MSE embeddings"] = loss.item()
        stats["R2 embeddings"] = r2.item()
        if self.equi_emb_size > 0:
            loss = F.mse_loss(predictor_angles_equi_emb_out,angles)
            total_loss += loss
            r2 = r2_score(predictor_angles_equi_emb_out,angles)
            stats["MSE embeddings-equi"] = loss.item()
            stats["R2 embeddings-equi"] = r2.item()

        return total_loss, stats

#--------------------- Predictor applied after the expander -------------------

class SIENoVar(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.equi_repr_size = self.args.equi
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        self.inv_repr_size = self.repr_size - self.equi_repr_size
        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        ratio_inv = self.inv_repr_size/float(self.repr_size)
        mlp_inv = [str(elt) for elt in list(np.round((mlp_dims*ratio_inv)).astype(int))]
        self.inv_emb_size = int(mlp_inv[-1])
        self.projector_inv  = Projector(self.inv_repr_size,"-".join(mlp_inv))
        
        ratio_equi = self.equi_repr_size/float(self.repr_size)
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims*ratio_equi)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.equi_repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        print("Invariant projector dims: ", mlp_inv)
        print("Equivariant projector dims: ", mlp_equi)


        self.evaluator = OnlineEvaluator(self.inv_repr_size,self.equi_repr_size,self.inv_emb_size,self.equi_emb_size,num_classes=num_classes)

       

    def forward(self, x, y, z, labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x_inv = x_emb[...,:self.inv_repr_size]
        y_inv = y_emb[...,:self.inv_repr_size]
        x_equi = x_emb[...,self.inv_repr_size:]
        y_equi = y_emb[...,self.inv_repr_size:]

        x_inv = self.projector_inv(x_inv)
        y_inv = self.projector_inv(y_inv)
        x_equi = self.projector_equi(x_equi)
        y_equi = self.projector_equi(y_equi)

        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # _________________
        # |        |      |
        # |   Inv  |Common|
        # |________|______|
        # |        |      |
        # | Common |  Eq  |
        # |________|______|
        #
        # Without this concatenation we would not regularize the common parts

        x = torch.cat((x_inv,x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        #======================================
        #           Stats logging
        #======================================

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        #======================================
        #           Inv part
        #======================================
        repr_loss_inv = F.mse_loss(x_inv, y_inv)

        #======================================
        #           Equi part
        #======================================
        
        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)


        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)
        
        #======================================
        #           Common part
        #======================================

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.args.sim_coeff * repr_loss_inv
                + self.args.equi_factor*self.args.sim_coeff * repr_loss_equi
                + self.args.std_coeff * std_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- Predictor applied after the expander. Variance on the pred output -------------------

class SIE(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.equi_repr_size = self.args.equi
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        self.inv_repr_size = self.repr_size - self.equi_repr_size
        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        ratio_inv = self.inv_repr_size/float(self.repr_size)
        mlp_inv = [str(elt) for elt in list(np.round((mlp_dims*ratio_inv)).astype(int))]
        self.inv_emb_size = int(mlp_inv[-1])
        self.projector_inv  = Projector(self.inv_repr_size,"-".join(mlp_inv))
        
        ratio_equi = self.equi_repr_size/float(self.repr_size)
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims*ratio_equi)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.equi_repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        print("Invariant projector dims: ", mlp_inv)
        print("Equivariant projector dims: ", mlp_equi)


        self.evaluator = OnlineEvaluator(self.inv_repr_size,self.equi_repr_size,self.inv_emb_size,self.equi_emb_size,num_classes=num_classes)

       

    def forward(self, x, y, z, labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x_inv = x_emb[...,:self.inv_repr_size]
        y_inv = y_emb[...,:self.inv_repr_size]
        x_equi = x_emb[...,self.inv_repr_size:]
        y_equi = y_emb[...,self.inv_repr_size:]

        x_inv = self.projector_inv(x_inv)
        y_inv = self.projector_inv(y_inv)
        x_equi = self.projector_equi(x_equi)
        y_equi = self.projector_equi(y_equi)

        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # _________________
        # |        |      |
        # |   Inv  |Common|
        # |________|______|
        # |        |      |
        # | Common |  Eq  |
        # |________|______|
        #
        # Without this concatenation we would not regularize the common parts

        x = torch.cat((x_inv,x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        #======================================
        #           Stats logging
        #======================================

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        #======================================
        #           Inv part
        #======================================
        repr_loss_inv = F.mse_loss(x_inv, y_inv)

        #======================================
        #           Equi part
        #======================================
        
        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)


        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)
        
        #======================================
        #           Common part
        #======================================

        y_equi_pred = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y_equi_pred = y_equi_pred - y_equi_pred.mean(dim=0)
        std_y_equi_pred = torch.sqrt(y_equi_pred.var(dim=0) + 0.0001)
        pred_std_loss = torch.mean(F.relu(1 - std_y_equi_pred)) / 2

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.args.sim_coeff * repr_loss_inv
                + self.args.equi_factor*self.args.sim_coeff * repr_loss_equi
                + self.args.std_coeff * std_loss
                + self.args.std_coeff * pred_std_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["pred_std_loss"] = pred_std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- Predictor applied after the expander. Variance on the pred output Rot + Color LV-------------------

class SIERotColor(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.equi_repr_size = self.args.equi
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        self.inv_repr_size = self.repr_size - self.equi_repr_size
        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        ratio_inv = self.inv_repr_size/float(self.repr_size)
        mlp_inv = [str(elt) for elt in list(np.round((mlp_dims*ratio_inv)).astype(int))]
        self.inv_emb_size = int(mlp_inv[-1])
        self.projector_inv  = Projector(self.inv_repr_size,"-".join(mlp_inv))
        
        ratio_equi = self.equi_repr_size/float(self.repr_size)
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims*ratio_equi)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.equi_repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)
        print("Invariant projector dims: ", mlp_inv)
        print("Equivariant projector dims: ", mlp_equi)


        self.evaluator = OnlineEvaluator(self.inv_repr_size,self.equi_repr_size,self.inv_emb_size,self.equi_emb_size,num_classes=num_classes)

       

    def forward(self, x, y, z, labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x_inv = x_emb[...,:self.inv_repr_size]
        y_inv = y_emb[...,:self.inv_repr_size]
        x_equi = x_emb[...,self.inv_repr_size:]
        y_equi = y_emb[...,self.inv_repr_size:]

        x_inv = self.projector_inv(x_inv)
        y_inv = self.projector_inv(y_inv)
        x_equi = self.projector_equi(x_equi)
        y_equi = self.projector_equi(y_equi)

        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # _________________
        # |        |      |
        # |   Inv  |Common|
        # |________|______|
        # |        |      |
        # | Common |  Eq  |
        # |________|______|
        #
        # Without this concatenation we would not regularize the common parts

        x = torch.cat((x_inv,x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        #======================================
        #           Stats logging
        #======================================

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z[:,:4])

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        #======================================
        #           Inv part
        #======================================
        repr_loss_inv = F.mse_loss(x_inv, y_inv)

        #======================================
        #           Equi part
        #======================================
        
        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)


        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)
        
        #======================================
        #           Common part
        #======================================

        y_equi_pred = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y_equi_pred = y_equi_pred - y_equi_pred.mean(dim=0)
        std_y_equi_pred = torch.sqrt(y_equi_pred.var(dim=0) + 0.0001)
        pred_std_loss = torch.mean(F.relu(1 - std_y_equi_pred)) / 2

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                  self.args.sim_coeff * repr_loss_inv
                + self.args.equi_factor*self.args.sim_coeff * repr_loss_equi
                + self.args.std_coeff * std_loss
                + self.args.std_coeff * pred_std_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["pred_std_loss"] = pred_std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- Predictor applied after the expander. Variance on the pred output -------------------

class SIEOnlyEqui(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,latent_dim=4,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,0,self.equi_emb_size,num_classes=num_classes)

       

    def forward(self, x, y, z, labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        # Concatenate both parts to apply the regularization on the whole vectors
        # This helps remove information that would be redundant in both parts
        # _________________
        # |        |      |
        # |   Inv  |Common|
        # |________|______|
        # |        |      |
        # | Common |  Eq  |
        # |________|______|
        #
        # Without this concatenation we would not regularize the common parts

        x = x_equi
        y = y_equi

        #======================================
        #           Stats logging
        #======================================

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        #======================================
        #           Equi part
        #======================================
        
        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)


        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)
        
        #======================================
        #           Common part
        #======================================

        y_equi_pred = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y_equi_pred = y_equi_pred - y_equi_pred.mean(dim=0)
        std_y_equi_pred = torch.sqrt(y_equi_pred.var(dim=0) + 0.0001)
        pred_std_loss = torch.mean(F.relu(1 - std_y_equi_pred)) / 2

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(cov_x.shape[0]) \
            + off_diagonal(cov_y).pow_(2).sum().div(cov_x.shape[0])

        loss = (
                + self.args.sim_coeff * repr_loss_equi
                + self.args.std_coeff * std_loss
                + self.args.std_coeff * pred_std_loss
                + self.args.cov_coeff * cov_loss
                )

        stats["repr_loss_inv"] = torch.Tensor([0])
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss
        stats["pred_std_loss"] = pred_std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- Predictor applied after the expander. Variance on the pred output -------------------

class SimCLROnlyEqui(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,0,self.equi_emb_size,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels
    

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        x = x_equi
        y = y_equi

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)

        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        x = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_equi), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss = self.criterion(logits, labels)
        
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


class SimCLROnlyEquiRotColor(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,latent_dim=6,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,0,self.equi_emb_size,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels
    

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        x = x_equi
        y = y_equi

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z[:,:4])
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)

        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        x = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_equi), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss = self.criterion(logits, labels)
        
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval



#--------------------- EquiMod Variations -------------------
class VICRegEquiMod(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.emb_size = int(args.mlp.split("-")[-1])
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])
        self.equi_factor = args.equi_factor

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        self.projector_inv  = Projector(self.repr_size,"-".join(mlp_equi))

        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,self.equi_emb_size,self.equi_emb_size,num_classes=num_classes)
    # z unused but present for compatibility
    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x_inv = self.projector_inv(x_emb)
        y_inv = self.projector_inv(y_emb)

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        x = torch.cat((x_inv, x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)
        x_all = torch.cat(FullGatherLayer.apply(x), dim=0)
        y_all = torch.cat(FullGatherLayer.apply(y), dim=0)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        repr_loss_inv = F.mse_loss(x_inv, y_inv)

        x = x_all[...,:self.emb_size]
        y = y_all[...,:self.emb_size]
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_inv = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss_inv = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)

        loss_inv = (
            self.args.sim_coeff * repr_loss_inv
            + self.args.std_coeff * std_loss_inv
            + self.args.cov_coeff * cov_loss_inv
        )

        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)

        repr_loss_equi = F.mse_loss(y_equi_pred, y_equi)

        x = x_all[...,self.emb_size:]
        y = y_all[...,self.emb_size:]
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss_equi = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss_equi = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)

        loss_equi = (
            self.args.sim_coeff * repr_loss_equi
            + self.args.std_coeff * std_loss_equi
            + self.args.cov_coeff * cov_loss_equi
        )


        loss = loss_inv+ self.equi_factor*loss_equi

        stats["repr_loss_inv"] = repr_loss_inv
        stats["repr_loss_equi"] = repr_loss_equi
        stats["std_loss"] = std_loss_inv + std_loss_equi
        stats["cov_loss"] = cov_loss_inv + cov_loss_equi
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


class SimCLREquiMod(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        self.projector_inv  = Projector(self.repr_size,"-".join(mlp_equi))
        self.equi_factor = args.equi_factor


        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,4,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,self.equi_emb_size,self.equi_emb_size,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels
    

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)



        x_inv = self.projector_inv(x_emb)
        y_inv = self.projector_inv(y_emb)

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        x = torch.cat((x_inv, x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)

        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        x = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_equi), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_equi = self.criterion(logits, labels)

        x = torch.cat(FullGatherLayer.apply(x_inv), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_inv), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_inv = self.criterion(logits, labels)
        
        loss = self.equi_factor*loss_equi + loss_inv

        stats["loss"] = loss
        stats["repr_loss_equi"] = loss_equi
        return loss, loss_eval, stats, stats_eval


class SimCLREquiModRotColor(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.emb_size = int(args.mlp.split("-")[-1])

        mlp_dims = np.array([int(dim) for dim in args.mlp.split('-')])
        
        mlp_equi = [str(elt) for elt in list(np.round((mlp_dims)).astype(int))]
        self.equi_emb_size = int(mlp_equi[-1])
        self.projector_equi  = Projector(self.repr_size,"-".join(mlp_equi))

        self.projector_inv  = Projector(self.repr_size,"-".join(mlp_equi))


        if args.predictor_type == "hypernetwork":
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)
        elif args.predictor_type == "mlp":
            self.predictor = predictors.MLPPredictor(repr_dim=self.equi_emb_size,latent_dim=6,n_layers=args.tf_num_layers)
        else:
            self.predictor = hypernetwork.ParametrizedNet(self.equi_emb_size,6,self.args)

        self.evaluator = OnlineEvaluator(0,self.repr_size,self.equi_emb_size,self.equi_emb_size,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels
    

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)



        x_inv = self.projector_inv(x_emb)
        y_inv = self.projector_inv(y_emb)

        x_equi = self.projector_equi(x_emb)
        y_equi = self.projector_equi(y_emb)

        x = torch.cat((x_inv, x_equi),dim=1)
        y = torch.cat((y_inv, y_equi),dim=1)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z[:,:4])
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        if self.args.predictor_type == "hypernetwork":
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)
        elif self.args.predictor_type in ["mlp"]:
            y_equi_pred = self.predictor(x_equi, z)
        else:
            # Unsqueeze is very important here (see ParametrizedNetwork class for more details)
            y_equi_pred = self.predictor(x_equi.unsqueeze(1), z)

        with torch.no_grad():
            stats = std_losses(stats, self.args, "_pred", y_equi_pred)


        x = torch.cat(FullGatherLayer.apply(y_equi_pred), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_equi), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_equi = self.criterion(logits, labels)

        x = torch.cat(FullGatherLayer.apply(x_inv), dim=0)
        y = torch.cat(FullGatherLayer.apply(y_inv), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_inv = self.criterion(logits, labels)
        
        loss = loss_equi + loss_inv

        stats["loss"] = loss
        stats["repr_loss_equi"] = loss_equi
        return loss, loss_eval, stats, stats_eval

#--------------------- Standard VICReg -------------------
class VICReg(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.emb_size = int(args.mlp.split("-")[-1])
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.repr_size,args.mlp)
        self.evaluator = OnlineEvaluator(self.repr_size,0,self.emb_size,0,num_classes=num_classes)

    # z unused but present for compatibility
    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x = self.projector(x_emb)
        y = self.projector(y_emb)

        repr_loss = F.mse_loss(x, y)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )

        stats["repr_loss_inv"] = repr_loss
        stats["std_loss"] = std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- Standard SimCLR -------------------
class SimCLR(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.emb_size = int(args.mlp.split("-")[-1])
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.repr_size,args.mlp)
        self.evaluator = OnlineEvaluator(self.repr_size,0,self.emb_size,0,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x = self.projector(x_emb)
        y = self.projector(y_emb)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss = self.criterion(logits, labels)
        
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#--------------------- SimCLR + AugSelf -------------------
class SimCLRAugSelf(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.emb_size = int(args.mlp.split("-")[-1])
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.repr_size,args.mlp)
        self.predictor = nn.Sequential(
                    nn.Linear(2*self.repr_size,2*self.repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.repr_size,2*self.repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.repr_size, 4),
                )

        self.evaluator = OnlineEvaluator(self.repr_size,0,self.emb_size,0,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x = self.projector(x_emb)
        y = self.projector(y_emb)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_inv = self.criterion(logits, labels)

        angle_pred = self.predictor(torch.cat([x_emb,y_emb],axis=1))
        loss_equi = F.mse_loss(angle_pred,z)

        loss = loss_inv + self.args.ec_weight*loss_equi
    

        
        stats["loss"] = loss
        stats["repr_loss_inv"] = loss_inv
        stats["repr_loss_equi"] = loss_equi

        return loss, loss_eval, stats, stats_eval


#--------------------- SimCLR + AugSelf, Rot + color LV -------------------
class SimCLRAugSelfRotColor(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.emb_size = int(args.mlp.split("-")[-1])
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.repr_size,args.mlp)
        self.predictor = nn.Sequential(
                    nn.Linear(2*self.repr_size,2*self.repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.repr_size,2*self.repr_size),
                    nn.ReLU(),
                    nn.Linear(2*self.repr_size, 6),
                )

        self.evaluator = OnlineEvaluator(self.repr_size,0,self.emb_size,0,num_classes=num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.gpu = 0

    def info_nce_loss(self, features,batch_size=0):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits = logits / self.args.simclr_temp
        return logits, labels

    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)


        x = self.projector(x_emb)
        y = self.projector(y_emb)

        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z[:,:4])
        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        
        features = torch.cat([x,y],axis=0)
        logits, labels = self.info_nce_loss(features,batch_size=x.shape[0])
        loss_inv = self.criterion(logits, labels)

        angle_pred = self.predictor(torch.cat([x_emb,y_emb],axis=1))
        loss_equi = F.mse_loss(angle_pred,z)

        loss = loss_inv + self.args.ec_weight*loss_equi
    

        
        stats["loss"] = loss
        stats["repr_loss_inv"] = loss_inv
        stats["repr_loss_equi"] = loss_equi

        return loss, loss_eval, stats, stats_eval



#--------------------- VICReg inv only on a part -------------------
class VICRegPartInv(nn.Module):
    def __init__(self, args,num_classes=55):
        super().__init__()
        self.args = args
        self.equi_emb_size = self.args.equi
        self.emb_size = int(args.mlp.split("-")[-1])
        self.inv_emb_size = self.emb_size - self.equi_emb_size 
        self.backbone, self.repr_size = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(self.repr_size,args.mlp)
        self.evaluator = OnlineEvaluator(self.repr_size,0,self.inv_emb_size,self.equi_emb_size,num_classes=num_classes)

    # z unused but present for compatibility
    def forward(self, x, y, z ,labels):
        x_emb = self.backbone(x)
        y_emb = self.backbone(y)

        x = self.projector(x_emb)
        y = self.projector(y_emb)

        repr_loss = F.mse_loss(x[...,:self.inv_emb_size], y[...,:self.inv_emb_size])
        
        loss_eval, stats_eval = self.evaluator([x_emb.detach(),y_emb.detach()],[x.detach(),y.detach()],labels,z)

        stats = {}
        with torch.no_grad():
            stats = std_losses(stats, self.args, "_view1", x_emb, proj_out=x)
            stats = std_losses(stats, self.args, "_view2", y_emb, proj_out=y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.emb_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.emb_size)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        stats["repr_loss_inv"] = repr_loss
        stats["std_loss"] = std_loss
        stats["cov_loss"] = cov_loss
        stats["loss"] = loss
        return loss, loss_eval, stats, stats_eval


#====================================================
#               Helper Functions
#=====================================================

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
        return res

# To use for logging, see main_vicreg_shared.py
def cor_metrics(outputs, args, suffix, embedding, proj_out=None):
    if proj_out is not None:
        proj_out = torch.cat(FullGatherLayer.apply(proj_out), dim=0)
        p = (proj_out - proj_out.mean(dim=0)) / (proj_out.std(dim=0) + 1e-05)
        outputs["corhead" + suffix] = torch.mean(off_diagonal((p.T @ p) / (p.size(0) - 1)))
    
    embedding = torch.cat(FullGatherLayer.apply(embedding), dim=0)
    e = (embedding - embedding.mean(dim=0)) / (embedding.std(dim=0) + 1e-05)
    outputs["coremb" + suffix] = torch.mean(off_diagonal((e.T @ e) / (e.size(0) - 1)))

    return outputs


def std_losses(outputs, args, suffix, embedding, proj_out=None):
    outputs = cor_metrics(outputs, args, suffix, embedding, proj_out=proj_out)

    embedding = F.normalize(embedding, p=2, dim=1)
    outputs["stdemb" + suffix] = torch.mean(embedding.std(dim=0))

    if proj_out is not None:
        proj_out = F.normalize(proj_out, p=2, dim=1)
        if args.std_coeff > 0.0:
            proj_out = torch.cat(FullGatherLayer.apply(proj_out), dim=0)
        outputs["stdhead" + suffix] = torch.mean(proj_out.std(dim=0))

    return outputs


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


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)

# Useful when you need to do computations on the whole batch, like the variance/covariance regularization
# or any contrastive kind of thing for example
# It basically aggregates and synchronizes the tensors between all devices
# Analogous to all_gather but with gradient propagation
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

#def custom_pairwise_dist(x1,x2,p=2.0,eps=1e-6):
#    return (torch.sum((x1-x2).pow(p),axis=1)+eps).pow(1/p)
def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2