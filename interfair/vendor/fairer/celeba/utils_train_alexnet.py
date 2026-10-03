import os,sys
import argparse
import pandas as pd
import numpy as np
from numpy.random import beta
from tqdm import tqdm
# import pickle5 as pickle
import pickle

from pprint import pprint

import torch
import torch.nn as nn
from torch import optim

import random
import torch.backends.cudnn as cudnn

random.seed(6)
torch.manual_seed(6)
np.random.seed(6)

cudnn.deterministic = True


from model_prune_tv import *

from utils import *


from torch.utils.tensorboard import SummaryWriter

import warnings

def fit_model_eo(epochs, model, dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11,
                 mode='mixup', lam=30, lam2=30,
                 pruning_engine=None, group_wd_optimizer=None, args=None, pretest_call=lambda x: x, criterion=None, 
                 writer=None, optimizer=None, optimizer_linear=None, sl=0):

    print ("epoch:", epochs, "total:", args.epochs, "--->", "args.pruning", args.pruning)

    len_dataloader = min(len(dataloader), len(dataloader_00), len(dataloader_01), len(dataloader_10),
                         len(dataloader_11))
    data_iter = iter(dataloader)
    data_iter_00 = iter(dataloader_00)
    data_iter_01 = iter(dataloader_01)
    data_iter_10 = iter(dataloader_10)
    data_iter_11 = iter(dataloader_11)

    model.train()

    metainfo = None
    for it in tqdm(range(len_dataloader)):

        inputs_00, target_00 = data_iter_00.next()
        inputs_01, target_01 = data_iter_01.next()
        inputs_10, target_10 = data_iter_10.next()
        inputs_11, target_11 = data_iter_11.next()

        inputs_00, inputs_01 = inputs_00.to(device), inputs_01.to(device)
        inputs_10, inputs_11 = inputs_10.to(device), inputs_11.to(device)
        target_00, target_01 = target_00.float().to(device), target_01.float().to(device)
        target_10, target_11 = target_10.float().to(device), target_11.float().to(device)
        
        inputs_0 = torch.cat((inputs_00, inputs_01), 0)
        inputs_1 = torch.cat((inputs_10, inputs_11), 0)
        target_0 = torch.cat((target_00, target_01), 0)
        target_1 = torch.cat((target_10, target_11), 0)

        inputs_0_ = [inputs_00, inputs_01]
        inputs_1_ = [inputs_10, inputs_11]

        inputs_list = [inputs_00, inputs_01, inputs_10, inputs_11]
        target_list = [target_00, target_01, target_10, target_11]

        inputs = torch.cat((inputs_00, inputs_01, inputs_10, inputs_11), 0)
        target = torch.cat((target_00, target_01, target_10, target_11), 0)
        feat = model(inputs)
        ops = feat

        loss_all = 0
        loss_gap = 0
        grads = []

        loss_sup = criterion(ops[:, 0], target)

        if mode == 'GapReg':
            loss_gap = 0
            for g in range(2):
                # model.eval()
                # model_linear.eval()
                inputs_0 = inputs_0_[g]
                inputs_1 = inputs_1_[g]
                feat = model(inputs_0)
                ops_0 = feat
                feat = model(inputs_1)
                ops_1 = feat

                loss_gap += torch.abs(ops_0.mean() - ops_1.mean())
            #                 print("Mean:", ops_0.mean().item(), ops_1.mean().item())

            loss = loss_sup + lam * loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

            writer.add_scalar('Train/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Train/loss_gap', loss_gap, len_dataloader * epochs + it)


        elif mode == 'CAIGA':
            losses = []
            ops_c_mean = []

            for g in range(4):
                # model.eval()
                # model_linear.eval()
                inputs_c = inputs_list[g]
                target_c = target_list[g]
                feat_c = model(inputs_c)
                ops_c = feat_c
                loss_c = criterion(ops_c[:, 0], target_c)
                ops_c_mean.append(ops_c.mean())
                loss_all += loss_c
                losses.append(loss_c)
                env_grad = torch.autograd.grad(loss_c, model.parameters(), create_graph=True)
                grads.append(env_grad)
            # print(ops_c_mean)

            # compute trace penalty
            loss_reg_s = 0
            for g1, g2, (name, para) in zip(grads[0], grads[2], model.named_parameters()):
                if "weight" in name and "gate" not in name:
                    
                    nunits = para.shape[0]
                    importance1 = (para * g1).pow(2).view(nunits, -1).sum(dim=1)
                    importance2 = (para * g2).pow(2).view(nunits, -1).sum(dim=1)
                    loss_reg_s += torch.cosine_similarity(importance1, importance2, dim=0)
#                     print(name, torch.cosine_similarity(importance1, importance2, dim=0))

            for g1, g2, (name, para) in zip(grads[1], grads[3], model.named_parameters()):
                if "weight" in name and "gate" not in name:
                    nunits = para.shape[0]
                    importance1 = (para * g1).pow(2).view(nunits, -1).sum(dim=1)
                    importance2 = (para * g2).pow(2).view(nunits, -1).sum(dim=1)
                    loss_reg_s += torch.cosine_similarity(importance1, importance2, dim=0)
#                     print(name, torch.cosine_similarity(importance1, importance2, dim=0))

            loss_reg = torch.abs(ops_c_mean[0] - ops_c_mean[2]) + torch.abs(ops_c_mean[1] - ops_c_mean[3])
            # loss_gap = torch.abs(losses[0] - losses[2]) + torch.abs(losses[1] - losses[3])
            objective = loss_sup - lam * loss_reg_s + lam2 * loss_reg
            loss = objective

            writer.add_scalar('Train/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Train/loss_reg_s', loss_reg_s, epochs)
            writer.add_scalar('Train/loss_reg', loss_reg, epochs)

            if (it % 20) == 0:
                print(f"loss_sup:{loss_sup}, penalty_loss: {loss_reg_s}, loss_gap: {loss_reg}")

        

        elif mode == 'mixup':
            alpha = 1
            loss_gap = 0
            for g in range(2):
                inputs_0 = inputs_0_[g]
                inputs_1 = inputs_1_[g]
                gamma = beta(alpha, alpha)
                inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
                inputs_mix = inputs_mix.requires_grad_(True)

                feat = model(inputs_mix)
                ops = feat
                ops = ops.sum()

                gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
                x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_gap += torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup: {:.7f}".format(loss_sup, loss_gap))

            writer.add_scalar('Train/loss', loss, len_dataloader * epochs + it)
            writer.add_scalar('Train/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Train/loss_gap', loss_gap, len_dataloader * epochs + it)

        else:
            loss = loss_sup
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f}".format(loss_sup))
        optimizer.zero_grad()
#         optimizer_linear.zero_grad()

        # final loss
        if args.pruning:
            assert pruning_engine.pruning_method not in [40, 50]
            # # useful for method 40 and 50 that calculate oracle
            # pruning_engine.run_full_oracle(model, data, target, criterion, initial_loss=loss.item())
        if args.pruning:
            if pruning_engine.needs_hessian:
                pruning_engine.compute_hessian(loss)
        if args.pruning:
            group_wd_optimizer.step()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        # step_after will calculate flops and number of parameters left
        # needs to be launched before the main optimizer,
        # otherwise weight decay will make numbers not correct
        if args.pruning:
            if it % args.log_interval == 0:
                group_wd_optimizer.step_after()

        optimizer.step()
#         optimizer_linear.step()

        if args.pruning:
            pruning_engine.do_step(loss=loss_sup.item(), optimizer=optimizer, pretest=pretest_call,
                                   test=(it == len_dataloader - 1))
            model_name = args.model_name
            if 1 == 1:
                if (
                        pruning_engine.maximum_pruning_iterations == pruning_engine.pruning_iterations_done) and pruning_engine.set_moment_zero:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            param_state = optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                del param_state['momentum_buffer']

                    pruning_engine.set_moment_zero = False

        writer.add_scalar('Loss/loss', loss, len_dataloader * epochs + it)
        if it == len_dataloader - 1:
            metainfo = pretest_call(namespace="Testing")
            if mode == "CAIGA":
                metainfo["loss_reg_s"] = loss_reg_s 

    return metainfo


def fit_model_dp(epochs, model, dataloader, dataloader_0, dataloader_1, mode='mixup_smooth', lam=100, lam2=0,
                 pruning_engine=None, group_wd_optimizer=None, args=None, pretest_call=lambda x: x, criterion=None, 
                 writer=None, optimizer=None, optimizer_linear=None):
    # if epochs >= warmup_epoch:
    #     args.pruning = True
    #     # freezen_gate(model, freeze_=False)
    #     # c_count_param = count_parameters(model)
    #     # assert c_count_param <64601
    #     # print ("total param:", c_count_param )
    # else:
    #     args.pruning = False
    #     # freezen_gate(model, freeze_=True)

    print ("epoch:", epochs, "total:", args.epochs, "--->", "args.pruning", args.pruning)

    len_dataloader = min(len(dataloader), len(dataloader_0), len(dataloader_1))
    len_dataloader = int(len_dataloader)
    data_iter = iter(dataloader)
    data_iter_0 = iter(dataloader_0)
    data_iter_1 = iter(dataloader_1)

    model.train()

    metainfo = None

    for it in tqdm(range(len_dataloader)):
        inputs_0, target_0 = data_iter_0.next()
        inputs_1, target_1 = data_iter_1.next()
        inputs_0, target_0 = inputs_0.cuda(), target_0.float().cuda()
        inputs_1, target_1 = inputs_1.cuda(), target_1.float().cuda()

        # supervised loss
        inputs = torch.cat((inputs_0, inputs_1), 0)
        target = torch.cat((target_0, target_1), 0)
        feat = model(inputs)
        ops = feat
        loss_sup = criterion(ops[:, 0], target)
#         if it % 100 == 0:
#             print(ops[:, 0], target)

        if mode == 'GapReg':
            feat = model(inputs_0)
            ops_0 = feat
            feat = model(inputs_1)
            ops_1 = feat

            loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
            loss = loss_sup + lam * loss_gap

            writer.add_scalar('Loss/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Loss/loss_grad', loss_gap, len_dataloader * epochs + it)

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} | Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif mode == 'mixup':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Input Mixup
            inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            feat = model(inputs_mix)
            ops = feat.sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            writer.add_scalar('Loss/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Loss/loss_grad', loss_grad, len_dataloader * epochs + it)

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup {:.7f}".format(loss_sup, loss_grad))

        elif mode == "CAIGA":

            # ERM loss
            batch_x_0 = inputs_0
            batch_x_1 = inputs_1
            batch_y_0 = target_0
            batch_y_1 = target_1

            batch_x = torch.cat((batch_x_0, batch_x_1), 0)
            batch_y = torch.cat((batch_y_0, batch_y_1), 0)

            batch_x_ = [batch_x_0, batch_x_1]
            batch_y_ = [target_0, target_1]

            losses = []
            grads = []
            loss_reg = 0

            feat_0 = model(batch_x_0)
            output_0 = feat_0
            feat_1 = model(batch_x_1)
            output_1 = feat_1
            loss_reg += torch.abs(output_0.mean() - output_1.mean())
            
            
            loss_reg_s = 0
            output_mean = []
            for g in range(2):
                feat = model(batch_x_[g])
                output = feat
                output_mean.append(output.mean())
                loss_c = criterion(output[:, 0], batch_y_[g])
                losses.append(loss_c)
                env_grad = torch.autograd.grad(loss_c, model.parameters(), create_graph=True)
                grads.append(env_grad)

            g_abs_0 = 0
            g_abs_1 = 0
            loss_reg_s = 0
            for g1, g2, (name, para) in zip(grads[0], grads[1], model.named_parameters()):
                if "weight" in name and "gate" not in name:
                    nunits = para.shape[0]
                    importance1 = (para * g1).pow(2).view(nunits, -1).sum(dim=1)
                    importance2 = (para * g2).pow(2).view(nunits, -1).sum(dim=1)
                    if lam == 0:
                        importance1 = importance1.detach()
                        importance2 = importance2.detach()
#                         print(importance1, importance2.shape)
                    loss_reg_s += torch.cosine_similarity(importance1, importance2, dim=0)
#                         loss_reg_s += torch.dist(importance1, importance2, p=2)

            loss = loss_sup - lam * loss_reg_s + lam2 * loss_reg

            writer.add_scalar('Loss/loss_sup', loss_sup, len_dataloader * epochs + it)
            writer.add_scalar('Loss/loss_reg_s', loss_reg_s, len_dataloader * epochs + it)
            writer.add_scalar('Loss/loss_reg', loss_reg, len_dataloader * epochs + it)
            
            if it % 100 == 0:
                pprint("Loss Sup: {:.4f} loss_reg {:.7f}".format(loss_sup, loss_reg))
            

        else:
            loss = loss_sup
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f}".format(loss_sup))
        # model.train()
        # model_linear.train()
        optimizer.zero_grad()
#         optimizer_linear.zero_grad()

        # final loss
        if args.pruning:
            assert pruning_engine.pruning_method not in [40, 50]
            # # useful for method 40 and 50 that calculate oracle
            # pruning_engine.run_full_oracle(model, data, target, criterion, initial_loss=loss.item())
        if args.pruning:
            if pruning_engine.needs_hessian:
                pruning_engine.compute_hessian(loss)
        if args.pruning:
            group_wd_optimizer.step()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)
        # step_after will calculate flops and number of parameters left
        # needs to be launched before the main optimizer,
        # otherwise weight decay will make numbers not correct
        if args.pruning:
            if it % args.log_interval == 0:
                group_wd_optimizer.step_after()

        optimizer.step()
#         optimizer_linear.step()

        if args.pruning:
            # pruning_engine.update_flops(stats=group_wd_optimizer.per_layer_per_neuron_stats)
            # pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)
            pruning_engine.do_step(loss=loss_sup.item(), optimizer=optimizer, pretest=pretest_call,
                                   test=(it == len_dataloader - 1))
            #             if it == len_dataloader-1 :
            #                 pretest_call(namespace="MiddleTest")
            model_name = args.model_name
            # if "resnet" in model_name :
            if 1 == 1:
                if (
                        pruning_engine.maximum_pruning_iterations == pruning_engine.pruning_iterations_done) and pruning_engine.set_moment_zero:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            param_state = optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                del param_state['momentum_buffer']

                    pruning_engine.set_moment_zero = False

        writer.add_scalar('Loss/loss', loss, len_dataloader * epochs + it)
        if it == len_dataloader - 1:
            metainfo = pretest_call(namespace="Test")
            if mode == "CAIGA":
                metainfo["loss_reg_s"] = loss_reg_s.item()

    return metainfo