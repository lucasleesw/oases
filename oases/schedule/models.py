# coding=utf-8
# Copyright (c) 2023 All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
import time
from .. import mpu
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from ..mpu.tmplayers import ParallelTransformerLayer
from .overlap_checkpoint import overlap_tmp_checkpoint, comm_optimized_checkpoint, comm_optimized_overlap_checkpoint, DPTP_optimized_overlap_checkpoint
from ..utils import init_process


""" We use the following notation throughout this file:
    h: hidden size
    n: number of attention heads
    p: number of model parallel partitions
    np: n/p
    hp: h/p
    hn: h/n
    b: batch size
    s: sequence length
    l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""



class LM_Model(nn.Module):
    def __init__(self, num_layer, h, n, cutpoint=None, changed_tp=None, changed_dp=None, ac=False):
        super(LM_Model, self).__init__()
        layers = []
        for idx in range(num_layer):
            if cutpoint is not None and idx < cutpoint:
                with mpu.initialize.change_model_parallel(changed_tp,changed_dp):
                    layers.append(ParallelTransformerLayer(h, n))
            else:
                layers.append(ParallelTransformerLayer(h, n))
        self.layers = torch.nn.ModuleList(layers)
        self.ac=ac

    
    def forward(self, input):
        x = input
        for layer in self.layers:
            if self.ac:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
        


class BenchmarkLM(nn.Module):
    def __init__(self, schedule, num_l, h, n, tp, dp, auto_plan=False, cutpoint=0):
        super(BenchmarkLM, self).__init__()

        if auto_plan and cutpoint == 0:
            changed_tp, changed_dp, cutpoint = mpu.planner.get_plan_args()
            self.cutpoint=cutpoint
            self.batch_size_mul = int(os.environ['DP'])//dp
            self.schedule='oases'
        else:
            # tp = int(os.environ['TP'])
            # dp = int(os.environ['DP'])
            if cutpoint > 0 and schedule=='oases':
                changed_tp, changed_dp = (tp//2, dp*2) 
                self.cutpoint = cutpoint
            else:
                changed_tp, changed_dp = 1, 1
                self.cutpoint = None
            self.schedule=schedule
            self.batch_size_mul = 1
        init_process(tp=tp, dp=dp)
        
        if self.schedule == 'default_checkpoint':
            ac = True
        else:
            ac = False
        self.dp = dp
        self.model = LM_Model(num_layer=num_l, h=h, n=n, cutpoint=self.cutpoint, changed_tp=changed_tp, changed_dp=changed_dp, ac=ac)
        self.changed_tp, self.changed_dp = changed_tp, changed_dp
        # print(self.changed_tp, self.changed_tp, self.cutpoint)
    def forward(self, input):
        if self.schedule == 'overlap_checkpoint':
            x = input.chunk(2, dim=1)
            output = overlap_tmp_checkpoint(self.model, *x)
            x = torch.cat(output, dim=1)
        elif self.schedule == 'optim_comm_checkpoint':
            x = comm_optimized_checkpoint(self.model, input)
        elif self.schedule == 'oases':
            x = input.chunk(2, dim=1)
            output = comm_optimized_overlap_checkpoint(self.model, self.cutpoint, *x)
            x = torch.cat(output, dim=1)
        else:
            x = self.model(input)
        return x

    def reduce_grad(self):
        if self.cutpoint is not None and self.cutpoint > 0:
            if self.changed_dp > 1:
                with mpu.initialize.change_model_parallel(self.changed_tp, self.changed_dp):
                    grads = [param.grad.data for l in self.model.layers[:self.cutpoint] for param in l.parameters()]
                    coalesced = _flatten_dense_tensors(grads)
                    coalesced /= mpu.get_data_parallel_world_size()
                    torch.distributed.all_reduce(
                        coalesced, group=mpu.get_data_parallel_group())
                    for buf, synced in zip(grads, _unflatten_dense_tensors(
                            coalesced, grads)):
                        buf.copy_(synced)
            if self.dp > 1:
                grads = [param.grad.data for l in self.model.layers[self.cutpoint:] for param in l.parameters()]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
        elif self.dp > 1:
            grads = [param.grad.data for param in self.model.parameters()]
            coalesced = _flatten_dense_tensors(grads)
            coalesced /= mpu.get_data_parallel_world_size()
            torch.distributed.all_reduce(
                coalesced, group=mpu.get_data_parallel_group())
            for buf, synced in zip(grads, _unflatten_dense_tensors(
                    coalesced, grads)):
                buf.copy_(synced)
    
    def gen_rand_inputs(self, b, h):
        if self.changed_dp > self.dp:
            input = torch.rand(1024, self.batch_size_mul*b//(self.changed_dp//self.dp), h, device=torch.cuda.current_device()).detach()
        else:
            input = torch.rand(1024, b//self.changed_dp, h, device=torch.cuda.current_device()).detach()
        return input