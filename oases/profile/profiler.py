# coding=utf-8
# Copyright (c) 2023.  All rights reserved.
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


# parts of code are adopted from https://github.com/NVIDIA/Megatron-LM/blob/v2.0/megatron/model/language_model.py

import sys 
import time
from .. import mpu
from ..utils import see_memory_usage, print_rank_0, init_process
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import LayerNorm
import argparse
import os
from ..mpu.tmplayers import RowParallelLinear, ColumnParallelLinear, Timer

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=1)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--num_layer", type=int, default=24)
parser.add_argument("--activation_checkpoint", action='store_true')
args = parser.parse_args()

tp = int(os.environ['TP'])
dp = int(os.environ['DP'])



SLURM_NTASKS=int(os.environ['SLURM_NTASKS'])
NODE_RANK=int(os.environ['SLURM_PROCID'])
args.local_rank = int(os.environ['SLURM_LOCALID'])

dist.init_process_group(backend='nccl', world_size=SLURM_NTASKS,rank=NODE_RANK)


init_process(tp=tp, dp=dp)


torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
world_size = dist.get_world_size()
global_rank = dist.get_rank()

mpu.initialize._set_random_seed(42+global_rank)




def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores

class ParallelAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, hidden_size, num_attention_heads, no_transformer=False, skip_comm=False):
        super(ParallelAttention, self).__init__()

        projection_size = hidden_size

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
  
        self.projection_size = projection_size
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, num_attention_heads)

        self.num_attention_heads = num_attention_heads
        # linear layer.
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * projection_size,
            gather_output=False,
            skip_comm=skip_comm,
            )

        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(0.7)

        # Output.
        self.dense = RowParallelLinear(
            projection_size,
            hidden_size,
            input_is_parallel=True,
            skip_bias_add=False,
            skip_comm=skip_comm,
            )

        self.no_transformer=no_transformer
        if no_transformer:
            self.input_layernorm = LayerNorm(hidden_size, eps=1e-5, device=device)



    def forward(self, hidden_states, attention_mask=None):


        if self.no_transformer:

            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)


        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]

        num_attention_heads_per_partition = mpu.divide(
            self.num_attention_heads, mpu.get_model_parallel_world_size())
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer,
            key_layer,
            value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.7, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        mask_output = attention_mask_func(attention_scores, attention_mask) if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(mask_output)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # with mpu.get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        hidden_size_per_partition = mpu.divide(self.projection_size,
                                mpu.get_model_parallel_world_size())
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)


        if self.no_transformer:

            # re-enable torch grad to enable fused optimization.
            out = torch.nn.functional.dropout(output, p=0.7)
            layernorm_input = residual + out
            return layernorm_input


        return output, bias

class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, hidden_size, ffn_hidden_size, no_transformer=False, skip_comm=False):
        super(ParallelMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            gather_output=False,
            skip_bias_add=False,
            skip_comm=skip_comm)

        self.activation_func = F.gelu


        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            input_is_parallel=True,
            skip_bias_add=False,
            skip_comm=skip_comm)

        self.no_transformer = no_transformer
        if no_transformer:
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=1e-5, device=device)

        

    def forward(self, hidden_states):
        if self.no_transformer:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)


        intermediate_parallel = \
                self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)

        if self.no_transformer:
            out = torch.nn.functional.dropout(output, p=0.7)
            output = residual + out
            return output

        return output, output_bias

class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, h, n):

        super(ParallelTransformerLayer, self).__init__()

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            h, eps=1e-5, device=device)

        # Self attention.
        self.self_attention = ParallelAttention(h, n)
        self.hidden_dropout = 0.7

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            h, eps=1e-5, device=device)


        # MLP
        self.mlp = ParallelMLP(h, h*4)
        
    def forward(self, hidden_states):
            # hidden_states: [b, s, h]

            # Layer norm at the beginning of the transformer layer.
            layernorm_output = self.input_layernorm(hidden_states)
            # Self attention.
            attention_output, attention_bias = \
                self.self_attention(layernorm_output)

            residual = hidden_states

            # re-enable torch grad to enable fused optimization.
            out = torch.nn.functional.dropout(attention_output, p=0.7)
            layernorm_input = residual + out

            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

            # MLP.
            mlp_output, mlp_bias = self.mlp(layernorm_output)

            residual = layernorm_input

            out = torch.nn.functional.dropout(mlp_output, p=0.7)
            output = residual + out
            return output


timers = {}
timers['attn_fp'] = Timer('attn_fp')
timers['attn_bp'] = Timer('attn_bp')
timers['mlp_fp'] = Timer('mlp_fp')
timers['mlp_bp'] = Timer('attn_bp')
timers['comm'] = Timer('comm')
if dist.get_rank() == 0:
    mem_prof = None
else:
    mem_prof = 1

class test_model(nn.Module):
    def __init__(self, num_layer, h, n):
        super(test_model, self).__init__()
        layers = []
        for _ in range(num_layer):
            layers.append(ParallelAttention(h, n, no_transformer=True, skip_comm=True))
            layers.append(ParallelMLP(h, h*4, no_transformer=True, skip_comm=True))
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, input):
        x = input
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                timers['attn_fp'].start()
                x = layer(x)
                timers['attn_fp'].stop()
                if mem_prof is None:
                    see_memory_usage(f'attn fp layer {idx} ', True, ranks=[0])
            else:
                timers['mlp_fp'].start()
                x = layer(x)
                timers['mlp_fp'].stop()
                if mem_prof is None:
                    added_mem = see_memory_usage(f'attn fp layer {idx} ', True, ranks=[0])
            timers['comm'].start()
            dist.all_reduce(x)
            timers['comm'].stop()
        return x
        

def timerhook(module, grad_input, grad_output):
    if isinstance(module, ParallelAttention):
        timers['attn_bp'].stop()
    elif isinstance(module, ParallelMLP) and timers['mlp_bp'].started_:
        timers['mlp_bp'].stop()

    timers['comm'].start()
    grad_input_ = grad_input[0]
    dist.all_reduce(grad_input_)
    timers['comm'].stop()
    if isinstance(module, ParallelAttention):
        if module.index != 0:
            timers['mlp_bp'].start()
    elif isinstance(module, ParallelMLP):
        timers['attn_bp'].start()
    return (grad_input_,)


# see_memory_usage('before init ', True)
model = test_model(num_layer=args.num_layer, h=args.hidden_size, n=args.hidden_size//64)
model.cuda()
print_rank_0(model)

for idx, l in enumerate(model.layers):
    l.index = idx
    l.register_full_backward_hook(timerhook)

# see_memory_usage('after init ', True)
optim = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()

for key, val in timers.items():
    val.warmup_start()
step = 50
step_time = []
for _ in range(step):

    input = torch.rand(1024, args.batch_size, args.hidden_size, device=device).detach()
    input.requires_grad = True
    label = torch.randint(100, (args.batch_size, args.hidden_size), device=device).long().detach()
    # see_memory_usage('after data ', True)
    
    torch.cuda.synchronize()
    start_time = time.time()

    optim.zero_grad()
    x = model(input)
    # see_memory_usage('after fp ', True)
    loss = loss_fn(x.transpose(0, 1).contiguous(), label)

    loss.backward()
    if mem_prof is None:
        mem_prof = see_memory_usage(f'mlp bp layer ', True, ranks=[0])
    # see_memory_usage('after bp ', True)

    optim.step()

    if _ == 5:
        for key, val in timers.items():
            val.warmup_stop()

    torch.cuda.synchronize()
    step_time.append(time.time()-start_time)
    print_rank_0(step_time[-1])


print_rank_0('average: ')
print_rank_0(sum(step_time[10:])/len(step_time[10:]))



for key, val in timers.items():
    print('Rank', dist.get_rank(), key, val.average()*1000)
        # print(sorted(val.alltime(), reverse=True)[:30])
print('Rank', dist.get_rank(), 'runtime_memory', mem_prof)
