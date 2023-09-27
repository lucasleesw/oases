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

# Parts of code are adopted from https://github.com/NVIDIA/Megatron-LM/blob/v2.0/megatron/model/language_model.py
# and https://github.com/NVIDIA/Megatron-LM/blob/v2.0/megatron/mpu/layers.py





import importlib
import time
from .. import mpu
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.init as init
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    
    with mpu.random.get_cuda_rng_tracker().fork():
        init_method(weight)
class ColumnParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, skip_comm=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.output_size_per_partition = mpu.divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        self.weight = nn.Parameter(torch.empty(
            self.output_size_per_partition, self.input_size,
            device=torch.cuda.current_device(), dtype=torch.float32))
        _initialize_affine_weight_gpu(self.weight, init_method,
                                        partition_dim=1, stride=stride)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=torch.float32))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.skip_comm = skip_comm

    def forward(self, input_):
        if self.skip_comm:
            input_parallel = input_
        else:
        # Set up backprop all-reduce.
            input_parallel = mpu.mappings.copy_to_model_parallel_region(input_)

        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = mpu.mappings.gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel 
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
        

    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size, self.output_size_per_partition, self.bias is not None
        )


class RowParallelLinear(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 skip_comm=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_model_parallel_world_size()
        self.input_size_per_partition = mpu.divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.

        self.weight = nn.Parameter(torch.empty(
            self.output_size, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=torch.float32))
        _initialize_affine_weight_gpu(self.weight, init_method,
                                                partition_dim=1, stride=stride)
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size, device=torch.cuda.current_device(),
                dtype=torch.float32))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.skip_comm = skip_comm

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = mpu.mappings.scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        if not self.skip_comm:
            # All-reduce across all the partitions.
            output_ = mpu.mappings.reduce_from_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size_per_partition, self.output_size, self.bias is not None
        )


class Timer:
    """Timer."""
    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self.count = 0
        self.warmup_ = False

        # self.time_list = []

    def start(self):
        """Start the timer."""
        if self.warmup_:
            torch.cuda.synchronize()
            return
        assert not self.started_, 'timer has already been started'
        self.count += 1
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self, reset=False):
        """Stop the timer."""
        if self.warmup_:
            torch.cuda.synchronize()
            return
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        if reset:
            self.elapsed_ = (time.time() - self.start_time)
            self.count = 0
        else:
            self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_
    
    def average(self, reset=True):
        started_ = self.started_
        if self.started_:
            self.stop()
        elapsed_ = self.elapsed_
        avg = elapsed_ / self.count
        # Reset the elapsed time
        if reset:
            self.reset()
        if started_:
            self.start()
        return avg


    def warmup_start(self):
        self.warmup_ = True

    def warmup_stop(self):
        self.warmup_ = False

    




def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


class ParallelAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, hidden_size, num_attention_heads, no_transformer=False):
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
            skip_comm=True,
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
            skip_bias_add=True,
            skip_comm=True,
            )

        self.no_transformer=no_transformer
        if no_transformer:
            self.input_layernorm = LayerNorm(hidden_size, eps=1e-5, device=torch.cuda.current_device())



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
            beta=0.0, alpha=(1.0/self.norm_factor))

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

    def __init__(self, hidden_size, ffn_hidden_size, no_transformer=False):
        super(ParallelMLP, self).__init__()

        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            gather_output=False,
            skip_bias_add=False,
            skip_comm=True)

        self.activation_func = F.gelu


        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            input_is_parallel=True,
            skip_bias_add=True,
            skip_comm=True)

        self.no_transformer = no_transformer
        if no_transformer:
            self.post_attention_layernorm = LayerNorm(hidden_size, eps=1e-5, device=torch.cuda.current_device())

        

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


class AddDropNorm(nn.Module):
    def __init__(self, h, p=0.7):
        super(AddDropNorm, self).__init__()

        self.post_attention_layernorm = LayerNorm(
            h, eps=1e-5, device=torch.cuda.current_device())
        self.drop = torch.nn.Dropout(p)

    def forward(self, hidden_states, bias, residual):
        
        attention_output = hidden_states + bias

        # re-enable torch grad to enable fused optimization.
        out = self.drop(attention_output)
        layernorm_input = residual + out
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        return layernorm_output, layernorm_input

class AddDrop(nn.Module):
    def __init__(self, p=0.7):
        super(AddDrop, self).__init__()
        self.drop = torch.nn.Dropout(p)

    def forward(self, hidden_states, bias, residual):
        mlp_output = hidden_states + bias
        out = self.drop(mlp_output)
        output = residual + out

        return output



class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, h, n):

        super(ParallelTransformerLayer, self).__init__()



        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            h, eps=1e-5, device=torch.cuda.current_device())

        # Self attention.
        self.self_attention = ParallelAttention(h, n)
        # self.hidden_dropout = 0.7


        self.post_attn = AddDropNorm(h)

        # MLP
        self.mlp = ParallelMLP(h, h*4)

        self.post_mlp = AddDrop()
        
    def forward(self, hidden_states):


            # Layer norm at the beginning of the transformer layer.
            attn_ln_out = self.input_layernorm(hidden_states)


            # backward allreduce
            attn_input = mpu.mappings.copy_to_model_parallel_region(attn_ln_out)


            # Self attention.
            attention_output_parallel, attention_bias = self.self_attention(attn_input)


            # attention_output_ = mpu.mappings.reduce_from_model_parallel_region(attention_output_parallel)
            dist.all_reduce(attention_output_parallel, group=mpu.get_model_parallel_group())


            layernorm_output, residual = self.post_attn(attention_output_parallel, attention_bias, hidden_states)

            
            # backward allreduce
            mlp_input = mpu.mappings.copy_to_model_parallel_region(layernorm_output)


            # MLP.
            mlp_output_parallel, mlp_bias = self.mlp(mlp_input)


            # mlp_output_ = mpu.mappings.reduce_from_model_parallel_region(mlp_output_parallel)
            dist.all_reduce(mlp_output_parallel, group=mpu.get_model_parallel_group())


            output = self.post_mlp(mlp_output_parallel, mlp_bias, residual)


            return output


