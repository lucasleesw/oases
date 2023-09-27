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
import warnings
from typing import Any, Iterable, List, Tuple
from .. import mpu
import torch.distributed as dist

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def detach_tensor(input):
    x = input.detach()
    x.requires_grad = True
    return x

def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

def sync_comm():
    if mpu.mappings.ASYNC_OP != []:
        async_op = mpu.mappings.ASYNC_OP.pop(0)
        async_op.wait()


class OverlapTMPCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function

        input_1, input_2 = args

        # ctx.inputs = args

        to_saved = []

        with torch.no_grad():
            for idx, l in enumerate(run_function.layers):
                to_saved.append(input_1)
                layernorm_out_1 = l.input_layernorm(input_1)
                # Self attention.
                attn_out_1, attn_bias_1 = l.self_attention(layernorm_out_1)

                if idx > 0:
                    handler.wait()
                handler = dist.all_reduce(attn_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)


                if idx > 0:
                    input_2 = last_layer.post_mlp(mlp_out_2, mlp_bias_2, residual_2)
                
                to_saved.append(input_2)
                layernorm_out_2 = l.input_layernorm(input_2)
                # Self attention.
                attn_out_2, attn_bias_2 = l.self_attention(layernorm_out_2)

                handler.wait()
                handler = dist.all_reduce(attn_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)


                mlp_in_1, residual_1 = l.post_attn(attn_out_1, attn_bias_1, input_1)
                to_saved.append(residual_1)
                to_saved.append(mlp_in_1)
                # MLP.
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)

                handler.wait()
                handler = dist.all_reduce(mlp_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)


                mlp_in_2, residual_2 = l.post_attn(attn_out_2, attn_bias_2, input_2)
                to_saved.append(residual_2)
                to_saved.append(mlp_in_2)

                # MLP.
                mlp_out_2, mlp_bias_2 = l.mlp(mlp_in_2)

                handler.wait()
                handler = dist.all_reduce(mlp_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                input_1 = l.post_mlp(mlp_out_1, mlp_bias_1, residual_1)
                last_layer = l


            handler.wait()
            input_2 = last_layer.post_mlp(mlp_out_2, mlp_bias_2, residual_2)

        ctx.save_for_backward(*to_saved)

        return input_1, input_2

    @staticmethod
    def backward(ctx, *args):
        saved_tensors = list(ctx.saved_tensors)

        grad_1, grad_2 = args

        with torch.enable_grad():
            for idx, l in enumerate(reversed(ctx.run_function.layers)):
                mlp_in_2 = detach_tensor(saved_tensors.pop())
                mlp_out_2, mlp_bias_2 = l.mlp(mlp_in_2)
                if idx > 0:
                    handler.wait()
                handler = dist.all_reduce(mlp_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                residual_2 = detach_tensor(saved_tensors.pop())

                if idx > 0:
                    torch.autograd.backward((layernorm_out_1, ), 
                        (detached_ln_out1.grad, ))

                mlp_in_1 = detach_tensor(saved_tensors.pop())
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)
                handler.wait()
                handler = dist.all_reduce(mlp_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                residual_1 = detach_tensor(saved_tensors.pop())
                output_2 = l.post_mlp(mlp_out_2, mlp_bias_2, residual_2)
            
                # grad for mlp_in_2, residual_2
                torch.autograd.backward(output_2, grad_2)

                handler.wait()
                handler = dist.all_reduce(mlp_in_2.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                output_1 = l.post_mlp(mlp_out_1, mlp_bias_1, residual_1)

                torch.autograd.backward(output_1, grad_1)

                handler.wait()
                handler = dist.all_reduce(mlp_in_1.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                ln_in_2 = detach_tensor(saved_tensors.pop())
                layernorm_out_2 = l.input_layernorm(ln_in_2)

                detached_ln_out2 = detach_tensor(layernorm_out_2)
                attn_out_2, attn_bias_2 = l.self_attention(detached_ln_out2)

                handler.wait()
                handler = dist.all_reduce(attn_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                ln_in_1 = detach_tensor(saved_tensors.pop())
                layernorm_out_1 = l.input_layernorm(ln_in_1)

                detached_ln_out1 = detach_tensor(layernorm_out_1)
                attn_out_1, attn_bias_1 = l.self_attention(detached_ln_out1)

                handler.wait()
                handler = dist.all_reduce(attn_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                mlp_in_recom_2, residual_recom_2 = l.post_attn(attn_out_2, attn_bias_2, ln_in_2)
            
                # grad for detached_ln_out2, ln_in_2
                torch.autograd.backward((mlp_in_recom_2, residual_recom_2), 
                        (mlp_in_2.grad, residual_2.grad))

                handler.wait()
                handler = dist.all_reduce(detached_ln_out2.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                mlp_in_recom_1, residual_recom_1 = l.post_attn(attn_out_1, attn_bias_1, ln_in_1)

                torch.autograd.backward((mlp_in_recom_1, residual_recom_1), 
                        (mlp_in_1.grad, residual_1.grad))

                handler.wait()
                handler = dist.all_reduce(detached_ln_out1.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                # grad for ln_in_2
                torch.autograd.backward((layernorm_out_2, ), 
                        (detached_ln_out2.grad, ))

        handler.wait()
        torch.autograd.backward((layernorm_out_1, ), 
            (detached_ln_out1.grad, ))

        return (None, ln_in_1.grad, ln_in_2.grad)


def overlap_tmp_checkpoint(run_function, *args, **kwargs):
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return OverlapTMPCheckpointFunction.apply(run_function, *args)


class CommOptimizedCheckpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, *args):

        check_backward_validity(args)
        # ctx.inputs = args
        ctx.run_function = run_function

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        tensor_inputs = []
        func_inputs = []

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                func_inputs.append(arg)

        hidden_stats = args[0]
        with torch.no_grad():

            for idx, l in enumerate(run_function.layers):
                # Layer norm at the beginning of the transformer layer.
                layernorm_output = l.input_layernorm(hidden_stats)
                # backward allreduce
                attn_input = mpu.mappings.copy_to_model_parallel_region(layernorm_output)
                # Self attention.
                attention_output_parallel, attention_bias = l.self_attention(attn_input)

                dist.all_reduce(attention_output_parallel, group=mpu.get_model_parallel_group())
                tensor_inputs.append((attention_output_parallel, hidden_stats))
                
                layernorm_output, residual = l.post_attn(attention_output_parallel, attention_bias, hidden_stats)
                
                # backward allreduce
                mlp_input = mpu.mappings.copy_to_model_parallel_region(layernorm_output)
                # MLP.
                mlp_output_parallel, mlp_bias = l.mlp(mlp_input)
                
                tensor_inputs.append((mlp_output_parallel, residual))
                dist.all_reduce(mlp_output_parallel, group=mpu.get_model_parallel_group())

                hidden_stats = l.post_mlp(mlp_output_parallel, mlp_bias, residual)

        ctx.save_for_backward(*func_inputs)
        ctx.tensor_for_backward = tensor_inputs
        return hidden_stats


    @staticmethod
    def backward(ctx, *args):

        saved_tensors = ctx.tensor_for_backward
        hidden_grad = args[0]

        with torch.enable_grad():
            for idx, l in enumerate(reversed(ctx.run_function.layers)):
            

                if idx == 0:
                    detach_mlp_output_parallel, detach_residual = detach_variable(saved_tensors.pop())
                    hidden_stats = l.post_mlp(detach_mlp_output_parallel, l.mlp.dense_4h_to_h.bias, detach_residual)
                    torch.autograd.backward((hidden_stats, ), (hidden_grad, ))
                else:
                    detach_mlp_output_parallel, detach_residual = detach_variable(saved_tensors.pop())
                    hidden_stats = l.post_mlp(detach_mlp_output_parallel, l.mlp.dense_4h_to_h.bias, detach_residual)

                    layernorm_output = last_layer.input_layernorm(hidden_stats)
                    # backward allreduce
                    attn_input = mpu.mappings.copy_to_model_parallel_region(layernorm_output)
                    # Self attention.
                    attention_output_parallel, attention_bias = last_layer.self_attention(attn_input)
                    
                    # grad for detach_mlp_output_parallel, detach_residual
                    torch.autograd.backward((attention_output_parallel, hidden_stats), 
                            (detach_attention_output_parallel.grad, detached_hidden_stats.grad))

                detach_attention_output_parallel, detached_hidden_stats = detach_variable(saved_tensors.pop())
                post_attn_ln, residual = l.post_attn(detach_attention_output_parallel, l.self_attention.dense.bias, detached_hidden_stats)
                mlp_input = mpu.mappings.copy_to_model_parallel_region(post_attn_ln)
                mlp_output_parallel, mlp_bias = l.mlp(mlp_input)

                # grad for detach_attention_output_parallel, detached_hidden_stats
                torch.autograd.backward((mlp_output_parallel, residual), 
                        (detach_mlp_output_parallel.grad, detach_residual.grad))

                last_layer = l 
            
            detached_input = detach_tensor(ctx.saved_tensors[0])
            layernorm_output = last_layer.input_layernorm(detached_input)
            # backward allreduce
            attn_input = mpu.mappings.copy_to_model_parallel_region(layernorm_output)
            # Self attention.
            attention_output_parallel, attention_bias = last_layer.self_attention(attn_input)
            torch.autograd.backward((attention_output_parallel, ), 
                    (detach_attention_output_parallel.grad, ))

        return (None, detached_input.grad)


def comm_optimized_checkpoint(run_function, *args, **kwargs):
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return CommOptimizedCheckpoint.apply(run_function, *args)

class CommOptimizedOverlapCheckpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, cutpoint, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.cutpoint = len(run_function.layers)-cutpoint if cutpoint is not None else None
    
        if cutpoint is not None and cutpoint > 0:
            new_mp_degree, new_dp_degree, trans_group = mpu.initialize.get_degree_trans_group()
            trans_group_size, trans_group_rank = dist.get_world_size(trans_group), dist.get_rank(trans_group)
            ctx.split_setting = (new_mp_degree, new_dp_degree, trans_group_size, trans_group_rank)
            model_group_bak, data_group_bak = \
                mpu.initialize.get_model_parallel_group(), mpu.initialize.get_data_parallel_group()
            mp_group, dp_group = mpu.initialize.get_group_from_degree(new_mp_degree, new_dp_degree)
            mpu.initialize.set_model_parallel_group(mp_group)
            mpu.initialize.set_data_parallel_group(dp_group)

        input_1, input_2 = args

        # ctx.save_for_backward(*args)

        to_saved = []
        to_saved.append((input_1, input_2))

        with torch.no_grad():
            for idx, l in enumerate(run_function.layers):
                if cutpoint is not None and idx == cutpoint:
                    mpu.initialize.set_model_parallel_group(model_group_bak)
                    mpu.initialize.set_data_parallel_group(data_group_bak)
                    hidden_size = list(input_1.size())
                    hidden_size[1] = hidden_size[1]*dist.get_world_size(group=trans_group)
                    
                    gathered = torch.empty(hidden_size, device=torch.cuda.current_device())
                    dist._all_gather_base(gathered, input_1, group=trans_group)
                    input_1 = gathered


                layernorm_out_1 = l.input_layernorm(input_1)
                # Self attention.
                attn_out_1, attn_bias_1 = l.self_attention(layernorm_out_1)

                if idx > 0:
                    handler.wait()
                handler = dist.all_reduce(attn_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)
                
                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                if idx > 0:
                    to_saved.append((mlp_out_2, residual_2))
                    input_2 = last_layer.post_mlp(mlp_out_2, mlp_bias_2, residual_2)

                    if cutpoint is not None and idx == cutpoint:
                        gathered = torch.empty(hidden_size, device=torch.cuda.current_device())
                        dist._all_gather_base(gathered, input_2, group=trans_group)
                        input_2 = gathered
                
                layernorm_out_2 = l.input_layernorm(input_2)
                # Self attention.
                attn_out_2, attn_bias_2 = l.self_attention(layernorm_out_2)

                handler.wait()
                handler = dist.all_reduce(attn_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)
                
                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                to_saved.append((attn_out_1, input_1))
                mlp_in_1, residual_1 = l.post_attn(attn_out_1, attn_bias_1, input_1)

                # MLP.
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)

                handler.wait()
                handler = dist.all_reduce(mlp_out_1, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                to_saved.append((attn_out_2, input_2))
                mlp_in_2, residual_2 = l.post_attn(attn_out_2, attn_bias_2, input_2)


                # MLP.
                mlp_out_2, mlp_bias_2 = l.mlp(mlp_in_2)

                handler.wait()
                handler = dist.all_reduce(mlp_out_2, 
                    group=mpu.get_model_parallel_group(), async_op=True)

                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                to_saved.append((mlp_out_1, residual_1))
                input_1 = l.post_mlp(mlp_out_1, mlp_bias_1, residual_1)
                last_layer = l


            handler.wait()
            to_saved.append((mlp_out_2, residual_2))
            input_2 = last_layer.post_mlp(mlp_out_2, mlp_bias_2, residual_2)

        ctx.tensor_for_backward = to_saved

        return input_1, input_2


    @staticmethod
    def backward(ctx, *args):
        saved_tensors = ctx.tensor_for_backward
        grad_1, grad_2 = args


        with torch.enable_grad():
            for idx, l in enumerate(reversed(ctx.run_function.layers)):

                detach_mlp_out_2, detach_residual_2 = detach_variable(saved_tensors.pop())
                # print(idx, detach_mlp_out_2.shape, detach_residual_2.shape)
                hidden_stats_2 = l.post_mlp(detach_mlp_out_2, l.mlp.dense_4h_to_h.bias, detach_residual_2)


                if idx == 0:
                    torch.autograd.backward((hidden_stats_2, ), (grad_2, ))
                else:

                    if ctx.cutpoint is not None and idx == ctx.cutpoint:
                        ln_out_2 = last_layer.input_layernorm(detached_hidden_stats_2)
                    else:
                        ln_out_2 = last_layer.input_layernorm(hidden_stats_2)
                    detached_ln_out2 = detach_tensor(ln_out_2)

                    attn_out_2, attn_bias_2 = last_layer.self_attention(detached_ln_out2)

                    # grad for detached_ln_out2
                    torch.autograd.backward((attn_out_2, ), 
                           (detach_attn_out_2.grad, ))
                    handler.wait()


                    handler = dist.all_reduce(detached_ln_out2.grad, 
                        group=mpu.get_model_parallel_group(), async_op=True)

                    # Delay the start of intput gradient computation shortly (3us) to have
                    # gather scheduled first and have GPU resources allocated
                    _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                    torch.autograd.backward((post_attn_ln_1, residual_1), 
                            (mlp_in_1.grad, detach_residual_1.grad))

                    detach_mlp_out_1, detach_residual_1 = detach_variable(saved_tensors.pop())
                    hidden_stats_1 = l.post_mlp(detach_mlp_out_1, l.mlp.dense_4h_to_h.bias, detach_residual_1)
                    
                    if ctx.cutpoint is not None and idx == ctx.cutpoint:
                        ln_out_1 = last_layer.input_layernorm(detached_hidden_stats_1)
                    else:
                        ln_out_1 = last_layer.input_layernorm(hidden_stats_1)

                    detached_ln_out1 = detach_tensor(ln_out_1)


                    attn_out_1, attn_bias_1 = last_layer.self_attention(detached_ln_out1)

                    torch.autograd.backward((attn_out_1, ), 
                            (detach_attn_out_1.grad, ))

                    handler.wait()
                    handler = dist.all_reduce(detached_ln_out1.grad, 
                        group=mpu.get_model_parallel_group(), async_op=True)

                    # Delay the start of intput gradient computation shortly (3us) to have
                    # gather scheduled first and have GPU resources allocated
                    _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                    # grad for detach_mlp_out_2, detach_residual_2
                    # print(detached_hidden_stats_2.grad.shape)
                    if ctx.cutpoint is not None and idx == ctx.cutpoint:
                        new_mp_degree, new_dp_degree, trans_group_size, trans_group_rank = ctx.split_setting

                        torch.autograd.backward((ln_out_2, ), 
                            (detached_ln_out2.grad, ))
                        detached_hidden_stats_2.grad.data = detached_hidden_stats_2.grad.data.chunk(trans_group_size, dim=1)[trans_group_rank]

                        torch.autograd.backward((hidden_stats_2, ), (detached_hidden_stats_2.grad,))
                    else:
                        torch.autograd.backward((ln_out_2, hidden_stats_2), 
                            (detached_ln_out2.grad, detached_hidden_stats_2.grad))

                            
                if ctx.cutpoint is not None and idx == ctx.cutpoint:
                    model_group_bak, data_group_bak = \
                        mpu.initialize.get_model_parallel_group(), mpu.initialize.get_data_parallel_group()
                    mp_group, dp_group = mpu.initialize.get_group_from_degree(new_mp_degree, new_dp_degree)
                    mpu.initialize.set_model_parallel_group(mp_group)
                    mpu.initialize.set_data_parallel_group(dp_group)
                
                detach_attn_out_2, detached_hidden_stats_2 = detach_variable(saved_tensors.pop())
                post_attn_ln_2, residual_2= l.post_attn(detach_attn_out_2, l.self_attention.dense.bias, detached_hidden_stats_2)
                
                mlp_in_2 = detach_tensor(post_attn_ln_2)
                mlp_out_2, mlp_bias_2 = l.mlp(mlp_in_2)

                # grad for mlp_in_2
                torch.autograd.backward((mlp_out_2, ), 
                        (detach_mlp_out_2.grad, ))

                if idx > 0:
                    handler.wait()
                handler = dist.all_reduce(mlp_in_2.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)  

                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                if idx == 0:
                    detach_mlp_out_1, detach_residual_1 = detach_variable(saved_tensors.pop())
                    hidden_stats_1 = l.post_mlp(detach_mlp_out_1, l.mlp.dense_4h_to_h.bias, detach_residual_1)
                    torch.autograd.backward((hidden_stats_1, ), 
                        (grad_1, ))
                else:
                    if ctx.cutpoint is not None and idx == ctx.cutpoint:
                        torch.autograd.backward((ln_out_1, ), 
                            (detached_ln_out1.grad, ))
                        detached_hidden_stats_1.grad.data = detached_hidden_stats_1.grad.data.chunk(trans_group_size, dim=1)[trans_group_rank]

                        torch.autograd.backward((hidden_stats_1, ), (detached_hidden_stats_1.grad,))
                    else:

                        torch.autograd.backward((ln_out_1, hidden_stats_1), 
                            (detached_ln_out1.grad, detached_hidden_stats_1.grad))
                
                detach_attn_out_1, detached_hidden_stats_1 = detach_variable(saved_tensors.pop())
                post_attn_ln_1, residual_1= l.post_attn(detach_attn_out_1, l.self_attention.dense.bias, detached_hidden_stats_1)
                mlp_out_1, mlp_bias_1 = l.mlp(post_attn_ln_1)

                mlp_in_1 = detach_tensor(post_attn_ln_1)
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)

                torch.autograd.backward((mlp_out_1, ), 
                        (detach_mlp_out_1.grad, ))

                handler.wait()
                handler = dist.all_reduce(mlp_in_1.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)  

                # grad for detach_attn_out_2, detached_hidden_stats_2
                torch.autograd.backward((post_attn_ln_2, residual_2), 
                        (mlp_in_2.grad, detach_residual_2.grad))
                
                last_layer = l 

            detached_input_1, detached_input_2 = detach_variable(saved_tensors.pop())

            ln_out_2 = last_layer.input_layernorm(detached_input_2)

            detached_ln_out2 = detach_tensor(ln_out_2)
            attn_out_2, attn_bias_2 = last_layer.self_attention(detached_ln_out2)

            torch.autograd.backward((attn_out_2, ), 
                (detach_attn_out_2.grad, ))
            
            handler.wait()
            handler = dist.all_reduce(detached_ln_out2.grad, 
                group=mpu.get_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=torch.cuda.current_device()) + 1
            
            torch.autograd.backward((post_attn_ln_1, residual_1), 
                    (mlp_in_1.grad, detach_residual_1.grad))

            ln_out_1 = last_layer.input_layernorm(detached_input_1)

            detached_ln_out1 = detach_tensor(ln_out_1)
            attn_out_1, attn_bias_1 = last_layer.self_attention(detached_ln_out1)

            torch.autograd.backward((attn_out_1, ), 
                    (detach_attn_out_1.grad, ))

            handler.wait()
            handler = dist.all_reduce(detached_ln_out1.grad, 
                group=mpu.get_model_parallel_group(), async_op=True)

            
            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=torch.cuda.current_device()) + 1

            torch.autograd.backward((ln_out_2, ), 
                    (detached_ln_out2.grad, ))

            handler.wait()
            torch.autograd.backward((ln_out_1, ), 
                    (detached_ln_out1.grad, ))

            if ctx.cutpoint is not None:
                mpu.initialize.set_model_parallel_group(model_group_bak)
                mpu.initialize.set_data_parallel_group(data_group_bak)

        return (None, None, detached_input_1.grad, detached_input_2.grad)

def comm_optimized_overlap_checkpoint(run_function, cutpoint, *args, **kwargs):
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return CommOptimizedOverlapCheckpoint.apply(run_function, cutpoint, *args)





from ..mpu.model_trans import _memory_buffer, get_size
from ..mpu.tmplayers import ColumnParallelLinear, RowParallelLinear


class DPTPwithOptimizedOverlapCheckpoint(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function


        for n,p in run_function.named_modules():
            if isinstance(p, ColumnParallelLinear):
                # print(getattr(p, 'weight').grad.shape)
                tensor_list = [torch.empty_like(p.weight.data) for _ in range(mpu.get_model_parallel_world_size())]
                tensor_list[mpu.get_model_parallel_rank()] = p.weight.data
                torch.distributed.all_gather(tensor_list, p.weight.data, group=mpu.get_model_parallel_group())
                gather_w = torch.cat(tensor_list, dim=0).contiguous()
                p.weight.data = gather_w
                # p.weight.data = torch.cat(weight_map[n], dim=0).contiguous()
                # print('return', p.weight.shape)
                if hasattr(p, 'bias'):
                    tensor_list = [torch.empty_like(p.bias.data) for _ in range(mpu.get_model_parallel_world_size())]
                    tensor_list[mpu.get_model_parallel_rank()] = p.bias.data
                    torch.distributed.all_gather(tensor_list, p.bias.data, group=mpu.get_model_parallel_group())
                    gather_b_grad = torch.cat(tensor_list, dim=0).contiguous()
                    p.bias.data = gather_b_grad
                    # print('return', p.bias.shape)
            elif isinstance(p, RowParallelLinear):
                # print(getattr(p, 'weight').grad.shape)
                tensor_list = [torch.empty_like(p.weight.data) for _ in range(mpu.get_model_parallel_world_size())]
                tensor_list[mpu.get_model_parallel_rank()] = p.weight.data
                torch.distributed.all_gather(tensor_list, p.weight.data, group=mpu.get_model_parallel_group())
                gather_w_grad = torch.cat(tensor_list, dim=1).contiguous()
                p.weight.data = gather_w_grad


        input_1 = args[0].chunk(mpu.get_model_parallel_world_size(), dim=0)[mpu.get_model_parallel_rank()]

        # ctx.save_for_backward(*args)
        data_group = mpu.get_data_parallel_group()
        model_group = mpu.get_model_parallel_group()
        mpu.initialize.set_data_parallel_group(model_group)
        mpu.initialize.set_model_parallel_group(data_group)

        to_saved = []
      
        # to_saved.append((input_1, input_2))

        with torch.no_grad():
            for idx, l in enumerate(run_function.layers):
                allgather_buffer = _memory_buffer.get_tensor(get_size(input_1.size()), input_1.dtype, 'mpu')
                handler = torch.distributed._all_gather_base(allgather_buffer, input_1, group=mpu.get_data_parallel_group(), async_op=True)
                layernorm_out_1 = l.input_layernorm(input_1)
                # Self attention.
                attn_out_1, attn_bias_1 = l.self_attention(layernorm_out_1)

                handler.wait()
                if idx == 0:
                    to_saved.append(allgather_buffer.chunk(2, dim=0))
                saved_input_1, input_2 = allgather_buffer.chunk(2, dim=0)
                
                allgather_buffer = _memory_buffer.get_tensor(get_size(attn_out_1.size()), attn_out_1.dtype, 'mpu')
                handler = torch.distributed._all_gather_base(allgather_buffer, attn_out_1, group=mpu.get_data_parallel_group(), async_op=True)
                saved_attn_out_1, attn_out_2 = allgather_buffer.chunk(2, dim=0)

                to_saved.append((saved_attn_out_1, saved_input_1))
                mlp_in_1, residual_1 = l.post_attn(attn_out_1, attn_bias_1, input_1)


                allgather_buffer_2 = _memory_buffer.get_tensor(get_size(residual_1.size()), residual_1.dtype, 'mpu_')
                handler_2 = torch.distributed._all_gather_base(allgather_buffer, residual_1, group=mpu.get_data_parallel_group(), async_op=True)

                # MLP.
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)

                allgather_buffer = _memory_buffer.get_tensor(get_size(mlp_out_1.size()), mlp_out_1.dtype, 'mpu')
                handler_1 = torch.distributed._all_gather_base(allgather_buffer, mlp_out_1, group=mpu.get_data_parallel_group(), async_op=True)


                to_saved.append((attn_out_2, input_2))


                input_1 = l.post_mlp(mlp_out_1, mlp_bias_1, residual_1)
                handler_2.wait()
                residual_1, residual_2 = allgather_buffer_2.chunk(2, dim=0)
                handler_1.wait()
                mlp_out_1, mlp_out_2 = allgather_buffer.chunk(2, dim=0)

                to_saved.append((mlp_out_1, residual_1))
                to_saved.append((mlp_out_2, residual_2))

        ctx.tensor_for_backward = to_saved

        data_group = mpu.get_data_parallel_group()
        model_group = mpu.get_model_parallel_group()
        mpu.initialize.set_data_parallel_group(model_group)
        mpu.initialize.set_model_parallel_group(data_group)

        for n,p in run_function.named_modules():
            if isinstance(p, ColumnParallelLinear):
                p.weight.data = p.weight.data.chunk(mpu.get_model_parallel_world_size(), dim=0)[mpu.get_model_parallel_rank()].contiguous()
                p.weight.grad = None
                # print('changeto', p.weight.shape)
                if hasattr(p, 'bias'):

                    p.bias.data = p.bias.data.chunk(mpu.get_model_parallel_world_size())[mpu.get_model_parallel_rank()].contiguous()
                    p.bias.grad = None
                    # print('changeto', p.bias.shape)
            elif isinstance(p, RowParallelLinear):

                p.weight.data = p.weight.data.chunk(mpu.get_model_parallel_world_size(), dim=1)[mpu.get_model_parallel_rank()].contiguous()
                p.weight.grad = None
        
        return input_1


    @staticmethod
    def backward(ctx, *args):
        saved_tensors = ctx.tensor_for_backward
        input_grad = args[0]
        tensor_list = [torch.empty_like(input_grad) for _ in range(mpu.get_model_parallel_world_size())]
        tensor_list[mpu.get_model_parallel_rank()] = input_grad
        torch.distributed.all_gather(tensor_list, input_grad, group=mpu.get_model_parallel_group())
        grad_1, grad_2 = torch.cat(tensor_list, dim=0).chunk(2, dim=0)

        with torch.enable_grad():
            for idx, l in enumerate(reversed(ctx.run_function.layers)):

                detach_mlp_out_2, detach_residual_2 = detach_variable(saved_tensors.pop())
                hidden_stats_2 = l.post_mlp(detach_mlp_out_2, l.mlp.dense_4h_to_h.bias, detach_residual_2)
                if idx == 0:
                    torch.autograd.backward((hidden_stats_2, ), (grad_2, ))
                    
                else:
                    ln_out_2 = last_layer.input_layernorm(hidden_stats_2)

                    detached_ln_out2 = detach_tensor(ln_out_2)
                    attn_out_2, attn_bias_2 = last_layer.self_attention(detached_ln_out2)

                    # grad for detached_ln_out2
                    torch.autograd.backward((attn_out_2, ), 
                           (detach_attn_out_2.grad, ))
                    
                    handler.wait()
                    handler = dist.all_reduce(detached_ln_out2.grad, 
                        group=mpu.get_model_parallel_group(), async_op=True)

                    # Delay the start of intput gradient computation shortly (3us) to have
                    # gather scheduled first and have GPU resources allocated
                    _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                    torch.autograd.backward((post_attn_ln_1, residual_1), 
                            (mlp_in_1.grad, detach_residual_1.grad))

                    detach_mlp_out_1, detach_residual_1 = detach_variable(saved_tensors.pop())
                    hidden_stats_1 = l.post_mlp(detach_mlp_out_1, l.mlp.dense_4h_to_h.bias, detach_residual_1)
                    
                    ln_out_1 = last_layer.input_layernorm(hidden_stats_1)

                    detached_ln_out1 = detach_tensor(ln_out_1)
                    attn_out_1, attn_bias_1 = last_layer.self_attention(detached_ln_out1)

                    torch.autograd.backward((attn_out_1, ), 
                            (detach_attn_out_1.grad, ))

                    handler.wait()
                    handler = dist.all_reduce(detached_ln_out1.grad, 
                        group=mpu.get_model_parallel_group(), async_op=True)

                    # Delay the start of intput gradient computation shortly (3us) to have
                    # gather scheduled first and have GPU resources allocated
                    _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                    # grad for detach_mlp_out_2, detach_residual_2
                    torch.autograd.backward((ln_out_2, ), 
                            (detached_ln_out2.grad, ))

                
                detach_attn_out_2, detached_hidden_stats_2 = detach_variable(saved_tensors.pop())
                post_attn_ln_2, residual_2= l.post_attn(detach_attn_out_2, l.self_attention.dense.bias, detached_hidden_stats_2)
                
                mlp_in_2 = detach_tensor(post_attn_ln_2)
                mlp_out_2, mlp_bias_2 = l.mlp(mlp_in_2)

                # grad for mlp_in_2
                torch.autograd.backward((mlp_out_2, ), 
                        (detach_mlp_out_2.grad, ))

                if idx > 0:
                    handler.wait()
                handler = dist.all_reduce(mlp_in_2.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)  

                # Delay the start of intput gradient computation shortly (3us) to have
                # gather scheduled first and have GPU resources allocated
                _ = torch.empty(1, device=torch.cuda.current_device()) + 1

                if idx == 0:
                    detach_mlp_out_1, detach_residual_1 = detach_variable(saved_tensors.pop())
                    hidden_stats_1 = l.post_mlp(detach_mlp_out_1, l.mlp.dense_4h_to_h.bias, detach_residual_1)
                    torch.autograd.backward((hidden_stats_1, ), 
                        (grad_1, ))
                else:
                    torch.autograd.backward((ln_out_1, ), 
                        (detached_ln_out1.grad, ))
                
                detach_attn_out_1, detached_hidden_stats_1 = detach_variable(saved_tensors.pop())
                post_attn_ln_1, residual_1= l.post_attn(detach_attn_out_1, l.self_attention.dense.bias, detached_hidden_stats_1)
                mlp_out_1, mlp_bias_1 = l.mlp(post_attn_ln_1)

                mlp_in_1 = detach_tensor(post_attn_ln_1)
                mlp_out_1, mlp_bias_1 = l.mlp(mlp_in_1)

                torch.autograd.backward((mlp_out_1, ), 
                        (detach_mlp_out_1.grad, ))

                handler.wait()
                handler = dist.all_reduce(mlp_in_1.grad, 
                    group=mpu.get_model_parallel_group(), async_op=True)  

                # grad for detach_attn_out_2, detached_hidden_stats_2
                torch.autograd.backward((post_attn_ln_2, residual_2), 
                        (mlp_in_2.grad, detach_residual_2.grad))
                

                last_layer = l 

            detached_input_1, detached_input_2 = detach_variable(saved_tensors.pop())
            # detached_input_1, detached_input_2 = detach_variable(ctx.saved_tensors)

            ln_out_2 = last_layer.input_layernorm(detached_input_2)

            detached_ln_out2 = detach_tensor(ln_out_2)
            attn_out_2, attn_bias_2 = last_layer.self_attention(detached_ln_out2)

            torch.autograd.backward((attn_out_2, ), 
                (detach_attn_out_2.grad, ))
            
            handler.wait()
            handler = dist.all_reduce(detached_ln_out2.grad, 
                group=mpu.get_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=torch.cuda.current_device()) + 1
            
            torch.autograd.backward((post_attn_ln_1, residual_1), 
                    (mlp_in_1.grad, detach_residual_1.grad))

            ln_out_1 = last_layer.input_layernorm(detached_input_1)

            detached_ln_out1 = detach_tensor(ln_out_1)
            attn_out_1, attn_bias_1 = last_layer.self_attention(detached_ln_out1)

            torch.autograd.backward((attn_out_1, ), 
                    (detach_attn_out_1.grad, ))

            handler.wait()
            handler = dist.all_reduce(detached_ln_out1.grad, 
                group=mpu.get_model_parallel_group(), async_op=True)

            
            # Delay the start of intput gradient computation shortly (3us) to have
            # gather scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=torch.cuda.current_device()) + 1

            torch.autograd.backward((ln_out_2, ), 
                    (detached_ln_out2.grad, ))

            handler.wait()
            torch.autograd.backward((ln_out_1, ), 
                    (detached_ln_out1.grad, ))
            input_grad = torch.cat([detached_input_1.grad, detached_input_2.grad], dim=0)
    

        return (None, input_grad)

def DPTP_optimized_overlap_checkpoint(run_function, *args, **kwargs):
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    return DPTPwithOptimizedOverlapCheckpoint.apply(run_function, *args)
    