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

# Parts of the code here are adapted from https://github.com/alpa-projects/alpa/blob/5660516ad3a29e5760673e599fc84aa604589a82/alpa/shard_parallel/auto_sharding.py

import torch
import torch.distributed as dist
import pulp
import time
import multiprocessing
from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus
import os
import math
from collections import Counter


CHANGED_TP = None
CHANGED_DP = None
CUTPOINT = None

def set_plan_args(tp, dp, cutpoint):
    global CHANGED_DP
    global CHANGED_TP
    global CUTPOINT
    CHANGED_DP = dp
    CHANGED_TP = tp
    CUTPOINT = cutpoint

def get_plan_args():
    global CHANGED_DP
    global CHANGED_TP
    global CUTPOINT
    return CHANGED_TP, CHANGED_DP, CUTPOINT


def get_profiling_result(args, num_gpu, profile_path):

    s_len = int(math.log(num_gpu, 2))
    all_strategies = [2**(i) for i in range(s_len+1)]

    profile_result = torch.load(profile_path)
    cost = profile_result['cost']
    op_mapping = profile_result['op_mapping']
    hidden_mapping = profile_result['hidden_mapping']
    mp_mapping = profile_result['mp_mapping'] 
    bs_mapping = profile_result['bs_mapping'] 

    model_costs = cost[hidden_mapping[args.hidden_size]]

    # filter the possible_strategies
    dp = int(os.environ['DP'])
    gbs = dp * args.batch_size

    possible_strategies = []

    for i in all_strategies:
        tp_bs = gbs//(num_gpu//i)
        if tp_bs > 1:
            if torch.all(model_costs[mp_mapping[i]][bs_mapping[tp_bs]] > 0):
                possible_strategies.append(i)

    assert len(possible_strategies) > 0, 'cannot train'


    num_layer = 2*args.num_layer

    tp_bs = [gbs*i//num_gpu for i in possible_strategies]
    # print(tp_bs)
    lp_comp = []
    lp_comm = []
    lp_bpcomp = []


    # model_costs[mp_degree_idx][bs_idx][op_idx]   
    for op_idx in range(num_layer):
        if op_idx % 2 == 0:
            op_name = 'attn'
        else:
            op_name = 'mlp'
        lp_comp.append([model_costs[mp_mapping[i]][bs_mapping[gbs*i//num_gpu//2]][op_mapping[op_name+'_fp']]  \
                            for i in possible_strategies])
        lp_bpcomp.append([model_costs[mp_mapping[i]][bs_mapping[gbs*i//num_gpu//2]][op_mapping[op_name+'_bp']]  \
                            for i in possible_strategies])
        lp_comm.append([model_costs[mp_mapping[i]][bs_mapping[gbs*i//num_gpu//2]][op_mapping['comm']]  \
                            for i in possible_strategies])

    lp_mem = [model_costs[mp_mapping[i]][bs_mapping[gbs*i//num_gpu]][op_mapping['runtime_memory']] * 1024 * 1024 \
                            for i in possible_strategies]
    # print(lp_comp)
    # print(lp_bpcomp)
    # print(lp_comm)
    
    return lp_comp, lp_comm, lp_bpcomp, lp_mem, possible_strategies, tp_bs



def memory_profiler(h, layer_idx, tp_degree, bs):
    s = 1024
    if layer_idx % 2 == 0:
        param_overhead = 4 * (4 * h * h / tp_degree + 6 * h)
    else:
        param_overhead = 4 * (8 * h * h / tp_degree + 7 * h)
    
    ac_overhead=12*bs*s*h

    return ac_overhead+5*param_overhead


# goal: given devices and batch size, get mp_degrees 
# for each comp op to minimize the iteration time.
# restrictions: 
# 1. the memory overhead is limited by device capacity.
# 2. should consider reshard overhead.
# 
def _get_plan(args=None, profile_path=None, mem_budget=None):
    start_time = time.time()

    if mem_budget is None:
        mem_budget = torch.cuda.get_device_properties(0).total_memory * 0.95
    
    num_gpu = torch.distributed.get_world_size()

    # read from profiling result
    lp_comp, lp_comm, lp_bpcomp, lp_mem, possible_strategies, tp_bs = \
                    get_profiling_result(args, num_gpu, profile_path)

    num_layer = 2*args.num_layer

    num_node = num_layer
    num_edge = num_layer - 1

    s_len = len(possible_strategies)

    # print(possible_strategies)
    # create variables
    s = []
    e = []

    for i in range(num_node):
        s.append(LpVariable.matrix(f"s[{i}]", \
                (range(s_len),), cat="Binary"))
    
    for i in range(num_edge):
        e.append(LpVariable.matrix(f"e[{i}]",
                                  (range(s_len ** 2),),
                                  cat="Binary"))

    lp_edge = []
    lp_overhead_1 = []
    lp_overhead_2 = []

    # fp overhead
    for i in range(num_node-1):
        lp_overhead_1.append([max(lp_comp[i][j], lp_comm[i][j]) for j in range(s_len)])
        lp_overhead_2.append([max(lp_comp[i+1][j], lp_comm[i][j]) for j in range(s_len)])
    # print(lp_overhead_1, lp_overhead_2)

    # bp_overhead
    lp_bp_1 = []
    lp_bp_2 = []
    for i in range(num_node-1):
        lp_bp_1.append([max(lp_bpcomp[i][j]+lp_comp[i][j], lp_comm[i][j]) for j in range(s_len)])
        lp_bp_2.append([max(lp_bpcomp[i+1][j]+lp_comp[i+1][j], lp_comm[i][j]) for j in range(s_len)])
    # print(lp_bp_1, lp_bp_2)
    # mem model overhead
    lp_mem_model = []
    for i in range(num_node):
        lp_mem_model.append([memory_profiler(args.hidden_size, \
                    i, possible_strategies[j], tp_bs[j]) for j in range(s_len)])

    # print(lp_mem)
    # edge_i is the edge between node model[edge_i] to model[edge_i+1]
    for edge_i in range(num_edge):
        ret = []
        for i in range(s_len):
            for j in range(s_len):
                if i == j:
                    ret.append(0)
                else:
                    # if forward allgather
                    if i < j:
                        comm_ag=possible_strategies[j] // possible_strategies[i]
                        ret.append(comm_ag*lp_comm[edge_i][i] \
                            +min(lp_comm[edge_i][i], lp_comp[edge_i+1][j]))
                    # if backward allgather
                    elif i > j:
                        comm_ag=possible_strategies[i] // possible_strategies[j]
                        ret.append(comm_ag*lp_comm[edge_i+1][j]  \
                            +min(lp_comm[edge_i][i], lp_bpcomp[edge_i+1][j]))
        lp_edge.append(ret)            

    prob = LpProblem("myProblem", LpMinimize)
    

    # forward cost
    obj = lpDot(s[0], lp_comp[0])

    for i in range(1, num_node-1):
        obj += lpDot(s[i], lp_overhead_1[i])
        obj += lpDot(s[i], lp_overhead_2[i])

    obj += lpDot(s[-1], lp_overhead_1[-1])
    obj += lpDot(s[-1], lp_comm[-1])

    # backward cost 

    obj += lpDot(s[0], lp_bpcomp[0])

    for i in range(1, num_node-1):
        obj += lpDot(s[i], lp_bp_1[i])
        obj += lpDot(s[i], lp_bp_2[i])
        
    obj += lpDot(s[-1], lp_bp_1[-1])
    obj += lpDot(s[-1], lp_comm[-1])

    # edge cost
    for i in range(num_edge):
        obj += lpDot(e[i], lp_edge[i])

    prob += obj

    # mem cost
    mem = 0
    for i in range(num_node):
        mem += lpDot(s[i], lp_mem_model[i])
    mem += lpDot(s[-1], lp_mem)
    prob += mem <= mem_budget

    # ensure the individual strategy choices
    for i in range(num_node):
        prob += lpSum(s[i]) == 1

    for idx in range(num_edge):
        prob += lpSum(e[idx]) == 1
        # ensure the strategy matches
        i = idx
        j = idx + 1
        for s_idx in range(s_len):
            i_row = s_idx
            j_col = s_idx
            prob += lpSum(
                e[idx][i_row * s_len + col] for col in range(s_len)) <= s[i][i_row]
            prob += lpSum(
                e[idx][row * s_len + j_col] for row in range(s_len)) <= s[j][j_col]

    verbose = False
    time_limit = 600
    solver = pulp.PULP_CBC_CMD(mip=True,
                                msg=verbose,
                                timeLimit=time_limit,
                                threads=multiprocessing.cpu_count())

    prob.solve(solver)
    status = prob.status
    # print(LpStatus[status])
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    print(f"ILP Status: {LpStatus[status]}\t", 
                f"Time: {time.time() - start_time}")
    print('obj', objective)

    def get_non_zero_index(binary_vector):
        """Get the index of non-zero item in a vector."""
        ct = 0
        ret = None
        for i, elem in enumerate(binary_vector):
            if pulp.value(elem):
                ret = i
                ct += 1

        assert ct == 1
        return ret

    res_s = []
    for i in range(num_node):
        ret = get_non_zero_index(s[i])
        if ret == -1:
            return None
        res_s.append(ret)
        # print(f'node {i} result of strategies is {ret}')
    
    for i in range(num_edge):
        ret = get_non_zero_index(e[i])
        # print(f'edge {i} result of strategies is {ret}')

    strategies = [possible_strategies[res_s[i]] for i in range(num_node)]
    return strategies
    

def get_plan(args=None, profile_path=None, mem_budget=None):
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    if dist.get_rank() == 0:
        try:
            strategies = _get_plan(args=args, profile_path=profile_path, mem_budget=mem_budget)
        except:
            dist.barrier()
            res = torch.tensor([0, 0, 0, 0, 0], device=torch.cuda.current_device())
            dist.broadcast(res, src=0)
            raise RuntimeError
        degrees = Counter(strategies)
        d = degrees.most_common(2)
        if len(d) < 2:
            tp = d[0][0]
            dp = world_size // tp
            changed_tp, changed_dp, cutpoint = 1,1,0
        else:
            tp = max(d[0][0], d[1][0])   
            dp = world_size // tp
            changed_tp = min(d[0][0], d[1][0])
            changed_dp = world_size // changed_tp
            cutpoint = sorted(d)[0][1] // 2 
        print(f'planning result, tp {tp}, dp {dp}, changed_tp {changed_tp}, '
                             f'changed_dp {changed_dp}, cutpoint {cutpoint}')
        res = torch.tensor([tp, dp, changed_tp, changed_dp, cutpoint], device=torch.cuda.current_device())
        dist.barrier()
        dist.broadcast(res, src=0)
    else:
        res = torch.empty(5, dtype=int, device=torch.cuda.current_device())
        dist.barrier()
        dist.broadcast(res, src=0)
        if torch.sum(res) < 1:
            raise RuntimeError
    tp, dp, changed_tp, changed_dp, cutpoint = list(res.cpu().numpy())
    set_plan_args(changed_tp, changed_dp, cutpoint if cutpoint > 0 else None)
    return tp, dp
