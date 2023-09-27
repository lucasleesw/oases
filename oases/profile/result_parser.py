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



import torch
import argparse
import math
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type=str, default='./profile.out')
parser.add_argument("--save_name", type=str, default='./profile_res.pt')

parser.add_argument("--max_bs", type=int, default=128)
parser.add_argument("--max_hidden", type=int, default=None)

args = parser.parse_args()

class CounterDict(defaultdict):
    def __init__(self):
        super().__init__(int)
    
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = len(self)
        return self[key]

f = open(args.log_path, mode='r')

hiddens = [1024,2048,3072,4096,6144,8192,12288]
s_hidden = len(hiddens)
if args.max_hidden is not None:
    s_hidden = args.max_hidden//1024
    hiddens = [1024*(i+1) for i in range(s_hidden)]
hidden_mapping = CounterDict()

bs_len = int(math.log(args.max_bs, 2))+1
bs_list = [2**(i) for i in range(bs_len)]
bs_mapping = CounterDict()

line = f.readline()
node_list = line.split()
num_gpu=4*len(node_list)

s_len = int(math.log(num_gpu, 2)) + 1
tp_strategies = [2**(i) for i in range(s_len)]
mp_mapping = CounterDict()

cost = torch.empty(s_hidden, s_len, bs_len, 6)
to_saved = {}

op_idx=CounterDict()


while line:
    if line.startswith('using'):
        res_list = line.split()
        hidden_size = int(res_list[-1])
        tp_degree = int(res_list[2])
        batch_size = int(res_list[6])
        hidden_idx = hidden_mapping[hidden_size]
        mp_degree_idx = mp_mapping[tp_degree]
        bs_idx = bs_mapping[batch_size]
        for i in range(cost.size(-1)):
            cost[hidden_idx][mp_degree_idx][bs_idx][i] = -1.0
    elif line.startswith('RuntimeError'):
        for i in range(cost.size(-1)):
            cost[hidden_idx][mp_degree_idx][bs_idx][i] = -1.0
        while line.startswith('RuntimeError'):
            line=f.readline()
    elif line.startswith('Rank'):
        res_list = line.split()
        op_name = res_list[-2]
        overhead = float(res_list[-1])
        cost[hidden_idx][mp_degree_idx][bs_idx][op_idx[op_name]] = overhead
    line = f.readline()


to_saved['cost'] = cost
to_saved['op_mapping'] = dict(op_idx)
to_saved['hidden_mapping'] = dict(hidden_mapping)
to_saved['mp_mapping'] = dict(mp_mapping)
to_saved['bs_mapping'] = dict(bs_mapping)

torch.save(to_saved, args.save_name)

f.close()
