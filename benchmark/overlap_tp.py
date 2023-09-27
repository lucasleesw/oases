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

import os
import time
import oases.mpu as mpu
import oases
from oases.utils import see_memory_usage, print_rank_0
import torch
import torch.distributed as dist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--hidden_size", type=int, default=2048)
parser.add_argument("--num_layer", type=int, default=24)
parser.add_argument("--schedule_option", type=str, default='activation_checkpoint')
parser.add_argument("--auto_plan", action='store_true')
parser.add_argument("--gas", type=int, default=20)
parser.add_argument("--profile_path", type=str, default='./profile_res.pt')
parser.add_argument("--torch_profiler", action='store_true')
parser.add_argument("--torch_prof_path", type=str, default=None)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

if args.auto_plan:
    assert args.schedule_option == 'oases'
    tp, dp = oases.get_plan(args=args, profile_path=args.profile_path)
else:
    tp = int(os.environ['TP'])
    dp = int(os.environ['DP'])


if args.torch_profiler:
    assert args.torch_prof_path is not None
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=10, warmup=10, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.torch_prof_path),
        with_stack=True)
    prof.start()
else:    
    prof = None 

model = oases.BenchmarkLM(schedule=args.schedule_option,num_l=args.num_layer, h=args.hidden_size, n=args.hidden_size//64, tp=tp, dp=dp, auto_plan=args.auto_plan)
model.cuda()

print_rank_0(model)

optim = torch.optim.Adam(model.parameters(), lr=0.1)
optim.zero_grad()

loss_fn = torch.nn.CrossEntropyLoss()
model.train()

step = 100

step_time = []

for _ in range(step):
    
    input = model.gen_rand_inputs(args.batch_size, args.hidden_size)
    input.requires_grad = True
    label = torch.randint(100, (args.batch_size, args.hidden_size), device=device).long().detach()

    if prof is None:
        torch.cuda.synchronize()
        start_time = time.time()
        
    x = model(input)
    loss = loss_fn(x.transpose(0, 1).contiguous(), label)
    
    loss.backward()

    if prof is None:
        torch.cuda.synchronize()
        calc_time = time.time()
    else:
        torch.cuda.synchronize()
        prof.step()


    if (_+1) % args.gas == 0:
        model.reduce_grad()
        optim.step()
        optim.zero_grad()

    if prof is None:
        torch.cuda.synchronize()
        step_time.append(time.time()-start_time)
        print_rank_0(f'STEP {_}: step time {step_time[-1]}')
    else:
        if _ >25 :
            break

# see_memory_usage('Finished', True)

if prof is None:
    print_rank_0('average time: ')
    print_rank_0(sum(step_time[-20:])/len(step_time[-20:]))
    print_rank_0(f'Result: avg {sum(step_time[-20:])/len(step_time[-20:])}, min {min(step_time[-20:])}, max {max(step_time[-20:])}')
else:
    prof.stop()


