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

import logging
import sys
import torch
import torch.distributed as dist
import gc
from .mpu.topology import PipeModelDataParallelTopology, PipelineParallelGrid
from .mpu.initialize import _set_random_seed

def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


topo = None
communication_grid = None

def init_process(pp=1, tp=1, dp=1, backend='nccl'):
    """
    Initialized the distributed communication groups, include data parallel, 
    tensor model parallel and pipeline model parallel. Each parallel degree 
    has it own communication group, we can ge the rank or size through mpu API.

    Parameters:
    -   dp (int) -- Parallel degree of data parallelism.
    -   tp (int) -- Parallel degree of tensor model parallelism.
    -   pp (int) -- Parallel degree of pipeline model parallelism.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend)
    # we init topology and communication grid here
    global topo
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=tp, num_dp=dp)
    global communication_grid
    communication_grid = PipelineParallelGrid(topo, dist.new_group(ranks=range(dist.get_world_size())))
    

    # set mpu for transformers model
    from .mpu.initialize import set_data_parallel_group, set_model_parallel_group, set_pipe_parallel_group
    set_data_parallel_group(communication_grid.get_data_parallel_group())
    set_model_parallel_group(communication_grid.get_slice_parallel_group())
    set_pipe_parallel_group(communication_grid.get_pipe_parallel_group())

    print_rank_0(f'Tensor Model Parallel Size: {tp} \nData Parallel Size: {dp} \n')
    global_rank = dist.get_rank()
    _set_random_seed(42+global_rank)


def get_topo():
    global topo
    return topo

def get_grid():
    global communication_grid
    return communication_grid



log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] "
            "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="Oases", level=logging.INFO)


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when one of following condition meets

    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]

    Args:
        message (str)
        ranks (list)
        level (int)

    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        final_message = "[Rank {}] {}".format(my_rank, message)
        logger.log(level, final_message)


peak_memory = 0
def see_memory_usage(message, force=False, ram=False, ranks=[0], group=None):
    if not force:
        return
    if torch.distributed.is_initialized() and not torch.distributed.get_rank() in ranks:
        # torch.cuda.empty_cache()
        # torch.distributed.barrier(group=group)
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()
    global peak_memory
    max_ma = round(torch.cuda.max_memory_allocated() / (1024 * 1024),2)
    increased_mem = max(peak_memory, max_ma)-peak_memory
    peak_memory = max(peak_memory, max_ma)
    # Print message except when distributed but not rank 0
    log_dist(message, ranks=ranks)
    log_dist(
        f"MA {round(torch.cuda.memory_allocated() / (1024 * 1024),2 )} MB \
        Max_MA {max_ma} MB \
        CA {round(torch.cuda.memory_reserved() / (1024 * 1024),2)} MB \
        Max_CA {round(torch.cuda.max_memory_reserved() / (1024 * 1024))} MB \
        PEAK_MA {peak_memory} MB    ", ranks=ranks)
    
    # torch.cuda.empty_cache()
    # torch.distributed.barrier(group=group)
    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()

    return increased_mem