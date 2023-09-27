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
from .. import mpu
from functools import reduce
import operator




class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name 
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = \
                torch.empty(required_len,
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False)

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

_memory_buffer = GlobalMemoryBuffer()

#############################
# When incresing MP in BP
# After FP
# 1. Allgather activations for recomputation.
# 2. Chunk parameters for recomputation.
# After BP
# 3. Allgather grad for param update.
#
# When decresing MP in BP
# After FP
# 1. Chunk activations for recomputation.
# 2. Allgather parameters for recomputation.
# After BP
# 3. Chunk grad for param update.
##############################
def get_size(size):
    dim_size = list(size)
    dim_size[0] = dim_size[0] * mpu.get_data_parallel_world_size()
    return dim_size

