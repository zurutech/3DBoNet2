# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
"""
from pathlib import Path

import tensorflow as tf
from tensorflow.python.framework import ops

BASE_DIR = Path(__file__).absolute().parent
SAMPLING_MODULE = tf.load_op_library(str(BASE_DIR / "tf_sampling_so.so"))


def farthest_point_sample(npoint, inp):
    """
    input:
        int32
        batch_size * ndataset * 3   float32
    returns:
        batch_size * npoint         int32
    """
    return SAMPLING_MODULE.farthest_point_sample(inp, npoint)


ops.NoGradient("FarthestPointSample")
