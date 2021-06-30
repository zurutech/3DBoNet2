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

"""Point Feature Propagation for implementing PointNet++."""

from typing import Sequence, Tuple

import tensorflow as tf

from .ops import interpolate_nearest


class FeaturePropagation(tf.keras.layers.Layer):
    """Feature Propagation layers for PointNet++.

    Shape conventions:
    - N1: Number of input points.
    - N2: Number of points in the subset of input points.
    - B: Number of blocks or batch size.
    - F1: Number of point features for input points.
    - F2: Number of point features for the subset of input points.
    - F: Number of output point features.
    """

    def __init__(self, mlp_sizes: Sequence[int], **kwargs):
        """
        Constructor of FeaturePropagation layer.

        Args:
            mlp_sizes: Output sizes of consecutive fully-connected layers
                applied to interpolated features.
        """
        super().__init__(**kwargs)
        self.mlp_sizes = mlp_sizes
        self.mlp = None

    def build(self, inputs_shape: Sequence[tf.TensorShape]) -> None:
        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )

        self.mlp = tf.keras.Sequential()
        self.mlp.add(
            tf.keras.layers.InputLayer(
                input_shape=(None, inputs_shape[2][-1] + inputs_shape[3][-1])
            )
        )
        for dense_size in self.mlp_sizes:
            self.mlp.add(
                tf.keras.layers.Dense(
                    dense_size,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                )
            )

    @tf.function
    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Tuple of tensors composed of:
                - (B, N1, 3)-shaped point coordinates;
                - (B, N2, 3)-shaped subset of the previous coordinates;
                - (B, N1, F1)-shaped fine point features;
                - (B, N2, F2)-shaped coarse point features for the selected subset of points.

        Returns:
            Propagated point features obtained interpolating and processing fine and coarse point features.
            Shape (B, N1, F)
        """
        xyz1, xyz2, points1, points2 = inputs
        interpolated_points = interpolate_nearest(xyz1, xyz2, points2)
        new_points1 = tf.concat([interpolated_points, points1], axis=2)
        return self.mlp(new_points1)
