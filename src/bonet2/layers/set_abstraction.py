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

"""Set Abstraction layers for implementing PointNet++."""

from typing import Sequence, Tuple

import tensorflow as tf

from ..tf_ops.sampling.tf_sampling import farthest_point_sample
from .ops import knn_search


class SetAbstraction(tf.keras.layers.Layer):
    """Set Abstraction layer defined in PointNet++.
    The set abstraction layer aggregates multi-scale information according to local
    point densities.

    Shape conventions:
    - N1: Number of input points.
    - N2: Number of points sampled from input points.
    - B: Number of blocks or batch size.
    - F1: Number of point features for input points.
    - F2: Number of point features for the subset of input points.
    """

    def __init__(
        self,
        n_sampled_points: int,
        n_region_points: int,
        radius: float,
        mlp_sizes: Sequence[int],
        **kwargs,
    ):
        """
        Constructor of a SetAbstraction layer.

        Args:
            n_sampled_points: Number of points sampled from the input points.
            n_region_points: Number of neighbors whose features are aggregated to obtain
                higher-level features of sampled points.
            radius: Radius of a point neighborhood.
            mlp_sizes: Output sizes of consecutive fully-connected layers applied to sampled features.
        """
        super().__init__(**kwargs)
        self.n_sampled_points = n_sampled_points
        self.n_region_points = n_region_points
        self.radius = radius
        self.mlp_sizes = mlp_sizes

    def build(self, inputs_shape: Sequence[tf.TensorShape]) -> None:
        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )

        self.mlp = tf.keras.Sequential()
        self.mlp.add(
            tf.keras.layers.InputLayer(
                input_shape=(
                    self.n_sampled_points,
                    self.n_region_points,
                    inputs_shape[0][-1] + inputs_shape[-1][-1],
                )
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
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Args:
            inputs: Tuple of tensors composed of:
                - (B, N1, 3)-shaped point coordinates;
                - (B, N1, F1)-shaped point features.

        Returns:
            Tuple of tensors composed of:
            - (B, N2, 3)-shaped sampled point coordinates;
            - (B, N2, F2)-shaped sampled point features.
        """
        pointcloud_xyz, pointcloud_features = inputs
        #  Sample points with farthest point sampling
        selected_indices = farthest_point_sample(self.n_sampled_points, pointcloud_xyz)

        selected_xyz = tf.gather(
            pointcloud_xyz, selected_indices, batch_dims=1
        )  # (batch_size, self.n_sampled_points, 3)

        neighbors_indices, _ = knn_search(
            selected_xyz, pointcloud_xyz, self.n_region_points, self.radius
        )

        neighbors_xyz = tf.gather(
            pointcloud_xyz, neighbors_indices, batch_dims=1
        )  # (batch_size,  self.n_sampled_points, self.n_region_points, 3)
        neighbors_xyz -= selected_xyz[:, :, tf.newaxis, :]
        neighbors_features = tf.gather(
            pointcloud_features, neighbors_indices, batch_dims=1
        )  # (batch_size,  self.n_sampled_points, self.n_region_points, channel)
        neighbors_features = tf.concat(
            [neighbors_xyz, neighbors_features], axis=-1
        )  # (batch_size,  self.n_sampled_points, self.n_region_points, 3 + channel)

        # Process features of points in neighborhoods
        neighbors_features = self.mlp(neighbors_features)
        # Pool neighborhood features
        selected_features = tf.math.reduce_max(neighbors_features, axis=2)
        return selected_xyz, selected_features


class SetAbstractionAll(tf.keras.layers.Layer):
    """Set Abstraction All is very similar to SetAbstraction,
    but here the information from all points is summarized in a single point."""

    def __init__(
        self,
        mlp_sizes: Sequence[int],
        **kwargs,
    ):
        """
        Constructor of a SetAbstractionAll layer.

        Args:
            mlp_sizes: Output sizes of consecutive fully-connected layers applied to sampled features.
        """
        super().__init__(**kwargs)
        self.mlp_sizes = mlp_sizes
        self.mlp = None

    def build(self, inputs_shape: Sequence[tf.TensorShape]) -> None:
        mlp_input_shapes = [(None, inputs_shape[0][-1] + inputs_shape[1][-1])]
        for dense_size in self.mlp_sizes[:-1]:
            mlp_input_shapes.append((None, dense_size))
        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dense_size,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                    input_shape=input_shape,
                )
                for dense_size, input_shape in zip(self.mlp_sizes, mlp_input_shapes)
            ]
        )

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Args:
            inputs: Tuple of tensors composed of:
                - (B, N1, 3)-shaped point coordinates;
                - (B, N1, F1)-shaped point features.

        Returns:
            Tuple of tensors composed of:
            - (B, 1, 3)-shaped tensor of null values;
            - (B, 1, F2)-shaped point features.
        """
        pointcloud_xyz, pointcloud_features = inputs
        pointcloud = tf.concat([pointcloud_xyz, pointcloud_features], axis=-1)
        features = self.mlp(pointcloud)
        features = tf.math.reduce_max(features, axis=1, keepdims=True)
        origin = tf.zeros_like(features[:, :, :3])
        return origin, features
