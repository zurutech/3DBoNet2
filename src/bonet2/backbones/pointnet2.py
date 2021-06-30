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
"""PointNet++ implementation in pure TensorFlow.
It optionally depends only on a single custom op."""

from typing import Tuple

import tensorflow as tf

from ..layers import (
    FeaturePropagation,
    SetAbstraction,
    SetAbstractionAll,
)


class PointNet2(tf.keras.Model):
    """PointNet2 aka PointNet++.

    Reference: https://arxiv.org/abs/1706.02413

    Shape conventions:
    - N: Number of points in every block (usually 4096).
    - B: Number of blocks or batch size.
    - F: Number of point features.
    - L: Number of semantic classes.
    """

    def __init__(self, n_segmentation_classes: int):
        """PointNet backbone implemented as a Keras model.

        Args:
            n_segmentation_classes: Number of segmentation classes.
        """
        super().__init__()
        self.n_segmentation_classes = n_segmentation_classes

        self.set_abstraction1 = SetAbstraction(
            n_sampled_points=1024,
            radius=0.1,
            n_region_points=32,
            mlp_sizes=[32, 32, 64],
        )
        self.set_abstraction2 = SetAbstraction(
            n_sampled_points=256,
            radius=0.2,
            n_region_points=64,
            mlp_sizes=[64, 64, 128],
        )
        self.set_abstraction3 = SetAbstraction(
            n_sampled_points=64,
            radius=0.4,
            n_region_points=128,
            mlp_sizes=[128, 128, 256],
        )
        self.set_abstraction_all = SetAbstractionAll(mlp_sizes=[256, 256, 512])
        self.features_propagation3 = FeaturePropagation(mlp_sizes=[256, 256])
        self.features_propagation2 = FeaturePropagation(mlp_sizes=[256, 256])
        self.features_propagation1 = FeaturePropagation(mlp_sizes=[256, 128])
        self.features_propagation = FeaturePropagation(mlp_sizes=[128, 128, 128, 128])
        self.global_reshape = tf.keras.layers.Reshape([512])
        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )
        self.segmentator = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    64,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    self.n_segmentation_classes,
                    kernel_initializer=initializer,
                    name="segmentation",
                ),
            ]
        )

    @tf.function
    def call(self, pointcloud: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass.

        Args:
            pointcloud: Input point cloud. Shape (B, N, F).

        Return:
            point_features: Feature vector for each point. Shape (B, N, 1024).
            global_features: Max pooling over the point features. Shape (B, 1024).
            segmentation_logits: Unscaled probabilities. Shape (B, N, L).
        """
        xyz0, features0 = pointcloud[..., :3], pointcloud[..., 3:]
        xyz1, features1 = self.set_abstraction1((xyz0, features0))
        xyz2, features2 = self.set_abstraction2((xyz1, features1))
        xyz3, features3 = self.set_abstraction3((xyz2, features2))
        xyz4, features4 = self.set_abstraction_all((xyz3, features3))
        features3 = self.features_propagation3((xyz3, xyz4, features3, features4))
        features2 = self.features_propagation2((xyz2, xyz3, features2, features3))
        features1 = self.features_propagation1((xyz1, xyz2, features1, features2))
        point_features = self.features_propagation((xyz0, xyz1, pointcloud, features1))
        global_features = self.global_reshape(features4)
        segmentation_logits = self.segmentator(point_features)
        return point_features, global_features, segmentation_logits
