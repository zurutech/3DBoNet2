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
"""PointNet implementation as a tf.keras.Model."""

from typing import Tuple

import tensorflow as tf


class PointNet(tf.keras.Model):
    """PointNet backbone.

    Reference: https://arxiv.org/abs/1612.00593

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

        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )
        self.point_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    64,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    64,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    64,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    128,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    1024,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
            ]
        )
        self.global_pooling = tf.keras.layers.GlobalMaxPool1D()
        self.global_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256, activation=leaky_relu, kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(
                    128, activation=leaky_relu, kernel_initializer=initializer
                ),
            ]
        )
        self.segmentator = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    512,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    256,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    128,
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
        point_features = self.point_layers(pointcloud)
        global_features = self.global_pooling(point_features)
        g = self.global_layers(global_features)
        n_points = tf.shape(pointcloud)[1]
        local_global_features = tf.tile(g[:, tf.newaxis, :], [1, n_points, 1])
        local_global_features = tf.concat([point_features, g], axis=-1)
        segmentation_logits = self.segmentator(local_global_features)
        return point_features, global_features, segmentation_logits
