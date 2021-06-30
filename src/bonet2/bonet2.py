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
"""3D-BoNet implementation in pure TensorFlow.
It optionally depends only on a single custom op."""

from typing import Optional, Tuple

import tensorflow as tf

from .layers import BoundingBoxMatcher
from .backbones import PointNet2


class BoNet2(tf.keras.Model):
    """3DBoNet with PointNet++ as backbone.

    Reference: https://arxiv.org/abs/1906.01140

    Shape conventions:
    - N: Number of points in every block (usually 4096).
    - B: Number of blocks or batch size.
    - F: Number of point features.
    - H: Maximum number of supported instances in a point cloud block.
    - L: Number of semantic classes.
    """

    def __init__(
        self,
        n_segmentation_classes: int,
        n_bbox: int,
        theta1: float = 100.0,
        theta2: float = 20.0,
    ):
        """3DBoNet with PointNet++ as backbone.

        Args:
            n_segmentation_classes: Number of segmentation classes.
            n_bbox: Maximum number of bounding boxes to predict.
            theta1: Hyper-parameter for numerical stability used during the bounding box matching.
            theta2: Hyper-parameter for numerical stability used during the bounding box matching.
        """
        super().__init__()
        self.n_segmentation_classes = n_segmentation_classes
        self.n_bbox = n_bbox
        self.theta1 = tf.constant(theta1, tf.float32)
        self.theta2 = tf.constant(theta2, tf.float32)

        self.backbone = PointNet2(n_segmentation_classes)

        leaky_relu = tf.keras.layers.LeakyReLU(0.2)
        initializer = tf.keras.initializers.VarianceScaling(
            mode="fan_avg", distribution="uniform"
        )
        self.bbox_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    512, activation=leaky_relu, kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(
                    256, activation=leaky_relu, kernel_initializer=initializer
                ),
            ]
        )
        self.bbox_vertices_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256, activation=leaky_relu, kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(n_bbox * 2 * 3, kernel_initializer=initializer),
                tf.keras.layers.Reshape([n_bbox, 2, 3]),
            ]
        )
        self.bbox_score_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    256, activation=leaky_relu, kernel_initializer=initializer
                ),
                tf.keras.layers.Dense(
                    n_bbox,
                    activation="sigmoid",
                    kernel_initializer=initializer,
                    name="bbox_scores",
                ),
            ]
        )
        self.bbox_matcher = BoundingBoxMatcher(theta1, theta2)
        self.mask_global_layer = tf.keras.layers.Dense(
            256, activation=leaky_relu, kernel_initializer=initializer
        )
        self.mask_point_layer = tf.keras.layers.Dense(
            256, activation=leaky_relu, kernel_initializer=initializer
        )
        self.mask_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    128,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
            ]
        )
        self.mask_bbox_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    64,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    32,
                    activation=leaky_relu,
                    kernel_initializer=initializer,
                ),
                tf.keras.layers.Dense(
                    1,
                    activation="sigmoid",
                    kernel_initializer=initializer,
                ),
            ]
        )

    @tf.function
    def call(
        self,
        pointcloud: tf.Tensor,
        ground_truth_vertices: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass.
        Args:
            pointcloud: Input point cloud. Shape (B, N, F).
            ground_truth_vertices: Bounding Box ground truth vertices to use during the training.
                Shape (B, H, 2, 3).
            training: boolean flag, toggle the training mode.

        Return:
            vertices: Predicted bounding box vertices. Shape (B, H, 2, 3).
            scores: Probability scores associated with the predicted boxes. Shape (B, H).
            mask_prob: Point mask prediction branch. Used to check if the point is
                       inside or outside the predicted bounding box. Shape (B, H, N).
            semantic_logits: Unscaled probabilities. Shape (B, N, L).
        """
        n_points = tf.shape(pointcloud)[1]
        point_features, global_features, semantic_logits = self.backbone(pointcloud)
        bbox_features = self.bbox_layers(global_features)
        vertices = self.bbox_vertices_layers(bbox_features)
        min_vertices = tf.math.reduce_min(vertices, axis=2, keepdims=True)
        max_vertices = tf.math.reduce_max(vertices, axis=2, keepdims=True)
        vertices = tf.concat([min_vertices, max_vertices], axis=2, name="bbox_vertices")
        scores = self.bbox_score_layers(bbox_features)
        if training:
            if ground_truth_vertices is None:
                raise ValueError(
                    "During training ground truth bounding boxes are needed."
                )
            bbox_order = self.bbox_matcher(pointcloud, ground_truth_vertices, vertices)
            vertices = tf.gather(
                vertices, bbox_order, batch_dims=1, name="bbox_vertices"
            )
            scores = tf.gather(scores, bbox_order, batch_dims=1, name="bbox_scores")
        mask_global_features = self.mask_global_layer(global_features)
        mask_global_features = tf.tile(
            mask_global_features[:, tf.newaxis, :], [1, n_points, 1]
        )
        mask_point_features = self.mask_point_layer(point_features)
        mask_features = tf.concat([mask_point_features, mask_global_features], axis=-1)
        mask_features = self.mask_layers(mask_features)

        bbox_features = tf.concat(
            [
                tf.reshape(vertices, [-1, self.n_bbox, 6]),
                scores[:, :, tf.newaxis],
            ],
            axis=-1,
        )
        bbox_features = tf.tile(bbox_features[:, :, tf.newaxis, :], [1, 1, n_points, 1])
        mask_features = tf.tile(
            mask_features[:, tf.newaxis, :, :], [1, self.n_bbox, 1, 1]
        )
        mask_bbox_features = tf.concat([mask_features, bbox_features], axis=-1)
        mask_prob = self.mask_bbox_layers(mask_bbox_features)
        mask_prob = tf.squeeze(mask_prob, axis=-1, name="mask")
        return vertices, scores, mask_prob, semantic_logits

    @tf.function
    def predict_step(
        self, pointcloud: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """A single prediction step. Uses the call method under the hood
        but it returns only the predictions.

        Args:
            pointcloud: Input point cloud. Shape (B, N, F).

        Return:
            vertices: Predicted vertices of the bounding box. Shape (B, H, 2, 3).
            scores: Probability score assigned to every predicted bounding box. Shape (B, H).
            instance_pred: The predicted majority class of the instance. Shape (B, N).
            semantic_pred: The predicted semantic class. Shape (B, N).
        """
        vertices, scores, mask_prob, semantic_logits = self.call(pointcloud)
        instance_mask_prob = mask_prob * scores[:, :, tf.newaxis]
        # Every point is assigned to an instance,
        # so that can be problematic if a background is present.
        instance_pred = tf.math.argmax(instance_mask_prob, axis=1, output_type=tf.int32)
        semantic_pred = tf.math.argmax(semantic_logits, axis=-1, output_type=tf.int32)
        return vertices, scores, instance_pred, semantic_pred
