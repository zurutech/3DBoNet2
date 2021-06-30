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
"""Keras losses for 3D Bounding Boxes prediction."""

from typing import Tuple

import tensorflow as tf


class VerticesLosses(tf.keras.losses.Loss):
    """
    Loss on vertices predictions composed of:
    - binary cross-entropy loss on instance masks on points;
    - soft IoU instance loss;
    - L2 loss between ground-truth and predicted vertices.
    """

    def __init__(self, theta1: float = 100.0, theta2: float = 20.0, **kwargs):
        """
        Constructor of a VerticesLosses loss.

        Args:
            theta1: Hyper-parameter for numerical stability used in soft IoU computation. Defaults to 100.0.
            theta2: Hyper-parameter for numerical stability used in soft IoU computation. Defaults to 20.0.
        """
        super().__init__(**kwargs)
        self.theta1 = theta1
        self.theta2 = theta2
        self._bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )
        self._mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )

    @tf.function
    def __call__(
        self, pointcloud: tf.Tensor, vertices_true: tf.Tensor, vertices_pred: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass.

        Args:
            pointcloud: Input point cloud. Shape (B, N, F).
            vertices_true: Ground-truth bounding box vertices. Shape (B, H, 2, 3).
            vertices_pred: Predicted bounding box vertices. Shape (B, H, 2, 3).

        Returns:
            - Sum of all the single losses.
            - Binary cross-entropy loss on instance masks on points.
            - Soft IoU instance loss.
            - L2 loss between ground-truth and predicted vertices.
        """
        xyz = pointcloud[:, tf.newaxis, :, tf.newaxis, :3]

        # Compute hard point inclusion in ground truth bounding boxes
        point_difference_true = vertices_true[:, :, tf.newaxis, :, :] - xyz
        point_bbox_true = tf.math.reduce_all(
            tf.math.less_equal(tf.math.reduce_prod(point_difference_true, axis=3), 0.0),
            axis=-1,
        )
        point_bbox_true = tf.cast(point_bbox_true, tf.float32)

        # Compute smooth point inclusion in bounding boxes
        point_difference_pred = vertices_pred[:, :, tf.newaxis, :, :] - xyz
        point_bbox_logits = -self.theta1 * tf.math.reduce_prod(
            point_difference_pred, axis=3
        )
        point_bbox_logits = tf.clip_by_value(
            point_bbox_logits, -self.theta2, self.theta2
        )
        point_bbox_logits = tf.math.reduce_min(point_bbox_logits, axis=-1)

        # Identify valid bounding boxes
        valid_mask = tf.math.reduce_any(
            tf.math.not_equal(vertices_true, 0), axis=[2, 3]
        )
        valid_mask = tf.cast(valid_mask, tf.float32)
        mask_weights = valid_mask / tf.math.reduce_sum(valid_mask)

        # Compare inclusion probability with binary cross-entropy loss
        bbox_loss_ce = self._bce(point_bbox_true, point_bbox_logits, mask_weights)

        # Compute soft Intersection-Over-Union loss
        point_bbox_pred = tf.math.sigmoid(point_bbox_logits)
        true_positives = tf.math.reduce_sum(point_bbox_true * point_bbox_pred, axis=-1)
        false_positives = tf.math.reduce_sum(point_bbox_pred, axis=-1) - true_positives
        false_negatives = tf.math.reduce_sum(point_bbox_true, axis=-1) - true_positives
        iou = -true_positives / (
            true_positives + false_positives + false_negatives + 1e-6
        )
        bbox_loss_iou = tf.math.reduce_sum(iou * mask_weights)

        # Compute mean squared error loss
        # The coefficient 0.5 preserves the original loss scaling.
        positive_bbox_loss = 0.5 * self._mse(vertices_true, vertices_pred, mask_weights)
        # The coefficient 3 preserves the original loss scaling.
        negative_bbox_loss = 3 * self._mse(
            vertices_pred[:, :, 0],
            vertices_pred[:, :, 1],
            sample_weight=(1.0 - valid_mask) / tf.math.reduce_sum((1.0 - valid_mask)),
        )
        bbox_loss_l2 = positive_bbox_loss + negative_bbox_loss

        bbox_loss = bbox_loss_ce + bbox_loss_iou + bbox_loss_l2

        return bbox_loss, bbox_loss_ce, bbox_loss_iou, bbox_loss_l2


class ScoreLoss(tf.keras.losses.Loss):
    """
    Binary cross-entropy loss on instance confidence scores.
    """

    def __init__(
        self,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        **kwargs
    ):
        """
        Constructor of a ScoreLoss loss.

        Args:
            reduction: Reduction method. Defaults to tf.keras.losses.Reduction.AUTO.
        """
        super().__init__(reduction=reduction, **kwargs)
        self._bce = tf.keras.losses.BinaryCrossentropy(reduction=reduction)

    @tf.function
    def __call__(self, vertices_true: tf.Tensor, score_pred: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            vertices_true: Ground-truth bounding box vertices. Shape (B, H, 2, 3).
            score_pred: Confidence scores for predicted instances. Shape (B, H).

        Returns:
            Binary cross-entropy loss on instance confidence scores.
        """
        valid_mask = tf.math.reduce_any(
            tf.math.not_equal(vertices_true, 0), axis=[2, 3]
        )
        valid_mask = tf.cast(valid_mask, tf.float32)
        return self._bce(valid_mask, score_pred)
