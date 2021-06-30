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
"""Bounding Box Matching layer used only during the training phase,
to correctly assign the predicted boxes with the ground truth boxes."""

import tensorflow as tf
from scipy.optimize import linear_sum_assignment


class BoundingBoxMatcher(tf.keras.layers.Layer):
    """Bounding Box Matching layer used only during the training phase,
    to correctly assign the predicted boxes with the ground truth boxes."""

    def __init__(self, theta1: float = 100.0, theta2: float = 20.0, **kwargs):
        """
        Constructor of BoundingBoxMatcher layer.

        Args:
            theta1: Hyper-parameter for numerical stability used in soft IoU computation. Defaults to 100.0.
            theta2: Hyper-parameter for numerical stability used in soft IoU computation. Defaults to 20.0.
        """
        super().__init__(trainable=False, **kwargs)
        self.theta1 = theta1
        self.theta2 = theta2
        self._bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

    @tf.function
    def call(
        self, pointcloud: tf.Tensor, vertices_true: tf.Tensor, vertices_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            pointcloud: Input point cloud. Shape (B, N, F).
            vertices_true: Ground-truth bounding box vertices. Shape (B, H, 2, 3).
            vertices_pred: Predicted bounding box vertices. Shape (B, H, 2, 3).

        Returns:
            Indices of the ground-truth bounding box with the maximum overlap
            for every predicted bounding box. Shape (B, H).
        """
        xyz = pointcloud[:, tf.newaxis, :, tf.newaxis, :3]
        n_bbox = vertices_true.shape[1]

        # Compute hard point inclusion in ground truth bounding boxes
        point_difference_true = vertices_true[:, :, tf.newaxis, :, :] - xyz
        point_bbox_true = tf.math.reduce_all(
            tf.math.less_equal(tf.math.reduce_prod(point_difference_true, axis=3), 0.0),
            axis=-1,
        )
        point_bbox_true = tf.cast(point_bbox_true, tf.float32)
        point_bbox_true = tf.tile(
            point_bbox_true[:, :, tf.newaxis, :], [1, 1, n_bbox, 1]
        )

        # Compute smooth point inclusion in bounding boxes
        point_difference_pred = vertices_pred[:, :, tf.newaxis, :, :] - xyz
        point_bbox_logits = -self.theta1 * tf.math.reduce_prod(
            point_difference_pred, axis=3
        )
        point_bbox_logits = tf.clip_by_value(
            point_bbox_logits, -self.theta2, self.theta2
        )
        point_bbox_logits = tf.math.reduce_min(point_bbox_logits, axis=-1)
        point_bbox_logits = tf.tile(
            point_bbox_logits[:, tf.newaxis, :, :], [1, n_bbox, 1, 1]
        )

        # Compare inclusion probability with binary cross-entropy
        crossentropy_score = self._bce(point_bbox_true, point_bbox_logits)

        # Compute soft Intersection-Over-Union score
        point_bbox_pred = tf.math.sigmoid(point_bbox_logits)
        true_positives = tf.math.reduce_sum(point_bbox_true * point_bbox_pred, axis=-1)
        false_positives = tf.math.reduce_sum(point_bbox_pred, axis=-1) - true_positives
        false_negatives = tf.math.reduce_sum(point_bbox_true, axis=-1) - true_positives
        iou_score = -tf.math.divide_no_nan(
            true_positives, true_positives + false_positives + false_negatives
        )

        # Compute L2 squared distance between vertices
        mse = tf.math.pow(
            vertices_true[:, :, tf.newaxis, :, :]
            - vertices_pred[:, tf.newaxis, :, :, :],
            2,
        )
        mse_score = tf.math.reduce_mean(mse, axis=[3, 4])

        score = crossentropy_score + iou_score + mse_score

        bbox_order = hungarian(score, vertices_true)
        return bbox_order


@tf.function
def hungarian(loss_matrix: tf.Tensor, vertices_true: tf.Tensor) -> tf.Tensor:
    """
    Implementation of the Hungarian algorithm in TensorFlow.
    The Implementation still relies upon the linear_sum_assignment function from
    scipy.optimize, and it's called with tf.numpy_function.

    Args:
        loss_matrix: Matrix of bounding box matching scores. Shape (B, H, H).
        vertices_true: Ground-truth bounding box vertices. Shape (B, H, 2, 3).

    Returns:
        Indices of the ground-truth bounding box with the maximum overlap
        for every predicted bounding box. Shape (B, H).
    """
    mask = tf.equal(tf.norm(vertices_true, ord=1, axis=[-2, -1]), 0.0)
    max_loss = 2 * tf.math.reduce_max(loss_matrix)
    loss_matrix = tf.where(
        mask[:, :, tf.newaxis], max_loss * tf.ones_like(loss_matrix), loss_matrix
    )
    _, col_indices = tf.vectorized_map(
        lambda cost: tf.numpy_function(
            linear_sum_assignment, [cost], [tf.int64, tf.int64]
        ),
        loss_matrix,
    )
    col_indices.set_shape(loss_matrix.shape[:2])
    col_indices = tf.cast(col_indices, tf.int32)
    return col_indices
