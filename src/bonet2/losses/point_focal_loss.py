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
"""Point Focal Loss as a Keras loss."""

import tensorflow as tf


class PointFocalLoss(tf.keras.losses.Loss):
    """Point Focal Loss as a Keras loss.

    Reference: https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(
        self, *, alpha: float = 0.75, gamma: float = 2.0, scale: float = 30.0, **kwargs
    ):
        """
        Constructor of a PointFocalLoss loss.

        Args:
            alpha: Weighting factor. Defaults to 0.75.
            gamma: Exponent of modulating factor. Defaults to 2.0.
            scale: Scale factor. Defaults to 30.0.
        """
        super().__init__(**kwargs)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.scale = tf.constant(scale, dtype=tf.float32)

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            y_true: Binary ground-truth labels.
            y_pred: Predicted probabilities.

        Returns:
            PointFocalLoss value.
        """
        positive = tf.math.equal(y_true, 1)
        valid_mask = tf.math.reduce_any(positive, axis=-1)
        valid_mask = tf.cast(valid_mask, tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        alpha_factor = self.alpha * tf.ones_like(y_true)
        alpha_tensor = tf.where(positive, alpha_factor, 1 - alpha_factor)
        probability = tf.where(positive, y_pred, 1.0 - y_pred)
        weight = alpha_tensor * tf.math.pow((1.0 - probability), self.gamma)
        focal_loss = -weight * tf.math.log(probability)
        focal_loss = tf.math.reduce_mean(focal_loss, axis=-1)
        focal_loss = tf.math.reduce_sum(focal_loss * valid_mask) / tf.math.reduce_sum(
            valid_mask
        )
        return self.scale * focal_loss
