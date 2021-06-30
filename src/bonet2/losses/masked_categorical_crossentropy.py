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
"""Masked Categorial CrossEntropy as a Keras loss."""

import tensorflow as tf


class MaskedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Masked Categorial CrossEntropy as a Keras loss."""

    def __init__(self, ignore_index: int = -1, **kwargs):
        """
        Constructor of MaskedCategoricalCrossentropy loss.

        Args:
            ignore_index: Index to be ignored in loss computation. Defaults to -1.
        """
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self._crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        )

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Args:
            y_true: Categorical ground-truth labels.
            y_pred: Predicted categorical probabilities.

        Returns:
            Categorical cross-entropy loss computed on values with ground-truth index
            different from `ignore_index`. If all indices have to be ignored, the result is 0.
        """
        valid_mask = tf.math.not_equal(y_true, self.ignore_index)
        valid_mask = tf.cast(valid_mask, tf.float32)
        mask_weights = tf.math.divide_no_nan(valid_mask, tf.math.reduce_sum(valid_mask))
        return self._crossentropy(tf.math.maximum(y_true, 0), y_pred, mask_weights)
