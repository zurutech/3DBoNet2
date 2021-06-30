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
"""Metric computing precision and recall for every semantic class."""

from typing import Optional, Dict

import numpy as np
import tensorflow as tf

from .sparse_ops import group_reduce_most_common


class InstancePrecisionRecall(tf.keras.metrics.Metric):
    """Metric computing precision and recall for every semantic class.

    Shape conventions:
    - M: Number of points in a point cloud.
    - B: Number of blocks or batch size.
    - F: Number of point features.
    - H: Maximum number of supported instances in a point cloud block.
    """

    def __init__(
        self,
        n_classes: int,
        minimum_instance_sizes: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Constructor for the InstancePrecisionRecall metric.

        Args:
            n_classes: Number of semantic classes.
            minimum_instance_sizes: Minimum instance size for every semantic class.
                The size is determined by the number of points in the instance.
                Defaults to None, in this case no filtering is applied.
            threshold: IoU minimum value to consider the instance detected. Defaults to 0.5.
        """
        super().__init__(**kwargs)
        self.n_classes = n_classes
        if minimum_instance_sizes is None:
            minimum_instance_sizes = np.zeros(n_classes, dtype=np.int64)
        self.minimum_instance_sizes = tf.constant(
            minimum_instance_sizes, dtype=tf.int64
        )
        self.threshold = threshold

        self.true_positives = self.add_weight(
            shape=(n_classes,),
            dtype=tf.int32,
            name="tp",
            initializer="zeros",
        )
        self.false_positives = self.add_weight(
            shape=(n_classes,),
            dtype=tf.int32,
            name="fp",
            initializer="zeros",
        )
        self.total = self.add_weight(
            shape=(n_classes,),
            dtype=tf.int32,
            name="total",
            initializer="zeros",
        )

    @tf.function(experimental_relax_shapes=True)
    def update_state(
        self,
        instance_labels: tf.Tensor,
        semantic_labels: tf.Tensor,
        instance_preds: tf.Tensor,
        semantic_preds: tf.Tensor,
    ) -> None:
        """
        Update the metric state comparing ground-truth labels with predicted labels.

        Args:
            instance_labels: Instance segmentation ground-truth labels. Shape (M,)
            semantic_labels: Semantic segmentation ground-truth labels. Shape (M,)
            instance_preds: Instance segmentation predicted labels. Shape (M,)
            semantic_preds: Semantic segmentation predicted labels. Shape (M,)
        """
        (
            unique_instance_preds,
            instance_groups_preds,
            instance_sizes_preds,
        ) = tf.unique_with_counts(instance_preds, out_idx=tf.int64)
        instance_semantic_preds = group_reduce_most_common(
            semantic_preds, instance_groups_preds
        )

        # Remove small instances
        size_mask = tf.math.greater(
            instance_sizes_preds,
            tf.gather(self.minimum_instance_sizes, instance_semantic_preds),
        )
        # Remove invalid default instance with index 0
        valid_preds_mask = tf.math.not_equal(unique_instance_preds, 0)
        unique_instance_preds_mask = tf.math.logical_and(size_mask, valid_preds_mask)
        instance_semantic_preds = tf.where(
            unique_instance_preds_mask, instance_semantic_preds, -1
        )

        (unique_instance_labels, instance_groups_labels) = tf.unique(
            instance_labels, out_idx=tf.int64
        )
        instance_semantic_labels = group_reduce_most_common(
            semantic_labels, instance_groups_labels
        )
        valid_labels_mask = tf.math.not_equal(unique_instance_labels, -1)
        instance_semantic_labels = tf.where(
            valid_labels_mask, instance_semantic_labels, -1
        )

        sample_true_positives = tf.TensorArray(dtype=tf.int32, size=self.n_classes)
        sample_false_positives = tf.TensorArray(dtype=tf.int32, size=self.n_classes)
        sample_total = tf.TensorArray(dtype=tf.int32, size=self.n_classes)
        for semantic_class in tf.range(self.n_classes):
            semantic_preds_mask = tf.math.equal(instance_semantic_preds, semantic_class)
            selected_instance_preds = unique_instance_preds[semantic_preds_mask]
            instance_class_preds_mask = tf.math.equal(
                selected_instance_preds[:, tf.newaxis],
                instance_preds[tf.newaxis, :],
            )

            semantic_labels_mask = tf.math.equal(
                instance_semantic_labels, semantic_class
            )
            selected_instance_labels = unique_instance_labels[semantic_labels_mask]
            instance_class_labels_mask = tf.math.equal(
                selected_instance_labels[:, tf.newaxis],
                instance_labels[tf.newaxis, :],
            )

            active_preds = tf.math.reduce_any(instance_class_preds_mask, axis=0)
            active_labels = tf.math.reduce_any(instance_class_labels_mask, axis=0)
            active_mask = tf.math.logical_or(active_preds, active_labels)
            active_preds_mask = tf.cast(
                tf.boolean_mask(instance_class_preds_mask, active_mask, axis=1),
                dtype=tf.float32,
            )
            active_labels_mask = tf.cast(
                tf.boolean_mask(instance_class_labels_mask, active_mask, axis=1),
                dtype=tf.float32,
            )
            intersection = tf.linalg.matmul(
                active_preds_mask, active_labels_mask, transpose_b=True
            )
            count_preds = tf.math.reduce_sum(active_preds_mask, axis=-1)
            count_labels = tf.math.reduce_sum(active_labels_mask, axis=-1)
            union = (
                count_preds[:, tf.newaxis] + count_labels[tf.newaxis, :] - intersection
            )
            iou = tf.math.divide_no_nan(intersection, union)
            iou_max = tf.math.reduce_max(iou, axis=-1)
            accepted = tf.math.greater_equal(iou_max, self.threshold)
            n_accepted = tf.math.reduce_sum(tf.cast(accepted, dtype=tf.int32))
            sample_true_positives = sample_true_positives.write(
                semantic_class, n_accepted
            )
            sample_false_positives = sample_false_positives.write(
                semantic_class, tf.size(selected_instance_preds) - n_accepted
            )
            sample_total = sample_total.write(
                semantic_class, tf.size(selected_instance_labels)
            )
        self.true_positives.assign_add(sample_true_positives.stack())
        self.false_positives.assign_add(sample_false_positives.stack())
        self.total.assign_add(sample_total.stack())

    @tf.function
    def result(self) -> Dict[str, tf.Tensor]:
        """
        Return the metric values associated to the current metric state.

        Returns:
            Dictionary of computed metric values with the following pairs:
            - precisions: vector of precisions for every semantic class;
            - recalls: vector of recalls for every semantic class;
            - average_precision: average precision computed on active semantic classes;
            - average_recall: average recall computed on active semantic classes.
        """
        true_positives = tf.cast(self.true_positives, dtype=tf.float32)
        false_positives = tf.cast(self.false_positives, dtype=tf.float32)
        total = tf.cast(self.total, dtype=tf.float32)

        precisions = tf.math.divide_no_nan(
            true_positives, true_positives + false_positives
        )
        recalls = tf.math.divide_no_nan(true_positives, total)
        average_precision = tf.math.reduce_mean(precisions[self.total > 0])
        average_recall = tf.math.reduce_mean(recalls[self.total > 0])
        results = {
            "precisions": precisions,
            "recalls": recalls,
            "average_precision": average_precision,
            "average_recall": average_recall,
        }
        return results

    @tf.function
    def reset_state(self) -> None:
        """
        Reset the metric state to the initial state.
        """
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.total.assign(tf.zeros_like(self.total))
