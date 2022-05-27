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
"""Post-processing module for merging of instances across blocks of the same point cloud."""

import math
from typing import Tuple

import numpy as np
import tensorflow as tf

from .sparse_ops import (
    group_reduce_most_common,
    group_reduce_sum,
    entries_in_sparse,
    sparse_gather,
    indices_in_sparse,
    unique_indices,
    unique_indices_with_counts,
)


class BlockMerger(tf.Module):
    """Module to merge instance predictions on blocks to global predictions.

    Shape conventions:
    - N: Number of points in every block (usually 4096).
    - B: Number of blocks or batch size.
    - F: Number of point features.
    - H: Maximum number of supported instances in a point cloud block.
    """

    def __init__(
        self,
        voxel_size: float,
        coordinates_slice: slice,
        minimum_instance_sizes: np.ndarray,
        overlap_threshold: int = 7,
        instance_points_threshold: int = 11,
    ):
        """
        Constructor of a BlockMerger module.

        Args:
            voxel_size: Size of voxels into which the space is divided.
            coordinates_slice: Slice to extract the normalized coordinates of the point cloud.
                These coordinates should be part of the input features.
            minimum_instance_sizes (np.ndarray): 1-D array of the minimum number of points
                needed to accept an instance for every semantic class.
            overlap_threshold: Minimum number of points in the same voxel and with the same semantic
                class from two separate blocks needed to consider them part of the same instance.
                Defaults to 7.
            instance_points_threshold: Minimum number of points in a block
                needed to form a new instance. Defaults to 11.
        """
        self.voxel_size = voxel_size
        self.coordinates_slice = coordinates_slice
        self.minimum_instance_sizes = tf.constant(
            minimum_instance_sizes, dtype=tf.int64
        )
        self.overlap_threshold = overlap_threshold
        self.instance_points_threshold = instance_points_threshold

        n_voxels = math.ceil(1 / self.voxel_size) + 1
        self.dense_shape = (n_voxels, n_voxels, n_voxels)
        self.grid_indices = tf.Variable(
            tf.zeros((0, 3), dtype=tf.int64),
            shape=[None, 3],
            trainable=False,
            name="grid_indices",
        )
        self.instance_grid_values = tf.Variable(
            tf.zeros(0, dtype=tf.int64),
            shape=[None],
            trainable=False,
            name="instance_grid_values",
        )
        self.semantic_grid_values = tf.Variable(
            tf.zeros(0, dtype=tf.int64),
            shape=[None],
            trainable=False,
            name="semantic_grid_values",
        )
        self.counter_grid_values = tf.Variable(
            tf.zeros(0, dtype=tf.int64),
            shape=[None],
            trainable=False,
            name="counter_grid_values",
        )
        self.instance_scores_sum = tf.Variable(
            tf.zeros(0, dtype=tf.float32),
            shape=[None],
            trainable=False,
            name="instance_scores_sum",
        )
        self.instance_scores_counter = tf.Variable(
            tf.zeros(0, dtype=tf.float32),
            shape=[None],
            trainable=False,
            name="instance_scores_counter",
        )

    @tf.function(reduce_retracing=True)
    def _block_update(
        self,
        block_pointcloud: tf.Tensor,
        block_instance_pred: tf.Tensor,
        block_semantic_pred: tf.Tensor,
        block_instance_score: tf.Tensor,
    ) -> None:
        """
        Update voxel grids using data from a point cloud block.

        Args:
            block_pointcloud: Point cloud block. Shape (N, F).
            block_instance_pred: Instance predictions for every point in the block. Shape (N,).
            block_semantic_pred: Semantic predictions for every point in the block. Shape (N,).
            block_instance_score: Confidence scores for predicted instances in the block. Shape (H,).
        """
        instance_grid = tf.SparseTensor(
            self.grid_indices, self.instance_grid_values, self.dense_shape
        )
        semantic_grid = tf.SparseTensor(
            self.grid_indices, self.semantic_grid_values, self.dense_shape
        )
        counter_grid = tf.SparseTensor(
            self.grid_indices, self.counter_grid_values, self.dense_shape
        )
        _, block_instance_indices = tf.unique(block_instance_pred, out_idx=tf.int64)

        instance_semantic_labels = group_reduce_most_common(
            tf.cast(block_semantic_pred, dtype=tf.int64),
            block_instance_indices,
        )
        uniform_semantic_pred = tf.gather(
            instance_semantic_labels, block_instance_indices
        )

        # Only points that belong to inactive voxels or that have the same semantic
        # label of their voxel can contribute to the update.
        voxel_ijk = tf.cast(
            block_pointcloud[:, self.coordinates_slice] / self.voxel_size,
            dtype=tf.int64,
        )
        overlap_mask = entries_in_sparse(
            voxel_ijk, uniform_semantic_pred, semantic_grid
        )
        new_voxel_mask = tf.math.logical_not(
            indices_in_sparse(voxel_ijk, instance_grid)
        )
        voxel_mask = tf.math.logical_or(overlap_mask, new_voxel_mask)
        if tf.math.reduce_any(voxel_mask):
            voxel_ijk = voxel_ijk[voxel_mask]
            overlap_mask = overlap_mask[voxel_mask]
            new_voxel_mask = new_voxel_mask[voxel_mask]

            (
                unique_block_instances,
                block_instance_indices,
                block_instance_counts,
            ) = tf.unique_with_counts(block_instance_pred[voxel_mask], out_idx=tf.int64)
            n_block_instances = tf.size(unique_block_instances, out_type=tf.int64)
            n_grid_instances = tf.where(
                tf.math.equal(tf.size(self.instance_grid_values), 0),
                tf.constant(0, dtype=tf.int64),
                tf.math.reduce_max(self.instance_grid_values) + 1,
            )
            if tf.math.greater(n_grid_instances, tf.constant(0, dtype=tf.int64)):
                instance_grid_selection = sparse_gather(
                    instance_grid, voxel_ijk[overlap_mask]
                )
                n_overlap = tf.math.reduce_sum(tf.cast(overlap_mask, tf.int64))
                overlap_range = tf.range(n_overlap, dtype=tf.int64)
                overlap_counts = tf.sparse.SparseTensor(
                    indices=tf.stack(
                        [
                            block_instance_indices[overlap_mask],
                            instance_grid_selection,
                            overlap_range,
                        ],
                        axis=-1,
                    ),
                    values=overlap_range,
                    dense_shape=(
                        n_block_instances,
                        n_grid_instances,
                        n_overlap,
                    ),
                )
                overlap_counts = tf.sparse.reorder(overlap_counts)
                overlap_counts = tf.sets.size(overlap_counts)
                max_overlap = tf.math.reduce_max(overlap_counts, axis=-1)
                local2global = tf.math.argmax(overlap_counts, axis=-1)
                local2global = tf.where(
                    tf.math.equal(max_overlap, 0),
                    -tf.ones_like(local2global),
                    local2global,
                )
            else:
                max_overlap = tf.zeros(n_block_instances, dtype=tf.int32)
                local2global = -tf.ones(n_block_instances, dtype=tf.int64)

            if tf.math.reduce_any(new_voxel_mask):
                local2global_mask = tf.math.logical_and(
                    tf.math.less(max_overlap, self.overlap_threshold),
                    tf.math.greater(
                        block_instance_counts, self.instance_points_threshold
                    ),
                )
                replacement = n_grid_instances + tf.math.cumsum(
                    tf.cast(local2global_mask, dtype=tf.int64), exclusive=True
                )
                local2global = tf.where(local2global_mask, replacement, local2global)

            block_global_instances = tf.gather(local2global, block_instance_indices)
            valid_voxels = tf.math.not_equal(block_global_instances, -1)
            valid_new_voxels = tf.ensure_shape(
                tf.math.logical_and(new_voxel_mask, valid_voxels), [None]
            )

            if tf.math.reduce_any(valid_new_voxels):
                new_voxel_ijk = voxel_ijk[valid_new_voxels]
                new_voxel_instances = block_global_instances[valid_new_voxels]

                unique_voxel_ijk, voxel_groups = unique_indices(new_voxel_ijk)
                unique_voxel_instances = group_reduce_most_common(
                    new_voxel_instances, voxel_groups
                )

                instance_grid_update = tf.SparseTensor(
                    indices=unique_voxel_ijk,
                    values=unique_voxel_instances,
                    dense_shape=self.dense_shape,
                )
                instance_grid_update = tf.sparse.reorder(instance_grid_update)
                instance_grid = tf.sparse.add(instance_grid, instance_grid_update)

                representatives_mask = entries_in_sparse(
                    new_voxel_ijk,
                    new_voxel_instances,
                    instance_grid_update,
                    subset=True,
                )
                new_voxel_semantic_pred = uniform_semantic_pred[voxel_mask][
                    valid_new_voxels
                ]
                (
                    unique_representatives_ijk,
                    representatives_groups,
                ) = unique_indices(new_voxel_ijk[representatives_mask])
                unique_voxel_semantic = group_reduce_most_common(
                    new_voxel_semantic_pred[representatives_mask],
                    representatives_groups,
                )
                semantic_grid_update = tf.SparseTensor(
                    indices=unique_representatives_ijk,
                    values=unique_voxel_semantic,
                    dense_shape=self.dense_shape,
                )
                semantic_grid_update = tf.sparse.reorder(semantic_grid_update)
                semantic_grid = tf.sparse.add(semantic_grid, semantic_grid_update)
                tf.debugging.assert_equal(
                    instance_grid.indices,
                    semantic_grid.indices,
                    message="Inconsistent instance and semantic grids",
                )

            updated_mask = entries_in_sparse(
                voxel_ijk, block_global_instances, instance_grid
            )
            if tf.math.reduce_any(updated_mask):
                unique_voxel_ijk, _, counter_voxel_ijk = unique_indices_with_counts(
                    voxel_ijk[updated_mask]
                )
                counter_grid_update = tf.SparseTensor(
                    indices=unique_voxel_ijk,
                    values=counter_voxel_ijk,
                    dense_shape=self.dense_shape,
                )
                counter_grid_update = tf.sparse.reorder(counter_grid_update)
                counter_grid = tf.sparse.add(counter_grid, counter_grid_update)
                tf.debugging.assert_equal(
                    instance_grid.indices,
                    counter_grid.indices,
                    message="Inconsistent instance and counter grids",
                )

            # The instance confidence score is the mean of the instance scores
            # of the single blocks weighted by the number of points assigned to the instance.
            # We keep track of the numerator of this mean in instance_scores_sum and of
            # the denominator in instance_scores_counter.
            unique_instance_scores = tf.gather(
                block_instance_score, unique_block_instances
            )
            unique_instance_scores_sum = (
                tf.cast(block_instance_counts, tf.float32) * unique_instance_scores
            )
            right_padding = tf.math.maximum(
                tf.math.reduce_max(local2global)
                + 1
                - tf.size(self.instance_scores_sum, out_type=tf.int64),
                0,
            )
            instance_scores_sum = tf.pad(self.instance_scores_sum, [[0, right_padding]])
            instance_scores_sum = tf.tensor_scatter_nd_add(
                instance_scores_sum,
                local2global[:, tf.newaxis],
                unique_instance_scores_sum,
            )

            instance_scores_counter = tf.pad(
                self.instance_scores_counter, [[0, right_padding]]
            )
            instance_scores_counter = tf.tensor_scatter_nd_add(
                instance_scores_counter,
                local2global[:, tf.newaxis],
                tf.cast(block_instance_counts, dtype=tf.float32),
            )

            self.grid_indices.assign(instance_grid.indices)
            self.instance_grid_values.assign(instance_grid.values)
            self.semantic_grid_values.assign(semantic_grid.values)
            self.counter_grid_values.assign(counter_grid.values)
            self.instance_scores_sum.assign(instance_scores_sum)
            self.instance_scores_counter.assign(instance_scores_counter)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        ]
    )
    def update(
        self,
        blocks: tf.Tensor,
        instance_preds: tf.Tensor,
        semantic_preds: tf.Tensor,
        instance_scores: tf.Tensor,
    ) -> tf.Tensor:
        """
        Update voxel grids using data from a batch of point cloud blocks.

        Args:
            blocks: Point cloud blocks. Shape (B, N, F).
            instance_preds: Instance predictions for every point in the blocks. Shape (B, N).
            semantic_preds: Semantic predictions for every point in the blocks. Shape (B, N).
            instance_scores: Confidence scores for predicted instances in blocks. Shape (B, H).

        Returns:
            Number of active voxels in the grid.
        """
        n_blocks = tf.shape(blocks)[0]
        iteration = tf.constant(0, dtype=tf.int32)

        def loop_body(it):
            self._block_update(
                blocks[it], instance_preds[it], semantic_preds[it], instance_scores[it]
            )
            # The tuple is necessary to make the while_loop works in eager mode.
            return (it + 1,)

        tf.while_loop(
            cond=lambda it: tf.math.less(it, n_blocks),
            body=loop_body,
            loop_vars=(iteration,),
        )
        # To call the function from a loaded SavedModel,
        # it is necessary to return a tf.Tensor.
        return tf.shape(self.grid_indices)[0]

    @tf.function(reduce_retracing=True)
    def __call__(self, coordinates: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get instance and semantic predictions from grid voxels.

        Args:
            coordinates: Point coordinates. Shape (..., 3).

        Returns:
            - Instance predictions for the given points. Shape (...)
            - Semantic predictions for the given points. Shape (...)
        """
        instance_grid = tf.SparseTensor(
            self.grid_indices, self.instance_grid_values, self.dense_shape
        )
        semantic_grid = tf.SparseTensor(
            self.grid_indices, self.semantic_grid_values, self.dense_shape
        )
        input_shape = tf.shape(coordinates)
        voxel_ijk = tf.cast(coordinates / self.voxel_size, dtype=tf.int64)
        voxel_ijk = tf.reshape(voxel_ijk, (-1, 3))

        instance_pred = sparse_gather(
            instance_grid, voxel_ijk, implicit=True, implicit_value=-1
        )
        instance_pred = tf.cast(instance_pred, dtype=tf.int32)
        instance_pred = tf.reshape(instance_pred, input_shape[:-1])
        semantic_pred = sparse_gather(
            semantic_grid, voxel_ijk, implicit=True, implicit_value=-1
        )
        semantic_pred = tf.cast(semantic_pred, dtype=tf.int32)
        semantic_pred = tf.reshape(
            semantic_pred,
            input_shape[:-1],
        )

        return instance_pred, semantic_pred

    @tf.function(input_signature=[])
    def get_instance_scores(self) -> tf.Tensor:
        """
        Obtain a confidence score for every detected instance.

        Returns:
            Confidence scores of detected instances.
        """
        sorted_instances = tf.sort(tf.unique(self.instance_grid_values)[0])
        tf.debugging.assert_equal(
            sorted_instances,
            tf.range(tf.size(self.instance_scores_counter), dtype=tf.int64),
        )
        return tf.math.divide_no_nan(
            self.instance_scores_sum, self.instance_scores_counter
        )

    @tf.function(input_signature=[])
    def remove_small_instances(self) -> tf.Tensor:
        """
        Remove voxels associated to instances with less point than the minimum size.
        Additionally, it removes voxels associated to the invalid default instance with index 0.

        Returns:
            Number of active voxels in the grid.
        """
        unique_instance_values, instance_grid_values = tf.unique(
            self.instance_grid_values, out_idx=tf.int64
        )
        unique_instance_counter = group_reduce_sum(
            self.counter_grid_values, instance_grid_values
        )
        first_instance_index = tf.math.argmax(
            tf.math.equal(
                tf.range(tf.size(unique_instance_values), dtype=tf.int64)[
                    :, tf.newaxis
                ],
                instance_grid_values[tf.newaxis, :],
            ),
            axis=-1,
            output_type=tf.int64,
        )
        unique_instance_semantic = tf.gather(
            self.semantic_grid_values, first_instance_index
        )
        min_instance_counter = tf.gather(
            self.minimum_instance_sizes, unique_instance_semantic
        )

        instance_inclusion_mask = tf.math.greater(
            unique_instance_counter, min_instance_counter
        )

        included_instances = tf.sort(unique_instance_values[instance_inclusion_mask])
        indices_inclusion_mask = tf.math.reduce_any(
            tf.math.equal(
                self.instance_grid_values[:, tf.newaxis],
                included_instances[tf.newaxis, :],
            ),
            axis=-1,
        )
        self.grid_indices.assign(self.grid_indices[indices_inclusion_mask])
        # Shift values to fill holes left by removed indices
        shift_mapping = -tf.ones(
            tf.math.reduce_max(unique_instance_values) + 1, dtype=tf.int64
        )
        shift_mapping = tf.tensor_scatter_nd_update(
            shift_mapping,
            included_instances[:, tf.newaxis],
            tf.range(tf.size(included_instances), dtype=tf.int64),
        )
        updated_values = tf.gather(
            shift_mapping, self.instance_grid_values[indices_inclusion_mask]
        )
        self.instance_grid_values.assign(updated_values)

        self.semantic_grid_values.assign(
            self.semantic_grid_values[indices_inclusion_mask]
        )
        self.counter_grid_values.assign(
            self.counter_grid_values[indices_inclusion_mask]
        )
        self.instance_scores_sum.assign(
            tf.gather(self.instance_scores_sum, included_instances)
        )
        self.instance_scores_counter.assign(
            tf.gather(self.instance_scores_counter, included_instances)
        )
        # To call the function from a loaded SavedModel,
        # it is necessary to return a tf.Tensor.
        return tf.shape(self.grid_indices)[0]

    @tf.function(input_signature=[])
    def reset(self) -> tf.Tensor:
        """
        Remove all active voxels resetting the grid.

        Returns:
            Number of active voxels in the grid, i.e. 0.
        """
        self.grid_indices.assign(tf.zeros((0, 3), dtype=tf.int64))
        self.instance_grid_values.assign(tf.zeros(0, dtype=tf.int64))
        self.semantic_grid_values.assign(tf.zeros(0, dtype=tf.int64))
        self.counter_grid_values.assign(tf.zeros(0, dtype=tf.int64))
        self.instance_scores_sum.assign(tf.zeros(0, dtype=tf.float32))
        self.instance_scores_counter.assign(tf.zeros(0, dtype=tf.float32))
        # To call the function from a loaded SavedModel,
        # it is necessary to return a tf.Tensor.
        return tf.shape(self.grid_indices)[0]
