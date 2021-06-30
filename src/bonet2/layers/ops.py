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
"""Operations used for creating custom layers."""

from typing import Optional, Tuple

import tensorflow as tf


@tf.function
def knn_search(
    query_xyz: tf.Tensor,
    pointcloud_xyz: tf.Tensor,
    k: tf.Tensor,
    radius: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """K-NearestNeighbor search.
    Select a fixed number of points in a neighborhood of selected query points
    The nearest points to the query point are selected, this differs from the
    original implementation where the first appearing points were selected.

    Args:
        query_xyz: The Query point.
        pointcloud_xyz: The dataset to use for the search.
        k: The number of neighbors to find. Integer value.
        radius: The radius to use in case of radius search. Float value.

    Return:
        neighbors_indices: The indices of the knn found in the pointcloud_xyz.
        neighbors_count: The number of pointcloud found.
    """
    distances = tf.norm(
        query_xyz[:, :, tf.newaxis, :] - pointcloud_xyz[:, tf.newaxis, :, :],
        axis=-1,
    )
    top_negative_distances, neighbors_indices = tf.math.top_k(-distances, k=k)
    if radius is not None:
        mask = tf.greater(top_negative_distances, -radius)
        repeated_base_index = tf.tile(neighbors_indices[:, :, 0, tf.newaxis], [1, 1, k])
        neighbors_indices = tf.where(mask, neighbors_indices, repeated_base_index)
        neighbors_count = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
    else:
        neighbors_count = tf.fill(tf.shape(distances)[:2], k, dtype=tf.int32)
    return neighbors_indices, neighbors_count


@tf.function
def interpolate_nearest(
    target_xyz: tf.Tensor,
    source_xyz: tf.Tensor,
    source_features: tf.Tensor,
    n_neighbors: tf.Tensor = tf.constant(3, dtype=tf.int32),
) -> tf.Tensor:
    """
    Linear interpolation of feature vectors from neighbors.

    Args:
        target_xyz: Coordinates of output points whose features
            are obtained through interpolation. Shape (B, T, 3).
        source_xyz: Coordinates of input points. Shape (B, S, 3).
        source_features: Features of input points. Shape (B, S, F).
        n_neighbors: Number of neighbors used for interpolation.
            Defaults to tf.constant(3, dtype=tf.int32).

    Returns:
        Interpolated output features. Shape (B, T, F).
    """
    squared_difference = tf.math.squared_difference(
        target_xyz[:, :, tf.newaxis, :], source_xyz[:, tf.newaxis, :, :]
    )  # Shape: (B, T, S, 3)
    squared_distances = tf.math.reduce_sum(
        squared_difference, axis=-1
    )  # Shape: (B, T, S)
    k = tf.math.minimum(tf.shape(source_xyz)[1], n_neighbors)
    negative_squared_distances, neighbors_indices = tf.math.top_k(
        -squared_distances, k, sorted=False
    )  # Shapes: (B, T, K), (B, T, K)
    nn_squared_distances = tf.math.maximum(
        -negative_squared_distances, 1e-10
    )  # Shape: (B, T, K)
    weights, _ = tf.linalg.normalize(
        1 / nn_squared_distances, ord=1, axis=-1
    )  # Shape: (B, T, K)
    neighbors_features = tf.gather(
        source_features, neighbors_indices, batch_dims=1
    )  # Shape: (B, T, K, F)
    interpolated_features = tf.math.reduce_sum(
        neighbors_features * weights[..., tf.newaxis], axis=2
    )  # Shape: (B, T, F)
    return interpolated_features


@tf.function
def farthest_point_sample(n_points: tf.Tensor, pointcloud: tf.Tensor) -> tf.Tensor:
    """Farthest point sample.

    This pure TensorFlow function can replace the analogous custom op.
    In particular it works on CPU, while the custom op is implemented for GPU only.
    On the other hand, this implementation is much slower than the custom op,
    so we suggest to use the provided custom op when possible.

    Args:
        n_points: Scalar integer tensor. The number of points.
        pointcloud: The input pointcloud.
    Returns:
        indices_tensor: The indices of the sampled points in pointcloud.
    """
    batch_size = tf.shape(pointcloud)[0]
    indices = tf.TensorArray(
        dtype=tf.int32,
        size=n_points * batch_size,
    )
    squared_distances = tf.reduce_sum(
        tf.math.squared_difference(
            pointcloud[:, :, tf.newaxis, :], pointcloud[:, tf.newaxis, :, :]
        ),
        axis=-1,
    )
    initial_distances = squared_distances[:, 0]
    iteration = 1
    next_indices = tf.argmax(initial_distances, axis=-1, output_type=tf.int32)
    indices = indices.scatter(
        batch_size * iteration + tf.range(batch_size), next_indices
    )

    def loop_body(iteration, min_distances, indices):
        latest_indices = indices.gather(batch_size * iteration + tf.range(batch_size))
        latest_distances = tf.gather(squared_distances, latest_indices, batch_dims=1)
        min_distances = tf.math.minimum(min_distances, latest_distances)
        next_indices = tf.argmax(min_distances, axis=-1, output_type=tf.int32)
        indices = indices.scatter(
            batch_size * iteration + batch_size + tf.range(batch_size), next_indices
        )
        return iteration + 1, min_distances, indices

    _, _, indices = tf.while_loop(
        cond=lambda *_: tf.constant(True),
        body=loop_body,
        loop_vars=(iteration, initial_distances, indices),
        maximum_iterations=n_points - 2,
    )
    indices_tensor = indices.stack()
    indices_tensor = tf.reshape(indices_tensor, [n_points, batch_size])
    indices_tensor = tf.transpose(indices_tensor)
    return indices_tensor
