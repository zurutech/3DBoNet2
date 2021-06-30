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

"""Operations on sparse tensors used for creating custom post-processing modules and metrics."""

from typing import Tuple

import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def group_reduce_most_common(
    values: tf.Tensor, group_indices: tf.Tensor, validate: bool = False
) -> tf.Tensor:
    """
    Finds the most common value for every group.
    If there are many values with the maximum frequency,
    then the lowest value is chosen.

    Args:
        values: 1-D tensor of values.
        group_indices: 1-D int64 tensor of group indices of same shape of values.
            Values associated to the same group index belong to the same group.
            Group indices should cover all numbers in [0, max(group_indices)].
        validate: If true, then check that the group indices are valid.

    Returns:
        1-D tensor of most common values per group.
    """
    if validate:
        unique_group_indices, _ = tf.unique(group_indices)
        indices_range = tf.range(
            tf.squeeze(tf.shape(unique_group_indices, dtype=tf.int64))
        )
        tf.debugging.assert_equal(
            unique_group_indices, indices_range, message="Invalid group indices."
        )
    order = tf.argsort(values)
    values = tf.gather(values, order)
    group_indices = tf.gather(group_indices, order)
    unique_values, values_indices = tf.unique(values)

    group_order = tf.argsort(group_indices)
    groups = tf.RaggedTensor.from_value_rowids(
        tf.gather(values_indices, group_order),
        value_rowids=tf.gather(group_indices, group_order),
        nrows=tf.math.reduce_max(group_indices) + 1,
    )
    counter = tf.math.bincount(groups, axis=-1)
    most_common = tf.math.argmax(counter, axis=-1)
    common_values = tf.gather(unique_values, most_common)
    common_values = tf.ensure_shape(common_values, [None])
    return common_values


@tf.function(experimental_relax_shapes=True)
def group_reduce_sum(
    values: tf.Tensor, group_indices: tf.Tensor, validate: bool = False
) -> tf.Tensor:
    """
    Sum values in every group.

    Args:
        values: 1-D tensor of values.
        group_indices: 1-D int64 tensor of group indices of same shape of values.
            Values associated to the same group index belong to the same group.
            Group indices should cover all numbers in [0, max(group_indices)].
        validate: If true, then check that the group indices are valid.

    Returns:
        1-D tensor of sums per group.
    """
    if validate:
        unique_group_indices, _ = tf.unique(group_indices)
        indices_range = tf.range(
            tf.squeeze(tf.shape(unique_group_indices, dtype=tf.int64))
        )
        tf.debugging.assert_equal(
            unique_group_indices, indices_range, message="Invalid group indices."
        )

    group_order = tf.argsort(group_indices)
    groups = tf.RaggedTensor.from_value_rowids(
        tf.gather(values, group_order),
        value_rowids=tf.gather(group_indices, group_order),
        nrows=tf.math.reduce_max(group_indices) + 1,
    )
    return tf.math.reduce_sum(groups, axis=-1)


@tf.function(experimental_relax_shapes=True)
def unique_indices(
    indices: tf.Tensor, validate: bool = False
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Finds the unique rows of in a 2D int64 tensor.
    Equivalent to row-wise tf.unique.

    Args:
        indices: 2-D int64 tensor of positive matrix indices.
        validate: If True the code raises and error if some indices are negative. Defaults to False.

    Returns:
        - 2-D int64 tensor of unique rows of matrix indices.
        - 1-D int64 tensor mapping unique indices rows to the input indices.
    """
    if validate:
        tf.debugging.assert_non_negative(
            indices, message="Negative indices in unique_indices."
        )
    min_shape = tf.math.reduce_max(indices, axis=0) + 1
    encoder_vector = tf.math.cumprod(min_shape, exclusive=True, reverse=True)
    encoded_indices = tf.math.reduce_sum(indices * encoder_vector, axis=-1)
    unique_encoded_indices, mapping = tf.unique(encoded_indices, out_idx=tf.int64)
    n_unique = tf.size(unique_encoded_indices, out_type=tf.int64)
    first_occurrence = tf.math.argmax(
        tf.math.equal(tf.range(n_unique)[:, tf.newaxis], mapping), axis=-1
    )
    unique_indices = tf.gather(indices, first_occurrence, axis=0)
    return unique_indices, mapping


@tf.function(experimental_relax_shapes=True)
def unique_indices_with_counts(
    indices: tf.Tensor, validate: bool = False
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Finds the unique rows of in a 2D int64 tensor and count them.
    Equivalent to row-wise tf.unique_with_counts.

    Args:
        indices: 2-D int64 tensor of positive matrix indices.
        validate: If True the code raises and error if some indices are negative. Defaults to False.

    Returns:
        - 2-D int64 tensor of unique rows of matrix indices.
        - 1-D int64 tensor mapping unique indices rows to the input indices.
        - 1-D int64 tensor counting the number of rows for every unique value.
    """
    if validate:
        tf.debugging.assert_non_negative(
            indices, message="Negative indices in unique_indices."
        )
    min_shape = tf.math.reduce_max(indices, axis=0) + 1
    encoder_vector = tf.math.cumprod(min_shape, exclusive=True, reverse=True)
    encoded_indices = tf.math.reduce_sum(indices * encoder_vector, axis=-1)
    unique_encoded_indices, mapping, counts = tf.unique_with_counts(
        encoded_indices, out_idx=tf.int64
    )
    n_unique = tf.size(unique_encoded_indices, out_type=tf.int64)
    first_occurrence = tf.math.argmax(
        tf.math.equal(tf.range(n_unique)[:, tf.newaxis], mapping), axis=-1
    )
    unique_indices = tf.gather(indices, first_occurrence, axis=0)
    return unique_indices, mapping, counts


@tf.function(experimental_relax_shapes=True)
def sparse_gather(
    sp_input: tf.SparseTensor,
    indices: tf.Tensor,
    validate: bool = False,
    implicit: bool = False,
    implicit_value: int = 0,
) -> tf.Tensor:
    """
    Gathers elements from a sparse tensor.

    Args:
        sp_input: n-D sparse tensor with ordered indices.
        indices: 2-D int64 tensor containing the list of indices to be gathered.
        validate: If True, then it checks that indices are valid for the dense shape of the sparse tensor.
            Defaults to False.
        implicit: If True, the value implicit_value is returned in correspondence of valid indices
            that do not explicitly appear in the sparse tensor. If False an error is raised.
            Defaults to False.
        implicit_value: When implicit is True, this value is used in correspondence of valid indices
            that do not explicitly appear in the sparse tensor. Defaults to 0.

    Returns:
        1-D tensor of gathered values.
    """
    encoder_vector = tf.math.cumprod(sp_input.dense_shape, exclusive=True, reverse=True)
    encoded_indices = tf.math.reduce_sum(indices * encoder_vector, axis=-1)
    if validate:
        tf.debugging.assert_less(
            indices,
            sp_input.dense_shape,
            message="Out-of-range indices in sparse_gather.",
        )
        tf.debugging.assert_non_negative(
            indices, message="Negative indices in sparse_gather."
        )
    encoded_sparse_indices = tf.math.reduce_sum(
        sp_input.indices * encoder_vector, axis=-1
    )
    search_indices = tf.searchsorted(
        encoded_sparse_indices, encoded_indices, out_type=tf.int64
    )
    gathered = tf.gather(sp_input.values, search_indices)
    in_range_mask = tf.math.less(
        search_indices, tf.size(encoded_sparse_indices, out_type=tf.int64)
    )
    search_indices = tf.where(in_range_mask, search_indices, 0)
    selected_sparse_indices = tf.gather(encoded_sparse_indices, search_indices)
    correspondence_mask = tf.math.equal(selected_sparse_indices, encoded_indices)
    mask = tf.math.logical_and(in_range_mask, correspondence_mask)
    if implicit:
        gathered = tf.where(mask, gathered, implicit_value)
    else:
        tf.debugging.assert_equal(
            mask,
            True,
            message="Some indices point to non-explicit entries in sparse tensor.",
        )
    mask = tf.ensure_shape(gathered, [None])
    return gathered


@tf.function(experimental_relax_shapes=True)
def select_entries(
    sp_input: tf.SparseTensor,
    entries_indices: tf.Tensor,
    entries_values: tf.Tensor,
) -> tf.SparseTensor:
    """
    Selects entries in a sparse tensor from a list of possible entries.

    Args:
        sp_input: n-D sparse tensor with ordered indices.
        entries_indices: 2-D int64 tensor of acceptable indices.
            Indices can be repeated. If that is not the case please use sparse_intersection.
        entries_values: 1-D tensor of acceptable values paired with corresponding indices.

    Returns:
        Sparse tensor with the same shape of the input sparse tensor
        containing a subset of the original entries.
    """
    n_values = tf.size(entries_values, out_type=tf.int64)
    acceptable_set = tf.sparse.SparseTensor(
        indices=tf.concat(
            [
                entries_indices,
                tf.range(n_values)[:, tf.newaxis],
            ],
            axis=-1,
        ),
        values=entries_values,
        dense_shape=tf.concat([sp_input.dense_shape, n_values[tf.newaxis]], axis=0),
    )
    acceptable_set = tf.sparse.reorder(acceptable_set)
    selection = tf.sets.intersection(tf.sparse.expand_dims(sp_input), acceptable_set)
    selection = tf.sparse.reset_shape(
        selection, tf.concat([sp_input.dense_shape, [1]], axis=0)
    )
    selection = tf.sparse.reshape(selection, selection.dense_shape[:-1])
    return selection


@tf.function(experimental_relax_shapes=True)
def sparse_intersection(
    sp_input1: tf.SparseTensor, sp_input2: tf.SparseTensor
) -> tf.SparseTensor:
    """
    Performs the intersection between two sparse tensors.

    Args:
        sp_input1: n-D sparse tensor with ordered indices.
        sp_input2: n-D sparse tensor of the same shape and type of sp_input1 with ordered indices.

    Returns:
        Sparse tensor of the common entries in the input sparse tensors.
    """
    intersection = tf.sets.intersection(
        tf.sparse.expand_dims(sp_input1), tf.sparse.expand_dims(sp_input2)
    )
    intersection = tf.sparse.reset_shape(
        intersection,
        tf.concat([sp_input1.dense_shape, [1]], axis=0),
    )
    intersection = tf.sparse.reshape(intersection, intersection.dense_shape[:-1])
    return intersection


@tf.function(experimental_relax_shapes=True)
def indices_in_sparse(
    indices: tf.Tensor,
    sp_input: tf.SparseTensor,
) -> tf.Tensor:
    """
    Checks if the given indices appears in a sparse tensor.

    Args:
        indices: 2-D int64 tensor of indices.
            Indices can be repeated and in arbitrary order.
        sp_input: n-D sparse tensor with ordered indices.

    Returns:
        1-D bool tensor of the same length of indices with True
        in correspence of indices found in the sparse tensor.
    """
    if tf.math.equal(tf.size(sp_input.values), 0):
        return tf.ensure_shape(tf.zeros(tf.shape(indices)[0], dtype=tf.bool), [None])
    encoder_vector = tf.math.cumprod(sp_input.dense_shape, exclusive=True, reverse=True)
    encoded_indices = tf.math.reduce_sum(indices * encoder_vector, axis=-1)
    encoded_sparse_indices = tf.math.reduce_sum(
        sp_input.indices * encoder_vector, axis=-1
    )
    search_indices = tf.searchsorted(
        encoded_sparse_indices, encoded_indices, out_type=tf.int64
    )
    upper_bound_mask = tf.math.less_equal(encoded_indices, encoded_sparse_indices[-1])
    search_indices = tf.where(upper_bound_mask, search_indices, 0)
    selected_indices = tf.gather(encoded_sparse_indices, search_indices)
    indices_mask = tf.math.equal(selected_indices, encoded_indices)
    mask = tf.logical_and(upper_bound_mask, indices_mask)
    mask = tf.ensure_shape(mask, [None])
    return mask


@tf.function(experimental_relax_shapes=True)
def entries_in_sparse(
    entries_indices: tf.Tensor,
    entries_values: tf.Tensor,
    sp_input: tf.SparseTensor,
    subset: bool = False,
) -> tf.Tensor:
    """
    Checks if the given entries appears in a sparse tensor.

    Args:
        entries_indices: 2-D int64 tensor of entries indices.
            Indices can be repeated and in arbitrary order.
        entries_values: 1-D tensor of entries values associated to the indices.
        sp_input: n-D sparse tensor with ordered indices of the same type of entries_values.
        subset: If True it is assumed that all entries indices appear among
            the indices of the sparse tensor.

    Returns:
        1-D bool tensor of the same shape of entries_values with True
        in correspence of entries found in the sparse tensor.
    """
    if tf.math.equal(tf.size(sp_input.values), 0):
        return tf.ensure_shape(tf.zeros_like(entries_values, dtype=tf.bool), [None])
    encoder_vector = tf.math.cumprod(sp_input.dense_shape, exclusive=True, reverse=True)
    encoded_entries_indices = tf.math.reduce_sum(
        entries_indices * encoder_vector, axis=-1
    )
    if not subset:
        lower_bound_mask = tf.math.reduce_all(
            tf.math.greater_equal(entries_indices, 0), axis=-1
        )
        upper_bound_mask = tf.math.reduce_all(
            tf.math.less(entries_indices, sp_input.dense_shape), axis=-1
        )
        valid_mask = tf.math.logical_and(lower_bound_mask, upper_bound_mask)
        encoded_entries_indices = tf.where(valid_mask, encoded_entries_indices, -1)
    encoded_sparse_indices = tf.math.reduce_sum(
        sp_input.indices * encoder_vector, axis=-1
    )
    search_indices = tf.searchsorted(
        encoded_sparse_indices, encoded_entries_indices, out_type=tf.int64
    )
    acceptable_indices_mask = tf.math.less(
        search_indices, tf.size(encoded_sparse_indices, out_type=tf.int64)
    )
    search_indices = tf.where(acceptable_indices_mask, search_indices, 0)
    selected_values = tf.gather(sp_input.values, search_indices)
    mask = tf.math.equal(selected_values, entries_values)
    if not subset:
        selected_indices = tf.gather(encoded_sparse_indices, search_indices)
        indices_mask = tf.math.equal(selected_indices, encoded_entries_indices)
        mask = tf.logical_and(mask, indices_mask)
    mask = tf.ensure_shape(mask, [None])
    return mask
