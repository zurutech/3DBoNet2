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

"""Dataset configuration db."""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


class DataConfig:
    """Dataset configuration."""

    def __init__(
        self,
        name: str,
        feature_dim: int,
        max_instances: int,
        block_points: int,
        block_size: float,
        block_stride: float,
        coordinates_slice_limits: Tuple[int, int],
        semantic_classes: Sequence[str],
        semantic_indices: Optional[Sequence[int]] = None,
        train_patterns: Sequence[str] = tuple(),
        eval_patterns: Sequence[str] = tuple(),
        subsampler_voxel_size: Optional[float] = None,
        feature_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """[summary]

        Args:
            name: The dataset name, used also for file caching.
            feature_dim: Feature dimension.
            max_instances: Maximum number of instances per block.
            block_points: Number of points per block.
            block_size: Spatial dimension of a block.
            block_stride: Spatial stride between consecutive blocks.
            coordinates_slice_limits: Limits of the slice of features used to identify point locations in a block.
            semantic_classes: List of semantic class names.
            semantic_indices: Optional list of indices associated to the semantic classes.
                Defaults to None. If not specified the indices are deduced from the ordered list of semantic classes.
            train_patterns: Sequence of glob patterns used to select training files.
                Defaults to tuple() and in such case all HDF5 files that are not evaluation files are used.
            eval_patterns: Sequence of glob patterns used to select evaluation files.
                Defaults to tuple() and in such case no file is used for evaluation.
            subsampler_voxel_size: Size of the voxels used to subsample the point cloud.
                Defaults to None and in such case the sampling is disabled.
            feature_processor: Function to be applied to raw pointcloud features. Defaults to None.

        Raises:
            ValueError: If both `semantic_classes` and `semantic_indices` are provided,
                but they have different lengths.
        """
        self.name = name
        self.feature_dim = feature_dim
        self.max_instances = max_instances
        self.block_points = block_points
        self.block_size = block_size
        self.block_stride = block_stride
        self.coordinates_slice = slice(
            coordinates_slice_limits[0], coordinates_slice_limits[1]
        )
        self.semantic_classes = semantic_classes
        self.semantic_indices = semantic_indices
        if semantic_indices is not None and len(semantic_classes) != len(
            semantic_indices
        ):
            raise ValueError(
                "The number of given semantic indices does not correspond to the number of semantic classes."
            )
        self.n_classes = len(semantic_classes)
        self.train_patterns = train_patterns
        self.eval_patterns = eval_patterns
        self.subsampler_voxel_size = subsampler_voxel_size
        self.feature_processor = feature_processor
