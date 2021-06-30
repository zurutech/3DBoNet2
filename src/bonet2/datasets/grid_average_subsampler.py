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
"""Transformation to subsample a point cloud averaging points in the same voxel."""

from typing import Union, Optional, Tuple
from collections import Counter

import numpy as np


class GridAverageSubsampler:
    """Transformation to subsample a point cloud averaging points in the same voxel.

    Shape conventions:
    - N: Number of points in the point cloud.
    - M: Number of active voxels.
    - F: Number of point features.
    """

    def __init__(self, voxel_size: float = 0.01) -> None:
        """
        Constructor of a GridAverageSubsampler object.

        Args:
            voxel_size: Size of a voxel in the sub-sampling grid. Defaults to 0.01.
        """
        self.voxel_size = voxel_size

    def __call__(
        self, pointcloud: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Aggregate points, and associated labels if provided, to keep a single point per voxel.
        Points in the same voxel are aggregated through averaging.
        Voxel labels are the most frequent label associated to points belonging to the voxel.

        Args:
            pointcloud: Input point cloud. Shape (N, 3 + F).
            labels: Point cloud labels. Defaults to None.

        Returns:
            Voxel coordinates and features obtained through averaging. Shape (M, 3 + F).
            If labels are provided, the additional array of aggregated labels is returned.
        """
        coordinates = pointcloud[:, :3]
        voxel_coordinates = np.floor(coordinates / self.voxel_size)
        order = np.lexsort(np.fliplr(voxel_coordinates).T)
        group_splits = np.unique(voxel_coordinates[order], axis=0, return_index=True)[
            1
        ][1:]

        ordered_pointcloud = pointcloud[order]
        pointcloud_groups = np.split(ordered_pointcloud, group_splits)
        pointcloud_groups = np.stack(
            [np.mean(group, axis=0) for group in pointcloud_groups], axis=0
        )
        if labels is None:
            return pointcloud_groups

        ordered_labels = labels[order]
        labels_groups = np.split(ordered_labels, group_splits)
        labels_groups = np.stack(
            [Counter(map(tuple, group)).most_common()[0][0] for group in labels_groups],
            axis=0,
        )

        return pointcloud_groups, labels_groups
