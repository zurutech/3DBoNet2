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
"""Transformation to split a point cloud into several fixed-sized blocks."""

import numpy as np


class BlockSplitter:
    """Transformation to split a point cloud into several fixed-sized blocks.

    Shape conventions:
    - N: Number of points in every block (usually 4096).
    - B: Number of blocks or batch size.
    - F: Number of point features.
    """

    def __init__(
        self,
        block_points: int,
        block_size: float = 1.0,
        block_stride: float = 0.5,
        threshold: int = 100,
        seed: int = 42,
    ):
        """
        Constructor of a BlockSplitter object.

        Args:
            block_points: Number of points in every block.
            block_size: Spatial dimension of a block. Defaults to 1.0.
            block_stride: Spatial stride between consecutive blocks. Defaults to 0.5.
            threshold: Minimum number of points in a block to accept it. Defaults to 100.
            seed: Random seed. Defaults to 42.
        """
        self.block_points = block_points
        self.block_size = block_size
        self.block_stride = block_stride
        self.threshold = threshold
        self._random_source = np.random.default_rng(seed)

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        Extract blocks of fixed size from a point cloud.

        Args:
            pointcloud: Input point cloud. Shape (N, 3 + F).

        Returns:
            Batch of blocks with shape (B, N, 9 + F + 2) where:
            [0:3] - global coordinates
            [3:6] - block normalized coordinates
            [6:9] - room normalized coordinates
            [9:9+F] - features
            [9+F:9+F+2] - labels (semantic and instance)
        """
        cloud_min = np.amin(pointcloud[:, :3], axis=0)
        cloud_max = np.amax(pointcloud[:, :3], axis=0)
        cloud_size = np.maximum(cloud_max - cloud_min, 1e-3)
        n_cells = np.maximum(np.ceil(cloud_size / self.block_stride) - 1, 1)
        x_cells = np.arange(n_cells[0]) * self.block_stride + cloud_min[0]
        y_cells = np.arange(n_cells[1]) * self.block_stride + cloud_min[1]
        cells = np.transpose(
            [np.tile(x_cells, y_cells.size), np.repeat(y_cells, x_cells.size)]
        )
        blocks = []
        for cell in cells:
            block_mask = np.all(
                np.logical_and(
                    pointcloud[:, :2] <= cell + self.block_size,
                    pointcloud[:, :2] >= cell,
                ),
                axis=1,
            )
            n_points_in_block = np.sum(block_mask)
            if n_points_in_block < self.threshold:
                continue
            block_points = pointcloud[block_mask]
            repetitions = self.block_points // n_points_in_block
            if repetitions > 0:
                block_points = np.tile(block_points, [repetitions, 1])
            if block_points.shape[0] % self.block_points != 0:
                n_missing = (
                    self.block_points - block_points.shape[0] % self.block_points
                )
                sampled_points = block_points[
                    self._random_source.choice(
                        n_points_in_block, n_missing, replace=False
                    )
                ]
                block_points = np.concatenate((block_points, sampled_points), axis=0)
            blocks.extend(
                np.split(block_points, block_points.shape[0] // self.block_points)
            )
        num_blocks = len(blocks)
        batch = np.zeros(
            (num_blocks, self.block_points, pointcloud.shape[1] + 6), dtype=np.float32
        )
        if num_blocks != 0:
            blocks = np.stack(blocks, axis=0)
            block_min = np.min(pointcloud[..., :3], axis=-2, keepdims=True)
            block_max = np.max(pointcloud[..., :3], axis=-2, keepdims=True)
            block_size = np.maximum((block_max - block_min), 1e-3)
            batch[:, :, 0:3] = blocks[:, :, 0:3]
            batch[:, :, 3:6] = (blocks[:, :, 0:3] - block_min) / block_size
            batch[:, :, 6:9] = (blocks[:, :, 0:3] - cloud_min) / cloud_size
            batch[:, :, 9:] = blocks[:, :, 3:]
        return batch
