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
"""A PlyInferenceDataset manages ply files with no labels for inference"""

import math
from pathlib import Path
from typing import Union

import numpy as np
import numpy.lib.recfunctions as rfn
import tensorflow as tf
from plyfile import PlyData
from tqdm import tqdm

from .data_config import DataConfig
from .block_splitter import BlockSplitter


class PlyInferenceDataset:
    """
    A PlyInferenceDataset manages ply files with no labels for inference

    Shape conventions:
    - N: Number of points in the point cloud.
    - F: Number of point features.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        config: DataConfig,
        batch_size: int,
    ):
        """
        Constructor of a PlyInferenceDataset object.

        Args:
            dataset_path: Path to the root directory of the dataset.
            config: Dataset configuration.
            batch_size: Batch size.
        """
        self.dataset_path = Path(dataset_path).expanduser()
        self.config = config
        self.batch_size = batch_size
        self.block_splitter = BlockSplitter(
            self.config.block_points, self.config.block_size, self.config.block_stride
        )

        self.paths = tuple(self.dataset_path.glob("**/*.ply"))
        if len(self.paths) == 0:
            raise ValueError("No Ply file found.")

    def load_ply(
        self,
        pointcloud_path: Path,
    ) -> np.ndarray:
        """
        Load and process a point cloud from a Ply file.

        Args:
            pointcloud_path: Ply file path.

        Returns:
            Coordinates and features of the point cloud. Shape (N, 3 + F).
        """
        if not pointcloud_path.exists():
            raise FileNotFoundError(
                f"The point cloud file {pointcloud_path} is missing."
            )
        ply_content = PlyData.read(pointcloud_path)
        pointcloud = rfn.structured_to_unstructured(
            ply_content["vertex"].data, dtype=np.float32
        )
        coordinates, features = pointcloud[:, :3], pointcloud[:, 3:]
        features = self.config.feature_processor(features)
        return np.column_stack((coordinates, features))

    def get_named_block_dataset(self) -> tf.data.Dataset:
        """
        Obtain a tf.data.Dataset to iterate over batches of point cloud blocks extracted
        from a point cloud loaded from a .ply file.
        After all the batches of blocks from the same point cloud, the dataset produces
        a single batch with the 3-D coordinates of the full point cloud for the final predictions.

        Returns:
            If no data is available None, otherwise a tf.data.Dataset of tuples composed of:
            - path of the original point cloud .ply file;
            - index of batch of the current point cloud;
            - number of batches for the current point cloud;
            - batch of point cloud blocks or 3-D coordinates of the full point cloud.
        """
        datasets = None
        for path in tqdm(self.paths, desc="Loading point clouds"):
            pointcloud = self.load_ply(path)
            coordinates = pointcloud[:, :3]
            pointcloud_blocks = self.block_splitter(pointcloud)
            # Remove global coordinates
            pointcloud_blocks = pointcloud_blocks[:, :, 3:]

            dataset = tf.data.Dataset.from_tensor_slices(pointcloud_blocks)
            dataset = dataset.batch(self.batch_size)
            n_batches = math.ceil(len(pointcloud_blocks) / self.batch_size)
            dataset = dataset.enumerate()
            dataset = dataset.map(
                lambda c, e: (path.name, tf.cast(c, dtype=tf.int32), n_batches, e),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

            # Compute room normalized coordinates
            cloud_min = np.amin(coordinates, axis=0)
            cloud_max = np.amax(coordinates, axis=0)
            cloud_size = np.maximum(cloud_max - cloud_min, 1e-3)
            room_coordinates = (coordinates - cloud_min) / cloud_size
            coordinates_dataset = tf.data.Dataset.from_tensors(
                (path.name, n_batches, n_batches, room_coordinates)
            )
            dataset = dataset.concatenate(coordinates_dataset)
            datasets = dataset if datasets is None else datasets.concatenate(dataset)
        datasets = datasets.prefetch(tf.data.AUTOTUNE)
        return datasets
