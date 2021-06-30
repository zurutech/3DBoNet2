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
"""A CloudBlocksDataset object manages a dataset of HDF5 files."""

import itertools
import math
from pathlib import Path
from typing import Optional, Iterator, Tuple, Union

import h5py
import numpy as np
import tensorflow as tf

from .data_config import DataConfig


class CloudBlocksDataset:
    """
    A CloudBlocksDataset object manages a dataset of HDF5 files,
    where every HDF5 files contains a point cloud splitted in possibly overlapping blocks
    with a fixed number of points (usually 4096 points).

    Shape conventions:
    - N: Number of points in every block (usually 4096).
    - B: Number of blocks or batch size.
    - F: Number of point features.
    - L: Number of semantic segmentation classes.
    - H: Maximum number of supported instances in a point cloud block.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        config: DataConfig,
        batch_size: int,
        eval_batch_size: Optional[int] = None,
    ):
        """
        Constructor of a CloudBlocksDataset object.

        Args:
            dataset_path: Path to the root directory of the dataset.
            config: Dataset configuration.
            batch_size: Batch size used for training.
            eval_batch_size: Batch size used for evaluation.
               Defaults to None and in such case it is used the same batch size used for training.
        """
        self.dataset_path = Path(dataset_path).expanduser()
        self.name = config.name
        self.n_classes = config.n_classes
        self.max_instances = config.max_instances
        self.feature_dim = config.feature_dim
        self.coordinates_slice = config.coordinates_slice
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size

        self._cache_path = Path("dataset_cache") / self.name
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.eval_paths = tuple(
            itertools.chain.from_iterable(
                self.dataset_path.glob(f"{eval_pattern}*.h5")
                for eval_pattern in config.eval_patterns
            )
        )
        if not config.train_patterns:
            self.train_paths = tuple(
                p for p in self.dataset_path.glob("*.h5") if p not in self.eval_paths
            )
        else:
            self.train_paths = tuple(
                itertools.chain.from_iterable(
                    self.dataset_path.glob(f"{train_pattern}*.h5")
                    for train_pattern in config.train_patterns
                )
            )
        self._average_instance_sizes = None

    @staticmethod
    def pointcloud_loader(
        file_path: bytes,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the entire point cloud from a HDF5 file.

        Args:
            file_path: HDF5 file path encoded with UTF-8 encoding.

        Returns:
            - Original coordinates of the point cloud useful for visualization. Shape (B, N, 3).
            - Processed point cloud. Shape (B, N, F).
            - Semantic segmentation labels. Shape (B, N).
            - Instance segmentation labels. Shape (B, N).
        """
        path = Path(file_path.decode("utf-8"))
        h5f = h5py.File(path, "r")
        coords = h5f["coords"][:]
        pointcloud = h5f["points"][:]
        semantic_labels = h5f["labels"][:, :, 0].astype(np.int32)
        instance_labels = h5f["labels"][:, :, 1].astype(np.int32)
        return coords, pointcloud, semantic_labels, instance_labels

    @property
    def average_instance_sizes(self) -> np.ndarray:
        """
        Compute the average number of points belonging to instances from the same class.

        Returns:
            Average number of points belonging to instances from the same class. Shape (L,).
        """
        if self._average_instance_sizes is None:
            instance_points_counter = np.zeros(self.n_classes, dtype=np.int32)
            instances_counter = np.zeros(self.n_classes, dtype=np.int32)
            for path in self.train_paths:
                h5f = h5py.File(path, "r")
                segmentation_labels = h5f["labels"][:].reshape([-1, 2])
                semantic_labels = segmentation_labels[:, 0]
                instance_labels = segmentation_labels[:, 1]
                unique_results = np.unique(
                    instance_labels, return_index=True, return_counts=True
                )
                for instance_label, instance_index, instance_size in zip(
                    *unique_results
                ):
                    if instance_label < 0:
                        continue
                    semantic_label = semantic_labels[instance_index]
                    instance_points_counter[semantic_label] += instance_size
                    instances_counter[semantic_label] += 1
            instances_counter = np.maximum(instances_counter, 1)
            self._average_instance_sizes = instance_points_counter / instances_counter
        return self._average_instance_sizes

    def count_batches(self, train: bool) -> int:
        """
        Count the number of batches in the dataset training or eval splits.

        Args:
            train: Whether the count is relative to the training split or the eval split.

        Raises:
            ValueError: If points from the same instance have not the same semantic label.

        Returns:
            Number of batches.
        """
        n_blocks = 0
        file_paths = self.train_paths if train else self.eval_paths
        for path in file_paths:
            h5f = h5py.File(path, "r")
            segmentation_labels = h5f["labels"][:].reshape([-1, 2])
            semantic_labels = segmentation_labels[:, 0]
            instance_labels = segmentation_labels[:, 1]

            unique_instance_labels = np.unique(instance_labels)
            for label in unique_instance_labels:
                if label < 0:
                    continue
                label_mask = instance_labels == label
                masked_labels = semantic_labels[label_mask]
                if not np.all(masked_labels == masked_labels[0]):
                    raise ValueError("Inconsistent instance and segmentation labels")
            n_blocks += h5f["coords"].shape[0]
        batch_size = self.batch_size if train else self.eval_batch_size
        return math.ceil(n_blocks / batch_size)

    def block_generator(
        self, train: bool
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generator that loops over point cloud blocks from a dataset split.

        Args:
            train: Whether the iterator loops over the training split or the eval split.

        Yields:
            - Processed point cloud block. Shape (N, F).
            - Semantic segmentation labels. Shape (N,).
            - Vertices of instance bounding boxes. Shape (H, 2, 3).
            - Boolean masks indicating the membership of points in instances. Shape (H, N).
        """
        file_paths = self.train_paths if train else self.eval_paths
        for path in file_paths:
            h5f = h5py.File(path, "r")
            n_blocks = h5f["coords"].shape[0]
            for block_id in range(n_blocks):
                points = h5f["points"][block_id]
                semantic_labels = h5f["labels"][block_id, :, 0].astype(np.int32)
                instance_labels = h5f["labels"][block_id, :, 1].astype(np.int32)
                vertices, point_masks = self.instance_vertices_and_mask(
                    points, instance_labels
                )
                yield points, semantic_labels, vertices, point_masks

    def get_block_dataset(self, train: bool) -> Tuple[tf.data.Dataset, tf.Tensor]:
        """
        Obtain a tf.data.Dataset to iterate over blocks from a dataset split.

        Args:
            train: Whether the tf.data.Dataset loops over the training split or the eval split.

        Returns:
            - Dataset of batches of point cloud blocks from a data split.
            - Number of batches in the dataset split.
        """
        n_batches = self.count_batches(train)
        dataset = tf.data.Dataset.from_generator(
            self.block_generator,
            output_signature=(
                tf.TensorSpec(shape=[None, self.feature_dim], dtype=tf.float32),
                tf.TensorSpec(shape=[None], dtype=tf.int64),
                tf.TensorSpec(shape=[self.max_instances, 2, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[self.max_instances, None], dtype=tf.float32),
            ),
            args=(train,),
        )
        dataset = dataset.cache(str(self._cache_path))
        if train:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
        batch_size = self.batch_size if train else self.eval_batch_size
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, tf.convert_to_tensor(n_batches, dtype=tf.int64)

    def instance_vertices_and_mask(
        self, pointcloud: np.ndarray, instance_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtain instance bounding boxes and membership boolean masks from instance labels.

        Args:
            pointcloud: Processed point cloud block. Shape (N, F).
            instance_labels: Block instance segmentation labels. Shape (N,).

        Returns:
            - Vertices of instance bounding boxes. Shape (H, 2, 3).
            - Boolean masks indicating the membership of points in instances. Shape (H, N).
        """
        vertices = np.zeros((self.max_instances, 2, 3), dtype=np.float32)
        point_masks = np.zeros(
            (self.max_instances, pointcloud.shape[0]), dtype=np.float32
        )
        unique_labels, label_count = np.unique(instance_labels, return_counts=True)
        valid_mask = unique_labels != -1
        unique_labels = unique_labels[valid_mask]
        label_count = label_count[valid_mask]
        if unique_labels.size > self.max_instances:
            print(
                f"A block contains {unique_labels.size} instances. "
                f"The smallest {unique_labels.size - self.max_instances} instances are ignored."
            )
            sorted_count_indices = np.argsort(-label_count)
            unique_labels = unique_labels[sorted_count_indices[: self.max_instances]]
        xyz = pointcloud[:, :3]
        for idx, label in enumerate(unique_labels):
            instance_mask = instance_labels == label
            point_masks[idx, :] = instance_mask
            vertices[idx, 0] = np.min(xyz[instance_mask], axis=0)
            vertices[idx, 1] = np.max(xyz[instance_mask], axis=0)
        return vertices, point_masks

    def get_pointcloud_dataset(self, train: bool) -> tf.data.Dataset:
        """
        Obtain a tf.data.Dataset to iterate over point clouds from a dataset split.

        Args:
            train:  Whether the tf.data.Dataset loops over the training split or the eval split.

        Returns:
            Dataset of point clouds (divided into blocks) from a data split.
        """
        file_paths = self.train_paths if train else self.eval_paths
        dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in file_paths])
        dataset = dataset.map(
            lambda p: tf.numpy_function(
                CloudBlocksDataset.pointcloud_loader,
                [p],
                (tf.float32, tf.float32, tf.int32, tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        if train:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
        return dataset

    def get_indexed_block_dataset(self, train: bool) -> Optional[tf.data.Dataset]:
        """
        Obtain a tf.data.Dataset to iterate over indexed batches of point cloud blocks.
        A batch contains only blocks from the same point cloud.

        Args:
            train:  Whether the point clouds belong to the training split or the eval split.

        Returns:
            If no data is available None, otherwise a tf.data.Dataset of triplets composed of:
            - index of batch of the current point cloud;
            - number of batches for the current point cloud;
            - batch of point cloud blocks.
        """
        file_paths = self.train_paths if train else self.eval_paths
        datasets = None
        for path in file_paths:
            pointcloud_blocks = CloudBlocksDataset.pointcloud_loader(bytes(path))
            dataset = tf.data.Dataset.from_tensor_slices(pointcloud_blocks)
            if train:
                dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
            batch_size = self.batch_size if train else self.eval_batch_size
            dataset = dataset.batch(batch_size)
            n_batches = math.ceil(len(pointcloud_blocks[0]) / batch_size)
            dataset = dataset.enumerate()
            dataset = dataset.map(
                lambda c, e: (tf.cast(c, dtype=tf.int32), n_batches, e),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            if datasets is None:
                datasets = dataset
            else:
                datasets = datasets.concatenate(dataset)
        if datasets is not None:
            datasets = datasets.prefetch(tf.data.AUTOTUNE)
        return datasets
