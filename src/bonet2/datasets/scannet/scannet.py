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
"""Scannet dataset."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
from plyfile import PlyData
from tqdm import tqdm


from ..dataset import DataConfig, Dataset
from ..block_splitter import BlockSplitter
from ..grid_average_subsampler import GridAverageSubsampler
from .download import download_dataset


class ScanNet(Dataset):
    """ScanNet dataset."""

    def __init__(self):
        self._config = DataConfig(
            name="scannet",
            feature_dim=9,
            max_instances=24,
            block_points=4096,
            block_size=2.0,
            block_stride=1.0,
            coordinates_slice_limits=(3, 6),
            semantic_classes=(
                "wall",
                "floor",
                "cabinet",
                "bed",
                "chair",
                "sofa",
                "table",
                "door",
                "window",
                "bookshelf",
                "picture",
                "counter",
                "desk",
                "curtain",
                "refridgerator",
                "shower curtain",
                "toilet",
                "sink",
                "bathtub",
                "otherfurniture",
            ),
            semantic_indices=(
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                14,
                16,
                24,
                28,
                33,
                34,
                36,
                39,
            ),
            # These evaluation scenes are the same used by 3D-BoNet authors.
            eval_patterns=(
                "scene0015_00",
                "scene0077_01",
                "scene0100_02",
                "scene0196_00",
                "scene0256_02",
                "scene0342_00",
                "scene0025_01",
                "scene0081_00",
                "scene0131_00",
                "scene0203_00",
                "scene0278_01",
                "scene0351_01",
                "scene0030_00",
                "scene0081_02",
                "scene0144_01",
                "scene0207_00",
                "scene0307_00",
                "scene0353_01",
                "scene0046_01",
                "scene0084_02",
                "scene0149_00",
                "scene0208_00",
                "scene0314_00",
                "scene0355_00",
                "scene0050_01",
                "scene0086_01",
                "scene0164_01",
                "scene0221_01",
                "scene0328_00",
                "scene0378_00",
                "scene0063_00",
                "scene0088_00",
                "scene0169_01",
                "scene0222_01",
                "scene0329_01",
                "scene0064_00",
                "scene0095_01",
                "scene0187_00",
                "scene0231_00",
                "scene0338_01",
            ),
            subsampler_voxel_size=0.05,
            feature_processor=(lambda x: x[..., :-1] / 255.0),
        )

        self._raw_dataset_path = Path(f"{self.config.name}_raw")
        self._h5_dataset_path = Path(f"{self.config.name}_h5")

    @property
    def config(self) -> DataConfig:
        """Return the ScanNet DataConfig."""
        return self._config

    @property
    def h5_path(self) -> Path:
        """Return the path of the folder containing the h5 version of the dataset."""
        return self._h5_dataset_path

    def download(self) -> None:
        """Download the dataset from the internet and save it (raw)
        in dest."""
        if self._raw_dataset_path.exists():
            return
        download_dataset(self._raw_dataset_path)

    def convert(self) -> None:
        """Convert the raw ScanNet dataset to h5 format, put it in dest."""
        if self._h5_dataset_path.exists():
            return

        self._convert_to_h5(self._raw_dataset_path, self._h5_dataset_path)

    def read_label_mapping(self, label_mapping_path: Path) -> Dict[str, int]:
        """
        Reads the label mapping file and returns its Python representation.
        This method uses the config property.

        Args:
            label_mapping_path: Path to the label mapping .tsv file.

        Returns:
            Dictionary mapping semantic class names to indices.
        """
        with label_mapping_path.open() as tsv:
            reader = csv.DictReader(tsv, delimiter="\t")
            mapping = {}
            for row in reader:
                if row["nyu40class"] in self.config.semantic_classes:
                    label_id = self.config.semantic_classes.index(row["nyu40class"])
                else:
                    label_id = -1
                mapping[row["raw_category"]] = label_id
        return mapping

    @staticmethod
    def rotate_pointcloud(
        pointcloud_vertices: np.ndarray, pointcloud_path: Path
    ) -> np.ndarray:
        """
        Rotate the ScanNet point cloud using the axisAlignment data provided by the dataset.

        Args:
            pointcloud_vertices: A matrix with only vertex values.
            pointcloud_path: The pointcloud path from which extract the txt filename.

        Returns:
            The rotated point cloud.
        """
        if pointcloud_vertices.shape[1] > 3:
            raise FileNotFoundError(
                f"The point cloud should have only x,y and z values."
            )

        # Open the .txt file and get the axisAligment data
        with open(str(pointcloud_path).rsplit("_", 3)[0] + ".txt", "r") as txtfile:
            lines = txtfile.readlines()
        for line in lines:
            line = line.split()
            if line[0] == "axisAlignment":
                align_mat = np.array(
                    [float(x) for x in line[2:]], dtype=np.float32
                ).reshape([4, 4])
                break
        R = align_mat[:3, :3]
        T = align_mat[:3, 3]

        # Rotate the point cloud
        rotated_point_cloud_vertices = pointcloud_vertices.dot(R.T) + T

        return rotated_point_cloud_vertices

    def read_pointcloud(self, pointcloud_path: Path) -> np.ndarray:
        """
        Read the pointcloud from the pointcloud_path and return the pointcloud in XYZRGB format.

        Args:
            pointcloud_path: The pointcloud path.

        Returns:
            The pointcloud in XYZRGB format.
        """
        ply_content = PlyData.read(pointcloud_path)
        pointcloud = rfn.structured_to_unstructured(
            ply_content["vertex"].data, dtype=np.float32
        )
        coordinates, features = pointcloud[:, :3], pointcloud[:, 3:]
        features = self.config.feature_processor(features)

        # Rotate the pointcloud point cloud
        rotated_coordinates = ScanNet.rotate_pointcloud(coordinates, pointcloud_path)
        return np.column_stack((rotated_coordinates, features))

    @staticmethod
    def read_aggregation(
        aggregation_path: Path, label_mapping: Dict[str, int]
    ) -> Tuple[Dict[int, List[int]], Dict[str, List[int]]]:
        """
        Read ScanNet aggregation file.

        Args:
            aggregation_path: Path to aggregation file.
            label_mapping: Dictionary mapping semantic class names to indices.

        Returns:
            Tuple composed of:
            - a dictionary mapping objects to segments;
            - a dictionary mapping labels to segments.
        """
        object2segments = {}
        label2segments = {}
        with aggregation_path.open() as file:
            data = json.load(file)
        for instance in data["segGroups"]:
            label = label_mapping[instance["label"]]
            if label != -1:
                object_id = instance["objectId"]
                segments = instance["segments"]
                object2segments[object_id] = segments
                label2segments.setdefault(label, []).extend(segments)
        return object2segments, label2segments

    @staticmethod
    def read_segments(segments_path: Path) -> Dict[str, List[int]]:
        """
        Read ScanNet segments file.

        Args:
            segments_path: Path to segments file.

        Returns:
            Dictionary mapping segments to vertices.
        """
        segments2vertices = {}
        with segments_path.open() as file:
            data = json.load(file)
        for vertex_index, segment in enumerate(data["segIndices"]):
            segments2vertices.setdefault(segment, []).append(vertex_index)
        return segments2vertices

    def get_labeled_pointcloud(
        self, scan_directory: Path, label_mapping: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the point cloud and associated labels.

        Args:
            scan_directory: Path to the directory of a single scan.
            label_mapping: Dictionary mapping semantic class names to indices.

        Returns:
            Tuple of `np.array`s composed of:
            - loaded point cloud;
            - loaded semantic labels;
            - loaded instance labels.
        """
        scan_id = scan_directory.name
        pointcloud_path = scan_directory / f"{scan_id}_vh_clean_2.ply"
        aggregation_path = scan_directory / f"{scan_id}.aggregation.json"
        segments_path = scan_directory / f"{scan_id}_vh_clean_2.0.010000.segs.json"
        pointcloud = self.read_pointcloud(pointcloud_path)
        instances2segments, semantic2segments = ScanNet.read_aggregation(
            aggregation_path, label_mapping
        )
        segments2vertices = ScanNet.read_segments(segments_path)

        semantic_labels = -np.ones(shape=(pointcloud.shape[0],), dtype=np.int32)
        for semantic_label, segments in semantic2segments.items():
            vertices = [v for segment in segments for v in segments2vertices[segment]]
            semantic_labels[vertices] = semantic_label

        instance_labels = -np.ones(shape=(pointcloud.shape[0],), dtype=np.int32)
        for instance_label, segments in instances2segments.items():
            vertices = [v for segment in segments for v in segments2vertices[segment]]
            instance_labels[vertices] = instance_label
        return pointcloud, semantic_labels, instance_labels

    def _convert_to_h5(self, dataset_dir: Path, out_dir: Path):
        """
        Convert the raw dataset to its h5 version splitting the point cloud into blocks.

        Args:
            dataset_dir: Path to the dataset directory.
            out_dir: Path to the output directory.

        Raises:
            ValueError: If the code produces an invalid semantic index.
        """
        dataset_dir = Path(dataset_dir).expanduser()
        output_dir = Path(out_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        scans_dir = dataset_dir / "scans"
        label_mapping_path = dataset_dir / "scannetv2-labels.combined.tsv"

        label_mapping = self.read_label_mapping(label_mapping_path)
        if self.config.subsampler_voxel_size:
            subsampler = GridAverageSubsampler(
                voxel_size=self.config.subsampler_voxel_size
            )
        block_splitter = BlockSplitter(
            self.config.block_points,
            block_size=self.config.block_size,
            block_stride=self.config.block_stride,
        )
        for scan_dir in tqdm(
            sorted(scans_dir.glob("scene*/")), desc="ScanNet conversion"
        ):
            h5_path = output_dir / f"{scan_dir.name}.h5"
            if h5_path.exists():
                print(f"WARNING: skipping conversion to existing file {h5_path}")
                continue
            pointcloud, semantic_labels, instance_labels = self.get_labeled_pointcloud(
                scan_dir, label_mapping
            )
            labels = np.column_stack([semantic_labels, instance_labels])
            if self.config.subsampler_voxel_size:
                print("Points before subsampling:", len(pointcloud))
                pointcloud, labels = subsampler(pointcloud, labels)
                print("Points after subsampling:", len(pointcloud))
            labeled_pointcloud = np.column_stack([pointcloud, labels])
            blocks = block_splitter(labeled_pointcloud)
            print("Number of blocks:", len(blocks))
            if len(blocks) == 0:
                print(
                    "WARNING: skipping block splitting of "
                    f"widely-spaced point cloud {scan_dir.name}"
                )
                continue
            if np.any(blocks[..., 12] >= self.config.n_classes):
                raise ValueError(
                    "There's a wrong label inside the block. "
                    f"Expected at most {self.config.n_classes} classes, found more."
                )
            with h5py.File(h5_path, "w") as file:
                file.create_dataset(
                    "coords",
                    data=blocks[:, :, 0:3],
                    compression="gzip",
                    dtype="float32",
                )
                file.create_dataset(
                    "points",
                    data=blocks[:, :, 3:12],
                    compression="gzip",
                    dtype="float32",
                )
                file.create_dataset(
                    "labels",
                    data=blocks[:, :, 12:14],
                    compression="gzip",
                    dtype="int64",
                )
