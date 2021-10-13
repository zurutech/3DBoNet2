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
            eval_patterns=(
                "scene0568_00",
                "scene0568_01",
                "scene0568_02",
                "scene0304_00",
                "scene0488_00",
                "scene0488_01",
                "scene0412_00",
                "scene0412_01",
                "scene0217_00",
                "scene0019_00",
                "scene0019_01",
                "scene0414_00",
                "scene0575_00",
                "scene0575_01",
                "scene0575_02",
                "scene0426_00",
                "scene0426_01",
                "scene0426_02",
                "scene0426_03",
                "scene0549_00",
                "scene0549_01",
                "scene0578_00",
                "scene0578_01",
                "scene0578_02",
                "scene0665_00",
                "scene0665_01",
                "scene0050_00",
                "scene0050_01",
                "scene0050_02",
                "scene0257_00",
                "scene0025_00",
                "scene0025_01",
                "scene0025_02",
                "scene0583_00",
                "scene0583_01",
                "scene0583_02",
                "scene0701_00",
                "scene0701_01",
                "scene0701_02",
                "scene0580_00",
                "scene0580_01",
                "scene0565_00",
                "scene0169_00",
                "scene0169_01",
                "scene0655_00",
                "scene0655_01",
                "scene0655_02",
                "scene0063_00",
                "scene0221_00",
                "scene0221_01",
                "scene0591_00",
                "scene0591_01",
                "scene0591_02",
                "scene0678_00",
                "scene0678_01",
                "scene0678_02",
                "scene0462_00",
                "scene0427_00",
                "scene0595_00",
                "scene0193_00",
                "scene0193_01",
                "scene0164_00",
                "scene0164_01",
                "scene0164_02",
                "scene0164_03",
                "scene0598_00",
                "scene0598_01",
                "scene0598_02",
                "scene0599_00",
                "scene0599_01",
                "scene0599_02",
                "scene0328_00",
                "scene0300_00",
                "scene0300_01",
                "scene0354_00",
                "scene0458_00",
                "scene0458_01",
                "scene0423_00",
                "scene0423_01",
                "scene0423_02",
                "scene0307_00",
                "scene0307_01",
                "scene0307_02",
                "scene0606_00",
                "scene0606_01",
                "scene0606_02",
                "scene0432_00",
                "scene0432_01",
                "scene0608_00",
                "scene0608_01",
                "scene0608_02",
                "scene0651_00",
                "scene0651_01",
                "scene0651_02",
                "scene0430_00",
                "scene0430_01",
                "scene0689_00",
                "scene0357_00",
                "scene0357_01",
                "scene0574_00",
                "scene0574_01",
                "scene0574_02",
                "scene0329_00",
                "scene0329_01",
                "scene0329_02",
                "scene0153_00",
                "scene0153_01",
                "scene0616_00",
                "scene0616_01",
                "scene0671_00",
                "scene0671_01",
                "scene0618_00",
                "scene0382_00",
                "scene0382_01",
                "scene0490_00",
                "scene0621_00",
                "scene0607_00",
                "scene0607_01",
                "scene0149_00",
                "scene0695_00",
                "scene0695_01",
                "scene0695_02",
                "scene0695_03",
                "scene0389_00",
                "scene0377_00",
                "scene0377_01",
                "scene0377_02",
                "scene0342_00",
                "scene0139_00",
                "scene0629_00",
                "scene0629_01",
                "scene0629_02",
                "scene0496_00",
                "scene0633_00",
                "scene0633_01",
                "scene0518_00",
                "scene0652_00",
                "scene0406_00",
                "scene0406_01",
                "scene0406_02",
                "scene0144_00",
                "scene0144_01",
                "scene0494_00",
                "scene0278_00",
                "scene0278_01",
                "scene0316_00",
                "scene0609_00",
                "scene0609_01",
                "scene0609_02",
                "scene0609_03",
                "scene0084_00",
                "scene0084_01",
                "scene0084_02",
                "scene0696_00",
                "scene0696_01",
                "scene0696_02",
                "scene0351_00",
                "scene0351_01",
                "scene0643_00",
                "scene0644_00",
                "scene0645_00",
                "scene0645_01",
                "scene0645_02",
                "scene0081_00",
                "scene0081_01",
                "scene0081_02",
                "scene0647_00",
                "scene0647_01",
                "scene0535_00",
                "scene0353_00",
                "scene0353_01",
                "scene0353_02",
                "scene0559_00",
                "scene0559_01",
                "scene0559_02",
                "scene0593_00",
                "scene0593_01",
                "scene0246_00",
                "scene0653_00",
                "scene0653_01",
                "scene0064_00",
                "scene0064_01",
                "scene0356_00",
                "scene0356_01",
                "scene0356_02",
                "scene0030_00",
                "scene0030_01",
                "scene0030_02",
                "scene0222_00",
                "scene0222_01",
                "scene0338_00",
                "scene0338_01",
                "scene0338_02",
                "scene0378_00",
                "scene0378_01",
                "scene0378_02",
                "scene0660_00",
                "scene0553_00",
                "scene0553_01",
                "scene0553_02",
                "scene0527_00",
                "scene0663_00",
                "scene0663_01",
                "scene0663_02",
                "scene0664_00",
                "scene0664_01",
                "scene0664_02",
                "scene0334_00",
                "scene0334_01",
                "scene0334_02",
                "scene0046_00",
                "scene0046_01",
                "scene0046_02",
                "scene0203_00",
                "scene0203_01",
                "scene0203_02",
                "scene0088_00",
                "scene0088_01",
                "scene0088_02",
                "scene0088_03",
                "scene0086_00",
                "scene0086_01",
                "scene0086_02",
                "scene0670_00",
                "scene0670_01",
                "scene0256_00",
                "scene0256_01",
                "scene0256_02",
                "scene0249_00",
                "scene0441_00",
                "scene0658_00",
                "scene0704_00",
                "scene0704_01",
                "scene0187_00",
                "scene0187_01",
                "scene0131_00",
                "scene0131_01",
                "scene0131_02",
                "scene0207_00",
                "scene0207_01",
                "scene0207_02",
                "scene0461_00",
                "scene0011_00",
                "scene0011_01",
                "scene0343_00",
                "scene0251_00",
                "scene0077_00",
                "scene0077_01",
                "scene0684_00",
                "scene0684_01",
                "scene0550_00",
                "scene0686_00",
                "scene0686_01",
                "scene0686_02",
                "scene0208_00",
                "scene0500_00",
                "scene0500_01",
                "scene0552_00",
                "scene0552_01",
                "scene0648_00",
                "scene0648_01",
                "scene0435_00",
                "scene0435_01",
                "scene0435_02",
                "scene0435_03",
                "scene0690_00",
                "scene0690_01",
                "scene0693_00",
                "scene0693_01",
                "scene0693_02",
                "scene0700_00",
                "scene0700_01",
                "scene0700_02",
                "scene0699_00",
                "scene0231_00",
                "scene0231_01",
                "scene0231_02",
                "scene0697_00",
                "scene0697_01",
                "scene0697_02",
                "scene0697_03",
                "scene0474_00",
                "scene0474_01",
                "scene0474_02",
                "scene0474_03",
                "scene0474_04",
                "scene0474_05",
                "scene0355_00",
                "scene0355_01",
                "scene0146_00",
                "scene0146_01",
                "scene0146_02",
                "scene0196_00",
                "scene0702_00",
                "scene0702_01",
                "scene0702_02",
                "scene0314_00",
                "scene0277_00",
                "scene0277_01",
                "scene0277_02",
                "scene0095_00",
                "scene0095_01",
                "scene0015_00",
                "scene0100_00",
                "scene0100_01",
                "scene0100_02",
                "scene0558_00",
                "scene0558_01",
                "scene0558_02",
                "scene0685_00",
                "scene0685_01",
                "scene0685_02",
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
