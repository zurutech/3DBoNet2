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
"""S3Dis dataset."""
from pathlib import Path
import shutil

import gdown
import h5py
import numpy as np
from tqdm import tqdm

from ..dataset import DataConfig, Dataset


class S3Dis(Dataset):
    """S3Dis dataset."""

    def __init__(self):
        self._config = DataConfig(
            name="s3dis",
            feature_dim=9,
            max_instances=24,
            block_points=4096,
            block_size=1.0,
            block_stride=0.5,
            coordinates_slice_limits=(6, 9),
            semantic_classes=(
                "ceiling",
                "floor",
                "wall",
                "beam",
                "column",
                "window",
                "door",
                "table",
                "chair",
                "sofa",
                "bookcase",
                "board",
                "clutter",
            ),
            train_patterns=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
            eval_patterns=("Area_5",),
        )

        self._raw_dataset_path = Path(f"{self.config.name}_raw")
        self._h5_dataset_path = Path(f"{self.config.name}_h5")

    @property
    def config(self) -> DataConfig:
        """Return the S3Dis DataConfig."""
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
        url = "https://drive.google.com/uc?id=1hOsoOqOWKSZIgAZLu2JmOb_U8zdR04v0"
        output = "Data_S3DIS.zip"
        gdown.download(url, output, quiet=False)
        gdown.extractall(output, to=str(self._raw_dataset_path))
        Path(output).unlink()

    def convert(self) -> None:
        """Convert the raw S3Dis dataset to h5 format, put it in dest."""
        if self._h5_dataset_path.exists():
            return
        self._h5_dataset_path.mkdir(parents=True)

        for raw_path in tqdm(
            sorted(self._raw_dataset_path.glob("*.h5")), desc="S3Dis conversion"
        ):
            dest_path = self._h5_dataset_path / raw_path.name
            shutil.copy(raw_path, dest_path)
            with h5py.File(dest_path, "r+") as f:
                data = f["points"]
                coords = f["coords"][..., :3]
                block_min = np.min(coords, axis=-2, keepdims=True)
                block_max = np.max(coords, axis=-2, keepdims=True)
                block_size = np.maximum((block_max - block_min), 1e-3)
                features = data[:]
                features[..., :3] = (coords - block_min) / block_size
                data[...] = features
