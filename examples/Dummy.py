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
"""Dummy dataset."""

# Usage:
# train-bonet --dataset Dummy.py

from pathlib import Path

from bonet2.datasets import DataConfig, Dataset


class Dummy(Dataset):
    """Dummy Dataset."""

    def __init__(self):
        self._config = DataConfig(
            name="dummy",
            feature_dim=9,
            max_instances=10,
            block_points=4096,
            block_size=1.0,
            block_stride=0.5,
            coordinates_slice_limits=(3, 6),
            semantic_classes=(
                "ceiling",
                "floor",
                "wall",
                "beam",
                "column",
                "window",
                "door",
                "table",
            ),
            # example patterns
            train_patterns=("Pat1", "Pat2"),
            eval_patterns=("Pat3",),
        )

        self._h5_dataset_path = Path(f"{self.config.name}_h5")
        self._raw_dataset_path = Path(f"{self.config.name}_raw")

    @property
    def config(self) -> DataConfig:
        """Return the Dummy DataConfig."""
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
        # Add the download here and save the result in self._raw_dataset_path

    def convert(self) -> None:
        """Convert the raw Dummy dataset to h5 format, put it in h5_path."""
        if self._h5_dataset_path.exists():
            return
        # Add conversion to h5 here and save the result in the self._h5_dataset_path folder
        return
