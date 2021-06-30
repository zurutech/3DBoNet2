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
"""Dataset abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path

from .data_config import DataConfig


class Dataset(ABC):
    """Set of methods and properties every dataset must have."""

    @property
    @abstractmethod
    def config(self) -> DataConfig:
        """Get the dataset configuration."""

    @property
    @abstractmethod
    def h5_path(self) -> Path:
        """Get the path of the folder containing the dataset files in h5 format."""

    @abstractmethod
    def download(self) -> None:
        """Download the raw dataset if not present."""

    @abstractmethod
    def convert(self) -> None:
        """Convert the raw dataset (obtained with download) to h5 format, if needed."""
