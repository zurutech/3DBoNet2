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
"""ScanNet download utilities."""

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence

from tqdm import tqdm

BASE_URL = "http://kaldir.vc.in.tum.de/scannet/"


def download_scans(
    scan_list_url: str, file_types: Sequence[str], out_dir: Path
) -> None:
    """
    Given scan list URLs, download them inside out_dir.
    """
    with urllib.request.urlopen(scan_list_url) as scan_list_file:
        scans = tuple(
            scan_line.decode("utf8").rstrip("\n") for scan_line in scan_list_file
        )
    for scan_id in tqdm(scans, desc="Download"):
        scan_dir = out_dir / scan_id
        scan_dir.mkdir(parents=True, exist_ok=True)
        scan_url = f"{BASE_URL}v2/scans/{scan_id}"
        for file_type in file_types:
            file_name = f"{scan_id}{file_type}"
            file_path = scan_dir / file_name
            file_url = f"{scan_url}/{file_name}"
            download_file(file_url, file_path)


def download_file(source_url: str, destination_path: Path) -> None:
    """Download the remote resource in source_url, inside destination_path."""

    if destination_path.exists():
        print("WARNING: skipping download of existing file ", destination_path)
        return
    handler, temporary_file = tempfile.mkstemp(dir=destination_path.parent)
    file = os.fdopen(handler, "w")
    file.close()
    retry_attempts = 0
    while True:
        try:
            urllib.request.urlretrieve(source_url, temporary_file)
            break
        except Exception as exc:
            retry_attempts += 1
            if retry_attempts >= 3:
                raise exc from exc
    os.rename(temporary_file, destination_path)


def download_dataset(out_dir: Path) -> None:
    """Download the ScanNet dataset in out_dir."""

    print(
        "By pressing any key to continue you confirm that you have "
        "agreed to the ScanNet terms of use as described at:",
        BASE_URL + "ScanNet_TOS.pdf",
    )
    print("***")
    _ = input("Press any key to continue, or CTRL-C to exit.")

    base_dir = out_dir.expanduser()

    scans_list_url = BASE_URL + "v2/scans.txt"
    scans_types = (
        ".txt",
        ".aggregation.json",
        "_vh_clean_2.0.010000.segs.json",
        "_vh_clean_2.ply",
    )
    scans_dir = base_dir / "scans"
    print(f"Downloading ScanNet v2 train files to {scans_dir}...")
    download_scans(scans_list_url, scans_types, scans_dir)
    print("Downloaded ScanNet v2 train files.")

    test_scans_list_url = BASE_URL + "v2/scans_test.txt"
    test_scans_types = ("_vh_clean_2.ply",)
    test_scans_dir = base_dir / "scans_test"
    print(f"Downloading ScanNet v2 test files to {test_scans_dir}...")
    download_scans(test_scans_list_url, test_scans_types, test_scans_dir)
    print("Downloaded ScanNet v2 test files.")

    print("Downloading ScanNet v2 label mapping file...")
    label_file_name = "scannetv2-labels.combined.tsv"
    label_url = f"{BASE_URL}v2/tasks/{label_file_name}"
    download_file(label_url, base_dir / label_file_name)
    print("Downloaded ScanNet v2 label mapping file.")
