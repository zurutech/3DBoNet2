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

"""Package BoNet - Extendible TensorFlow 2 implementation of 3DBoNet."""
import os
import re
import subprocess
import sys
from distutils.spawn import find_executable
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf
from setuptools import find_packages, setup


def find_in_path(name: str, path: str) -> Optional[Path]:
    """Find a file in a search path.

    Source: https://stackoverflow.com/a/13300714/2891324
    """
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for directory in path.split(os.pathsep):
        path = Path(directory) / name
        if path.exists():
            return path.absolute()
    return None


def locate_cuda() -> Dict[str, Path]:
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.

    Source: https://stackoverflow.com/a/13300714/2891324
    """

    # first check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = Path(os.environ["CUDAHOME"])
        nvcc = home / "bin" / "nvcc"
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, or set $CUDAHOME"
            )
        home = nvcc.parent.parent

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": home / "include",
        "lib64": home / "lib64",
    }
    for key, value in cudaconfig.items():
        if not value.exists():
            raise EnvironmentError(
                f"The CUDA {key} path could not be located in {value}"
            )

    return cudaconfig


def compile_custom_op_cuda_code(cuda_file: Path, dest_path: Path) -> None:
    """Compiles the cuda code used in the custom op.

    Args:
        cuda_file: The path of the .cu file to compile
        dest_path: The path of the folder where to put the .o file.
                   This path must exist.

    Raises:
        ValueError if nvcc is not in path or compilation fails.
        ValueError if dest_path does not exist.
    """

    nvcc = find_executable("nvcc")
    if not nvcc:
        raise ValueError("nvcc executable required in PATH")

    if not dest_path.exists():
        raise ValueError(f"{dest_path} does not exist.")

    cmd = [
        nvcc,
        cuda_file,
        "-o",
        dest_path / f"{cuda_file.name}.o",
        "-c",
        "-O3",
        "-DGOOGLE_CUDA=1",
        "-x",
        "cu",
        "-Xcompiler",
        "-fPIC",
    ]

    subprocess.run(cmd, check=True)


def tf_sampling_build() -> None:
    """Build the custom op and put the .so next to the python script that loads it."""

    cuda = locate_cuda()
    compile_custom_op_cuda_code(
        Path("src/bonet2/tf_ops/sampling/tf_sampling_g.cu"),
        Path("src/bonet2/tf_ops/sampling/"),
    )

    cmd = (
        [
            "gcc",
            "-std=c++11",
            "src/bonet2/tf_ops/sampling/tf_sampling.cpp",  # cpp
            "src/bonet2/tf_ops/sampling/tf_sampling_g.cu.o",  # cuda lib to link
            "-o",
            "src/bonet2/tf_ops/sampling/tf_sampling_so.so",  # output to load
            "-shared",
            "-fPIC",
            f"-I{cuda['include']}",
        ]
        + tf.sysconfig.get_compile_flags()
        + [
            "-lcudart",
            f"-L{cuda['lib64']}",
        ]
        + tf.sysconfig.get_link_flags()
        + ["-O3"]
    )
    subprocess.run(cmd, check=True)


def run() -> None:
    """Run the cuda compilation, the extension compilation and crates the wheel."""

    # Meta
    init_py = open("src/bonet2/__init__.py").read()
    metadata = dict(re.findall(r"__([a-z]+)__ = \"([^\"]+)\"", init_py))

    # Info
    readme = open("README.md").read()

    # Requirements
    requirements = open("requirements.in").read().split()

    # Build the .so
    tf_sampling_build()

    setup(
        author_email=metadata["email"],
        author=metadata["author"],
        description=("Extendible TensorFlow 2 implementation of 3DBoNet."),
        install_requires=requirements,
        keywords=[
            "bonet2",
            "tensorflow",
            "tensorflow-2.0",
            "pointcloud",
            "3d",
            "deep-learning",
        ],
        license="Apache License, Version 2.0",
        long_description_content_type="text/markdown",
        long_description=readme,
        name="bonet2",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        url=metadata["url"],
        version=metadata["version"],
        zip_safe=False,
        scripts=["bin/bonet2-train.py"],  # TODO: rename and remove .py
        # https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
        # The line below allows creating the wheel with the .o and .so
        # build just before calling this function
        package_data={
            "": ["*.so", "*.o"],
        },
    )


if __name__ == "__main__":
    sys.exit(run())
