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

#!/usr/bin/env python

"""Configure the predict phase and executes it."""

import importlib
import sys
from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from plyfile import PlyData, PlyElement
import numpy.lib.recfunctions as rfn
import numpy as np
import matplotlib.pyplot as plt

from bonet2.bonet2 import BoNet2
from bonet2.datasets import PlyInferenceDataset, Dataset
from bonet2.predictor import Predictor


def main():
    """Configure the predict phase and executes it."""

    parser = ArgumentParser(
        "bonet-predict", "Execute a 3DBoNet2 model to get predictions"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to a model checkpoint",
    )
    parser.add_argument("--dataset", required=True, type=str, help="The dataset to use")
    parser.add_argument(
        "--test-dir",
        required=True,
        type=str,
        help="Directory containing .ply test files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory containing predicted labels",
    )
    parser.add_argument(
        "--blockmerger",
        type=str,
        help="Path to a BlockMerger SavedModel directory. It can be inferred from the checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    # Instantiate dataset config from dataset name
    if args.dataset.endswith(".py"):
        file_path = Path(args.dataset).absolute()
        name = file_path.stem
        sys.path.append(str(file_path.parent))

        dataset_manager = getattr(__import__(name), name)()
        if not isinstance(dataset_manager, Dataset):
            raise ValueError(
                f"Your class {dataset_manager} must implement the bonet2.dataset.Dataset interface"
            )
    else:
        dataset_manager = getattr(
            importlib.import_module(f"bonet2.datasets.{args.dataset.lower()}"),
            args.dataset,
        )()

    dataset_manager.download()

    data = PlyInferenceDataset(
        args.test_dir,
        dataset_manager.config,
        batch_size=args.batch_size,
    )

    model = BoNet2(
        dataset_manager.config.n_classes, dataset_manager.config.max_instances
    )

    if args.blockmerger is None:
        checkpoint_path = Path(args.checkpoint)
        block_merger_path = checkpoint_path.parents[1] / "blockmerger"
        if not block_merger_path.exists():
            raise ValueError(
                "Path to BlockMerger SavedModel cannot be inferred from checkpoint path."
            )
    else:
        block_merger_path = Path(args.blockmerger)

    block_merger = tf.saved_model.load(str(block_merger_path))

    predictor = Predictor(
        model,
        block_merger,
        data,
    )
    predictor.load(args.checkpoint)
    for path, instance_scores, instance_preds, semantic_preds in predictor.predict():
        input_filename = Path(path.numpy().decode("utf-8"))
        unique_instances, _ = tf.unique(instance_preds)
        first_instance_index = tf.math.argmax(
            tf.math.equal(
                unique_instances[:, tf.newaxis],
                instance_preds[tf.newaxis, :],
            ),
            axis=-1,
        )
        input_path = next(data.dataset_path.glob(f"**/{input_filename}"))
        instance2semantic = tf.gather(semantic_preds, first_instance_index)
        if data.config.semantic_indices is None:
            instance2original = instance2semantic
        else:
            semantic2original = tf.constant(
                data.config.semantic_indices, dtype=tf.int32
            )
            instance2original = tf.gather(semantic2original, instance2semantic)
        write_semantic_segmentation(args.output_dir, input_path, semantic_preds)
        write_instance_segmentation(args.output_dir, input_path, instance_preds)
        write_predictions(
            args.output_dir,
            input_path,
            instance_scores,
            instance_preds,
            instance2original,
        )


def write_semantic_segmentation(
    output_dir: Path, input_path: Path, semantic_preds: tf.Tensor
) -> None:
    """
    Write semantic segmentation predictions on a .ply file.
    The colors are associated to semantic classes according to the colormap tab20,
    see https://matplotlib.org/stable/gallery/color/colormap_reference.html.
    Points that are not labelled are shown in gray.

    Args:
        output_dir: Path to output directory.
        input_path: Path to the input .ply file.
        semantic_preds: Index of the predicted semantic class for every point.
            If a point is not associated to a semantic class, the index is -1.
    """
    semantic_dir = Path(output_dir).expanduser() / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    scene_name = "_".join(input_path.stem.split("_")[:2])
    output_semantic_path = semantic_dir / f"{scene_name}_semantic.ply"
    ply_content = PlyData.read(input_path)
    coordinates = rfn.structured_to_unstructured(
        ply_content["vertex"].data, dtype=np.float32
    )[:, :3]
    rgb_colors = np.array([plt.cm.tab20(i) for i in range(20)])[:, :3]
    rgb_colors = (rgb_colors * 255).astype(np.int32)
    rgb_colors = np.row_stack((rgb_colors, np.array([128, 128, 128])))
    semantic_colors = rgb_colors[semantic_preds.numpy()]
    ply_coordinates = rfn.unstructured_to_structured(
        coordinates, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    )
    ply_colors = rfn.unstructured_to_structured(
        semantic_colors,
        dtype=np.dtype([("red", "u1"), ("green", "u1"), ("blue", "u1")]),
    )
    ply_vertices = rfn.merge_arrays((ply_coordinates, ply_colors), flatten=True)
    vertex_element = PlyElement.describe(ply_vertices, "vertex")
    elements = [vertex_element]
    if "face" in ply_content:
        elements.append(ply_content["face"])
    PlyData(elements, text=True).write(output_semantic_path)


def write_instance_segmentation(
    output_dir: Path, input_path: Path, instance_preds: tf.Tensor
) -> None:
    """
    Write instance segmentation predictions on a .ply file.
    The colors are taken from the colormap tab20 (they can be repeated),
    see https://matplotlib.org/stable/gallery/color/colormap_reference.html.
    Points that are not labelled are shown in gray.

    Args:
        output_dir: Path to output directory.
        input_path: Path to the input .ply file.
        semantic_preds: Index of the predicted instance for every point.
            If a point is not associated to an instance, the index is -1.
    """
    instance_dir = Path(output_dir).expanduser() / "instance"
    instance_dir.mkdir(parents=True, exist_ok=True)
    scene_name = "_".join(input_path.stem.split("_")[:2])
    output_instance_path = instance_dir / f"{scene_name}_instance.ply"
    ply_content = PlyData.read(input_path)
    coordinates = rfn.structured_to_unstructured(
        ply_content["vertex"].data, dtype=np.float32
    )[:, :3]
    rgb_colors = np.array([plt.cm.tab20(i) for i in range(20)])[:, :3]
    rgb_colors = (rgb_colors * 255).astype(np.int32)
    null_color = np.array([128, 128, 128])
    instance_colors = rgb_colors[instance_preds.numpy() % 20]
    instance_colors[instance_preds.numpy() == -1] = null_color
    ply_coordinates = rfn.unstructured_to_structured(
        coordinates, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    )
    ply_colors = rfn.unstructured_to_structured(
        instance_colors,
        dtype=np.dtype([("red", "u1"), ("green", "u1"), ("blue", "u1")]),
    )
    ply_vertices = rfn.merge_arrays((ply_coordinates, ply_colors), flatten=True)
    vertex_element = PlyElement.describe(ply_vertices, "vertex")
    elements = [vertex_element]
    if "face" in ply_content:
        elements.append(ply_content["face"])
    PlyData(elements, text=True).write(output_instance_path)


def write_predictions(
    output_dir: Path,
    input_filename: Path,
    instance_scores: tf.Tensor,
    instance_preds: tf.Tensor,
    instance2semantic: tf.Tensor,
) -> None:
    """
    Write predictions on file in the ScanNet output format,
    see http://kaldir.vc.in.tum.de/scannet_benchmark/documentation#format-instance3d.

    Args:
        output_dir: Path to output directory.
        input_filename: Name of the input .ply file.
        instance_scores: Confidence scores associated to the predicted instances by the model.
        instance_preds: Index of the predicted instance for every point.
            If a point is not associated to an instance, the index is -1.
        instance2semantic: Index of the predicted semantic class for every point.
            If a point is not associated to a semantic class, the index is -1.
    """
    output_dir = Path(output_dir).expanduser()
    masks_dir = output_dir / "predicted_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    scene_name = "_".join(input_filename.stem.split("_")[:2])
    output_instance_path = output_dir / f"{scene_name}.txt"
    n_instances = tf.math.reduce_max(instance_preds) + 1
    with output_instance_path.open("w") as instance_file:
        for i in tf.range(n_instances):
            mask_path = masks_dir / f"{scene_name}_{i:03d}.txt"
            instance_file_line = f"{mask_path.relative_to(output_dir)} {instance2semantic[i].numpy()} {instance_scores[i].numpy():.4f}\n"
            instance_file.write(instance_file_line)
            with mask_path.open("w") as mask_file:
                mask = tf.cast(tf.math.equal(instance_preds, i), tf.int32).numpy()
                mask_lines = "\n".join(map(str, mask.tolist()))
                mask_file.write(mask_lines)


if __name__ == "__main__":
    sys.exit(main())
