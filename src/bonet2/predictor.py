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
"""Predictor produces the predictions of a loaded model on a dataset of .ply point clouds."""

from typing import Iterator, Tuple

import tensorflow as tf
from tqdm import tqdm

from .bonet2 import BoNet2
from .datasets import PlyInferenceDataset


class Predictor:
    """Predictor produces the predictions of a loaded model on a dataset of .ply point clouds.

    Shape conventions:
    - M: Number of points in a point cloud.
    - I: Number of predicted instances.
    """

    def __init__(
        self,
        model: BoNet2,
        block_merger_saved_model,
        data: PlyInferenceDataset,
    ):
        """
        Constructor of a Predictor object.

        Args:
            model: 3D-BoNet model with PointNet++ backbone.
            block_merger_saved_model: Loaded SavedModel of a BlockMerger tf.Module.
            data: Dataset of point clouds in .ply format.
        """
        self.model = model
        self.block_merger = block_merger_saved_model
        self.data = data

    def load(self, checkpoint_path: str) -> None:
        """Load the weights of the model from a checkpoint.
        It calls the predict step on dummy data to define the model
        and calls `model.load_weight` afterward.

        Args:
            checkpoint_path: path of the model weights.
        """
        batch = tf.zeros(
            (
                self.data.batch_size,
                self.data.config.block_points,
                self.data.config.feature_dim,
            )
        )
        self.model.predict_step(batch)
        self.model.load_weights(checkpoint_path)

    def predict(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Compute instance and semantic segmentation predictions on the given dataset.

        Yields:
            Iterator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]: [description]
            - Input file name UTF-8 encoded in a tf.string tf.Tensor.
            - Instance confidence scores. Shape (I,).
            - Instance segmentation predicted labels. Shape (M,).
            - Semantic segmentation predicted labels. Shape (M,).
        """
        test_dataset = self.data.get_named_block_dataset()

        pbar = tqdm(desc="Predicting instances", total=len(self.data.paths))

        for path, batch_idx, n_batches, batch_blocks in test_dataset:
            if tf.math.equal(batch_idx, 0):
                self.block_merger.signatures["reset"]()
            if tf.math.less(batch_idx, n_batches):
                (
                    _,
                    batch_scores,
                    batch_instance_pred,
                    batch_semantic_pred,
                ) = self.model.predict_step(batch_blocks)
                self.block_merger.signatures["update"](
                    blocks=batch_blocks,
                    instance_preds=batch_instance_pred,
                    semantic_preds=batch_semantic_pred,
                    instance_scores=batch_scores,
                )
            else:
                self.block_merger.signatures["remove_small_instances"]()
                cloud_preds = self.block_merger.signatures["call"](
                    coordinates=batch_blocks
                )
                cloud_instance_preds = cloud_preds["output_0"]
                cloud_semantic_preds = cloud_preds["output_1"]
                instance_scores = self.block_merger.signatures["get_instance_scores"]()[
                    "output_0"
                ]
                pbar.update()
                yield path, instance_scores, cloud_instance_preds, cloud_semantic_preds

        pbar.close()
