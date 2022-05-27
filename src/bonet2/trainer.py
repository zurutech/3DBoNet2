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
"""Train and evaluate a 3D-BoNet model"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .bonet2 import BoNet2
from .datasets import CloudBlocksDataset
from .losses import (
    MaskedCategoricalCrossentropy,
    PointFocalLoss,
    ScoreLoss,
    VerticesLosses,
)
from .processing import BlockMerger, InstancePrecisionRecall


class Trainer:
    """Trainer manages the training and evaluation of a 3D-BoNet model."""

    def __init__(
        self,
        model: BoNet2,
        data: CloudBlocksDataset,
        output_directory: Union[str, Path],
        voxel_size: float = 5e-3,
    ):
        """
        Constructor of a Trainer object.

        Args:
            model: 3D-BoNet model with PointNet++ backbone.
            data: Dataset of point cloud blocks.
            output_directory: Path to the directory used to save model checkpoints and TensorBoard logging.
            voxel_size: Size of voxels used by BlockMerger to merge instances from many blocks. Defaults to 5e-3.
        """
        self.model = model
        self.data = data
        self.output_directory = Path(output_directory).expanduser()
        self.voxel_size = voxel_size

        self.losses = {
            "bbox_vertices": VerticesLosses(),
            "bbox_scores": ScoreLoss(),
            "mask": PointFocalLoss(scale=30.0),
            "segmentation": MaskedCategoricalCrossentropy(),
        }
        minimum_instance_sizes = np.ceil(0.3 * data.average_instance_sizes)

        self.block_merger = BlockMerger(
            voxel_size, data.coordinates_slice, minimum_instance_sizes
        )
        self.instance_pr = InstancePrecisionRecall(
            data.n_classes, minimum_instance_sizes
        )

        self.writer = tf.summary.create_file_writer(str(output_directory))

        # The BlockMerger tf.Module is exported for inference
        tf.saved_model.save(
            self.block_merger,
            str(output_directory / "blockmerger"),
            signatures={
                "call": self.block_merger.__call__.get_concrete_function(
                    tf.TensorSpec([None, 3], dtype=tf.float32)
                ),
                "update": self.block_merger.update,
                "reset": self.block_merger.reset,
                "get_instance_scores": self.block_merger.get_instance_scores,
                "remove_small_instances": self.block_merger.remove_small_instances,
            },
        )

    @tf.function
    def _train_step(
        self,
        batch: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> Tuple[tf.Tensor]:
        """
        Train the model for a single step.

        Args:
            train_dataset: Dataset of training point cloud blocks.
            optimizer: Optimizer of model parameters.
            epoch: Index of the current epoch.
            n_batches: Number of batches per epoch.
        """
        (
            pointcloud,
            segmentation_labels,
            vertices_true,
            mask_true,
        ) = batch
        with tf.GradientTape() as tape:
            vertices, scores, mask_prob, segmentation_prob = self.model(
                pointcloud, ground_truth_vertices=vertices_true, training=True
            )
            (
                bbox_vertices_loss,
                bbox_vertices_ce_loss,
                bbox_vertices_iou_loss,
                bbox_vertices_l2_loss,
            ) = self.losses["bbox_vertices"](pointcloud, vertices_true, vertices)
            bbox_scores_loss = self.losses["bbox_scores"](vertices_true, scores)
            mask_loss = self.losses["mask"](mask_true, mask_prob)
            segmentation_loss = self.losses["segmentation"](
                segmentation_labels, segmentation_prob
            )
            loss_value = (
                bbox_vertices_loss + bbox_scores_loss + mask_loss + segmentation_loss
            )
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return (
            loss_value,
            segmentation_loss,
            bbox_vertices_ce_loss,
            bbox_vertices_iou_loss,
            bbox_vertices_l2_loss,
            bbox_scores_loss,
            mask_loss,
        )

    def _train_epoch(
        self,
        train_dataset: tf.data.Dataset,
        optimizer: tf.keras.optimizers.Optimizer,
        epoch: tf.Tensor,
        n_batches: tf.Tensor,
    ) -> None:
        """
        Train the model for a single epoch.

        Args:
            train_dataset: Dataset of training point cloud blocks.
            optimizer: Optimizer of model parameters.
            epoch: Index of the current epoch.
            n_batches: Number of batches per epoch.
        """
        log_interval = 100
        for step, batch in enumerate(train_dataset):
            (
                loss_value,
                segmentation_loss,
                bbox_vertices_ce_loss,
                bbox_vertices_iou_loss,
                bbox_vertices_l2_loss,
                bbox_scores_loss,
                mask_loss,
            ) = self._train_step(batch, optimizer)

            if step % log_interval != 0:
                continue
            log_step = epoch * n_batches + step
            tf.print(
                "Epoch:",
                epoch,
                "- Iteration:",
                step,
                "- Total loss:",
                loss_value,
                "- Segmentation loss:",
                segmentation_loss,
                "- Cross-entropy loss:",
                bbox_vertices_ce_loss,
                "- Soft IoU loss:",
                bbox_vertices_iou_loss,
                "- L2 loss:",
                bbox_vertices_l2_loss,
                "- Score loss:",
                bbox_scores_loss,
                "- Mask loss:",
                mask_loss,
            )
            with self.writer.as_default(step=log_step):
                tf.summary.scalar(
                    "train/loss",
                    loss_value,
                )
                tf.summary.scalar(
                    "train/segmentation_loss",
                    segmentation_loss,
                )
                tf.summary.scalar(
                    "train/crossentropy_loss",
                    bbox_vertices_ce_loss,
                )
                tf.summary.scalar(
                    "train/soft_iou_loss",
                    bbox_vertices_iou_loss,
                )
                tf.summary.scalar(
                    "train/l2_loss",
                    bbox_vertices_l2_loss,
                )
                tf.summary.scalar(
                    "train/score_loss",
                    bbox_scores_loss,
                )
                tf.summary.scalar(
                    "train/mask_loss",
                    mask_loss,
                )
            self.writer.flush()

    def train(self, epochs: int, lr: float = 0.0005) -> None:
        """
        Train the model for a fixed number of epochs using the given initial learning rate.
        Every 20 epochs, the learning rate is halved.
        A checkpoint is saved at the end of every epoch.

        Args:
            epochs: Number of training epochs.
            lr: Initial learning rate. Defaults to 0.0005.
        """
        train_dataset, n_batches = self.data.get_block_dataset(train=True)
        tf.print("Number of batches per epoch:", n_batches)
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=20 * n_batches,
            decay_rate=0.5,
            staircase=True,
        )
        radam = tfa.optimizers.RectifiedAdam(lr_scheduler)
        for epoch in tf.range(epochs, dtype=tf.int64):
            self._train_epoch(train_dataset, radam, epoch, n_batches)
            self.model.save_weights(
                str(
                    self.output_directory
                    / "checkpoints"
                    / f"bonet2_ckpt_{(epoch.numpy() + 1) * n_batches:09d}"
                )
            )
            self.eval((epoch + 1) * n_batches)

    def eval(self, log_step: Optional[tf.Tensor] = None) -> None:
        """
        Evaluate the model on the validation set.

        Args:
            log_step: Optional number of training step performed before the evaluation.
                Defaults to None.
        """
        self.instance_pr.reset_state()
        eval_dataset = self.data.get_indexed_block_dataset(train=False)
        if eval_dataset is None:
            return

        semantic_labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=[None],
        )
        instance_labels = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=[None],
        )
        semantic_preds = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=[None],
        )
        instance_preds = tf.TensorArray(
            dtype=tf.int32,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=[None],
        )

        for batch_idx, n_batches, batch in eval_dataset:
            if tf.math.equal(batch_idx, 0):
                semantic_labels = tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    infer_shape=False,
                    element_shape=[None],
                )
                instance_labels = tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    infer_shape=False,
                    element_shape=[None],
                )
                semantic_preds = tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    infer_shape=False,
                    element_shape=[None],
                )
                instance_preds = tf.TensorArray(
                    dtype=tf.int32,
                    size=0,
                    dynamic_size=True,
                    infer_shape=False,
                    element_shape=[None],
                )
                self.block_merger.reset()

            (_, batch_blocks, batch_semantic_labels, batch_instance_labels) = batch
            instance_labels = instance_labels.write(
                batch_idx, tf.reshape(batch_instance_labels, (-1,))
            )
            semantic_labels = semantic_labels.write(
                batch_idx, tf.reshape(batch_semantic_labels, (-1,))
            )
            (
                _,
                batch_scores,
                batch_instance_pred,
                batch_semantic_pred,
            ) = self.model.predict_step(batch_blocks)
            self.block_merger.update(
                batch_blocks, batch_instance_pred, batch_semantic_pred, batch_scores
            )
            merged_instance_preds, merged_semantic_preds = self.block_merger(
                batch_blocks[..., self.data.coordinates_slice]
            )
            instance_preds = instance_preds.write(
                batch_idx, tf.reshape(merged_instance_preds, (-1,))
            )
            # The original version of the code used directly batch_semantic_pred
            # but it is inconsistent with the final prediction phase.
            # Using merged_semantic_preds the measured performance can change.
            semantic_preds = semantic_preds.write(
                batch_idx, tf.reshape(merged_semantic_preds, (-1,))
            )

            if tf.math.equal(batch_idx, n_batches - 1):
                cloud_semantic_labels = semantic_labels.concat()
                cloud_instance_labels = instance_labels.concat()

                cloud_instance_preds = instance_preds.concat()
                cloud_semantic_preds = semantic_preds.concat()

                self.instance_pr.update_state(
                    cloud_instance_labels,
                    cloud_semantic_labels,
                    cloud_instance_preds,
                    cloud_semantic_preds,
                )
        results = self.instance_pr.result()

        # Log results
        with self.writer.as_default(step=log_step):
            for label in range(self.data.n_classes):
                if log_step is not None:
                    tf.summary.scalar(
                        f"eval/precision_class_{label:02d}",
                        results["precisions"][label],
                    )
                    tf.summary.scalar(
                        f"eval/recall_class_{label:02d}",
                        results["recalls"][label],
                    )
                tf.print(
                    f"Class {label}: precision",
                    results["precisions"][label],
                    "- recall",
                    results["recalls"][label],
                )
            if log_step is not None:
                tf.summary.scalar(
                    "eval/precision",
                    results["average_precision"],
                )
                tf.summary.scalar(
                    "eval/recall",
                    results["average_recall"],
                )

        self.writer.flush()
        tf.print(
            "Mean precision",
            results["average_precision"],
            "- Mean recall",
            results["average_recall"],
        )
