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

"""Configure the training phase and executes it."""

import importlib
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from bonet2.bonet2 import BoNet2
from bonet2.datasets import CloudBlocksDataset, Dataset
from bonet2.trainer import Trainer


def main():
    """Configure the training phase and executes it."""

    parser = ArgumentParser("bonet-train", "Train 3DBoNet2")
    parser.add_argument(
        "--dataset", required=True, type=str, help="The dataset to use."
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=1234,
        help="TensorFlow RNG seed. Set it to a fixed value to make the experiments reproducible",
    )
    parser.add_argument(
        "--epochs", type=int, default=51, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--eval-batch-size", type=int, help="Evaluation batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Initial learning rate. A decay every 20 epoch is applied",
    )
    parser.add_argument(
        "--rundir",
        type=str,
        default="runs",
        help="Path where to save the logs and the checkpoints during the experiments",
    )
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
    dataset_manager.convert()

    tf.random.set_seed(args.rng_seed)

    data = CloudBlocksDataset(
        dataset_manager.h5_path,
        dataset_manager.config,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    model = BoNet2(
        dataset_manager.config.n_classes, dataset_manager.config.max_instances
    )

    time_string = datetime.now().strftime("%Y-%m-%d_%H:%M")
    trainer = Trainer(
        model,
        data,
        Path(args.rundir) / f"{time_string}",
    )
    trainer.train(args.epochs, lr=args.lr)


if __name__ == "__main__":
    sys.exit(main())
