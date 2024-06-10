import datetime
import json
import os
import tempfile
from typing import Tuple

import numpy as np
import ray
import ray.train as train
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer 

from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.data import Dataset
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    DataConfig,
    RunConfig,
    ScalingConfig
)

from ray.train.torch import TorchTrainer
from torch.nn.parallel.distributed import DistributedDataParallel
from transformers import BertModel
from typing_extensions import Annotated

from src import data, utils
from src.config import EFS_DIR, ML_FLOW_TRACKING_URI, logger
from src.models import FinetunedLLM

app = typer.Typer()


def train_model(
    experiment_name: str,
    dataset_loc: str,
    train_loop_config: str,
    num_workers: int, 
    cpu_per_worker: int,
    gpu_per_worker: int,
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    results_fp: str
) -> ray.air.result.Result:
    # Setup
    training_loop_config = json.loads(train_loop_config)
    training_loop_config["num_samples"] = num_samples
    training_loop_config["num_epochs"] = num_epochs
    training_loop_config["batch_size"] = batch_size

    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(gpu_per_worker),
        resources_per_worker={"CPU": cpu_per_worker, "GPU": gpu_per_worker}
    )

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min"
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=ML_FLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True
    )

    # Run config

    run_config = RunConfig(callbacks=[mlflow_callback], checkpoint_config=checkpoint_config, storage_path=EFS_DIR, local_dir=EFS_DIR)

    # Dataset
    ds = data.load_data(dataset_loc=dataset_loc, num_samples=training_loop_config["num_samples"])
    train_ds, val_ds = data.stratify_split(ds, stratify="tag", test_size=0.2)
    tags = train_ds.unique(column="tag")
    training_loop_config["num_classes"] = len(tags)




if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init(runtime_env={"env_vars": {"GITHUB_USERNAME": os.environ["GITHUB_USERNAME"]}})
    app()