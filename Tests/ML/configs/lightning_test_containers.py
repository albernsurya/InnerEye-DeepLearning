#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import param
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.metrics import MeanSquaredError
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import DataLoader, Dataset

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.lightning_container import InnerEyeInference, LightningContainer, LightningModuleWithOptimizer


class DummyContainerWithDatasets(LightningContainer):
    def __init__(self, has_local_dataset: bool = False, has_azure_dataset: bool = False):
        super().__init__()
        self.local_dataset = full_ml_test_data_path("lightning_module_data") if has_local_dataset else None
        self.azure_dataset_id = "azure_dataset" if has_azure_dataset else ""

    def create_model(self) -> LightningModule:
        return LightningModuleWithOptimizer()


class DummyContainerWithAzureDataset(DummyContainerWithDatasets):
    def __init__(self) -> None:
        super().__init__(has_azure_dataset=True)


class DummyContainerWithoutDataset(DummyContainerWithDatasets):
    pass


class DummyContainerWithLocalDataset(DummyContainerWithDatasets):
    def __init__(self) -> None:
        super().__init__(has_local_dataset=True)


class DummyContainerWithAzureAndLocalDataset(DummyContainerWithDatasets):
    def __init__(self) -> None:
        super().__init__(has_local_dataset=True, has_azure_dataset=True)


class InferenceWithParameters(LightningModule):
    model_param = param.String(default="bar")

    def __init__(self, container_param: str):
        super().__init__()


class DummyContainerWithParameters(LightningContainer):
    container_param = param.String(default="foo")

    def __init__(self) -> None:
        super().__init__()

    def create_model(self) -> LightningModule:
        return InferenceWithParameters(self.container_param)


class DummyRegressionPlainLightning(LightningModuleWithOptimizer):
    """
    A class that only implements plain Lightning training and test. Ideally, we want to support importing any plain
    Lightning module without further methods added. This class here inherits LightningWithInference, but does not
    implement the inference_step method
    """

    def __init__(self, in_features: int = 1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.l_rate = 1e-1
        activation = Identity()
        layers = [
            torch.nn.Linear(in_features=in_features, out_features=1, bias=True),
            activation
        ]
        self.model = torch.nn.Sequential(*layers)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch: Any, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:  # type: ignore
        Path("test_step.txt").touch()
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("test_loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_test_epoch_end(self) -> None:
        Path("on_test_epoch_end.txt").touch()
        pass


class DummyRegression(DummyRegressionPlainLightning, InnerEyeInference):
    def __init__(self, in_features: int = 1, *args, **kwargs) -> None:  # type: ignore
        super().__init__(in_features=in_features, *args, **kwargs)  # type: ignore
        self.l_rate = 1e-1
        self.dataset_split = ModelExecutionMode.TRAIN
        activation = Identity()
        layers = [
            torch.nn.Linear(in_features=in_features, out_features=1, bias=True),
            activation
        ]
        self.model = torch.nn.Sequential(*layers)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch, *args, **kwargs) -> torch.Tensor:  # type: ignore
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_inference_start(self) -> None:
        Path("on_inference_start.txt").touch()
        self.inference_mse: Dict[ModelExecutionMode, float] = {}

    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        self.dataset_split = dataset_split
        Path(f"on_inference_start_{self.dataset_split.value}.txt").touch()
        self.mse = MeanSquaredError()

    def inference_step(self, item: Tuple[Tensor, Tensor], batch_idx: int, model_output: torch.Tensor) -> None:
        input, target = item
        prediction = self.forward(input)
        self.mse(prediction, target)
        with Path(f"inference_step_{self.dataset_split.value}.txt").open(mode="a") as f:
            f.write(f"{prediction.item()},{target.item()}\n")

    def on_inference_epoch_end(self) -> None:
        Path(f"on_inference_end_{self.dataset_split.value}.txt").touch()
        self.inference_mse[self.dataset_split] = self.mse.compute().item()
        self.mse.reset()

    def on_inference_end(self) -> None:
        Path("on_inference_end.txt").touch()
        df = pd.DataFrame(columns=["Split", "MSE"],
                          data=[[split.value, mse] for split, mse in self.inference_mse.items()])
        df.to_csv("metrics_per_split.csv", index=False)


class FixedDataset(Dataset):
    def __init__(self, inputs_and_targets: List[Tuple[Any, Any]]):
        super().__init__()
        self.inputs_and_targets = inputs_and_targets

    def __len__(self) -> int:
        return len(self.inputs_and_targets)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        input = torch.tensor([float(self.inputs_and_targets[item][0])])
        target = torch.tensor([float(self.inputs_and_targets[item][1])])
        return input, target


class FixedRegressionData(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_data = [(i, i) for i in range(1, 20, 3)]
        self.val_data = [(i, i) for i in range(2, 20, 3)]
        self.test_data = [(i, i) for i in range(3, 20, 3)]

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(FixedDataset(self.train_data))  # type: ignore

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(FixedDataset(self.val_data))  # type: ignore

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(FixedDataset(self.test_data))  # type: ignore


class DummyContainerWithModel(LightningContainer):

    def __init__(self) -> None:
        super().__init__()
        self.perform_training_set_inference = True
        self.num_epochs = 50
        self.l_rate = 1e-1

    def setup(self) -> None:
        assert self.local_dataset is not None
        (self.local_dataset / "setup.txt").touch()

    def create_model(self) -> LightningModule:
        return DummyRegression()

    def get_data_module(self) -> LightningDataModule:
        return FixedRegressionData()  # type: ignore

    def create_report(self) -> None:
        Path("create_report.txt").touch()


class DummyContainerWithInvalidTrainerArguments(LightningContainer):

    def create_model(self) -> LightningModule:
        return DummyRegression()

    def get_trainer_arguments(self) -> Dict[str, Any]:
        return {"no_such_argument": 1}


class DummyContainerWithPlainLightning(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        self.num_epochs = 100
        self.l_rate = 1e-2

    def create_model(self) -> LightningModule:
        return DummyRegressionPlainLightning()

    def get_data_module(self) -> LightningDataModule:
        return FixedRegressionData()  # type: ignore


class DummyContainerWithHooks(LightningContainer):

    def __init__(self) -> None:
        super().__init__()
        self.num_epochs = 1
        self.l_rate = 1e-1
        # Let hooks write to current working directory, they should be executed with changed working directory
        # that points to outputs folder.
        self.hook_global_zero = Path("global_rank_zero.txt")
        self.hook_local_zero = Path("local_rank_zero.txt")
        self.hook_all = Path("all_ranks.txt")

    def create_model(self) -> LightningModule:
        return DummyRegression()

    def get_data_module(self) -> LightningDataModule:
        return FixedRegressionData()  # type: ignore

    def before_training_on_global_rank_zero(self) -> None:
        assert not self.hook_global_zero.is_file(), "before_training_on_global_rank_zero should only be called once"
        self.hook_global_zero.touch()

    def before_training_on_local_rank_zero(self) -> None:
        assert self.hook_global_zero.is_file(), "before_training_on_global_rank_zero should have been called already"
        assert not self.hook_local_zero.is_file(), "before_training_on_local_rank_zero should only be called once"
        self.hook_local_zero.touch()

    def before_training_on_all_ranks(self) -> None:
        assert self.hook_local_zero.is_file(), "before_training_on_local_rank_zero should have been called already"
        assert not self.hook_all.is_file(), "before_training_on_all_ranks should only be called once"
        self.hook_all.touch()
