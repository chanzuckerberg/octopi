import os
import argparse
import copick
import torch
from tqdm import tqdm
from typing import Optional, Union, Tuple, List
from collections import defaultdict
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from dotenv import load_dotenv
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    NormalizeIntensityd,
    RandCropByLabelClassesd,    
)
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
import optuna
from optuna.integration import PyTorchLightningPruningCallback


def get_args():
    parser = argparse.ArgumentParser(
        description = "Hyperparamter tuning using PyTorch Lightning distributed data-parallel and Optuna."
    )
    parser.add_argument('--copick_config_path', type=str, default='copick_config_dataportal_10439.json')
    parser.add_argument('--copick_user_name', type=str, default='user0')
    parser.add_argument('--copick_segmentation_name', type=str, default='paintedPicks')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_random_samples_per_batch', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--num_optuna_trials', type=int, default=10)
    parser.add_argument('--pruning', action="store_true", help="Activate the pruning feature. `MedianPruner` stops unpromising trials at the early stages of training.")
    return parser.parse_args()


class Model(pl.LightningModule):
    def __init__(
        self, 
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 8,
        channels: Union[Tuple[int, ...], List[int]] = (48, 64, 80, 80),
        strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
        num_res_units: int = 1,
        lr: float=1e-3):
    
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
        )
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): # This ensures that gradients are not stored in memory
            x, y = batch['image'], batch['label']
            y_hat = self(x)
            metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i) for i in decollate_batch(y_hat)]
            metric_val_labels = [AsDiscrete(to_onehot=self.hparams.out_channels)(i) for i in decollate_batch(y)]

            # compute metric for current iteration
            self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
            metrics = self.metric_fn.aggregate(reduction="mean_batch")
            for i,m in enumerate(metrics):
                self.log(f"validation metric class {i+1}", m, prog_bar=True, on_epoch=True, sync_dist=True)
            metric = torch.mean(metrics) # cannot log ndarray 
            self.log('val_metric', metric, prog_bar=True, on_epoch=True, sync_dist=True) # sync_dist=True for distributed training
        return {'val_metric': metric}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

class CopickDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        copick_config_path: str, 
        train_batch_size: int,
        val_batch_size: int,
        num_random_samples_per_batch: int):
    
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        
        self.data_dicts, self.nclasses = self.data_from_copick(copick_config_path)
        self.train_files = self.data_dicts[:int(len(self.data_dicts)//2)]
        self.val_files = self.data_dicts[int(len(self.data_dicts)//2):]
        print(f"Number of training samples: {len(self.train_files)}")
        print(f"Number of validation samples: {len(self.val_files)}")

        # Non-random transforms to be cached
        self.non_random_transforms = Compose([
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS")
        ])

        # Random transforms to be applied during training
        self.random_transforms = Compose([
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[96, 96, 96],
                num_classes=self.nclasses,
                num_samples=num_random_samples_per_batch 
            ),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_ds = CacheDataset(data=self.train_files, transform=self.non_random_transforms, cache_rate=1.0)
        self.train_ds = Dataset(data=self.train_ds, transform=self.random_transforms)
        self.val_ds = CacheDataset(data=self.val_files, transform=self.non_random_transforms, cache_rate=1.0)
        self.val_ds = Dataset(data=self.val_ds, transform=self.random_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                self.train_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=torch.cuda.is_available(),
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                self.val_ds,
                batch_size=self.val_batch_size,
                shuffle=False,  # Ensure the data order remains consistent
                num_workers=4,
                persistent_workers=True,
                pin_memory=torch.cuda.is_available(),
            )

    @staticmethod
    def data_from_copick(copick_config_path):
        root = copick.from_file(copick_config_path)
        nclasses = len(root.pickable_objects) + 1
        data_dicts = []
        target_objects = defaultdict(dict)
        for object in root.pickable_objects:
            if object.is_particle:
                target_objects[object.name]['label'] = object.label
                target_objects[object.name]['radius'] = object.radius
        
        data_dicts = []
        for run in tqdm(root.runs[:8]):
            tomogram = run.get_voxel_spacing(10).get_tomogram('wbp').numpy()
            segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=10, is_multilabel=True)[0].numpy()
            membrane_seg = run.get_segmentations(name='membrane', user_id="data-portal")[0].numpy()
            segmentation[membrane_seg==1] = 1  
            data_dicts.append({"image": tomogram, "label": segmentation})
        
        return data_dicts, nclasses


def objective(trial: optuna.trial.Trial) -> float:
    args = get_args()
    mlf_logger = MLFlowLogger(experiment_name='training-3D-UNet-model-for-the-cryoET-ML-Challenge',
                              tracking_uri='http://mlflow.mlflow.svc.cluster.local:5000',
                              #run_name='test1'
                              )

    # Trainer callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_metric', save_top_k=1, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Detect distributed training environment
    devices = list(range(args.num_gpus))

    #channels, strides_pattern, num_res_units = sync_hyperparameters(trial)
    # We optimize the number of layers, strides, and number of residual units
    num_layers = trial.suggest_int("num_layers", 3, 5)
    base_channel = trial.suggest_categorical("base_channel", [8, 16, 32, 64])
    channels = [base_channel * (2 ** i) for i in range(num_layers)]
    num_downsampling_layers = trial.suggest_int("num_downsampling_layers", 1, num_layers - 1)
    strides_pattern = [2] * num_downsampling_layers + [1] * (num_layers - num_downsampling_layers - 1)        
    num_res_units = trial.suggest_int("num_res_units", 1, 3)
    
    model = Model(channels=channels, strides=strides_pattern, num_res_units=num_res_units, lr=args.learning_rate)
    datamodule = CopickDataModule(args.copick_config_path, args.train_batch_size, args.val_batch_size, args.num_random_samples_per_batch)
    callback = PyTorchLightningPruningCallback(trial, monitor="val_metric")

    # Priotize performace over precision
    torch.set_float32_matmul_precision('medium') # or torch.set_float32_matmul_precision('high')    
    
    # Trainer for distributed training with DDP
    trainer = Trainer(
        max_epochs=args.num_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy="ddp_spawn",
        accelerator="gpu",
        devices=devices,
        num_nodes=1, #int(os.environ.get("WORLD_SIZE", 1)) // args.num_gpus,
        log_every_n_steps=1
    )

    hyperparameters = dict(op_num_layers=num_layers, op_base_channel=base_channel, op_num_downsampling_layers=num_downsampling_layers, op_num_res_units=num_res_units)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)
    callback.check_pruned()
    return trainer.callback_metrics["val_metric"].item()


if __name__ == "__main__":
    args = get_args()
    # MLflow setup
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if not password or not username:
        print("Password not found in environment, loading from .env file...")
        load_dotenv()  # Loads environment variables from a .env file
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        
    # Check again after loading .env file
    if not password:
        raise ValueError("Password is not set in environment variables or .env file!")
    else:
        print("Password loaded successfully")
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password    
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=pruner
    )
    study.optimize(objective, n_trials=args.num_optuna_trials)

    # Print the best hyperparameters
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")