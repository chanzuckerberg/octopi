import os
import torch
import copick
import numpy as np
from tqdm import tqdm
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    ToTensord,
    AsDiscrete,
    RandFlipd,
    RandRotate90d,
    RandGridPatchd,
    NormalizeIntensityd,
    RandGridPatchd,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    Resized, 
    RandZoomd,
    Activations, 
    CropForegroundd, 
    ScaleIntensityRanged, 
    RandCropByPosNegLabeld,      
)
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from copick_utils.segmentation.segmentation_from_picks import segmentation_from_picks
import mlflow


my_num_samples = 16
train_batch_size = 1
val_batch_size = 1

# Non-random transforms to be cached
non_random_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS")
])

# Random transforms to be applied during training
random_transforms = Compose([
    RandCropByLabelClassesd(
        keys=["image", "label"],
        label_key="label",
        spatial_size=[96, 96, 96],
        num_classes=8,
        num_samples=my_num_samples
    ),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),    
])


def train(train_loader,
          model,
          loss_function,
          metrics_function,
          optimizer,
          nclasses=8,
          max_epochs=100):

    post_pred = AsDiscrete(argmax=True, to_onehot=nclasses)
    post_label = AsDiscrete(to_onehot=nclasses)
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)  # Shape: [B, C, H, W, D]
            labels = batch_data["label"].to(device)  # Shape: [B, C, H, W, D]  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"batch {step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch+1)


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    metric_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    metric_val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # compute metric for current iteration
                    metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)

                metrics = metrics_function.aggregate(reduction="mean_batch")
                metric_per_class = ["{:.4g}".format(x) for x in metrics]
                metric = torch.mean(metrics).numpy(force=True)
                mlflow.log_metric("validation metric", metric, step=epoch+1)
                for i,m in enumerate(metrics):
                    mlflow.log_metric(f"validation metric class {i+1}", m, step=epoch+1)
                metrics_function.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join('./', "best_metric_model.pth"))

                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean recall per class: {', '.join(metric_per_class)}"
                    f"\nbest mean recall: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


if __name__ == "__main__":
    copick_config_path = "copick_config_dataportal_10439.json"
    root = copick.from_file(copick_config_path)

    nclasses = len(root.pickable_objects) + 1
    voxel_spacing = 10
    tomo_type = "wbp"
    painting_segmentation_name = "paintedPicks"
    data_dicts = []
    paint_scale = 0.8  # how large the paint ball size wrt particle size
    end = 2
    lr = 1e-3
    epochs = 20


    from copick_utils.segmentation import target_generator
    import copick_utils.writers.write as write
    from collections import defaultdict

    target_objects = defaultdict(dict)
    for object in root.pickable_objects:
        if object.is_particle:
            target_objects[object.name]['label'] = object.label
            target_objects[object.name]['radius'] = object.radius


    # for run in tqdm(root.runs):
    #     tomo = run.get_voxel_spacing(10)
    #     tomo = tomo.get_tomogram('wbp').numpy()
    #     target = np.zeros(tomo.shape, dtype=np.uint8)
    #     for pickable_object in root.pickable_objects:
    #         pick = run.get_picks(object_name=pickable_object.name, user_id="data-portal")
    #         if len(pick):  
    #             target = target_generator.from_picks(pick[0], 
    #                                                 target, 
    #                                                 target_objects[pickable_object.name]['radius'] * 0.8,
    #                                                 target_objects[pickable_object.name]['label']
    #                                                 )
    #     write.segmentation(run, target, "user0", segmentationName='paintedPicks')

    
    data_dicts = []
    for run in tqdm(root.runs[:2]):
        tomogram = run.get_voxel_spacing(10).get_tomogram('wbp').numpy()
        segmentation = run.get_segmentations(name='paintedPicks', user_id='user0', voxel_size=10, is_multilabel=True)[0].numpy()
        membrane_seg = run.get_segmentations(name='membrane', user_id="data-portal")[0].numpy()
        segmentation[membrane_seg==1]=1  
        data_dicts.append({"image": tomogram, "label": segmentation})


    train_files, val_files = data_dicts[:int(end/2)], data_dicts[int(end/2):end]
    print(f"Number of training samples: {len(train_files)}")
    print(f"Number of validation samples: {len(val_files)}")

    # Create the cached dataset with non-random transforms
    train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1.0)

    # Wrap the cached dataset to apply random transforms during iteration
    train_ds = Dataset(data=train_ds, transform=random_transforms)

    # DataLoader remains the same
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Create validation dataset
    val_ds = CacheDataset(data=val_files, transform=non_random_transforms, cache_rate=1.0)

    # Wrap the cached dataset to apply random transforms during iteration
    val_ds = Dataset(data=val_ds, transform=random_transforms)

    # Create validation DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,  # Ensure the data order remains consistent
    )
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Create UNet, DiceLoss and Adam optimizer
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=nclasses,
        channels=(48, 64, 80, 80),
        strides=(2, 2, 1),
        num_res_units=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)  # must use onehot for multiclass
    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="None")

    mlflow.set_experiment('training 3D U-Net model for the cryoET ML Challenge')
    with mlflow.start_run():    
        train(train_loader, model, loss_function, dice_metric, optimizer, max_epochs=epochs)
    mlflow.end_run()
