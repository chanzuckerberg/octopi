import os
import torch
import copick
from tqdm import tqdm
from monai.data import DataLoader, CacheDataset, decollate_batch
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
)
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from copick_utils.segmentation.segmentation_from_picks import segmentation_from_picks
from model_explore.utils import get_tomogram_array, get_segmentation_array, stack_patches


transforms = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    NormalizeIntensityd(keys=["image"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    EnsureTyped(keys=["image", "label"]),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandGridPatchd(keys=["image", "label"], patch_size=(96, 96, 96), patch_overlap=(32, 32, 32)),  # Tiling into patches
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
            inputs, labels = (
                stack_patches(batch_data["image"]).to(device),
                stack_patches(batch_data["label"]).to(device),
            )
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

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        stack_patches(val_data["image"]).to(device),
                        stack_patches(val_data["label"]).to(device),
                    )
                    val_outputs = model(val_inputs)
                    metric_val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    metric_val_labels = [post_label(i) for i in decollate_batch(val_labels)]


                    # compute metric for current iteration
                    metrics_function(y_pred=metric_val_outputs, y=metric_val_labels)

                metrics = metrics_function.aggregate(reduction="mean_batch")
                metric_per_class = ["{:.4g}".format(x) for x in metrics]
                metric = torch.mean(metrics).numpy(force=True)
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

    for run in tqdm(root.runs[:end]):
        for pickable_object in root.pickable_objects:
            print(f"Painting {pickable_object.name}")
            # radius = pickable_object.radius / voxel_spacing * paint_scale
            radius = 10
            painting_segmentation_name = "paintedPicks"
            try:
                pick_set = run.get_picks(object_name=pickable_object.name, user_id="data-portal")[0]
                segmentation_from_picks(radius, painting_segmentation_name, run, voxel_spacing, tomo_type, pickable_object, pick_set, user_id="paintedPicks", session_id="0")
            except:
                pass

    data_dicts = []
    for run in tqdm(root.runs[:end]):
        tomogram = get_tomogram_array(run)
        segmentation = get_segmentation_array(run, painting_segmentation_name)
        data_dicts.append({"image": tomogram, "label": segmentation})


    train_files, val_files = data_dicts[:int(end/2)], data_dicts[int(end/2):end]
    print(f"Number of training samples: {len(train_files)}")
    print(f"Number of validation samples: {len(val_files)}")

    train_ds = CacheDataset(data=train_files, transform=transforms)
    val_ds = CacheDataset(data=val_files, transform=transforms)

    train_batch_size = 1
    val_batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=4, pin_memory=torch.cuda.is_available())


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
    #loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
    loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  # softmax=True for multiclass
    dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)  # must use onehot for multiclass
    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="recall", reduction="None")

    train(train_loader, model, loss_function, dice_metric, optimizer, max_epochs=epochs)
