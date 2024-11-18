from monai.losses import DiceLoss, FocalLoss, TverskyLoss
from model_explore.pytorch import io, trainer, utils, data
from monai.metrics import ConfusionMatrixMetric
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
import torch, mlflow, os

model_save_path = '/mnt/simulations/ml_challenge/resunet_results'
copick_config_path = "/mnt/simulations/ml_challenge/ml_config.json"
segmentation_name = 'segmentation'
my_channels = [32,64,128,128]
my_strides = [2,2,1]
my_num_res_units = 2
my_num_samples = 16
reload_frequency = 5
lr = 1e-3

# Split Experiment into Train and Validation Runs
Nclass = io.get_num_classes(copick_config_path)
data_generator = data.train_generator(copick_config_path, 
                                      segmentation_name, 
                                      Nclasses = Nclass,
                                      tomo_batch_size = 20)
myRunIDs = data_generator.get_data_splits()

# Monai Functions
loss_function = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)  
metrics_function = ConfusionMatrixMetric(include_background=False, metric_name=["recall",'precision','f1 score'], reduction="none")

# Create UNet Model and Load Weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=Nclass,
    channels=my_channels,
    strides=my_strides,
    num_res_units=my_num_res_units,
).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr)

# Create UNet-Trainer
train = trainer.unet(model, device, loss_function, metrics_function, optimizer)

results = train.local_train(data_generator, 
                            model_save_path, 
                            max_epochs = 100,
                            my_num_samples = my_num_samples,
                            reload_frequency = reload_frequency,
                            val_interval = 5,
                            verbose=True)

parameters_save_name = os.path.join(model_save_path, 'training_parameters.json')
io.save_parameters_to_json(model, train, data_generator, parameters_save_name)

results_save_name = os.path.join(model_save_path, 'results.json')
io.save_results_to_json(results, results_save_name)