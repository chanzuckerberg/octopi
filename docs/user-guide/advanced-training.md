# Training Octopi Models

This guide covers everything you need to know about training 3D U-Net models with Octopi. Here, we are fixing the model architecture. For those interested in writing their own training functions, refer to the API. 

## Single Model Training

For specific use cases or when you have a known good architecture, you can train a single model directly. In this case, the command only allows for training U-Net models. To play around with more unique model configurations, or to try importing new model designs refer to the API or `octopi model-explore`. 

### Basic Training Command

```bash
octopi train \
    --config config.json \
    --voxel-size 10 --tomo-alg wbp --Nclass 8 \
    --tomo-batch-size 50 --num-epochs 100 --val-interval 10 \
    --target-info targets,octopi,1
```

### Training Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--config` | Path to copick configuration file  | `config.json` |
| `--voxel-size` | Tomogram voxel size (Angstroms) | `10` |
| `--tomo-alg` | Tomographic reconstruction algorithm | `wbp` (weighted back projection) |
| `--Nclass` | Number of output classes for segmentation | `8` |
| `--tomo-batch-size` | Batch size for tomographic processing | `50` |
| `--num-epochs` | Total number of training epochs | `100` |
| `--val-interval` | Validation frequency (every N epochs) | `10` |
| `--target-info` | Target information in format: name,framework,version | `targets,octopi,1` |

!!! important "Remember that Nclass is number of classes + 1 for background."

## Fine Tuning Models

If we have base weights that we would like to fine-tune for new datasets, we can still use the `train` command. Instead of specifying the model architecture, we can simple point to the configuration file and weights to load the existing model to fine tune.

```bash
octopi train \
    --config config.json \
    --voxel-size 10 --tomo-alg wbp \
    --model-config results/model_config.yaml \
    --model-weights results/best_model_weights.pth
```

### Fine-Tuning Parameters

The parameters for fine-tuning is equivalent to the base training, with the added option to provide the pre-trained model configuration and its weights.

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--model-config` | Path to configuration file that the previous training run generated. | `results/model_config.yaml` |
| `--model-weights` | Path to trained model weights. | `results/best_model_weights.pth` |

#### Training Outputs

Training Output
During training, you'll see:

* Progress indicators: Real-time loss and accuracy metrics
* Validation results: Periodic evaluation on validation set
* Model checkpoints: Saved to results/ directory by default
* Training logs: Detailed logs for monitoring and debugging
