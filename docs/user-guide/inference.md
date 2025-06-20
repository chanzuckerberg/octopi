# Inference Guide

This guide covers how to apply your trained Octopi models to generate predictions and extract particle coordinates from new tomograms. Inference is a two-step process: segmentation followed by localization.

## Overview 

Octopi inference follows a systematic two-step approach:

1. **Segmentation** - Apply trained model to generate 3D probability masks with test-time augmentation (TTA).
2. **Localization** - Convert probability masks into 3D coordinates using size-based filtering.
3. **Evaluation (Optional)** - Compare predicted coordinates against ground truth annotations.

## Segmentation 

Generate segmentation prediction masks for tomograms using your trained model.

```bash
octopi segment \
    --config config.json \
    --model-config best_model_config.yaml \
    --model-weights best_model.pth \
    --voxel-size 10 --tomo-alg wbp \
    --seg-info predict,unet,1
```

By default, OCTOPI performs **4 Test-Time Augmentation (4TTA)** during segmentation to improve prediction robustness. 

### Segmentation Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--config` | Path to copick configuration file | `config.json` |
| `--model-config` | Path to model configuration file | `best_model_config.yaml` |
| `--model-weights` | Path to trained model weights | `best_model.pth` |
| `--voxel-size` | Voxel size of tomograms | `10` | `10` |
| `--tomo-alg` | Tomogram algorithm for predictions | `wbp` | `wbp`, `denoised` |
| `--seg-info` | Segmentation output specification | `predict,octopi,1` |
| `--tomo-batch-size` | Batch size for processing | `15` |
| `--run-ids` | Specific runs to process | All runs | `run1,run2,run3` |

## Localization 

Convert segmentation masks into 3D particle coordinates using peak detection.

```bash
octopi localize \
    --config config.json \
    --seg-info predict,unet,1 \
    --pick-session-id 1 --pick-user-id octopi
```

The localization algorithm uses **particle size information** from your copick configuration to filter predictions. For each protein type, Octopi reads the expected particle radius from the copick config file. Predicted candidates smaller than `radius * radius_min_scale` or larger than `radius * radius_max_scale` are discarded as noise.

### Localization Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--config` | Path to copick configuration file | `config.json` |
| `--seg-info` | Segmentation input specification | `predict,unet,1` |
| `--method` | Localization method | `watershed` |
| `--voxel-size` | Voxel size for localization | `10` |
| `--pick-session-id` | Session ID for particle picks | `1`|
| `--pick-user-id` | User ID for particle picks | `octopi` |
| `--radius-min-scale` | Minimum particle radius scale | `0.5` |
| `--radius-max-scale` | Maximum particle radius scale | `1.0` |
| `--filter-size` | Filter size for peak detection  (for watershed algorithm) | `10` |
| `--pick-objects` | Specific objects to localize | All objects | `apoferritin,ribosome` |
| `--runIDs` | Specific runs to process | All runs | `run1,run2,run3` |

## Evaluate Results

Evaluate the particle coordinates against the coordinates that were used to generate the segmentation masks. 

```bash
octopi evaluate 
    --config config.json
    --ground-truth-user-id data-portal --ground-truth-session-id 0
    --predict-user-id octopi --predict-session-id 1
    --save-path evaluate_results
```

### Evaluation Metrics

- **Precision**: Fraction of predicted particles that are correct (TP / (TP + FP))
- **Recall**: Fraction of true particles that were detected (TP / (TP + FN))  
- **F1-Score**: Harmonic mean of precision and recall
- **F-beta Score**: Weighted harmonic mean emphasizing recall (configurable β parameter)
- **True Positives (TP)**: Correctly detected particles within distance threshold
- **False Positives (FP)**: Predicted particles with no nearby ground truth
- **False Negatives (FN)**: Ground truth particles with no nearby predictions

## Visualization

To visualize your results and validate the quality of segmentations and coordinates, refer to our interactive notebook:

**📓 [Inference Notebook](https://github.com/chanzuckerberg/octopi/blob/main/notebooks/inference.ipynb)**

With this notebook, we can overlay the segmentation masks or coordiantes the tomograms. 

![Coordinates Visualization](../assets/coordinates.png)
*Example of predicted particle coordinates displayed on a holdout tomogram from cryo-ET training dataset. The visualization shows Octopi's localization results overlaid on tomographic data from [DatasetID: 10440](https://cryoetdataportal.czscience.com/datasets/10440).*

## Next Steps

You now have a complete workflow for applying Octopi models to new tomographic data. The inference pipeline transforms your trained models into actionable scientific results through robust segmentation, intelligent localization, and comprehensive evaluation.

For users who want to integrate Octopi into custom analysis pipelines or automate large-scale processing workflows, refer to the [API Tutorial](api-tutorial.md) to learn how to script new workflows programmatically with OCTOPI's Python interface.