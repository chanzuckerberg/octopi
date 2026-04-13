# nnUNet Integration for CoPick

This module integrates [nnUNet](https://github.com/MIC-DKFZ/nnUNet) with CoPick projects, exposing three subcommands under `octopi nnunet`:

| Command | Description |
|---|---|
| `octopi nnunet prepare` | Convert a CoPick project → nnUNet raw dataset (`.nii.gz` files) |
| `octopi nnunet train` | Plan, preprocess, and train nnUNet on the converted dataset |
| `octopi nnunet predict` | Run nnUNet inference and optionally write predictions back to CoPick |

---

## Installation

### 1. Install nnUNet

```bash
pip install nnunetv2
```

### 2. (Optional) Install MedNeXt

Required only if you want to train with a MedNeXt variant (e.g. `--model mednext_m`):

```bash
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git
```

### 3. Install SimpleITK

Required for reading/writing `.nii.gz` files:

```bash
pip install SimpleITK
```

---

## Configuration

All three commands take a single `--config` / `-c` argument pointing to a YAML file. Create one based on the template below:

```yaml
# CoPick project settings
copick_config: /path/to/copick_config.json
tomo_algorithm: wbp
voxel_size: 10.0
segmentation_name: targets
segmentation_user_id: octopi
segmentation_session_id: "1"

# Run splits — leave empty to use all available runs for training
train_run_ids: []
test_run_ids: []

# nnUNet dataset settings
dataset_id: 100
dataset_name: CryoET           # Final dir: nnunet_raw/Dataset100_CryoET
nnunet_raw: /path/to/nnunet_raw
nnunet_preprocessed: /path/to/nnunet_preprocessed
nnunet_results: /path/to/nnunet_results

# Training settings
configuration: 3d_fullres      # 3d_fullres or 3d_lowres
folds: [0]                     # nnUNet fold indices (0–4); use [0,1,2,3,4] for full CV

# Model selection — overridden by --model CLI flag if provided
# Options: nnunet | mednext_s | mednext_b | mednext_m | mednext_l
#          mednext_s_k5 | mednext_b_k5 | mednext_m_k5 | mednext_l_k5
model: nnunet

# Prediction output
predictions_dir: /path/to/predictions
```

---

## Usage

### Step 1 — Prepare the dataset

Convert CoPick tomograms and segmentation masks into nnUNet's raw `.nii.gz` format:

```bash
octopi nnunet prepare -c config.yaml
```

This writes:
```
nnunet_raw/Dataset100_CryoET/
├── dataset.json
├── imagesTr/   # training tomograms  ({case}_0000.nii.gz)
├── labelsTr/   # training labels     ({case}.nii.gz)
└── imagesTs/   # test tomograms      ({case}_0000.nii.gz)
```

> `train_run_ids` and `test_run_ids` control which CoPick runs go into each split. If `train_run_ids` is empty, all runs not in `test_run_ids` are used for training.

---

### Step 2 — Train

Plan, preprocess, and train the model:

```bash
octopi nnunet train -c config.yaml
```

Override the model from the CLI (takes precedence over `config.yaml`):

```bash
octopi nnunet train -c config.yaml --model mednext_m
```

Skip the planning/preprocessing step if it has already been run:

```bash
octopi nnunet train -c config.yaml --skip-preprocess
```

#### Available models

| `--model` flag | Trainer class | Notes |
|---|---|---|
| `nnunet` | `nnUNetTrainer` | Default |
| `mednext_s` | `nnUNetTrainerMedNeXtS_kernel3` | Requires MedNeXt |
| `mednext_b` | `nnUNetTrainerMedNeXtB_kernel3` | Requires MedNeXt |
| `mednext_m` | `nnUNetTrainerMedNeXtM_kernel3` | Requires MedNeXt |
| `mednext_l` | `nnUNetTrainerMedNeXtL_kernel3` | Requires MedNeXt |
| `mednext_s_k5` | `nnUNetTrainerMedNeXtS_kernel5` | Requires MedNeXt |
| `mednext_b_k5` | `nnUNetTrainerMedNeXtB_kernel5` | Requires MedNeXt |
| `mednext_m_k5` | `nnUNetTrainerMedNeXtM_kernel5` | Requires MedNeXt |
| `mednext_l_k5` | `nnUNetTrainerMedNeXtL_kernel5` | Requires MedNeXt |

---

### Step 3 — Predict

Run inference on the test tomograms:

```bash
octopi nnunet predict -c config.yaml
```

Write predictions back into the CoPick project as segmentations (under `user_id="nnunet"`, `session_id="0"`):

```bash
octopi nnunet predict -c config.yaml --save-to-copick
```

Use a specific model (must match what was used during training):

```bash
octopi nnunet predict -c config.yaml --model mednext_m --save-to-copick
```

---

## Authors

Jonathan Schwartz, Kevin Zhao, Daniel Ji, Utz Ermel
