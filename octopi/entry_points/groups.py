import rich_click as click

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

click.rich_click.COMMAND_GROUPS = {
    "routines": [
        {
            "name": "Pre-Processing",
            "commands": ["download", "import", "create-targets"]
        },
        {
            "name": "Training",
            "commands": ["train", "model-explore"]
        },
        {
            "name": "Inference",
            "commands": ["segment", "localize", "membrane-extract", "evaluate"]
        }
    ]
}

# Define option groups for all subcommands
# Key format: "parent_command_name subcommand_name" or just "subcommand_name"
click.rich_click.OPTION_GROUPS = {
    "routines train": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--voxel-size", "--target-info", "--tomo-alg", 
                       "--trainRunIDs", "--validateRunIDs", "--data-split"]
        },
        {
            "name": "Fine-Tuning Arguments",
            "options": ["--model-config", "--model-weights"]
        },
        {
            "name": "Training Arguments",
            "options": ["--num-epochs", "--val-interval", "--ncache-tomos", "--best-metric", 
                       "--batch-size", "--lr", "--tversky-alpha", "--output"]
        },
        {
            "name": "UNet-Model Arguments",
            "options": ["--Nclass", "--channels", "--strides", "--res-units", "--dim-in"]
        }        
    ],
    "routines create-targets": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--target", "--picks-session-id", "--picks-user-id", 
                       "--seg-target", "--run-ids"]
        },
        {
            "name": "Parameters",
            "options": ["--tomo-alg", "--radius-scale", "--voxel-size"]
        },
        {
            "name": "Output Arguments",
            "options": ["--target-segmentation-name", "--target-user-id", "--target-session-id"]
        }
    ],
    "routines segment": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--voxel-size", "--tomo-alg"]
        },
        {
            "name": "Model Arguments",
            "options": ["--model-config", "--model-weights"]
        },
        {
            "name": "Inference Arguments",
            "options": ["--seg-info", "--tomo-batch-size", "--run-ids"]
        }
    ],
    "routines localize": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--method", "--seg-info", "--voxel-size", "--runIDs"]
        },
        {
            "name": "Localize Arguments",
            "options": ["--radius-min-scale", "--radius-max-scale", "--filter-size", 
                       "--pick-objects", "--n-procs"]
        },
        {
            "name": "Output Arguments",
            "options": ["--pick-session-id", "--pick-user-id"]
        }
    ],
    "routines model-explore": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--voxel-size", "--target-info", "--tomo-alg", 
                       "--mlflow-experiment-name", "--trainRunIDs", "--validateRunIDs", "--data-split"]
        },
        {
            "name": "Training Arguments",
            "options": ["--model-type", "--num-epochs", "--val-interval", 
                       "--ncache-tomos", "--best-metric", "--output",
                       "--num-trials", "--random-seed"]
        }
    ],
    "routines evaluate": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--ground-truth-user-id", "--ground-truth-session-id",
                       "--predict-user-id", "--predict-session-id", "--run-ids"]
        },
        {
            "name": "Evaluation Parameters",
            "options": ["--distance-threshold-scale", "--object-names"]
        },
        {
            "name": "Output Arguments",
            "options": ["--save-path"]
        }
    ],
    "routines membrane-extract": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--voxel-size", "--picks-info", "--membrane-info",
                       "--organelle-info", "--runIDs"]
        },
        {
            "name": "Parameters",
            "options": ["--distance-threshold", "--n-procs"]
        },
        {
            "name": "Output Arguments",
            "options": ["--save-user-id", "--save-session-id"]
        }
    ],
    "routines download": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--datasetID", "--overlay-path"]
        },
        {
            "name": "Tomogram Settings",
            "options": ["--dataportal-name", "--target-tomo-type"]
        },
        {
            "name": "Voxel Settings",
            "options": ["--input-voxel-size", "--output-voxel-size"]
        }
    ],    
}