import rich_click as click

# Configure rich-click
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True

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
            "options": ["--num-epochs", "--val-interval", "--tomo-batch-size", "--best-metric", 
                       "--num-tomo-crops", "--lr", "--tversky-alpha", "--model-save-path"]
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
    "routines inference": [
        {
            "name": "Input Arguments",
            "options": ["--config", "--voxel-size"]
        },
        {
            "name": "Model Arguments",
            "options": ["--model-config", "--model-weights"]
        },
        {
            "name": "Inference Arguments",
            "options": ["--tomo-alg", "--seg-info", "--tomo-batch-size", "--run-ids"]
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
}