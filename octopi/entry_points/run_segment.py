from octopi.entry_points import common
from typing import List
import rich_click as click

def inference(
    config: str,
    model_weights: str,
    model_config: str,
    tomo_uri: str,
    seg_uri: str,
    run_ids: List[str],
    swbs: int,
    overlap: float,
    ntta: int
    ):
    """
    Perform segmentation inference using a model on provided tomograms.

    Args:
        config (str): Path to CoPick configuration file.
        run_ids (List[str]): List of tomogram run IDs for inference.
        model_weights (str): Path to the trained model weights file, or a pretrained checkpoint
            alias (e.g. "tomogram-boundary") to auto-download from the Hugging Face Hub.
        model_config (str): Path to the model configuration file.
        tomo_uri (str): Tomogram URI in the form "algorithm@voxel_size".
        seg_uri (str): Segmentation output URI in the form "name:user_id/session_id".
    """
    from octopi.workflows import segment

    if ',' in model_weights:
        model_weights = model_weights.split(',')
    if model_config and ',' in model_config:
        model_config = model_config.split(',')
    if isinstance(model_weights, list) and isinstance(model_config, list):
        if len(model_weights) != len(model_config):
            raise ValueError("Number of model weights and model configs must match for ensemble prediction.")
        print("\nUsing Model Ensemble (Soup) Segmentation.")
        print('Model Weights:', model_weights)
        print('Model Configs:', model_config)
    else:
        print("Using Single Model Segmentation.")


    segment(
        config, model_weights, model_config,
        tomo_uri=tomo_uri, seg_uri=seg_uri,
        run_ids=run_ids, swbs = swbs, overlap=overlap, ntta=ntta
    )

    print("✅ Inference completed successfully.")


@click.command('segment', no_args_is_help=True)
# Inference Arguments
@common.inference_parameters()
# Model Arguments
@common.inference_model_parameters()
# Input Arguments
@click.option(
    "-c", "--config", type=click.Path(exists=True), required=True,
    help="Path to copick configuration file" )
@click.option(
    "-uri", "--tomo-uri", type=str, required=False, default='wbp@10.0',
    help="Tomogram URI for Inference (tomo-alg@voxel-size)" 
)
def cli(config, tomo_uri,
        model_config, model_weights, seg_uri, run_ids,
        sliding_window_batch_size, overlap, ntta):
    """
    Segment volumes using trained neural network models.
    
    It supports both single model inference and model ensembles 
    (model soups) for improved accuracy. Multi-GPU inference is automatically enabled when 
    multiple GPUs are available.
    
    The segmentation masks are saved as zarr arrays in your copick project, organized by 
    segmentation name, user ID, and session ID for easy tracking and comparison.
    
    \b
    Examples:
      # Segment with a single model
      octopi segment -c config.json \\
        --tomo-uri wbp@10.0 \\
        --model-config model.yaml --model-weights model.pth \\
        --seg-uri predictions:octopi/1

    \b
      # Segment with a pretrained checkpoint from the Hugging Face Hub (auto-downloaded)
      octopi segment -c config.json \\
        --tomo-uri wbp@10.0 \\
        --model-weights tomogram-boundary \\
        --seg-uri predictions:octopi/1

    \b
      # Segment with model ensemble (comma-separated)
      octopi segment -c config.json \\
        --tomo-uri wbp@10.0 \\
        --model-config model1.yaml,model2.yaml \\
        --model-weights model1.pth,model2.pth \\
        --seg-uri ensemble:octopi/1
    
    \b
      # Segment specific runs only
      octopi segment -c config.json \\
        --tomo-uri wbp@10.0 \\
        --model-config model.yaml --model-weights model.pth \\
        --run-ids TS_001,TS_002,TS_003
    """
    
    # Call the inference function with parsed arguments; tomo_uri/seg_uri parsing
    # happens once, inside octopi.workflows.segment().
    print('\n🚀 Starting inference with Octopi...\n')
    inference(
        config=config,
        model_weights=model_weights,
        model_config=model_config,
        tomo_uri=tomo_uri,
        seg_uri=seg_uri,
        run_ids=run_ids,
        swbs=sliding_window_batch_size,
        overlap=overlap,
        ntta=ntta
    )

if __name__ == "__main__":
    cli()