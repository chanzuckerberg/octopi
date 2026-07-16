"""Resolve pretrained checkpoint aliases against the biohub/octopi Hugging Face Hub repo.

Repo layout on the Hub is one subfolder per checkpoint, each containing a weights file and a
matching model config, e.g. `tomogram-boundary/weights.pth` + `tomogram-boundary/config.yaml`.
"""
from typing import List, Optional, Tuple, Union
import os

HF_REPO_ID = "biohub/octopi"
WEIGHTS_FILENAME = "weights.pth"
CONFIG_FILENAME = "config.yaml"

# Checkpoint alias -> subfolder name in HF_REPO_ID.
KNOWN_MODELS = {
    "tomogram-boundary": "tomogram-boundary",
}


def default_cache_dir() -> str:
    import octopi

    return os.path.join(os.path.dirname(os.path.abspath(octopi.__file__)), "cache")


def _resolve_one(
    weights: str, config: Optional[str], cache_dir: str
) -> Tuple[str, str]:
    if os.path.exists(weights):
        if config is None:
            raise ValueError(
                f"--model-config is required when --model-weights is a local path ({weights})."
            )
        return weights, config

    if weights not in KNOWN_MODELS:
        raise ValueError(
            f"'{weights}' is not a local file and not a known octopi checkpoint alias. "
            f"Known aliases: {sorted(KNOWN_MODELS)}"
        )

    from huggingface_hub import snapshot_download

    subfolder = KNOWN_MODELS[weights]
    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=f"{subfolder}/*",
            cache_dir=cache_dir,
        )
    except Exception as e:
        raise ValueError(
            f"Could not download checkpoint '{weights}' from the Hugging Face Hub repo "
            f"'{HF_REPO_ID}' (subfolder '{subfolder}'). It may not be published yet, or "
            f"there was a network/authentication issue. Original error: {e}"
        ) from e

    resolved_weights = os.path.join(local_dir, subfolder, WEIGHTS_FILENAME)
    resolved_config = os.path.join(local_dir, subfolder, CONFIG_FILENAME)
    if not os.path.exists(resolved_weights) or not os.path.exists(resolved_config):
        raise ValueError(
            f"Checkpoint '{weights}' is registered but no files were found under "
            f"'{subfolder}/' in the Hugging Face Hub repo '{HF_REPO_ID}'. It may not be "
            f"uploaded yet."
        )
    return resolved_weights, resolved_config


def resolve_model_source(
    model_weights: Union[str, List[str]],
    model_config: Optional[Union[str, List[str]]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Union[str, List[str]], Union[str, List[str]]]:
    """
    Resolve model_weights/model_config to local file paths, downloading from the Hub
    when a value is a known checkpoint alias rather than an existing local path.

    Accepts and returns either a single str or a list (for model-soup ensembles), matching
    the shape of the input `model_weights`. Cached downloads are skipped on repeat calls.
    """
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    is_list = isinstance(model_weights, list)
    weights_list = model_weights if is_list else [model_weights]

    if model_config is None:
        config_list = [None] * len(weights_list)
    elif isinstance(model_config, list):
        config_list = model_config
    else:
        config_list = [model_config] * len(weights_list)

    if len(config_list) != len(weights_list):
        raise ValueError("Number of model configs must match number of model weights.")

    resolved = [
        _resolve_one(w, c, cache_dir) for w, c in zip(weights_list, config_list)
    ]
    resolved_weights = [r[0] for r in resolved]
    resolved_config = [r[1] for r in resolved]

    if not is_list:
        return resolved_weights[0], resolved_config[0]
    return resolved_weights, resolved_config
