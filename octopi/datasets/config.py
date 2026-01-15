from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, field
from octopi.datasets import generators
from octopi.utils import parsers
import random, torch
import numpy as np

@dataclass
class DataGeneratorConfig:
    """
    Configuration class for creating a data generator for training.
    """
    # Core data description
    config: Union[str, Dict[str, str]]
    name: str           # Name of the target for segmentation
    voxel_size: float   # Voxel Size for training

    # Optional identifiers for the target query
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Tomogram + sampling
    tomo_algorithm: Union[str, List[str]] = "wbp"
    ntomo_cache: int = 15
    background_ratio: float = 0.0

    # Splits
    data_split: str = "0.8"
    trainRunIDs: Optional[List[str]] = field(default_factory=list)
    validateRunIDs: Optional[List[str]] = field(default_factory=list)

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataGeneratorConfig":
        """
        Safe constructor from a dict (e.g. JSON/YAML/Optuna params).
        Ignores extra keys automatically.
        """
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _normalize_tomo_algorithm(self) -> str:
        """
        Always return a comma-separated string for downstream code.
        """
        if isinstance(self.tomo_algorithm, list):
            return ",".join(self.tomo_algorithm)
        return self.tomo_algorithm

    # -------------------------
    # Factory
    # -------------------------
    def create_data_generator(self, verbose: bool = True):
        """
        Create and initialize a Copick or MultiCopick data module.
        Safe to call inside a multiprocessing worker.
        """
        tomo_alg = self._normalize_tomo_algorithm()

        if isinstance(self.config, dict):
            data_generator = generators.MultiCopickDataModule(
                self.config, tomo_alg,
                self.name, self.session_id, self.user_id,
                self.voxel_size, self.ntomo_cache, self.background_ratio,
                verbose
            )
        else:
            data_generator = generators.CopickDataModule(
                self.config, tomo_alg,
                self.name, self.session_id, self.user_id,
                self.voxel_size, self.ntomo_cache, self.background_ratio,
                verbose
            )

        # Parse split
        ratios = parsers.parse_data_split(self.data_split)

        data_generator.get_data_splits(
            trainRunIDs=self.trainRunIDs or None,
            validateRunIDs=self.validateRunIDs or None,
            train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2],
            create_test_dataset=False,
        )

        return data_generator

def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure reproducibility of operations by disabling certain optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 