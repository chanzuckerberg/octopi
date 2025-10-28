"""
Configuration management for SLURM job submission.

Provides flexible environment setup configuration with support for:
- Configuration files (~/.octopi/config.yaml or ~/.config/octopi/config.yaml)
- Multiple environment managers (conda, mamba, venv, custom)
- Command-line argument overrides
- Full backward compatibility with --conda-env
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class SLURMConfig:
    """
    Manages SLURM environment configuration.

    Supports multiple sources with precedence:
    1. CLI arguments (highest)
    2. Config file
    3. Defaults (lowest)
    """

    def __init__(self):
        """Initialize with default values."""
        self.env_type = "conda"
        self.env_path = "/hpc/projects/group.czii/conda_environments/pyUNET/"
        self.setup_commands = None

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> 'SLURMConfig':
        """
        Load configuration from file or defaults.

        Args:
            config_file: Optional path to config file. If None, searches standard locations.

        Returns:
            SLURMConfig instance with loaded configuration
        """
        config = cls()

        # Try to load from config file
        if config_file:
            config._load_from_file(config_file)
        else:
            # Search standard locations
            config._load_from_standard_locations()

        return config

    def _load_from_standard_locations(self):
        """Load config from standard file locations."""
        possible_locations = [
            Path.home() / ".octopi" / "config.yaml",
            Path.home() / ".config" / "octopi" / "config.yaml",
        ]

        for location in possible_locations:
            if location.exists():
                try:
                    self._load_from_file(str(location))
                    return
                except Exception as e:
                    print(f"Warning: Failed to load config from {location}: {e}")

    def _load_from_file(self, config_file: str):
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # Navigate to slurm.environment section
            slurm_config = data.get('slurm', {})
            env_config = slurm_config.get('environment', {})

            # Load environment settings
            if 'type' in env_config:
                self.env_type = env_config['type']
            if 'path' in env_config:
                self.env_path = env_config['path']
            if 'setup_commands' in env_config:
                self.setup_commands = env_config['setup_commands']

        except FileNotFoundError:
            pass  # File doesn't exist, use defaults
        except yaml.YAMLError as e:
            print(f"Warning: Invalid YAML in config file {config_file}: {e}")
        except Exception as e:
            print(f"Warning: Error loading config file {config_file}: {e}")

    def apply_cli_overrides(self, args):
        """
        Apply command-line argument overrides.

        Args:
            args: Argparse namespace with potential overrides
        """
        # New-style arguments
        if hasattr(args, 'env_type') and args.env_type:
            self.env_type = args.env_type
        if hasattr(args, 'env_path') and args.env_path:
            self.env_path = args.env_path
        if hasattr(args, 'env_setup') and args.env_setup:
            self.setup_commands = args.env_setup

        # Backward compatibility: --conda-env
        if hasattr(args, 'conda_env') and args.conda_env:
            self.env_type = 'conda'
            self.env_path = args.conda_env

    def get_environment_setup(self) -> str:
        """
        Generate bash commands for environment setup.

        Returns:
            Multi-line string with bash commands to set up environment
        """
        # If custom commands provided, use them directly
        if self.setup_commands:
            return self.setup_commands.strip()

        # Otherwise, use template based on env_type
        templates = {
            'conda': self._conda_template(),
            'mamba': self._mamba_template(),
            'venv': self._venv_template(),
        }

        template = templates.get(self.env_type)
        if template:
            return template

        # Fallback to conda if unknown type
        print(f"Warning: Unknown env_type '{self.env_type}', falling back to 'conda'")
        return self._conda_template()

    def _conda_template(self) -> str:
        """Generate conda environment setup commands."""
        return f"ml anaconda\nconda activate {self.env_path}"

    def _mamba_template(self) -> str:
        """Generate mamba environment setup commands."""
        return f"ml anaconda\nmamba activate {self.env_path}"

    def _venv_template(self) -> str:
        """Generate Python venv activation commands."""
        return f"source {self.env_path}/bin/activate"

    @classmethod
    def from_conda_path(cls, conda_path: str) -> 'SLURMConfig':
        """
        Create config from conda path (backward compatibility helper).

        Args:
            conda_path: Path to conda environment

        Returns:
            SLURMConfig configured for conda
        """
        config = cls()
        config.env_type = 'conda'
        config.env_path = conda_path
        return config

    def __repr__(self) -> str:
        return f"SLURMConfig(env_type='{self.env_type}', env_path='{self.env_path}')"


def create_example_config() -> Dict[str, Any]:
    """
    Create an example configuration dictionary.

    Returns:
        Dictionary with example configuration structure
    """
    return {
        'slurm': {
            'environment': {
                'type': 'conda',  # or 'mamba', 'venv', 'custom'
                'path': '/hpc/projects/group.czii/conda_environments/pyUNET/',
                # Alternatively, use custom setup commands:
                # 'setup_commands': |
                #   module load python/3.10
                #   source /path/to/venv/bin/activate
            }
        }
    }


def generate_example_config_file(output_path: str):
    """
    Generate an example config file at the specified path.

    Args:
        output_path: Path where to write the example config
    """
    example = create_example_config()

    # Create directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(example, f, default_flow_style=False, sort_keys=False)
        f.write("\n# Example custom setup:\n")
        f.write("# slurm:\n")
        f.write("#   environment:\n")
        f.write("#     setup_commands: |\n")
        f.write("#       module load gcc/11.2.0\n")
        f.write("#       module load cuda/11.8\n")
        f.write("#       source /path/to/venv/bin/activate\n")

    print(f"Example config file created at: {output_path}")
