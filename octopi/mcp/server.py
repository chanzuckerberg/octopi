"""octopi MCP Server — exposes octopi CLI commands as MCP tools."""

import logging
import os
import subprocess
import sys
from typing import Any

try: 
  from fastmcp import FastMCP
except ImportError:
  raise ImportError("MCP server is not installed. Please install it with `pip install copick-mcp`.")

logger = logging.getLogger("octopi-mcp")
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

mcp = FastMCP(
    "octopi MCP Server",
    instructions="""octopi is a deep learning framework for automated 3D particle picking in cryo-electron tomography (cryo-ET).

Always begin by asking the user what they want to do before proceeding. The typical workflow runs in this order:

URI FORMATS
  Users express resources using short URI notation. Translate these to CLI flags as follows:

  Tomogram URI  "algorithm@voxel_spacing"   e.g. "wbp@10.0"
    → --tomo-alg wbp --voxel-size 10.0

  Segmentation URI  "name:user_id/session_id"   e.g. "predict:octopi/1"
    → --seg-info predict,octopi,1  (comma-separated: name,user_id,session_id)

  Pick/Target URI  "name:user_id/session_id"   e.g. "ribosome:manual/1"
    → --target ribosome,manual,1  (for create-targets source picks)
    → --picks-info ribosome,manual,1  (for membrane-extract)
    → --pick-user-id manual --pick-session-id 1  (for localize output)

  Multiple URIs of the same type are passed as repeated flags, e.g.:
    --target ribosome,manual,1 --target virus-like-particle,tm,2

STEP 1 — create-targets
  Convert pick coordinates from a CoPick project into 3D segmentation masks (Zarr).
  This is a fast command — you can run it directly.
  Key params: --config, --target (pick URI → name,user_id,session_id), --voxel-size, --radius-scale

STEP 2 — train OR model-explore
  train: Train a 3D U-Net model on tomogram/segmentation pairs. GPU-intensive, takes hours.
  model-explore: Bayesian architecture search via Optuna — recommended over train for new datasets.
    Supports --submitit for SLURM job submission (njobs concurrent trials).
  IMPORTANT: For train and model-explore, ALWAYS suggest the command as a copy-pasteable block.
  NEVER call run_octopi_command for these unless the user says "run it", "go ahead", or "execute it".
  Key params for both: --config, --voxel-size, --target-info (seg URI → name,user_id,session_id), --tomo-alg, --output
  Key params for model-explore: --model-type, --num-trials, --submitit, --njobs, --gpu-constraint

STEP 3 — segment
  Run sliding-window inference on tomograms to produce probability maps.
  Supports model ensembling (comma-separated --model-config and --model-weights paths).
  GPU-intensive — always suggest rather than run unless the user explicitly asks.
  Key params: --config, --model-config, --model-weights, --tomo-uri (tomo URI), --seg-uri (seg URI)

STEP 4 — localize
  Convert segmentation probability maps to 3D particle coordinates using watershed or center-of-mass.
  This is a fast command — you can run it directly.
  Key params: --config, --seg-uri (seg URI), --method, --pick-user-id, --pick-session-id

STEP 5 (optional) — evaluate
  Measure Precision, Recall, and F1 against ground truth annotations.
  Fast command — can run directly.
  Key params: --config, --ground-truth-user-id, --predict-user-id

STEP 6 (optional) — membrane-extract
  Split picks by proximity to a membrane or organelle segmentation.
  Fast command — can run directly.
  Key params: --config, --picks-info (pick URI), --seg-info (seg URI), --threshold, --save-session-id

HOW TO RESPOND
- Use get_command_help to look up flags before suggesting a command.
- ALWAYS suggest long-running commands (train, model-explore, segment) as copy-pasteable code blocks.
  NEVER run them unless the user explicitly says "run it", "go ahead and run it", or "execute it".
- Describing what they want ("I'd like to train a model") is NOT permission to run — suggest instead.
- For fast commands (create-targets, localize, evaluate, membrane-extract), you may run them directly
  if the user has provided all required parameters.
- If the user provides all required parameters, go straight to the suggestion without asking follow-up questions.
""",
)

OCTOPI_COMMANDS = [
    ("create-targets", "Convert pick coordinates to 3D segmentation masks (Zarr) — fast"),
    ("train", "Train a 3D U-Net model on tomogram/segmentation pairs — GPU-intensive"),
    ("model-explore", "Bayesian architecture search via Optuna (recommended over train) — GPU-intensive"),
    ("segment", "Run sliding-window inference to produce probability maps — GPU-intensive"),
    ("localize", "Convert segmentation maps to 3D particle coordinates — fast"),
    ("evaluate", "Measure Precision/Recall/F1 against ground truth — fast"),
    ("membrane-extract", "Split picks by membrane proximity (alias: mb-extract) — fast"),
]

LONG_RUNNING = {"train", "model-explore", "segment"}


# ============================================================================
# Discovery
# ============================================================================


@mcp.tool()
def list_octopi_commands() -> dict[str, Any]:
    """List all octopi commands available via this MCP server."""
    return {
        "success": True,
        "commands": [{"command": f"octopi {cmd}", "description": desc} for cmd, desc in OCTOPI_COMMANDS],
        "workflow_order": ["create-targets", "train or model-explore", "segment", "localize", "evaluate (optional)", "membrane-extract (optional)"],
        "tip": "Call get_command_help with a command name (e.g. 'train') to see all options.",
    }


@mcp.tool()
def get_command_help(command: str) -> dict[str, Any]:
    """Get the full --help output for an octopi command.

    Args:
        command: Command name, e.g. 'train', 'segment', 'create-targets', 'model-explore'.
    """
    cmd = ["octopi", command, "--help"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        help_text = result.stdout or result.stderr
        return {"success": True, "command": " ".join(cmd), "help": help_text}
    except FileNotFoundError:
        return {"success": False, "error": "'octopi' not found. Is octopi installed in the active environment?"}
    except Exception as e:
        logger.exception("get_command_help failed")
        return {"success": False, "error": str(e)}


# ============================================================================
# Execution
# ============================================================================


@mcp.tool()
def run_octopi_command(args: list[str], working_dir: str | None = None) -> dict[str, Any]:
    """Run an octopi command and return its output.

    Use this for fast commands: create-targets, localize, evaluate, membrane-extract.
    For long-running GPU jobs (train, model-explore, segment), suggest the command as a
    copy-pasteable block instead — only run them if the user explicitly asks you to.

    Args:
        args: Arguments after 'octopi', e.g. ['create-targets', '--config', 'config.json', '--voxel-size', '10'].
        working_dir: Directory to run the command in. Defaults to cwd.
    """
    cwd = working_dir or os.getcwd()
    cmd = ["octopi"] + args
    subcommand = args[0] if args else ""
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)

    timeout = 60 if subcommand not in LONG_RUNNING else 3600

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return {
            "success": result.returncode == 0,
            "command": " ".join(cmd),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s. For GPU jobs, suggest the command for the user to run directly.",
        }
    except FileNotFoundError:
        return {"success": False, "error": "'octopi' not found. Is octopi installed in the active environment?"}
    except Exception as e:
        logger.exception("run_octopi_command failed")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
