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
    → --tomo-uri wbp@10.0
    Repeatable for multi-source training (train / model-explore only) — combine different
    voxel sizes and/or different reconstruction algorithms of the same tomogram:
      --tomo-uri wbp@10.0 --tomo-uri wbp@5.0         (multi-resolution)
      --tomo-uri wbp@10.0 --tomo-uri denoised@10.0   (multi-algorithm, same voxel size)
      --tomo-uri wbp@10.0 --tomo-uri denoised@5.0    (both at once)

  Segmentation URI  "name:user_id/session_id"   e.g. "predict:octopi/1"
    → --seg-uri predict:octopi/1

  Pick/Target URI  "name:user_id/session_id"   e.g. "ribosome:manual/1"
    → --target ribosome:manual/1  (for create-targets, repeatable — accepts EITHER a particle pick set
      or a continuous segmentation, e.g. "membrane:membrane-seg/1"; the type is auto-detected from the
      CoPick config, so there is no separate flag for segmentation targets)
    → --picks-uri ribosome:manual/1  (for extract mb-picks)
    → --pick-user-id manual --pick-session-id 1  (for localize output)
    → --target-uri targets:octopi/1  (for create-targets / train / model-explore target segmentation)

  Multiple URIs of the same type are passed as repeated flags, e.g.:
    --target ribosome:manual/1 --target virus-like-particle:tm/2
    (labels are assigned sequentially in the order --target is given, so mixing particle and
    segmentation names in one command preserves their true relative order)

STEP 1 — create-targets
  Convert pick coordinates from a CoPick project into 3D segmentation masks (Zarr).
  This is a fast command — you can run it directly.
  Key params: --config, --target (repeatable; particle or segmentation, auto-detected), --tomo-uri, --target-uri (output seg URI), --radius-scale

STEP 2 — train OR model-explore
  train: Train a 3D U-Net model on tomogram/segmentation pairs. GPU-intensive, takes hours.
  model-explore: Bayesian architecture search via Optuna — recommended over train for new datasets.
    Supports --submitit True for SLURM job submission (njobs concurrent trials).
    IMPORTANT: --submitit is NOT a bare flag — it's a boolean option that requires an explicit
    value, e.g. --submitit True (or --submitit False, the default). `--submitit` alone errors
    with "Option '--submitit' requires an argument."
  IMPORTANT: For train and model-explore, ALWAYS suggest the command as a copy-pasteable block.
  NEVER call run_octopi_command for these unless the user says "run it", "go ahead", or "execute it".
  Key params for both: --config, --tomo-uri (tomogram URI, repeatable for multi-source training — see URI FORMATS), --target-uri (target seg URI), --output
  Key params for model-explore: --model-type, --num-trials, --submitit True, --njobs, --gpu-constraint

STEP 3 — segment
  Run sliding-window inference on tomograms to produce probability maps.
  Supports model ensembling (comma-separated --model-config and --model-weights paths).
  --model-weights also accepts a pretrained checkpoint alias (e.g. 'tomogram-boundary') to
  auto-download from the biohub/octopi Hugging Face Hub repo — in that case omit
  --model-config, since its config is bundled and downloaded automatically.
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

STEP 6 (optional) — extract
  Command group for post-processing existing pipeline outputs. Fast — can run directly.
  extract mb-picks: split picks by proximity to a membrane or organelle segmentation.
    Key params: --config, --picks-uri (pick URI), --seg-uri (seg URI), --threshold, --save-session-id
  extract seg: isolate a single object's mask from a multi-class `segment` prediction.
    Key params: --config, --name (object name), --seg-uri (source seg URI), --session-id

HOW TO RESPOND
- Use get_command_help to look up flags before suggesting a command.
- ALWAYS suggest long-running commands (train, model-explore, segment) as copy-pasteable code blocks.
  NEVER run them unless the user explicitly says "run it", "go ahead and run it", or "execute it".
- Describing what they want ("I'd like to train a model") is NOT permission to run — suggest instead.
- For fast commands (create-targets, localize, evaluate, extract), you may run them directly
  if the user has provided all required parameters.
- If the user provides all required parameters, go straight to the suggestion without asking follow-up questions.
- Default to CLI commands (this server only knows the CLI). Only suggest the Python API
  (octopi.workflows) if the user is clearly programming — writing a script or notebook, not
  running a one-off command — e.g. "how do I call segment from my script/notebook."
""",
)

OCTOPI_COMMANDS = [
    ("create-targets", "Convert pick coordinates to 3D segmentation masks (Zarr) — fast"),
    ("train", "Train a 3D U-Net model on tomogram/segmentation pairs — GPU-intensive"),
    ("model-explore", "Bayesian architecture search via Optuna (recommended over train) — GPU-intensive"),
    ("segment", "Run sliding-window inference to produce probability maps — GPU-intensive"),
    ("localize", "Convert segmentation maps to 3D particle coordinates — fast"),
    ("evaluate", "Measure Precision/Recall/F1 against ground truth — fast"),
    ("extract mb-picks", "Split picks by membrane proximity — fast"),
    ("extract seg", "Isolate a single object's mask from a multi-class prediction — fast"),
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
        "workflow_order": ["create-targets", "train or model-explore", "segment", "localize", "evaluate (optional)", "extract (optional)"],
        "tip": "Call get_command_help with a command name (e.g. 'train') to see all options.",
    }


@mcp.tool()
def get_command_help(command: str) -> dict[str, Any]:
    """Get the full --help output for an octopi command.

    Args:
        command: Command name, e.g. 'train', 'segment', 'create-targets', or a group
            subcommand like 'extract seg'.
    """
    cmd = ["octopi", *command.split(), "--help"]
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

    Use this for fast commands: create-targets, localize, evaluate, extract.
    For long-running GPU jobs (train, model-explore, segment), suggest the command as a
    copy-pasteable block instead — only run them if the user explicitly asks you to.

    Args:
        args: Arguments after 'octopi', e.g. ['create-targets', '--config', 'config.json', '--tomo-uri', 'wbp@10.0'].
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
