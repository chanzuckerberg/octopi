# Claude Code / Claude Desktop Integration

`octopi mcp` starts a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that connects Claude Code or Claude Desktop directly to your cryo-ET particle picking workflows. Claude can look up command options, suggest fully-formed CLI invocations, and run fast processing steps — all from a conversation.

---

## What Claude Can Do

Once connected, Claude has access to the full octopi CLI through these tools:

| Tool | Purpose |
|------|---------|
| `list_octopi_commands` | Browse all exposed commands and their descriptions |
| `get_command_help` | Fetch the full `--help` output for any command |
| `run_octopi_command` | Execute an octopi command directly |

!!! info "Default behaviour: suggest, not run"
    By default Claude will give you the exact command to copy and paste — you stay in control of what runs and when. If you'd prefer Claude to run a command directly, just ask: *"go ahead and run it"*.

    For long-running GPU jobs (`train`, `model-explore`, `segment`), Claude always hands off to you regardless — it prints the command for you to run, with all flags filled in.

---

## Setup

Install the `mcp` extras first:

```bash
pip install "octopi[mcp]"
```

Then run `octopi mcp install` once from the terminal. It automatically registers the server in the right config file — you never need to start the server manually.

=== "Claude Code (project)"

    Registers octopi for the current directory only. Other projects won't see it.

    ```bash
    cd /path/to/your/project
    octopi mcp install
    ```

    This creates a `.mcp.json` file in the current directory. Open Claude Code in that directory and the server connects automatically.

=== "Claude Code (global)"

    Registers octopi for all Claude Code sessions on this machine.

    ```bash
    octopi mcp install --target code-global
    ```

    Start a new Claude Code session to pick up the change.

=== "Claude Desktop"

    ```bash
    octopi mcp install --target desktop
    ```

    Restart Claude Desktop to pick up the change.

You can verify the registration at any time:

```bash
octopi mcp status                        # check project-level
octopi mcp status --target code-global   # check global
```

To remove it:

```bash
octopi mcp uninstall --server-name octopi
```

---

## Example Workflow

A full particle picking run — from coordinates to evaluated picks — guided by Claude:

**Step 1 — Create segmentation targets**

> *"Create segmentation targets from my CoPick project. Config is at /data/config.json, particle name is ribosome."*

Claude suggests (or runs):

```bash
octopi create-targets \
    --config /data/config.json \
    --target ribosome \
    --voxel-size 10
```

**Step 2 — Explore model architectures**

> *"Run a Bayesian architecture search. I have 4 GPUs available."*

Claude suggests:

```bash
octopi model-explore \
    --config /data/config.json \
    --target-info targets,octopi,1 \
    --voxel-size 10 \
    --model-type Unet \
    --num-trials 50 \
    --output explore_results
```

!!! tip "SLURM support"
    Add `--submitit --njobs 5 --gpu-constraint a6000` to distribute trials across SLURM nodes instead of running them locally.

**Step 3 — Segment tomograms**

> *"Segment all my tomograms using the best model from the search."*

Claude suggests:

```bash
octopi segment \
    --config /data/config.json \
    --model-config explore_results/best_model_config.yaml \
    --model-weights explore_results/best_model.pt \
    --voxel-size 10 \
    --seg-info predict,octopi,1
```

**Step 4 — Localize particles**

> *"Extract coordinates from the segmentation."*

Claude suggests (or runs):

```bash
octopi localize \
    --config /data/config.json \
    --seg-info predict,octopi,1 \
    --voxel-size 10 \
    --method watershed \
    --pick-user-id octopi \
    --pick-session-id 1
```

**Step 5 — Evaluate (optional)**

> *"Evaluate against ground truth from user 'expert'."*

Claude runs:

```bash
octopi evaluate \
    --config /data/config.json \
    --ground-truth-user-id expert \
    --predict-user-id octopi \
    --predict-session-id 1
```
