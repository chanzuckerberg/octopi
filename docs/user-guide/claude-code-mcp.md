# Claude Code / Claude Desktop Integration

`octopi mcp` starts a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that connects Claude Code or Claude Desktop directly to your cryo-ET particle picking workflows. Claude can look up command options, suggest fully-formed CLI invocations, and run fast processing steps — all from a conversation.

---

## What Claude Can Do

Once connected, you can describe what you want in plain language and Claude handles the rest:

<div class="grid cards" markdown>

-   :material-sitemap:{ .lg .middle } **Guide the full workflow**

    ---

    Walks you through create-targets → train/model-explore → segment → localize → evaluate, asking only for what it needs.

-   :material-flag-checkered:{ .lg .middle } **Fill in the flags**

    ---

    Looks up the correct options for every command so you don't have to memorise them.

-   :material-link-variant:{ .lg .middle } **Understand URI shorthand**

    ---

    Accepts tomogram, segmentation, and pick URIs directly in your prompt and translates them to the right CLI flags.

-   :material-lightning-bolt:{ .lg .middle } **Run fast steps for you**

    ---

    Executes `create-targets`, `localize`, `evaluate`, and `membrane-extract` directly and reports back.

-   :material-chip:{ .lg .middle } **Hand off GPU jobs**

    ---

    For `train`, `model-explore`, and `segment`, prints a ready-to-run command block for you to submit yourself.

</div>

!!! info "You stay in control"
    Claude suggests commands rather than running them by default — you copy, review, and paste. For fast, non-destructive commands you can say *"go ahead and run it"* to skip the copy-paste step. GPU-intensive jobs (`train`, `model-explore`, `segment`) are always handed off regardless.

??? note "MCP tools used under the hood"
    | Tool | Purpose |
    |------|---------|
    | `list_octopi_commands` | Browse all exposed commands and their descriptions |
    | `get_command_help` | Fetch the full `--help` output for any command |
    | `run_octopi_command` | Execute an octopi command directly |

---

## Setup

The MCP server ships with octopi — no extra install needed. Run `octopi mcp install` once to register it for your preferred destination. It writes to the right config file automatically and you never need to start the server manually.

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

!!! tip "Also register copick-mcp"
    [copick-mcp](https://github.com/copick/copick-mcp) is bundled with octopi and gives Claude direct access to your CoPick project — runs, picks, segmentations, and tomograms. Register it the same way:

    ```bash
    copick setup mcp --target code-global   # all Claude Code sessions
    copick setup mcp --target code-project  # current directory only
    ```

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

> *"Help me create segmentation targets from my CoPick project at /data/config.json. I want targets for ribosome:manual/1, virus-like-particle:tm/2, and membranes:membrane-seg/1."*

Claude suggests (or runs):

```bash
octopi create-targets \
    --config /data/config.json \
    --target ribosome,manual,1 \
    --target virus-like-particle,tm,2 \
    --target membranes,membrane-seg,1 \
    --voxel-size 10
```

??? tip "URI shorthand for tomograms, segmentations, and picks"
    Throughout the workflow you can refer to data resources using compact URI strings rather than spelling out individual flags. Claude understands both formats.

    | Resource | URI format | Example |
    |----------|-----------|---------|
    | Tomogram | `algorithm@voxel_spacing` | `wbp@10.0` |
    | Segmentation | `name:user_id/session_id` | `predict:octopi/1` |
    | Picks / targets | `name:user_id/session_id` | `ribosome:manual/1` |

    Multiple picks or segmentation targets can be listed together:

    > *"… targets for ribosome:manual/1, virus-like-particle:tm/2, and membranes:membrane-seg/1"*

**Step 2 — Explore model architectures**

> *"Run a Bayesian architecture search on those targets (targets:octopi/1). I'm on a SLURM cluster — distribute 50 trials across 5 jobs, constrained to a6000 GPUs."*

Claude suggests:

```bash
octopi model-explore \
    --config /data/config.json \
    --tomo-uri wbp@10.0 \
    --target-uri targets:octopi/1 \
    --model-type Unet \
    --num-trials 50 \
    --output explore_results \
    --submitit \
    --njobs 5 \
    --gpu-constraint a6000
```

**Step 3 — Segment tomograms**

> *"Segment all tomograms (wbp@10.0) with the best model from the search. Save predictions as predict:octopi/1."*

Claude suggests:

```bash
octopi segment \
    --config /data/config.json \
    --model-config explore_results/best_model_config.yaml \
    --model-weights explore_results/best_model.pt \
    --tomo-uri wbp@10.0 \
    --seg-uri predict:octopi/1
```

**Step 4 — Localize particles**

> *"Extract particle coordinates from predict:octopi/1 using watershed. Save picks with session 2."*

Claude suggests (or runs):

```bash
octopi localize \
    --config /data/config.json \
    --seg-uri predict:octopi/1 \
    --method watershed \
    --pick-user-id octopi \
    --pick-session-id 2
```

**Step 5 — Evaluate (optional)**

> *"Evaluate my picks (octopi, session 1) against the ground truth annotations from user 'expert'."*

Claude runs:

```bash
octopi evaluate \
    --config /data/config.json \
    --ground-truth-user-id expert \
    --predict-user-id octopi \
    --predict-session-id 1
```
