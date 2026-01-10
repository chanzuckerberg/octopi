from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

# Minimal helper to get a shared Console without top-level Rich dependency.
def get_console():
    return Console()

def _progress(iterable, description="Processing", unit=None):
    """
    Wrap an iterable with a Rich progress bar.

    Args:
        iterable: Any iterable object (e.g., list, generator).
        description: Text label to display above the progress bar.

    Yields:
        Each item from the iterable, while updating the progress bar.

    Example:
        for x in _progress(range(10), "Doing work"):
            time.sleep(0.5)
    """

    console = Console()

    # The generator itself yields items while advancing the progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(iterable), unit=unit if unit is not None else "item")
        for item in iterable:
            yield item
            progress.advance(task)

def print_summary(process: str, **kwargs):
    """
    Pretty-print download parameters using Rich in a clean table with green highlights.
    """

    console = Console()

    # ---- Header ----
    console.rule(f"[bold green]{process} Parameters Summary[/bold green]")

    # ---- Table (no outer box) ----
    table = Table(
        show_header=True,
        header_style="bold magenta",
        expand=False,
        border_style="green",   # table borders only
    )
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, indent=2)
        table.add_row(str(key), str(value))

    console.print(table)   # Print table directly, NO panel wrapper
