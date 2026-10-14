"""Execute all demo cells in the project's locked environment and save verified outputs."""

import nbformat
from nbclient import NotebookClient

from src.common import ROOT, write_json


def execute():
    """Verify orchestration through measured results; do not initiate repeat training."""
    notebook = nbformat.read(ROOT / "notebooks/pipeline_demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    NotebookClient(
        notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    ).execute()
    destination = ROOT / "results/phase4/pipeline_demo.executed.ipynb"
    nbformat.write(notebook, destination)
    report = {
        "status": "completed",
        "executed_notebook": str(destination),
        "cells": len(notebook.cells),
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
        "all_code_cells_executed": all(
            c.execution_count is not None for c in notebook.cells if c.cell_type == "code"
        ),
    }
    write_json(ROOT / "results/phase4/notebook_execution.json", report)
    print(report)


if __name__ == "__main__":
    execute()
