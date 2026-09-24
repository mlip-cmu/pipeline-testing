from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).parents[1]


def test_notebook_runs_without_exceptions():
    nb = nbformat.read(ROOT / "wrangling.ipynb", as_version=4)
    NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
