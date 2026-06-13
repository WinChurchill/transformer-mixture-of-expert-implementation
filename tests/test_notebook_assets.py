import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "assignment_transformer_moe.ipynb"


def test_notebook_figure_paths_exist():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    paths = re.findall(r"\.\./(assets/figures/[^)\s]+)", text)

    assert paths, "notebook should reference generated figure assets"
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert missing == []
