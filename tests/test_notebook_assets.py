import re
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "assignment_transformer_moe.ipynb"
NOTEBOOKS = ROOT / "notebooks"


def test_notebook_figure_paths_exist():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join("".join(cell.source) for cell in notebook.cells)
    paths = re.findall(r"\.\./(assets/figures/[^)\s]+)", text)

    assert paths, "notebook should reference generated figure assets"
    missing = [path for path in paths if not (ROOT / path).exists()]
    assert missing == []


def test_all_notebooks_parse_and_referenced_figures_exist():
    notebooks = sorted(NOTEBOOKS.rglob("*.ipynb"))
    assert notebooks

    missing = []
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        text = "\n".join("".join(cell.source) for cell in notebook.cells)
        for figure_ref in re.findall(r"!\[[^\]]*\]\(([^)]+assets/figures/[^)\s]+)\)", text):
            figure_path = (path.parent / figure_ref).resolve()
            if not figure_path.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {figure_ref}")

    assert missing == []
