"""Execute every CODE cell of a notebook top to bottom in one namespace (no Jupyter needed).  Used by CI and by the authors to prove the
notebook runs.   python scripts/run_notebook_cells.py notebooks/01_walkthrough.ipynb
The value of the last expression of a cell is printed, like Jupyter does."""
import ast
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ns: dict = {"__name__": "__main__"}
n = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    tree = ast.parse(src)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, f"<cell {i}>", "exec"), ns)
    if last is not None:
        val = eval(compile(ast.Expression(last.value), f"<cell {i}>", "eval"), ns)
        if val is not None:
            print(val)
    n += 1
print(f"\nALL {n} CODE CELLS EXECUTED")
