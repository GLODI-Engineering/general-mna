#!/usr/bin/env python3
"""Execute every code cell in each generated notebook without Jupyter."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebooks) != 3:
        raise AssertionError(f"expected three converter notebooks, found {len(notebooks)}")
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace: dict = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, str(path), "exec"), namespace)
        print(f"validated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
