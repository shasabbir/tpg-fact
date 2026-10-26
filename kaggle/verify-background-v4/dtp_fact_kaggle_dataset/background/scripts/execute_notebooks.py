from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute every project notebook in a clean kernel")
    parser.add_argument("--notebook-dir", default="notebooks")
    parser.add_argument("--output-dir", default="outputs/executed_notebooks")
    parser.add_argument("--kernel-name", default="tpg-fact")
    args = parser.parse_args()

    notebook_directory = Path(args.notebook_dir).resolve()
    output_directory = Path(args.output_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    for path in sorted(notebook_directory.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=300,
            kernel_name=args.kernel_name,
            resources={"metadata": {"path": str(notebook_directory)}},
        )
        client.execute()
        destination = output_directory / path.name
        nbformat.write(notebook, destination)
        print(f"executed={path.name} output={destination}")


if __name__ == "__main__":
    main()

