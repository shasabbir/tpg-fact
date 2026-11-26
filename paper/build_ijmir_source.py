"""Export one LaTeX manuscript and consecutively named figures for IJMIR."""

import argparse
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parent
INCLUDE = re.compile(r"(?m)^[ \t]*\\input\{([a-zA-Z0-9_/-]+)\}[ \t]*$")
GRAPHIC = re.compile(r"(\\includegraphics(?:\[[^\]\n]+\])?\{)([^{}]+)(\})")
GRAPHIC_PATH = r"\graphicspath{{figures/}}"


def inline_source(path, ancestors=()):
    if path in ancestors or not path.is_relative_to(ROOT):
        raise ValueError(f"Invalid LaTeX include: {path}")

    def expand(match):
        relative = match.group(1) + ".tex"
        included = inline_source((ROOT / relative).resolve(), (*ancestors, path))
        return f"% From {relative}\n{included.rstrip()}\n% End {relative}"

    return INCLUDE.sub(expand, path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = inline_source(ROOT / "paper.tex")
    if r"\input{" in source or r"\include{" in source:
        raise ValueError("An include could not be expanded")
    if source.count(GRAPHIC_PATH) != 1:
        raise ValueError("Unexpected graphic search path")
    source = source.replace(GRAPHIC_PATH + "\n", "", 1)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {out}")
    names = {}

    def rename_graphic(match):
        old_name = match.group(2)
        if Path(old_name).name != old_name:
            raise ValueError(f"Unexpected graphic path: {old_name}")
        if old_name not in names:
            figure = ROOT / "figures" / old_name
            name = f"Fig{len(names) + 1}{figure.suffix}"
            shutil.copyfile(figure, out / name)
            names[old_name] = name
        return match.group(1) + names[old_name] + match.group(3)

    source = GRAPHIC.sub(rename_graphic, source)
    (out / "paper.tex").write_text(source, encoding="utf-8", newline="\n")
    for filename in ("sn-jnl.cls", "sn-basic.bst", "references_used.bib"):
        shutil.copyfile(ROOT / filename, out / filename)
    print(f"Exported manuscript and {len(names)} figures to {out}")


if __name__ == "__main__":
    main()
