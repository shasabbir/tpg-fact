from __future__ import annotations

import argparse
from pathlib import Path

from trace_scv.data.download import download_huggingface_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("sciver", "musciclaims"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/public"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    print(
        download_huggingface_snapshot(
            args.dataset, args.root, metadata_only=args.metadata_only
        )
    )


if __name__ == "__main__":
    main()
