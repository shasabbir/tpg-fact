"""Resume and verify the pinned Gemma checkpoint used by the development judge."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download


REPO = "google/gemma-3-27b-it"
REVISION = "005ad3404e59d6023443cb575daa05336842228a"
SUPPORT = (
    "added_tokens.json", "chat_template.json", "config.json", "generation_config.json",
    "model.safetensors.index.json", "preprocessor_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)
REQUIRED = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
            "tokenizer_config.json"}
BLOCK = 8 * 1024 * 1024


def valid_file(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata.size:
        return False
    with path.open("rb") as stream:
        if metadata.lfs:
            return hashlib.file_digest(stream, "sha256").hexdigest() == metadata.lfs.sha256
        # Hugging Face exposes a Git blob object ID for small files.
        sha = hashlib.sha1(f"blob {metadata.size}\0".encode())
        while chunk := stream.read(BLOCK):
            sha.update(chunk)
        return sha.hexdigest() == metadata.blob_id


def fetch(metadata, cache, snapshot):
    name = metadata.rfilename
    path = snapshot / name
    if valid_file(path, metadata):
        print(f"Verified cached {name}", flush=True)
        return
    force = path.exists()
    for attempt in range(1, 6):
        try:
            print(f"Fetching {name} (attempt {attempt})", flush=True)
            result = Path(hf_hub_download(REPO, name, revision=REVISION, cache_dir=cache,
                                         token=True, force_download=force))
            if result.resolve() != path.resolve() or not valid_file(path, metadata):
                # A complete-sized but corrupted partial cannot be resumed.
                force = True
                raise ValueError("Downloaded file did not pass size and hash checks")
        except Exception as error:
            # Transport exceptions can include credential-bearing URLs.
            status = getattr(getattr(error, "response", None), "status_code", None)
            print(f"{name}: {type(error).__name__} (HTTP status {status})", flush=True)
            if status in (401, 403) or attempt == 5:
                raise RuntimeError(f"Could not verify {name}") from None
            time.sleep(min(2 ** attempt, 30))
        else:
            print(f"Verified {name}", flush=True)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cache = args.cache.resolve()
    snapshot = cache / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION
    try:
        model = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=True)
    except Exception as error:
        raise RuntimeError(f"Checkpoint metadata lookup failed: {type(error).__name__}") from None
    if model.sha != REVISION:
        raise ValueError("Pinned revision mismatch")
    metadata = {item.rfilename: item for item in model.siblings}
    if not REQUIRED <= metadata.keys() or not {"tokenizer.model", "tokenizer.json"} & metadata.keys():
        raise ValueError("Pinned checkpoint metadata lacks required model or processor files")
    # Download the index first so the serialized tensors determine the shard set.
    fetch(metadata["model.safetensors.index.json"], cache, snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 12 or not set(shards) <= metadata.keys():
        raise ValueError("Unexpected shard list in the pinned index")
    support = [name for name in SUPPORT if name in metadata and name != "model.safetensors.index.json"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda name: fetch(metadata[name], cache, snapshot), support + shards):
            pass
    print(f"Verified all {len(shards)} weight shards and {len(support) + 1} support files.", flush=True)


if __name__ == "__main__":
    main()
