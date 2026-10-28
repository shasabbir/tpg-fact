from __future__ import annotations

import tomllib
from pathlib import Path


def test_formal_dependencies_include_slow_siglip_tokenizer_backend() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    formal = configuration["project"]["optional-dependencies"]["formal"]
    assert any(requirement.startswith("sentencepiece") for requirement in formal)
