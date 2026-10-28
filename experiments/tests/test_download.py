import json
from pathlib import Path

from trace_scv.data import download


def test_sciver_release_files_uses_only_referenced_assets(tmp_path: Path) -> None:
    (tmp_path / "valset.json").write_text(
        json.dumps(
            [
                {"paperid": "paper-a", "image_path": "remote/figure-a.png"},
                {"paperid": "paper-b", "image_path": None},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "testset.json").write_text(
        json.dumps(
            [
                {"paperid": "paper-a", "image_path": "other/figure-a.png"},
                {"paperid": "paper-c", "image_path": "figure-c.png"},
            ]
        ),
        encoding="utf-8",
    )

    assert download._sciver_release_files(tmp_path) == [
        "images/figure-a.png",
        "images/figure-c.png",
        "papers/paper-a.json",
        "papers/paper-b.json",
        "papers/paper-c.json",
    ]


def test_hf_retry_preserves_transient_rate_limit(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    class RateLimitError(RuntimeError):
        response = type("Response", (), {"status_code": 429})()

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RateLimitError("rate limited")
        return "complete"

    monkeypatch.setattr(download.random, "random", lambda: 0.0)
    monkeypatch.setattr(download.time, "sleep", delays.append)

    assert (
        download._retry_hf_call(operation, description="test download", attempts=3)
        == "complete"
    )
    assert attempts == 3
    assert delays == [1.0, 2.0]

