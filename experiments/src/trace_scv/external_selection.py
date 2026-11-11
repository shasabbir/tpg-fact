from __future__ import annotations

from collections.abc import Sequence

from trace_scv.cefes import select_cefes_prefix


def select_evidence_indices(
    probabilities: Sequence[float],
    *,
    method: str,
    maximum_items: int = 10,
    fixed_k: int | None = None,
    threshold: float | None = None,
    require_nonempty: bool = True,
) -> list[int]:
    if maximum_items < 1:
        raise ValueError("maximum_items must be positive")
    values = [float(value) for value in probabilities]
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    if method == "fixed":
        if fixed_k is None or fixed_k < 1:
            raise ValueError("fixed selection requires fixed_k >= 1")
        selected = order[: min(fixed_k, maximum_items)]
    elif method == "threshold":
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold selection requires threshold in [0, 1]")
        selected = [index for index in order if values[index] >= threshold][:maximum_items]
        if require_nonempty and not selected:
            selected = order[:1]
    elif method == "expected_f1":
        selected = list(
            select_cefes_prefix(
                values,
                allow_empty=not require_nonempty,
                max_items=maximum_items,
            ).selected_indices
        )
    else:
        raise ValueError(f"unknown evidence-selection method: {method}")
    if len(selected) > maximum_items:
        raise AssertionError("selector exceeded the official evidence cap")
    return selected


def enforce_official_output_constraints(
    evidence: Sequence[dict[str, object]],
    *,
    maximum_items: int = 10,
    maximum_characters: int = 1500,
) -> None:
    if len(evidence) > maximum_items:
        raise ValueError(f"official output permits at most {maximum_items} evidence items")
    for index, row in enumerate(evidence):
        text = str(row.get("text", row.get("content", "")))
        if len(text) > maximum_characters:
            raise ValueError(f"evidence item {index} exceeds {maximum_characters} characters")
        images = row.get("images", [])
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise ValueError(f"evidence item {index} images must be a sequence")
        if not text.strip() and not images:
            raise ValueError(f"evidence item {index} contains neither text nor images")
        if any(not str(path).strip() for path in images):
            raise ValueError(f"evidence item {index} contains an empty image reference")
