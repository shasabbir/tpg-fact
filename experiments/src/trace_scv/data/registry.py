from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    name: str
    provider: str
    identifier: str
    revision: str | None
    required: bool
    license_expected: str | None
    homepage: str


DATASETS: dict[str, DatasetSpec] = {
    "sciver": DatasetSpec(
        key="sciver",
        name="SciVer",
        provider="huggingface",
        identifier="chengyewang/SciVer",
        revision="5695ac384d247ae5df2a7b73568dc63d189296b0",
        required=True,
        license_expected="cc-by-4.0",
        homepage="https://huggingface.co/datasets/chengyewang/SciVer",
    ),
    "musciclaims": DatasetSpec(
        key="musciclaims",
        name="MuSciClaims",
        provider="huggingface",
        identifier="StonyBrookNLP/MuSciClaims",
        revision="9134142d05fb473e0a10c4c05f0c2d5a2af51099",
        required=False,
        license_expected=None,
        homepage="https://huggingface.co/datasets/StonyBrookNLP/MuSciClaims",
    ),
    "mocheg": DatasetSpec(
        key="mocheg",
        name="MOCHEG",
        provider="github",
        identifier="PLUM-Lab/Mocheg",
        revision=None,
        required=False,
        license_expected="cc-by-4.0",
        homepage="https://github.com/PLUM-Lab/Mocheg",
    ),
    "aichartclaim": DatasetSpec(
        key="aichartclaim",
        name="AIChartClaim",
        provider="github",
        identifier="cezhang01/mever",
        revision=None,
        required=False,
        license_expected=None,
        homepage="https://github.com/cezhang01/mever",
    ),
}
