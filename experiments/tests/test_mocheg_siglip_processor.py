from __future__ import annotations

from trace_scv.mocheg_features import MochegEncoderConfig, _prepare_siglip_text_batch


def test_siglip_text_limit_matches_pinned_model_position_limit() -> None:
    config = MochegEncoderConfig()
    assert config.vision_max_text_tokens == 64


def test_siglip_slow_processor_receives_explicit_padding_length() -> None:
    captured = {}

    def processor(**kwargs):
        captured.update(kwargs)
        return kwargs

    result = _prepare_siglip_text_batch(processor, ["short", "a longer claim"], 64)
    assert result["padding"] == "max_length"
    assert result["truncation"] is True
    assert result["max_length"] == 64
    assert result["return_tensors"] == "pt"
