from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from trace_scv.data.sciver import _perceptual_dhash
from trace_scv.features import HashEmbedder
from trace_scv.image_io import load_rgb_on_white


def _transparent_figure(path: Path, diagonal: bool) -> None:
    image = Image.new("LA", (40, 40), (0, 0))
    draw = ImageDraw.Draw(image)
    if diagonal:
        draw.line((2, 2, 37, 37), fill=(0, 255), width=5)
    else:
        draw.rectangle((8, 8, 31, 31), fill=(80, 255))
    image.save(path)
    image.close()


def test_transparency_is_composited_before_hashing_and_encoding(tmp_path: Path) -> None:
    diagonal = tmp_path / "diagonal.png"
    square = tmp_path / "square.png"
    _transparent_figure(diagonal, diagonal=True)
    _transparent_figure(square, diagonal=False)

    loaded = load_rgb_on_white(diagonal)
    assert loaded.mode == "RGB"
    assert loaded.getpixel((39, 0)) == (255, 255, 255)
    assert loaded.getpixel((5, 5)) == (0, 0, 0)
    loaded.close()

    diagonal_hash = _perceptual_dhash(diagonal)
    square_hash = _perceptual_dhash(square)
    assert diagonal_hash not in {None, 0}
    assert square_hash not in {None, 0}
    assert diagonal_hash != square_hash

    embedder = HashEmbedder(dim=64)
    diagonal_features = embedder.image(str(diagonal), "same fallback")
    square_features = embedder.image(str(square), "same fallback")
    assert not np.allclose(diagonal_features, square_features)

