from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np


USER_AGENT = "DTP-Fact/0.1 research-contact: 26-94030-1@student.aiub.edu"
CONTENT_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}


class ImageAssetError(RuntimeError):
    pass


def _pillow_image(data: bytes):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageAssetError(
            'Pillow is required; install the collection extra with pip install -e ".[collection]"'
        ) from exc
    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image
    except Exception as exc:
        raise ImageAssetError(f"Downloaded content is not a readable image: {exc}") from exc


def perceptual_hash(data: bytes, hash_size: int = 8, high_frequency_factor: int = 4) -> str:
    """Compute a deterministic pHash using a two-dimensional DCT."""
    image = _pillow_image(data)
    size = hash_size * high_frequency_factor
    grayscale = image.convert("L").resize((size, size))
    pixels = np.asarray(grayscale, dtype=np.float64)
    positions = np.arange(size)
    frequencies = positions[:, None]
    transform = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    transform[0, :] *= 1.0 / np.sqrt(2.0)
    coefficients = (2.0 / size) * transform @ pixels @ transform.T
    low = coefficients[:hash_size, :hash_size]
    comparison = low > np.median(low[1:, 1:])
    bit_string = "".join("1" if value else "0" for value in comparison.ravel())
    return f"{int(bit_string, 2):0{hash_size * hash_size // 4}x}"


def inspect_image_bytes(data: bytes) -> dict[str, Any]:
    image = _pillow_image(data)
    return {
        "image_sha256": sha256(data).hexdigest(),
        "image_phash": perceptual_hash(data),
        "width": int(image.width),
        "height": int(image.height),
        "format": str(image.format or "").upper(),
    }


def _extension(content_type: str, url: str, image_format: str) -> str:
    mime = content_type.split(";", 1)[0].strip().casefold()
    if mime in CONTENT_EXTENSIONS:
        return CONTENT_EXTENSIONS[mime]
    suffix = Path(urlparse(url).path).suffix.casefold()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(
        image_format, ".img"
    )


def download_image_asset(
    url: str,
    candidate_id: str,
    output_directory: str | Path,
    *,
    timeout: int = 30,
    max_bytes: int = 25 * 1024 * 1024,
) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ImageAssetError("Image URL must use http or https")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > max_bytes:
                raise ImageAssetError("Image exceeds the configured size limit")
            data = response.read(max_bytes + 1)
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
    except HTTPError as exc:
        raise ImageAssetError(f"Image server returned HTTP {exc.code}") from None
    except URLError as exc:
        raise ImageAssetError(f"Image download failed: {exc.reason}") from None
    if len(data) > max_bytes:
        raise ImageAssetError("Image exceeds the configured size limit")

    details = inspect_image_bytes(data)
    output_root = Path(output_directory)
    output_root.mkdir(parents=True, exist_ok=True)
    extension = _extension(content_type, final_url, details["format"])
    output_path = output_root / f"{candidate_id}{extension}"
    if output_path.exists():
        existing = output_path.read_bytes()
        if sha256(existing).hexdigest() != details["image_sha256"]:
            raise ImageAssetError(f"Refusing to overwrite different file: {output_path}")
    else:
        output_path.write_bytes(data)
    return {
        "candidate_id": candidate_id,
        "status": "ok",
        "source_url": url,
        "final_url": final_url,
        "local_path": str(output_path),
        "content_type": content_type,
        **details,
    }
