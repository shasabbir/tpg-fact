from __future__ import annotations

from pathlib import Path


def load_rgb_on_white(path: str | Path):
    """Load RGB pixels after compositing transparent images onto white."""
    from PIL import Image

    with Image.open(path) as image:
        has_transparency = image.mode in {"RGBA", "LA"} or "transparency" in image.info
        if not has_transparency:
            return image.convert("RGB").copy()
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        rgba.close()
        rgb = canvas.convert("RGB")
        canvas.close()
        return rgb

