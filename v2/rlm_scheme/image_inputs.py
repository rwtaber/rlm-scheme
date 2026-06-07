"""Image input resolution and MIME validation."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def resolve_image(image: dict[str, Any] | str) -> dict[str, Any]:
    """Resolve an image input to a data URL dict.

    Accepts:
      - {"url": "https://..."} — pass through
      - {"url": "data:image/png;base64,..."} — pass through
      - {"path": "/local/file.png"} — read + base64 encode
      - "https://..." — wrap as url
      - "/local/file.png" — read + base64 encode
    """
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return {"url": image}
        image = {"path": image}

    if "url" in image:
        return {"url": image["url"]}

    if "path" in image:
        path = Path(image["path"])
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        data = path.read_bytes()
        validate_image_bytes(data, path.name)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return {"url": f"data:{mime};base64,{b64}"}

    if "data" in image:
        return {"url": image["data"]}

    raise ValueError(f"Cannot resolve image: {image!r}")


def validate_image_bytes(data: bytes, filename: str = "") -> None:
    """Validate image data: check size and MIME type."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large: {len(data)} bytes (max {MAX_IMAGE_BYTES})"
        )
    mime = _detect_mime(data, filename)
    if mime not in SUPPORTED_MIMES:
        raise ValueError(f"Unsupported image type: {mime}")


def _detect_mime(data: bytes, filename: str = "") -> str:
    """Detect MIME type from magic bytes or filename."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # Fall back to filename
    if filename:
        mime = mimetypes.guess_type(filename)[0]
        if mime:
            return mime
    return "application/octet-stream"
