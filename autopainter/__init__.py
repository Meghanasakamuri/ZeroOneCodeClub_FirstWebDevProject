"""AutoPainter Pix2Pix-based sketch-to-painting toolkit."""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("autopainter")
except PackageNotFoundError:  # pragma: no cover - local dev fallback
    __version__ = "0.1.0"


__all__ = ["__version__"]
