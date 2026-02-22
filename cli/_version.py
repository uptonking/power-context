"""Version management using package metadata."""
from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("power-context")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0.dev0"
