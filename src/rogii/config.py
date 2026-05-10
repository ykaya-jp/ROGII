"""Config loader — reads YAML config and returns a dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/exp001.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
