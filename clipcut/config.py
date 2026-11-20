from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "editorial.yaml"
DEFAULT_LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"
ASSETS_DIR = PROJECT_ROOT / "assets"


def load_local_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        path = config_path
    else:
        env_p = os.environ.get("CLIPCUT_LOCAL_CONFIG", "").strip()
        path = Path(env_p).expanduser() if env_p else DEFAULT_LOCAL_CONFIG_PATH
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_editorial_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Editorial config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_editorial_config(data: dict[str, Any], config_path: Path | None = None) -> Path:
    path = config_path or DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def load_editorial_raw(config_path: Path | None = None) -> str:
    path = config_path or DEFAULT_CONFIG_PATH
    return path.read_text(encoding="utf-8")


def save_editorial_raw(raw_yaml: str, config_path: Path | None = None) -> dict[str, Any]:
    parsed = yaml.safe_load(raw_yaml)
    if not isinstance(parsed, dict):
        raise ValueError("Editorial YAML must parse to a mapping")
    path = config_path or DEFAULT_CONFIG_PATH
    path.write_text(raw_yaml, encoding="utf-8")
    return parsed
