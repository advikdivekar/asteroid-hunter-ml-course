"""Load a YAML config from configs/ and turn its paths into absolute paths."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """Return the settings in configs/<name>.yaml as a dict.

    Every entry under a top-level `paths:` key is converted to an absolute
    Path rooted at the project folder, so code never hard-codes
    /Users/... again.
    """
    config_file = CONFIG_DIR / f"{name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"No config named '{name}' (looked for {config_file})")

    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    if "paths" in config:
        config["paths"] = {key: PROJECT_ROOT / value for key, value in config["paths"].items()}
    return config
