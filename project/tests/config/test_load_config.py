from pathlib import Path

import pytest

from asthunt.config.load_config import PROJECT_ROOT, load_config


def test_project_config_loads_with_seed():
    config = load_config("project")
    assert config["seed"] == 42


def test_paths_become_absolute_and_inside_project():
    config = load_config("project")
    for path in config["paths"].values():
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_missing_config_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="no_such_stage"):
        load_config("no_such_stage")
