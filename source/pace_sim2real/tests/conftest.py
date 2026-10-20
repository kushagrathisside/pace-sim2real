# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

import sys
from pathlib import Path

import pytest

# make the package and the test helpers importable without installing them
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR.parent))
sys.path.insert(0, str(TESTS_DIR))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="Run the slow multi-joint recovery benchmarks.")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: multi-joint parameter recovery benchmarks (minutes), run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow benchmark, run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def log_dir(tmp_path):
    return str(tmp_path / "logs")
