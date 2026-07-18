"""Shared pytest configuration.

Adds the ``java_comparison`` marker and a ``--run-java`` opt-in flag. Strict
Python-vs-Java behavioral-fidelity comparisons (which run the real Java Genius
agents through the bridge) are **deselected by default**: the ports are
AI-assisted approximations and are not expected to match the Java agents
bit-for-bit, so those comparisons are a fidelity/diagnostic tool rather than a
pass/fail gate. Run them explicitly with::

    pytest --run-java
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-java",
        action="store_true",
        default=False,
        help="run the strict Python-vs-Java behavioral fidelity comparisons "
        "(requires a running Genius bridge)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "java_comparison: strict Python-vs-Java behavioral comparison via the "
        "Genius bridge (opt-in via --run-java)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-java"):
        return
    skip = pytest.mark.skip(reason="needs --run-java (strict Java fidelity comparison)")
    for item in items:
        if "java_comparison" in item.keywords:
            item.add_marker(skip)
