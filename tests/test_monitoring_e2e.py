"""End-to-end monitoring test: the demo script must run green.

Marked ``slow`` because it stands up a local Ray cluster and the FastAPI app.
Mirrors what reviewers see when they run ``python demo.py`` (RFC §12 stage 2.5).
"""

import sys
from pathlib import Path

import pytest

# Make the repo-root demo.py importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.slow
def test_run_demo_passes():
    from demo import run_demo

    assert run_demo("greedy") == 0
