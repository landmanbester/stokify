"""Tests for the committed `hip-cargo transpile` output in src/stokify/transpiled/.

The structural tests pin the generated package to the hand-written exemplars
in runtime/ (same DAG, same in-memory targets); the slow test runs the
transpiled pipeline end-to-end on a local Ray cluster with monitoring and
asserts events + per-task diagnostics flow, mirroring the classic path.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
TRANSPILED = REPO / "src" / "stokify" / "transpiled"


def test_transpiled_matches_handwritten_dag():
    from stokify.runtime import runner as handwritten
    from stokify.transpiled import runner as generated

    assert generated.STEPS == handwritten.STEPS
    assert generated.EDGES == handwritten.EDGES


def test_transpiled_targets_same_inmem_functions():
    tasks_src = (TRANSPILED / "tasks.py").read_text()
    for target in ("init_inmem", "process_inmem", "image_inmem"):
        assert target in tasks_src


def test_committed_output_is_current():
    """Re-running the transpiler changes nothing (committed output is fresh)."""
    result = subprocess.run(
        [
            "hip-cargo",
            "transpile",
            "--recipe",
            str(REPO / "src" / "stokify" / "recipes" / "stokify.yml"),
            "--output-dir",
            str(TRANSPILED),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "output already up to date" in result.stdout


def test_transpiled_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "stokify", "transpiled", "run", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:  # no __main__; go through the console script
        result = subprocess.run(["stokify", "transpiled", "run", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--base-dir" in result.stdout
    assert "--memory-mode" in result.stdout
    assert "--monitor" in result.stdout


@pytest.mark.slow
def test_transpiled_pipeline_end_to_end(tmp_path):
    """Run the generated runner on a local Ray cluster with monitoring."""
    import ray
    from hip_cargo.monitoring.ray_backend import get_or_create_aggregator

    from stokify.transpiled.runner import run_pipeline

    if not ray.is_initialized():
        ray.init(num_cpus=2, namespace="stokify", ignore_reinit_error=True)
    try:
        job_id, final_ref = run_pipeline(
            base_dir=str(tmp_path / "out"),
            memory_mode="greedy",
            n_band=1,
            n_stokes=1,
            n_x=8,
            n_y=8,
            niter=2,
            monitor=True,
            job_id="transpiledtest",
        )
        final = ray.get(final_ref)
        assert final is not None

        aggregator = get_or_create_aggregator()
        events = ray.get(aggregator.get_events.remote(job_id))
        types = [e["event_type"] for e in events]
        assert "pipeline_started" in types
        assert types.count("step_started") == 3
        assert types.count("step_completed") == 3
        assert "diagnostic" in types

        report = ray.get(aggregator.get_diagnostics.remote(job_id))
        steps_seen = {t["step"] for t in report["tasks"]}
        assert steps_seen == {"init", "process", "image"}
        for task in report["tasks"]:
            assert task["requested"] == {"num_cpus": 1}
            assert task["import_s"] >= 0
            assert task["cpu_utilisation"] is not None
    finally:
        ray.shutdown()
