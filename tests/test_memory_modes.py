"""Unit tests for the synthetic core step contract.

Covers the two memory modes, the ``xr.Dataset`` return type, and the progress
events each step emits. A Ray ObjectRef round-trip test (marked ``slow``)
verifies that both memory-mode datasets survive being passed between Ray tasks
— the property the hand-written runner relies on.
"""

import pytest
import xarray as xr
from hip_cargo.utils.progress import EventType, NullBackend, set_backend

from stokify.core._synth import make_synth_dataset
from stokify.core.image import image_inmem
from stokify.core.init import init_inmem
from stokify.core.process import process_inmem

SMALL = dict(n_band=2, n_stokes=1, n_x=16, n_y=16)


class _Recorder:
    """Progress backend that records every emitted event."""

    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass

    def types(self) -> list[str]:
        return [e.event_type.value for e in self.events]

    def metric_names(self) -> set[str]:
        return {e.metric_name for e in self.events if e.event_type == EventType.METRIC}


@pytest.fixture
def recorder():
    rec = _Recorder()
    set_backend(rec)
    yield rec
    set_backend(NullBackend())


def test_make_synth_dataset_shape():
    ds = make_synth_dataset(4, 2, 32, 32)
    assert ds["vis"].shape == (4, 2, 32, 32)
    assert ds["vis"].dims == ("band", "stokes", "x", "y")


def test_init_greedy_returns_in_memory_dataset(recorder, tmp_path):
    ds = init_inmem("greedy", "job1", "job1", tmp_path, n_steps=3, sleep=0.0, **SMALL)
    assert isinstance(ds, xr.Dataset)
    assert ds["vis"].shape == (2, 1, 16, 16)
    # greedy is materialised in memory — no backing store
    assert ds["vis"].chunks is None


def test_init_conservative_returns_dataset(recorder, tmp_path):
    ds = init_inmem("conservative", "job2", "job2", tmp_path, n_steps=3, sleep=0.0, **SMALL)
    assert isinstance(ds, xr.Dataset)
    assert ds["vis"].shape == (2, 1, 16, 16)
    assert (tmp_path / "init_job2.zarr").exists()


def test_step_emits_full_lifecycle(recorder, tmp_path):
    init_inmem("greedy", "job3", "job3", tmp_path, n_steps=4, sleep=0.0, **SMALL)
    types = recorder.types()
    assert types[0] == "started"
    # track_progress emits COMPLETED then one trailing DIAGNOSTIC record
    assert types[-2:] == ["completed", "diagnostic"]
    assert types.count("progress") == 4


def test_metrics_emitted_per_step(recorder, tmp_path):
    ds0 = init_inmem("greedy", "job4", "job4", tmp_path, n_steps=2, sleep=0.0, **SMALL)
    ds1 = process_inmem(ds0, "greedy", "job4", "job4", tmp_path, n_iterations=3, sleep=0.0)
    image_inmem(ds1, "greedy", "job4", "job4", tmp_path, n_iterations=3, sleep=0.0)
    assert {"rows_read", "residual", "convergence", "rms", "peak"} <= recorder.metric_names()
    assert "image" in image_inmem(ds1, "greedy", "job4", "job4", tmp_path, n_iterations=1, sleep=0.0)


def test_pipeline_chains_in_memory(tmp_path):
    """The three steps compose: init -> process -> image, both modes."""
    set_backend(NullBackend())
    for mode in ("greedy", "conservative"):
        ds0 = init_inmem(mode, "c", "c", tmp_path, n_steps=2, sleep=0.0, **SMALL)
        ds1 = process_inmem(ds0, mode, "c", "c", tmp_path, n_iterations=2, sleep=0.0)
        ds2 = image_inmem(ds1, mode, "c", "c", tmp_path, n_iterations=2, sleep=0.0)
        assert ds2["image"].shape == (2, 1, 16, 16)


@pytest.mark.slow
def test_objectref_roundtrip_both_modes(tmp_path):
    """Both memory-mode datasets must survive a Ray ObjectRef round-trip."""
    import ray

    ray.init(num_cpus=2, ignore_reinit_error=True)
    try:
        for mode in ("greedy", "conservative"):
            ds = init_inmem(mode, f"rt_{mode}", f"rt_{mode}", tmp_path, n_steps=2, sleep=0.0, **SMALL)
            got = ray.get(ray.put(ds))
            assert got["vis"].shape == (2, 1, 16, 16)
            assert float(got["vis"].sum()) == pytest.approx(float(ds["vis"].sum()), rel=1e-5)
    finally:
        ray.shutdown()
