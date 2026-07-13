"""The ``process`` step: coherencies -> image-ready Stokes visibilities.

Synthetic transformation for the demonstrator. Emits a falling ``residual`` and
a rising ``convergence`` per iteration so the monitoring metrics look like a
real solver converging.

Two callables live here:

* :func:`process_inmem` — the in-memory step the Ray runner chains by ObjectRef.
* :func:`process` — the disk-persisting cab entry point.
"""

import time
import uuid
from pathlib import Path

import xarray as xr
from hip_cargo import track_progress

from stokify.core._synth import DIMS, MemoryMode, finalize


def process_inmem(
    dataset: xr.Dataset,
    memory_mode: MemoryMode,
    job_id: str,
    pipeline_run_id: str,
    work_dir: str | Path,
    n_iterations: int = 10,
    sleep: float = 0.15,
) -> xr.Dataset:
    """Transform the upstream cube into Stokes visibilities (synthetic).

    Args:
        dataset: Upstream ``xr.Dataset`` (Ray resolves the ObjectRef to the
            real dataset inside this worker before the body runs).
        memory_mode: ``"greedy"`` or ``"conservative"``.
        job_id: Shared id grouping all events of this pipeline run.
        pipeline_run_id: Pipeline run id (stored in event ``extra``).
        work_dir: Directory for conservative-mode zarr stores.
        n_iterations: Number of synthetic solver iterations.
        sleep: Per-iteration simulated compute time (seconds).

    Returns:
        An ``xr.Dataset`` whose kind depends on ``memory_mode``.
    """
    with track_progress("process", total_steps=n_iterations, job_id=job_id, pipeline_run_id=pipeline_run_id) as tracker:
        result = dataset["vis"].values  # materialise the upstream ref's data in this worker
        for i in range(n_iterations):
            time.sleep(sleep)
            result = result - 0.1 * result  # synthetic contraction toward a solution
            tracker.step(message=f"Stokes iteration {i + 1}/{n_iterations}")
            tracker.metric("residual", 1.0 / (i + 1))
            tracker.metric("convergence", 1.0 - 1.0 / (i + 1))
        out = dataset.copy(deep=False)
        out["vis"] = (DIMS, result)
    return finalize(out, memory_mode, work_dir, "process", job_id)


def process(
    output: str,
    input_data: str,
    memory_mode: MemoryMode = "greedy",
    n_iterations: int = 10,
) -> None:
    """Cab entry point: open ``input_data`` (zarr), process it, persist to ``output``."""
    job_id = uuid.uuid4().hex[:8]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_in = xr.open_zarr(str(input_data), chunks=None)
    ds = process_inmem(ds_in, memory_mode, job_id, job_id, out.parent, n_iterations=n_iterations)
    ds.to_zarr(str(out), mode="w")
