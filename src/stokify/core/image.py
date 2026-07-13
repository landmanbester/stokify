"""The ``image`` step: grid and image the Stokes visibilities.

Synthetic imaging for the demonstrator. Emits a falling ``rms`` and the current
``peak`` per major cycle.

Two callables live here:

* :func:`image_inmem` — the in-memory step the Ray runner chains by ObjectRef.
* :func:`image` — the disk-persisting cab entry point.
"""

import time
import uuid
from pathlib import Path

import numpy as np
import xarray as xr
from hip_cargo import track_progress

from stokify.core._synth import DIMS, MemoryMode, finalize


def image_inmem(
    dataset: xr.Dataset,
    memory_mode: MemoryMode,
    job_id: str,
    pipeline_run_id: str,
    work_dir: str | Path,
    n_iterations: int = 10,
    sleep: float = 0.15,
) -> xr.Dataset:
    """Grid and image the Stokes visibilities (synthetic).

    Args:
        dataset: Upstream ``xr.Dataset`` (Ray resolves the ObjectRef inside
            this worker before the body runs).
        memory_mode: ``"greedy"`` or ``"conservative"``.
        job_id: Shared id grouping all events of this pipeline run.
        pipeline_run_id: Pipeline run id (stored in event ``extra``).
        work_dir: Directory for conservative-mode zarr stores.
        n_iterations: Number of synthetic major cycles.
        sleep: Per-cycle simulated compute time (seconds).

    Returns:
        An ``xr.Dataset`` with an added ``image`` variable; its kind depends on
        ``memory_mode``.
    """
    with track_progress("image", total_steps=n_iterations, job_id=job_id, pipeline_run_id=pipeline_run_id) as tracker:
        vis = dataset["vis"].values
        img = np.zeros_like(vis)
        for i in range(n_iterations):
            time.sleep(sleep)
            img = img + (vis - img) * 0.3  # synthetic major-cycle gain -> geometric rms decay
            tracker.step(message=f"Major cycle {i + 1}/{n_iterations}")
            tracker.metric("rms", float(np.std(vis - img)))
            tracker.metric("peak", float(np.max(np.abs(img))))
        out = dataset.copy(deep=False)
        out["image"] = (DIMS, img)
    return finalize(out, memory_mode, work_dir, "image", job_id)


def image(
    output: str,
    input_data: str,
    memory_mode: MemoryMode = "greedy",
    n_iterations: int = 10,
) -> None:
    """Cab entry point: open ``input_data`` (zarr), image it, persist to ``output``."""
    job_id = uuid.uuid4().hex[:8]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_in = xr.open_zarr(str(input_data), chunks=None)
    ds = image_inmem(ds_in, memory_mode, job_id, job_id, out.parent, n_iterations=n_iterations)
    ds.to_zarr(str(out), mode="w")
