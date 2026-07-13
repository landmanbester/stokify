"""The ``init`` step: read a measurement set into an image cube.

For the demonstrator this *synthesises* a dataset rather than reading a real
measurement set (no casacore / python-casacore dependency — RFC §12 stage 2.5).
The contract that matters is preserved: a plain Python callable that takes a
``memory_mode``, returns an ``xr.Dataset``, and emits progress events.

Two callables live here:

* :func:`init_inmem` — the in-memory step the Ray runner chains by ObjectRef.
* :func:`init` — the disk-persisting cab entry point (``stokify.core.init.init``)
  that a standalone ``stokify init`` invocation runs.
"""

import time
import uuid
from pathlib import Path

import xarray as xr
from hip_cargo import track_progress

from stokify.core._synth import MemoryMode, finalize, make_synth_dataset


def init_inmem(
    memory_mode: MemoryMode,
    job_id: str,
    pipeline_run_id: str,
    work_dir: str | Path,
    n_band: int = 4,
    n_stokes: int = 2,
    n_x: int = 256,
    n_y: int = 256,
    n_steps: int = 8,
    sleep: float = 0.15,
) -> xr.Dataset:
    """Synthesise an image cube, emitting one PROGRESS + METRIC per read block.

    Args:
        memory_mode: ``"greedy"`` (reified) or ``"conservative"`` (store-backed).
        job_id: Shared id grouping all events of this pipeline run.
        pipeline_run_id: Pipeline run id (stored in event ``extra``).
        work_dir: Directory for conservative-mode zarr stores.
        n_band: Number of frequency bands.
        n_stokes: Number of Stokes parameters.
        n_x: Image width in pixels.
        n_y: Image height in pixels.
        n_steps: Number of simulated read blocks.
        sleep: Per-block simulated compute time (seconds).

    Returns:
        An ``xr.Dataset`` whose kind depends on ``memory_mode``.
    """
    total_rows = n_band * n_stokes * n_x * n_y
    with track_progress("init", total_steps=n_steps, job_id=job_id, pipeline_run_id=pipeline_run_id) as tracker:
        ds = make_synth_dataset(n_band, n_stokes, n_x, n_y)
        for i in range(n_steps):
            time.sleep(sleep)
            tracker.step(message=f"Reading visibility block {i + 1}/{n_steps}")
            tracker.metric("rows_read", float(int(total_rows * (i + 1) / n_steps)))
    return finalize(ds, memory_mode, work_dir, "init", job_id)


def init(
    output: str,
    memory_mode: MemoryMode = "greedy",
    n_band: int = 4,
    n_stokes: int = 2,
    n_x: int = 256,
    n_y: int = 256,
    n_steps: int = 8,
) -> None:
    """Cab entry point: synthesise a dataset and persist it to ``output`` (zarr).

    This is the importable ``flavour: python`` callable the generated cab's
    ``command`` points at. The pipeline runner uses :func:`init_inmem` directly;
    this disk-persisting sibling is what a standalone ``stokify init`` runs.
    """
    job_id = uuid.uuid4().hex[:8]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = init_inmem(
        memory_mode, job_id, job_id, out.parent, n_band=n_band, n_stokes=n_stokes, n_x=n_x, n_y=n_y, n_steps=n_steps
    )
    ds.to_zarr(str(out), mode="w")
