"""Synthetic datasets and the two memory-mode return strategies.

This module is the stand-in for the future ``hip_cargo.runtime`` memory helpers
(RFC §5.1, §5.5). It builds synthetic image cubes of realistic shape and
implements the two ways a step hands its result to the next step across the
Ray ObjectRef boundary:

* ``greedy``       — a reified, in-memory (plasma-resident) dataset; disk is
                     fully bypassed between steps.
* ``conservative`` — a small, lazily store-backed dataset (``xr.open_zarr``
                     without dask); reads are deferred to a shared store.

On a single node the distinction is largely cosmetic (plasma vs. a local zarr
on the same disk); the demonstrator proves the *contract*, not the disk-bypass
performance characteristic, which only a real cluster shows (RFC §12 stage 2.5).
"""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

MemoryMode = Literal["greedy", "conservative"]

DIMS = ("band", "stokes", "x", "y")


def make_synth_dataset(n_band: int, n_stokes: int, n_x: int, n_y: int, seed: int = 42) -> xr.Dataset:
    """Build a synthetic image cube of shape ``(band, stokes, x, y)``.

    Stands in for reading a measurement set — no radio-astronomy dependencies,
    just numpy noise shaped like an imaging cube.

    Args:
        n_band: Number of frequency bands.
        n_stokes: Number of Stokes parameters.
        n_x: Image width in pixels.
        n_y: Image height in pixels.
        seed: RNG seed for reproducibility.

    Returns:
        An in-memory ``xr.Dataset`` with a single ``vis`` data variable.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_band, n_stokes, n_x, n_y)).astype("float32")
    return xr.Dataset(
        data_vars={"vis": (DIMS, data)},
        coords={
            "band": np.arange(n_band),
            "stokes": np.arange(n_stokes),
            "x": np.arange(n_x),
            "y": np.arange(n_y),
        },
        attrs={"synthetic": 1},
    )


def reify_greedy(ds: xr.Dataset) -> xr.Dataset:
    """Greedy mode: materialise the dataset in memory.

    Once returned from a Ray task, the array data lives in the plasma object
    store and is handed to the next worker directly, bypassing disk.
    """
    return ds.load()


def store_conservative(ds: xr.Dataset, store: str | Path) -> xr.Dataset:
    """Conservative mode: persist to a zarr store, return a lazy handle.

    The returned dataset is opened with ``chunks=None`` (no dask) so it is a
    thin, store-backed handle whose reads resolve on access rather than holding
    the array data in memory.
    """
    store = str(store)
    ds.to_zarr(store, mode="w")
    return xr.open_zarr(store, chunks=None)


def finalize(
    ds: xr.Dataset,
    memory_mode: MemoryMode,
    work_dir: str | Path,
    name: str,
    job_id: str,
) -> xr.Dataset:
    """Apply the memory-mode return strategy.

    The static return type is ``xr.Dataset`` in both modes — the mode changes
    the *kind* of dataset, not the signature (RFC §5.5).
    """
    if memory_mode == "greedy":
        return reify_greedy(ds)
    store = Path(work_dir) / f"{name}_{job_id}.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    return store_conservative(ds, store)
