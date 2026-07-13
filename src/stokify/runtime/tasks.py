"""Per-step ``@ray.remote`` task wrappers (would-be transpiler output).

Each step becomes one ``@ray.remote`` task. The heavy import lives *inside* the
task body so it executes in the worker, not on the driver (RFC §5.3). The driver
imports only ``ray``; the science stack (xarray/numpy) is pulled in here, where
in a real transpiled package it would be present in the per-step container image.

Two things the transpiler would attach to ``@ray.remote`` are modelled here:

* **Resources** (``num_cpus``) — *applied*, because they cost nothing locally
  and demonstrate per-step heterogeneity (a cheap ``init`` next to heavier
  ``process``/``image`` steps).
* **``runtime_env``** (the cab's container image + env_vars + run_options) —
  carried as the ``*_RUNTIME_ENV`` constants below but **not applied**, because
  Ray's per-task ``image_uri`` is experimental and the local demo runs the steps
  in-process on synthetic data (honesty caveat, RFC §12 stage 2.5). A real
  transpiled package would pass ``runtime_env=INIT_RUNTIME_ENV`` to the decorator.
"""

import os

# Must be set before ray is imported / initialised (see conftest / runner).
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray  # noqa: E402

# The container shape the transpiler would synthesise from each cab's `image`,
# `env_vars`, and `run_options`. A single worker image serves all three steps in
# this single-package demonstrator. Carried as data; not applied locally.
_WORKER_IMAGE = "ghcr.io/landmanbester/stokify:latest"
_COMMON_RUN_OPTIONS = ["--cap-drop=ALL", "--shm-size=8g"]
_COMMON_ENV_VARS = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

INIT_RUNTIME_ENV = {
    "container": {"image": _WORKER_IMAGE, "run_options": _COMMON_RUN_OPTIONS},
    "env_vars": _COMMON_ENV_VARS,
}
PROCESS_RUNTIME_ENV = {
    "container": {"image": _WORKER_IMAGE, "run_options": _COMMON_RUN_OPTIONS},
    "env_vars": _COMMON_ENV_VARS,
}
IMAGE_RUNTIME_ENV = {
    "container": {"image": _WORKER_IMAGE, "run_options": _COMMON_RUN_OPTIONS},
    "env_vars": _COMMON_ENV_VARS,
}


def _activate_backend() -> None:
    """Point this worker's progress backend at the shared aggregator actor.

    Each Ray worker is a separate process with its own module-level progress
    backend (defaulting to ``NullBackend``), so the ``RayProgressBackend`` must
    be registered *inside* the task body — registering it only on the driver
    would leave worker-emitted events going nowhere. The aggregator is a named,
    detached actor, so every worker resolves the same one the driver and the
    monitoring server use.
    """
    from hip_cargo.monitoring.ray_backend import RayProgressBackend, get_or_create_aggregator
    from hip_cargo.utils.progress import set_backend

    set_backend(RayProgressBackend(get_or_create_aggregator()))


@ray.remote(num_cpus=1)
def init_task(
    memory_mode: str,
    job_id: str,
    work_dir: str,
    monitor: bool,
    n_band: int,
    n_stokes: int,
    n_x: int,
    n_y: int,
    n_steps: int,
    sleep: float,
) -> "object":
    if monitor:
        _activate_backend()
    import time as _time

    _t0 = _time.perf_counter()
    from stokify.core.init import init_inmem

    if monitor:
        from hip_cargo.utils.diagnostics import annotate_diagnostics

        annotate_diagnostics(
            import_s=_time.perf_counter() - _t0,
            requested={"num_cpus": 1},
            memory_mode=memory_mode,
        )

    return init_inmem(
        memory_mode,
        job_id,
        job_id,
        work_dir,
        n_band=n_band,
        n_stokes=n_stokes,
        n_x=n_x,
        n_y=n_y,
        n_steps=n_steps,
        sleep=sleep,
    )


@ray.remote(num_cpus=2)
def process_task(
    dataset: "object",
    memory_mode: str,
    job_id: str,
    work_dir: str,
    monitor: bool,
    n_iterations: int,
    sleep: float,
) -> "object":
    if monitor:
        _activate_backend()
    import time as _time

    _t0 = _time.perf_counter()
    from stokify.core.process import process_inmem

    if monitor:
        from hip_cargo.utils.diagnostics import annotate_diagnostics

        annotate_diagnostics(
            import_s=_time.perf_counter() - _t0,
            requested={"num_cpus": 2},
            memory_mode=memory_mode,
        )

    return process_inmem(dataset, memory_mode, job_id, job_id, work_dir, n_iterations=n_iterations, sleep=sleep)


@ray.remote(num_cpus=2)
def image_task(
    dataset: "object",
    memory_mode: str,
    job_id: str,
    work_dir: str,
    monitor: bool,
    n_iterations: int,
    sleep: float,
) -> "object":
    if monitor:
        _activate_backend()
    import time as _time

    _t0 = _time.perf_counter()
    from stokify.core.image import image_inmem

    if monitor:
        from hip_cargo.utils.diagnostics import annotate_diagnostics

        annotate_diagnostics(
            import_s=_time.perf_counter() - _t0,
            requested={"num_cpus": 2},
            memory_mode=memory_mode,
        )

    return image_inmem(dataset, memory_mode, job_id, job_id, work_dir, n_iterations=n_iterations, sleep=sleep)
