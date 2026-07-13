"""ObjectRef-chaining pipeline runner (would-be transpiler output).

Chains the three steps purely by Ray ObjectRef — init -> process -> image —
holding only opaque refs and never calling ``ray.get`` on an intermediate, so
the driver never materialises an ``xr.Dataset`` and a lightweight driver install
never needs the heavy stack (RFC §5.4). Pipeline-level lifecycle events
(``PIPELINE_STARTED`` / ``STEP_STARTED`` / ``STEP_COMPLETED`` / ``COMPLETED``)
are emitted around each submission; the per-iteration ``PROGRESS`` / ``METRIC``
events come from the ``track_progress`` blocks inside the core functions.
"""

import os

# Must be set before ray is imported / initialised.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import uuid  # noqa: E402

from hip_cargo.utils.progress import EventType, ProgressEvent, emit  # noqa: E402

STEPS = ["init", "process", "image"]
EDGES = [["init", "process"], ["process", "image"]]


def _emit(event_type: EventType, job_id: str, worker_name: str, message: str = "", extra: dict | None = None) -> None:
    """Emit a pipeline-level event through the active backend."""
    emit(
        ProgressEvent(
            job_id=job_id,
            worker_name=worker_name,
            event_type=event_type,
            message=message,
            extra=extra or {},
        )
    )


def run_pipeline(
    work_dir: str,
    memory_mode: str = "greedy",
    monitor: bool = False,
    job_id: str | None = None,
    ray_address: str | None = None,
    n_band: int = 4,
    n_stokes: int = 2,
    n_x: int = 256,
    n_y: int = 256,
    n_steps: int = 8,
    process_iterations: int = 10,
    image_iterations: int = 10,
    sleep: float = 0.12,
) -> "tuple[str, object]":
    """Run the linear stokify pipeline on a Ray cluster.

    Args:
        work_dir: Directory for conservative-mode zarr stores.
        memory_mode: ``"greedy"`` or ``"conservative"``.
        monitor: If True, register a ``RayProgressBackend`` so events flow to
            the shared aggregator (and onward to the monitoring server).
        job_id: Run id all events are grouped under. Auto-generated if None.
        ray_address: Ray cluster address; None starts/uses a local cluster.
        n_band: Number of frequency bands.
        n_stokes: Number of Stokes parameters.
        n_x: Image width in pixels.
        n_y: Image height in pixels.
        n_steps: Simulated read blocks in the ``init`` step.
        process_iterations: Synthetic solver iterations in ``process``.
        image_iterations: Synthetic major cycles in ``image``.
        sleep: Per-iteration simulated compute time (seconds).

    Returns:
        ``(job_id, final_ref)`` — the run id, and the ObjectRef of the final
        ``image`` dataset (deliberately *not* materialised here).
    """
    import ray

    from stokify.runtime import tasks

    if job_id is None:
        job_id = uuid.uuid4().hex[:8]

    if not ray.is_initialized():
        # Pin a named namespace so the detached aggregator actor is shared with
        # the monitoring server (which `stokify monitor` launches in the same
        # namespace) across processes in the live two-terminal flow.
        ray.init(address=ray_address, namespace="stokify", ignore_reinit_error=True)

    if monitor:
        from hip_cargo.monitoring.ray_backend import RayProgressBackend, get_or_create_aggregator
        from hip_cargo.utils.progress import set_backend

        set_backend(RayProgressBackend(get_or_create_aggregator()))

    work_dir = str(work_dir)
    _emit(
        EventType.PIPELINE_STARTED,
        job_id,
        "stokify",
        message=f"memory_mode={memory_mode}",
        extra={"steps": STEPS, "edges": EDGES, "memory_mode": memory_mode},
    )

    # init -> ref0
    _emit(EventType.STEP_STARTED, job_id, "init", extra={"step_index": 0})
    ref0 = tasks.init_task.remote(memory_mode, job_id, work_dir, monitor, n_band, n_stokes, n_x, n_y, n_steps, sleep)
    ray.wait([ref0])  # block on completion WITHOUT materialising the dataset
    _emit(EventType.STEP_COMPLETED, job_id, "init", extra={"step_index": 0})

    # process -> ref1 (consumes ref0 — Ray resolves the ObjectRef inside the worker)
    _emit(EventType.STEP_STARTED, job_id, "process", extra={"step_index": 1})
    ref1 = tasks.process_task.remote(ref0, memory_mode, job_id, work_dir, monitor, process_iterations, sleep)
    ray.wait([ref1])
    _emit(EventType.STEP_COMPLETED, job_id, "process", extra={"step_index": 1})

    # image -> ref2 (consumes ref1)
    _emit(EventType.STEP_STARTED, job_id, "image", extra={"step_index": 2})
    ref2 = tasks.image_task.remote(ref1, memory_mode, job_id, work_dir, monitor, image_iterations, sleep)
    ray.wait([ref2])
    _emit(EventType.STEP_COMPLETED, job_id, "image", extra={"step_index": 2})

    _emit(EventType.COMPLETED, job_id, "stokify", message="pipeline complete")
    return job_id, ref2
