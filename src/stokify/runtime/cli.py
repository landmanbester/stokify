"""The pipeline driver CLI (would-be transpiler output) + monitoring helpers.

``stokify run`` is the thin, lightweight driver the transpiler emits from the
recipe's top-level inputs: it parses arguments, optionally activates the
monitoring backend, and submits the ObjectRef-chained pipeline.

``stokify monitor`` and ``stokify watch`` are stokify glue (not transpiler
output). ``monitor`` is a thin launcher around hip-cargo's *unmodified*
``create_app`` that pins the Ray namespace to ``stokify`` so the server and
``stokify run --monitor`` share one detached ``ProgressAggregator`` across
processes. ``watch`` is a terminal live-view that polls the server's REST API —
a stand-in for the visual dashboard the transpiler will eventually emit (RFC
§5.9), useful for showing live progress without a browser frontend.
"""

import uuid
from typing import Annotated, Literal

import typer


def run(
    work_dir: Annotated[
        str,
        typer.Option(
            help="Directory for outputs / conservative-mode zarr stores.",
        ),
    ] = "stokify_run",
    memory_mode: Annotated[
        Literal["greedy", "conservative"],
        typer.Option(
            help="Memory mode.",
        ),
    ] = "greedy",
    monitor: Annotated[
        bool,
        typer.Option(
            help="Stream progress events to the monitoring aggregator.",
        ),
    ] = False,
    job_id: Annotated[
        str | None,
        typer.Option(
            help="Fixed job id so `stokify watch <job_id>` can follow it live.",
        ),
    ] = None,
    ray_address: Annotated[
        str | None,
        typer.Option(
            help="Ray cluster address; omit for a local cluster, 'auto' to join one.",
        ),
    ] = None,
    n_steps: Annotated[
        int,
        typer.Option(
            help="Simulated read blocks in the init step.",
        ),
    ] = 8,
    process_iterations: Annotated[
        int,
        typer.Option(
            help="Solver iterations in the process step.",
        ),
    ] = 10,
    image_iterations: Annotated[
        int,
        typer.Option(
            help="Major cycles in the image step.",
        ),
    ] = 10,
    sleep: Annotated[
        float,
        typer.Option(
            help="Per-iteration simulated compute time (s). Raise it (e.g. 1.0) to watch live.",
        ),
    ] = 0.3,
):
    """Run the linear stokify pipeline on a Ray cluster."""
    from stokify.runtime.runner import run_pipeline

    if job_id is None:
        job_id = uuid.uuid4().hex[:8]

    # Print the job id up front so it can be followed live while the run proceeds.
    typer.echo(f"job_id={job_id}")
    if monitor:
        typer.echo(f"follow it live:  stokify watch {job_id}")
        typer.echo(f"or in a browser: http://127.0.0.1:8321/api/progress/{job_id}/events")

    run_pipeline(
        work_dir,
        memory_mode=memory_mode,
        monitor=monitor,
        job_id=job_id,
        ray_address=ray_address,
        n_steps=n_steps,
        process_iterations=process_iterations,
        image_iterations=image_iterations,
        sleep=sleep,
    )
    typer.echo(f"pipeline complete: job_id={job_id}")


def monitor(
    port: Annotated[
        int,
        typer.Option(
            help="Port to serve the monitoring dashboard on.",
        ),
    ] = 8321,
    host: Annotated[
        str,
        typer.Option(
            help="Host to bind.",
        ),
    ] = "127.0.0.1",
    ray_address: Annotated[
        str | None,
        typer.Option(help="Ray cluster address; omit to start a local cluster."),
    ] = None,
):
    """Launch the hip-cargo monitoring server in the shared 'stokify' Ray namespace."""
    import os

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    import ray
    import uvicorn
    from hip_cargo.monitoring.config import MonitorSettings
    from hip_cargo.monitoring.server import create_app

    if not ray.is_initialized():
        ray.init(address=ray_address, namespace="stokify", ignore_reinit_error=True)

    typer.echo(f"stokify monitor: serving on http://{host}:{port}  (Ray namespace 'stokify')")
    typer.echo(f"  Swagger UI:      http://{host}:{port}/docs")
    typer.echo("  Ray Dashboard:   http://127.0.0.1:8265")
    typer.echo("  NOTE: the root page is a placeholder — the visual frontend is not built yet (RFC §5.9).")
    uvicorn.run(create_app(MonitorSettings(_env_file=None)), host=host, port=port)


def _format_event(event: dict) -> str:
    """One-line rendering of a progress event for the terminal live-view."""
    worker = event.get("worker_name", "")
    event_type = event.get("event_type", "")
    if event_type == "metric":
        value = event.get("metric_value")
        value_str = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        return f"  [{worker:<8}] metric   {event.get('metric_name')}={value_str}"
    if event_type == "progress":
        step = f"{event.get('current_step')}/{event.get('total_steps')}"
        return f"  [{worker:<8}] progress {step}  {event.get('message', '')}"
    return f"  [{worker:<8}] {event_type}  {event.get('message', '')}".rstrip()


def watch(
    job_id: Annotated[str, typer.Argument(help="The job id to follow (printed by `stokify run`).")],
    url: Annotated[str, typer.Option(help="Monitoring server base URL.")] = "http://127.0.0.1:8321",
    poll: Annotated[float, typer.Option(help="Poll interval in seconds.")] = 0.4,
    timeout: Annotated[float, typer.Option(help="Give up after this many seconds without completion.")] = 300.0,
):
    """Stream a pipeline's progress events to the terminal by polling the monitor REST API.

    A stand-in for the visual dashboard (RFC §5.9). Polls ``/events`` from the
    start, so it shows the full history whether started before, during, or after
    the run — no browser required.
    """
    import json
    import time
    import urllib.request

    typer.echo(f"watching job_id={job_id} at {url} (Ctrl-C to stop)\n")
    seen = 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/progress/{job_id}/events?since={seen}", timeout=5) as resp:
                events = json.load(resp)
        except Exception:
            time.sleep(poll)
            continue
        for event in events:
            typer.echo(_format_event(event))
            if event.get("event_type") == "completed" and event.get("worker_name") == "stokify":
                typer.echo("\npipeline complete.")
                return
        seen += len(events)
        time.sleep(poll)
    typer.echo("\nwatch timed out (no pipeline-level completion seen).")
