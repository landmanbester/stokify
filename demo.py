#!/usr/bin/env python
"""End-to-end monitoring demonstration for the stokify spike (RFC §12 stage 2.5).

Runs the hand-written stokify pipeline on a local Ray cluster with monitoring
enabled, then queries the hip-cargo monitoring server's REST and WebSocket
endpoints to prove that real progress events, metrics, and the pipeline DAG flow
end-to-end through the existing (apis-branch) monitoring stack.

This is the artifact to show reviewers: one self-verifying command that prints a
human-readable summary of what the monitoring API returns.

Run:
    uv run --extra full python demo.py [--memory-mode greedy|conservative]
"""

import argparse
import os
import tempfile
import threading
import time
import warnings
from collections import Counter

# Must be set before ray is imported / initialised.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
# Keep the demonstrator output clean for a live pitch.
warnings.filterwarnings("ignore")

RULE = "=" * 68

# The stable per-task diagnostics field set (RFC §5.10 agent contract).
REQUIRED_DIAG_FIELDS = {
    "step",
    "wall_s",
    "cpu_user_s",
    "cpu_system_s",
    "rss_entry_mb",
    "peak_rss_mb",
    "hostname",
    "pid",
    "import_s",
    "requested",
    "memory_mode",
    "queue_lag_s",
    "cpu_utilisation",
}


def _settle(aggregator, job_id: str, timeout: float = 12.0) -> int:
    """Wait until the (fire-and-forget) event stream for a job stops growing."""
    import ray

    deadline = time.time() + timeout
    last, stable = -1, 0
    while time.time() < deadline:
        n = len(ray.get(aggregator.get_events.remote(job_id)))
        if n == last and n > 0:
            stable += 1
            if stable >= 2:
                return n
        else:
            stable = 0
        last = n
        time.sleep(0.4)
    return last


def _try_live_ws(client) -> int:
    """Best-effort live WebSocket streaming demo. Returns frames received."""
    from stokify.runtime.runner import run_pipeline

    ws_job = "wsdemo01"
    work_dir = tempfile.mkdtemp(prefix="stokify_ws_")

    def _run():
        time.sleep(0.5)  # let the WebSocket subscribe before events flow
        run_pipeline(
            work_dir,
            memory_mode="greedy",
            monitor=True,
            job_id=ws_job,
            n_x=64,
            n_y=64,
            n_steps=3,
            process_iterations=4,
            image_iterations=3,
            sleep=0.05,
        )

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    frames = []
    try:
        with client.websocket_connect(f"/ws/progress/{ws_job}") as ws:
            for _ in range(300):
                msg = ws.receive_json()
                if msg.get("type") == "heartbeat":
                    continue
                if msg.get("type") == "close":
                    break
                frames.append(msg)
                if msg.get("event_type") == "completed" and msg.get("worker_name") == "stokify":
                    break
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  websocket (best-effort) error: {exc}")
    worker.join(timeout=10)
    return len(frames)


def run_demo(memory_mode: str = "greedy", transpiled: bool = False) -> int:
    """Run the pipeline with monitoring and verify the API. Returns exit code."""
    import ray
    from fastapi.testclient import TestClient
    from hip_cargo.monitoring.ray_backend import RayProgressBackend, get_or_create_aggregator
    from hip_cargo.monitoring.server import create_app
    from hip_cargo.utils.progress import set_backend

    if transpiled:
        from stokify.transpiled.runner import run_pipeline
    else:
        from stokify.runtime.runner import run_pipeline

    print(f"\n{RULE}\n stokify monitoring demonstrator  (RFC §12 stage 2.5)\n{RULE}")

    # A named namespace lets the detached aggregator actor be shared cleanly
    # (and silences Ray's anonymous-namespace notice).
    ray.init(namespace="stokify", ignore_reinit_error=True, log_to_driver=False)
    aggregator = get_or_create_aggregator()
    set_backend(RayProgressBackend(aggregator))

    work_dir = tempfile.mkdtemp(prefix="stokify_demo_")
    variant = "transpiled (hip-cargo transpile output)" if transpiled else "hand-written runtime"
    print(f"\n=== Running stokify pipeline (memory_mode={memory_mode}, runner: {variant}) ===")
    t0 = time.time()
    # The generated runner's signature comes from the recipe inputs; it is
    # call-compatible with the hand-written one for the arguments used here.
    job_id, _final_ref = run_pipeline(work_dir, memory_mode=memory_mode, monitor=True, n_x=128, n_y=128)
    n_events = _settle(aggregator, job_id)
    print(f"Pipeline complete in {time.time() - t0:.1f}s. job_id={job_id}, {n_events} events captured.")

    failures: list[str] = []
    with TestClient(create_app()) as client:
        dag_resp = client.get(f"/api/progress/{job_id}/dag")
        events_resp = client.get(f"/api/progress/{job_id}/events")
        latest_resp = client.get(f"/api/progress/{job_id}")
        residual_resp = client.get(f"/api/progress/{job_id}/metrics/residual")

        # --- DAG ---
        print("\n--- GET /api/progress/{job_id}/dag ---")
        if dag_resp.status_code == 200:
            dag = dag_resp.json()
            print(f"  steps: {' -> '.join(dag.get('steps', []))}")
            print(f"  edges: {', '.join('->'.join(e) for e in dag.get('edges', []))}")
            print(f"  memory_mode: {dag.get('memory_mode')}")
            if dag.get("steps") != ["init", "process", "image"]:
                failures.append("DAG steps mismatch")
        else:
            failures.append(f"/dag returned {dag_resp.status_code}")

        # --- Events ---
        print("\n--- GET /api/progress/{job_id}/events ---")
        if events_resp.status_code == 200:
            events = events_resp.json()
            counts = Counter(e["event_type"] for e in events)
            print(f"  total events: {len(events)}")
            for et in (
                "pipeline_started",
                "step_started",
                "started",
                "progress",
                "metric",
                "step_completed",
                "completed",
            ):
                print(f"    {et:<18}{counts.get(et, 0)}")
            if counts.get("step_started", 0) != 3:
                failures.append(f"expected 3 step_started, got {counts.get('step_started', 0)}")
            if counts.get("pipeline_started", 0) != 1:
                failures.append("missing pipeline_started")
            if not any(e["event_type"] == "completed" and e["worker_name"] == "stokify" for e in events):
                failures.append("missing pipeline-level completed")
        else:
            failures.append(f"/events returned {events_resp.status_code}")

        # --- Latest ---
        print("\n--- GET /api/progress/{job_id} (latest) ---")
        if latest_resp.status_code == 200:
            latest = latest_resp.json()
            print(f"  worker={latest['worker_name']}  event={latest['event_type']}  msg='{latest['message']}'")
        else:
            failures.append(f"/progress returned {latest_resp.status_code}")

        # --- Metric time series ---
        print("\n--- GET /api/progress/{job_id}/metrics/residual ---")
        if residual_resp.status_code == 200:
            series = residual_resp.json()
            for point in series:
                bar = "#" * max(1, int(point["value"] * 30))
                print(f"    step {point['step']:>2}  {point['value']:.4f}  {bar}")
            values = [p["value"] for p in series]
            decreasing = all(a >= b for a, b in zip(values, values[1:]))
            print(f"  monotonically decreasing: {'yes' if decreasing else 'no'}")
            if not values:
                failures.append("no residual metric series")
            elif not decreasing:
                failures.append("residual not decreasing")
        else:
            failures.append(f"/metrics returned {residual_resp.status_code}")

        # --- Per-task diagnostics (RFC §5.10 agent contract) ---
        diag_resp = client.get(f"/api/progress/{job_id}/diagnostics")
        print("\n--- GET /api/progress/{job_id}/diagnostics ---")
        if diag_resp.status_code == 200:
            report = diag_resp.json()
            print(f"    {'step':<10}{'wall_s':>8}{'cpu_s':>8}{'util':>6}{'peak_mb':>9}{'lag_s':>7}{'import_s':>9}")
            for task in report["tasks"]:
                cpu_s = task["cpu_user_s"] + task["cpu_system_s"]
                util = task["cpu_utilisation"]
                util_str = "n/a" if util is None else f"{util:.2f}"
                lag = task["queue_lag_s"]
                lag_str = "n/a" if lag is None else f"{lag:.2f}"
                print(
                    f"    {task['step']:<10}"
                    f"{task['wall_s']:>8.2f}"
                    f"{cpu_s:>8.2f}"
                    f"{util_str:>6}"
                    f"{task['peak_rss_mb']:>9.0f}"
                    f"{lag_str:>7}"
                    f"{task['import_s']:>9.3f}"
                )
            steps_seen = {t["step"] for t in report["tasks"]}
            if steps_seen != {"init", "process", "image"}:
                failures.append(f"diagnostics missing steps: {steps_seen}")
            for task in report["tasks"]:
                missing = REQUIRED_DIAG_FIELDS - set(task)
                if missing:
                    failures.append(f"diagnostics fields missing for {task['step']}: {sorted(missing)}")
        else:
            failures.append(f"/diagnostics returned {diag_resp.status_code}")

        # --- Live WebSocket (best effort) ---
        print("\n--- WS /ws/progress/{job_id} (live stream) ---")
        ws_frames = _try_live_ws(client)
        print(f"  streamed {ws_frames} live frames")
        if ws_frames == 0:
            print("  (note: WebSocket streamed no frames — REST proof above is authoritative)")

    ray.shutdown()

    print(f"\n{RULE}")
    if failures:
        print(" RESULT: FAIL")
        for f in failures:
            print(f"   - {f}")
        print(RULE)
        return 1
    print(" RESULT: PASS — events, metrics, DAG, and per-task diagnostics flow through the monitoring API.")
    print(RULE)
    print(
        "\nHonesty caveats (RFC §12 stage 2.5):\n"
        "  * Steps run in-process locally; tasks.py carries the per-step container\n"
        "    runtime_env shape as data but does not launch image_uri containers\n"
        "    (experimental; validated separately).\n"
        "  * On a single node greedy vs conservative is cosmetic (plasma vs a local\n"
        "    zarr on the same disk) — this proves the contract + event flow, not the\n"
        "    disk-bypass performance characteristic, which only a real cluster shows.\n"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-mode",
        choices=["greedy", "conservative"],
        default="greedy",
        help="Which memory mode to run the demonstrator in.",
    )
    parser.add_argument(
        "--transpiled",
        action="store_true",
        help="Drive the pipeline through the generated src/stokify/transpiled/ runner instead of runtime/.",
    )
    args = parser.parse_args()
    return run_demo(args.memory_mode, transpiled=args.transpiled)


if __name__ == "__main__":
    raise SystemExit(main())
