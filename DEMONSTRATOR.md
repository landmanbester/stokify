# stokify — the hip-cargo transpiler demonstrator

`stokify` is the runnable demonstrator for the `hip-cargo transpile` RFC
(`hip-cargo/docs/design/transpile-rfc.md`, §12 stages 1–3). It is a
purpose-built, strictly-conforming three-step package — **init → process →
image** — that exercises hip-cargo's existing (`apis`-branch) monitoring stack
end-to-end on **synthetic data** (no casacore / measurement-set dependencies).

What it shows reviewers:

1. A restricted-grammar recipe (`src/stokify/recipes/stokify.yml`) — the *input*
   a transpiler would consume.
2. Hand-written `src/stokify/runtime/{tasks,runner,cli}.py` — the *output* a
   transpiler would emit (plain, mechanical Python; no clever abstractions).
3. The pipeline running on Ray and streaming **live progress** — events,
   metrics, and the DAG — through hip-cargo's unmodified FastAPI monitoring
   server.

Compare the recipe (1) against the runtime modules (2): the code generation is
mechanical. That is the RFC's central claim, made concrete.

---

## Layout

```
src/stokify/
  core/_synth.py        synthetic datasets + the two memory-mode return strategies
  core/{init,process,image}.py
                        in-memory step (*_inmem, chained by the runner) +
                        disk-persisting cab entry point (the cab `command:` target)
  cli/{init,process,image}.py
                        thin Typer wrappers -> generated cabs
  cabs/{init,process,image}.yml
                        generated via `hip-cargo generate-cabs`
  recipes/stokify.yml   the restricted-grammar recipe (would-be transpiler input)
  runtime/tasks.py      3 @ray.remote wrappers (per-step runtime_env as data; resources applied)
  runtime/runner.py     ObjectRef chaining + pipeline-level events (driver never ray.gets intermediates)
  runtime/cli.py        `stokify run` (transpiler-emitted driver) + `stokify monitor` (launcher glue)
demo.py                 self-verifying end-to-end monitoring demonstration
```

---

## Setup (two repos, side by side)

stokify depends on the **local** hip-cargo `apis` checkout (an editable path
dependency in `pyproject.toml`) so it picks up the unreleased monitoring and
progress APIs.

```bash
# hip-cargo lives at ../hip-cargo on the `apis` branch.
cd ~/software/stokify
uv sync --extra full          # installs hip-cargo[monitoring] (editable) + xarray/numpy/zarr
```

> If you do not use `uv`: `pip install -e ".[full]"` from a checkout whose
> `[tool.uv.sources]` is adjusted (or pip-install the local hip-cargo first).

---

## One-command demo (recommended — self-verifying)

```bash
uv run --extra full python demo.py --memory-mode greedy
# or: --memory-mode conservative
```

This starts a local Ray cluster, runs the pipeline with monitoring on, then
queries the monitoring server's REST + WebSocket endpoints and prints a summary
(DAG, event-type counts, the residual metric series as an ASCII chart, and the
live WebSocket frame count). It ends with `RESULT: PASS` and the honesty caveats.

It is the most robust thing to show: one command, no port coordination.

---

## Live two-terminal monitoring (the "production shape")

**There is no graphical dashboard yet** — that is the deliberately-not-yet-built
piece the transpiler will emit (RFC §5.9). The monitoring *server* exposes a JSON
REST API, a WebSocket stream, and Swagger UI; `http://127.0.0.1:8321/` is a
placeholder page and will look empty. Watch live progress with the `stokify
watch` terminal view (below) or by refreshing a JSON endpoint in the browser.
Events also **persist** after the run, so you can query them any time the server
is up — the run finishing quickly does not lose them.

```bash
# Terminal 1 — monitoring server (pins the Ray namespace to 'stokify')
uv run --extra full stokify monitor --port 8321

# Terminal 2 — run the pipeline, slowed down so it is watchable, with a fixed id
uv run --extra full stokify run --monitor --ray-address auto --job-id demo --sleep 1.0

# Terminal 3 — live terminal view (the closest thing to the future dashboard)
uv run --extra full stokify watch demo
```

Or observe in a browser / with curl (the GET endpoints render JSON directly):

```bash
open  http://127.0.0.1:8321/docs                          # Swagger UI — execute any endpoint
open  http://127.0.0.1:8321/api/progress/demo/events      # refresh to watch it grow
curl  http://127.0.0.1:8321/api/progress/demo/dag
curl  http://127.0.0.1:8321/api/progress/demo/metrics/residual
```

The Ray Dashboard (cluster/task analytics) is at http://127.0.0.1:8265.

> **Why `stokify monitor` and not `hip-cargo monitor`?** Ray scopes detached
> actors by namespace, and each `ray.init()` job gets a fresh anonymous
> namespace by default. `stokify monitor` is a thin launcher around hip-cargo's
> *unmodified* `create_app` that pins the namespace to `stokify`; `stokify run
> --monitor` joins the same namespace, so both processes share one
> `ProgressAggregator`. (Single-process `demo.py` needs none of this.)

---

## What is and isn't demonstrated (honesty caveats, RFC §12 stage 2.5)

* **In-process, not containerised.** `tasks.py` carries the per-step container
  `runtime_env` *shape* (image + run_options + env_vars) as the `*_RUNTIME_ENV`
  constants, but does **not** apply Ray's experimental `image_uri`. The local
  run executes the steps in-process. Validating `image_uri` on the target Ray
  version is a separate milestone (RFC §11).
* **Memory modes are cosmetic on one node.** `greedy` (plasma-resident) vs.
  `conservative` (a local zarr opened with `chunks=None`) is a real contract —
  the static return type is `xr.Dataset` in both modes, and both are verified to
  survive a Ray ObjectRef round-trip (`tests/test_memory_modes.py`) — but on a
  single node the disk-bypass *performance* characteristic only a real cluster
  shows is not demonstrated here.
* **Synthetic data.** `init` synthesises an image cube instead of reading a
  measurement set. The contract (typed callable, returns `xr.Dataset`, emits
  progress) is what matters for validating the target shape.

---

## Manual monitoring wiring (reference for the future `hip-cargo init` templates)

`hip-cargo init` does not yet scaffold monitoring (RFC §9.7). stokify wires it by
hand; these are the steps the templates should eventually automate:

1. **Dependency:** add `hip-cargo[monitoring]` (pulls fastapi/uvicorn/ray[default]).
2. **Instrument core functions:** wrap work in `track_progress(worker_name,
   total_steps, job_id, pipeline_run_id)` and call `tracker.step()` /
   `tracker.metric()` (see `core/{init,process,image}.py`).
3. **Activate the backend in each worker:** a `@ray.remote` task runs in its own
   process whose module-level backend defaults to `NullBackend`, so each task
   body must call
   `set_backend(RayProgressBackend(get_or_create_aggregator()))`
   (see `runtime/tasks.py::_activate_backend`). The driver does the same once
   (`runtime/runner.py`).
4. **Emit pipeline-level events** around each step submission: `PIPELINE_STARTED`
   (carrying the DAG in `extra`), `STEP_STARTED`, `STEP_COMPLETED`, `COMPLETED`
   (see `runtime/runner.py`).
5. **Share one namespace** across the server and the runner so the detached
   aggregator is the same actor (see `stokify monitor` / the runner's
   `ray.init(namespace="stokify")`).

---

## Planned: per-task diagnostics (RFC §5.10, spike stage 2.6)

**Not yet built.** The next monitoring increment is the lightweight per-task
diagnostics wrapper designed in the RFC (`hip-cargo/docs/design/transpile-rfc.md`
§5.10): each task emits one `DIAGNOSTIC` event (stdlib `getrusage` deltas —
wall / user / system CPU, peak RSS, block I/O — plus lazy-import time), and the
monitoring server joins these with the step timeline and the declared per-step
resource requests at `GET /api/progress/{job_id}/diagnostics`. That
requested-vs-used breakdown is the machine-readable contract an optimising
agent consumes; Ray's Dashboard keeps the cluster-wide view, Stimela's coarse
profiling has no per-task equivalent. When implemented, `demo.py` will print a
per-task diagnostics table and gate `RESULT: PASS` on the documented field set.

---

## Tests

```bash
uv run --extra full python -m pytest -m "not slow"   # fast: contract, recipe, install
uv run --extra full python -m pytest -m slow         # Ray ObjectRef round-trip + e2e demo
```
