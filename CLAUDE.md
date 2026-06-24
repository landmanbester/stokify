# CLAUDE.md — Project Context

This project was bootstrapped with [`hip-cargo init`](https://github.com/landmanbester/hip-cargo).
It is a **hip-cargo package**: a Python CLI whose commands are decorated so that
Stimela cab definitions are generated automatically from the CLI source. Cabs
let the same commands be invoked from Stimela recipes and from `stokify`
on the command line interchangeably.

When working in this repo, treat the patterns in the rules below as load-bearing
— they are what makes the round-trip between CLI source, generated cabs, and
container fallback work. If you find yourself wanting to deviate, stop and check
[hip-cargo's own docs](https://github.com/landmanbester/hip-cargo) first.

*Note: Detailed architecture/domain logic, Python standards, and testing/CI
rules have been modularized into the `.claude/rules/` directory for progressive
disclosure. Read the relevant file before editing the matching files.*

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/stokify/**` — package layout, install modes, container fallback, cab generation. |
| `.claude/rules/python-standards.md` | any `**/*.py` — type hints, lazy imports, Typer syntax, hip-cargo types. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**` — round-trip tests, dev workflow, commits. |

---

## Where to Go Deeper
## 2. Lightweight vs Full Installation

This package supports two install modes. The split is what makes the
container-fallback pattern below work.

| Mode | Command | What it pulls | When to use |
|---|---|---|---|
| **Lightweight** | `pip install stokify` | `hip-cargo` + `typer` only | Cab consumers (Stimela), CI machines that only need to dispatch commands into containers, anyone who already has the project's container image available. |
| **Full** | `pip install stokify[full]` | Lightweight + everything listed under `[project.optional-dependencies].full` in `pyproject.toml` | Local development; native (non-container) execution. |

The lightweight install is **always sufficient to invoke any command** because
the generated CLI wrappers fall back to running the same command inside the
project's container when native imports fail (see §3).

### When you add a heavy dep

- Add it to `[project.optional-dependencies].full` in `pyproject.toml`, **not**
  to the top-level `dependencies`. The top-level list must stay tiny so the
  lightweight install remains lightweight.
- Import it **only from inside `core/`**. Never import it from `cli/` at module
  scope.

---

## 3. Container Fallback & Backends

Every generated CLI wrapper in `cli/*.py` follows the same shape (this is
emitted by `hip-cargo generate-function`, but the pattern matters when you
write a new command by hand too):

```python
def my_command(...):
    if backend == "native" or backend == "auto":
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist
            preflight_remote_must_exist(my_command, dict(...))
            from stokify.core.my_command import my_command as my_command_core
            my_command_core(...)
            return
        except ImportError:
            if backend == "native":
                raise
    # Heavy deps missing OR backend explicitly chose a container → run in container.
    from hip_cargo.utils.config import get_container_image
    from hip_cargo.utils.runner import run_in_container
    image = get_container_image("stokify")
    run_in_container(my_command, dict(...), image=image, backend=backend, ...)
```

### How `--backend` flows

Every command auto-grows two parameters via `hip-cargo generate-function`:

| Flag | Values | Effect |
|---|---|---|
| `--backend` | `auto` (default), `native`, `apptainer`, `singularity`, `docker`, `podman` | `auto` tries native then falls back to a detected container runtime. `native` forces in-process execution and surfaces the `ImportError` if `[full]` is not installed. The explicit backends skip the native attempt entirely and dispatch into the matching runtime. |
| `--always-pull-images` | bool | Forces a fresh `pull` before each container run. |

Both flags are decorated with `StimelaMeta(skip=True)` so they appear in the
Python CLI but **not** in the generated cab YAML — Stimela manages container
execution on its own side and doesn't need them.

### Image resolution

The image tag is owned by `src/stokify/_container_image.py`:

```python
CONTAINER_IMAGE = "ghcr.io/landmanbester/stokify:latest"
```

Three things keep this in sync — do not bypass them:

1. **Feature branches:** Edit `_container_image.py` by hand to point at your
   branch tag (e.g. `:my-feature`). The `publish-container.yml` workflow builds
   and pushes that tag on every push of the PR.
2. **Merge to `main`:** The `update-cabs.yml` workflow resets the
   tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **Releases:** `tbump <version>` rewrites the tag to the semantic version and
   regenerates cabs as a `before_commit` hook.

### GPU passthrough & extra run-args

For a GPU image, declare it in `src/stokify/_container_image.py` alongside
`CONTAINER_IMAGE`:

```python
GPU = True                  # True | False | "auto" | a device spec ("device=0,1")
RUN_ARGS_APPTAINER = []     # per-backend extras: _DOCKER / _PODMAN / _APPTAINER / _SINGULARITY
```

On the container-fallback path the runner translates `GPU` per runtime (`--gpus`
for docker, CDI `--device nvidia.com/gpu=...` for podman, `--nv` for
apptainer/singularity) and appends the matching `RUN_ARGS_*` verbatim. `"auto"`
only requests a GPU when one is detected (and, for docker/podman, the NVIDIA
Container Toolkit is present). Override per-invocation with `HIP_CARGO_GPUS`
(e.g. `HIP_CARGO_GPUS=none`) and `HIP_CARGO_RUN_ARGS`. These constants live only
in `_container_image.py` — they are deliberately kept out of the cab YAML, since
Stimela manages its own container execution. Explicit device specs (e.g. `device=0,1`) target docker and apptainer/singularity; on podman prefer `True`/`"auto"`/`"all"` since CDI uses a different device grammar.

### Remote URIs (S3 / GCS / Azure)

Path-typed parameters (`File`, `Directory`, `MS`, `URI`) accept both local
paths and remote URIs (`s3://...`, `gs://...`, `az://...`). When the path is
remote:

- `_resolve_mounts` skips it (nothing to bind-mount).
- `preflight_remote_must_exist` checks existence via fsspec.
- `run_in_container` forwards the matching credentials (`AWS_*`, `~/.aws`,
  `GOOGLE_APPLICATION_CREDENTIALS`, `~/.config/gcloud`, `AZURE_*`, `~/.azure`).

Users who want native remote access install the right extra: `pip install
hip-cargo[s3]`, `[gcs]`, or `[azure]`. Without it, the wrapper's existing
`try/except ImportError` routes them into the container, which already has the
backends.

---

## 4. Cab Generation is Automatic

**The `src/stokify/cabs/*.yml` files are generated artefacts. Never edit
them by hand and never run `hip-cargo generate-cabs` manually.**

Three automated paths keep them in sync with `cli/*.py`:

1. **Pre-commit hook** (`.pre-commit-config.yaml`): on every commit, runs
   `hip-cargo generate-cabs --module src/stokify/cli/*.py --output-dir
   src/stokify/cabs`. If it modifies files, pre-commit will "fail" the
   commit — re-run `git add -u && git commit` to include the updates.
2. **`update-cabs.yml` workflow**: on merge to `main`, resets the
   container tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **`tbump`**: on release, rewrites the container tag to the version and
   regenerates cabs.

If you ever see a cab YAML in a diff that wasn't generated by one of these
three paths, that's a bug — revert it and edit the corresponding `cli/*.py`
instead.

### How CLI source maps to cab YAML

- `@stimela_cab(name=..., info=...)` → the cab's name and top-level info.
- `@stimela_output(...)` → entries under `outputs:` in the cab.
- Each Typer parameter → an entry under `inputs:` (dtype inferred from the type
  hint, `info` from `help=`, defaults from `= ...`).
- `Annotated[..., StimelaMeta(skip=True)]` → omitted from the cab (used for
  `--backend`, `--always-pull-images`, etc.).
- `Annotated[..., StimelaMeta(metadata={"rich_help_panel": "Inputs", "tunable":
  True})]` → flows into the cab's `metadata:` dict.
- Inline comments after `Annotated[...]` rows are preserved through the round
  trip — they show up as `# noqa: ...` or similar on the matching cab field.

---

## 5. Round-Trip Tests

The round-trip test in `tests/test_roundtrip.py` is **not optional** — it is
how this project guarantees that `cli/*.py` and `cabs/*.yml` agree.

It runs:

```
cli/<cmd>.py  ──(generate-cabs)──►  cabs/<cmd>.yml  ──(generate-function)──►  <cmd>.py
```

…then asserts the regenerated `<cmd>.py` is byte-identical (after `ruff
format`) to the original `cli/<cmd>.py`. If you write a CLI wrapper in a shape
that hip-cargo cannot round-trip, the test fails and the cab is unreliable.
Fix the source, not the test.

Add a new round-trip case to `tests/test_roundtrip.py` whenever you add a new
command under `cli/`.

---

## 6. Mandatory Dev Workflow

After every code change run:

```bash
uv run ruff format . && uv run ruff check . --fix
```

This is non-negotiable — the pre-commit hook and CI both enforce it, and
generated code is formatted with the same configuration, so divergence breaks
the round-trip.

### Other rules

- **Python 3.10+.** Use modern syntax (`X | Y`, `list[int]`, etc.).
- **Type hints on every function signature.**
- **Lazy imports in `cli/`.** Heavy imports live in `core/` only, and `cli/`
  imports from `core/` inside the function body.
- **Typer Option syntax:**
  - Required: `Annotated[T, typer.Option(..., help="...")]` (no `= default`).
  - Optional w/ default: `Annotated[T, typer.Option(help="...")] = default`.
  - Optional None: `Annotated[T | None, typer.Option(help="...")] = None`.
  - **Never pass `None` as the positional default to `typer.Option()`** — it
    raises `AttributeError`.
- **Comma-separated lists:** use `ListInt`, `ListFloat`, `ListStr` from
  `hip_cargo`, with their matching `parse_list_*` parsers.
- **UPath-backed path types:** `File`, `Directory`, `MS`, `URI` are
  `NewType(..., UPath)`. Generated CLIs use `parser=parse_upath` so the same
  signature accepts local paths and remote URIs.
- **Commits:** use Conventional Commits (`feat:`, `fix:`, `chore:`, …). The
  `update-cabs` bot uses `[skip checks]` to bypass required status checks; do
  not add that tag to human commits.

---

## 7. Where to Go Deeper

- hip-cargo source & docs: <https://github.com/landmanbester/hip-cargo>
- Stimela: <https://github.com/caracal-pipeline/stimela>
- Twelve-factor principles guide most architectural decisions in this repo:
  <https://12factor.net/>
