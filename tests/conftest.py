"""Shared test fixtures and configuration for stokify."""

import os

# Disable Ray's automatic uv-run runtime env propagation before ray is imported.
# When the driver runs under `uv run`, Ray would otherwise package the project
# directory and re-resolve deps for workers via `uv run`, but workers spawned
# that way lose access to ray itself. The local in-process Ray used here shares
# the driver's venv, so the hook adds nothing and (in ray>=2.55) crashes on
# `working_dir=None`. This mirrors hip-cargo's own conftest.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
