"""CLI wrapper for the ``init`` cab."""

from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(name="init", info="Read a measurement set into a synthetic image cube.")
@stimela_output(name="output", dtype="Directory", info="Output dataset directory (zarr).", required=True, mkdir=True)
def init(
    output: Annotated[Directory, typer.Option(..., parser=parse_upath, help="Output dataset directory (zarr).")],
    memory_mode: Annotated[
        Literal["greedy", "conservative"],
        typer.Option(help="Memory mode: greedy (plasma-resident) or conservative (store-backed)."),
    ] = "greedy",
    n_band: Annotated[int, typer.Option(help="Number of frequency bands.")] = 4,
    n_stokes: Annotated[int, typer.Option(help="Number of Stokes parameters.")] = 2,
    n_x: Annotated[int, typer.Option(help="Image width in pixels.")] = 256,
    n_y: Annotated[int, typer.Option(help="Image height in pixels.")] = 256,
    n_steps: Annotated[int, typer.Option(help="Number of simulated read blocks.")] = 8,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(help="Execution backend."),
        StimelaMeta(skip=True),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(help="Always pull container images, even if cached locally."),
        StimelaMeta(skip=True),
    ] = False,
):
    """Read a measurement set into a synthetic image cube."""
    if backend == "native" or backend == "auto":
        try:
            from stokify.core.init import init as init_core  # noqa: E402

            init_core(
                output, memory_mode=memory_mode, n_band=n_band, n_stokes=n_stokes, n_x=n_x, n_y=n_y, n_steps=n_steps
            )
            return
        except ImportError:
            if backend == "native":
                raise

    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("stokify")
    if image is None:
        raise RuntimeError("No Container URL in stokify metadata.")

    run_in_container(
        init,
        dict(
            output=output, memory_mode=memory_mode, n_band=n_band, n_stokes=n_stokes, n_x=n_x, n_y=n_y, n_steps=n_steps
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
