"""CLI wrapper for the ``process`` cab."""

from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(name="process", info="Transform coherencies into image-ready Stokes visibilities.")
@stimela_output(name="output", dtype="Directory", info="Output dataset directory (zarr).", required=True, mkdir=True)
def process(
    input_data: Annotated[Directory, typer.Option(..., parser=parse_upath, help="Upstream dataset directory (zarr).")],
    output: Annotated[Directory, typer.Option(..., parser=parse_upath, help="Output dataset directory (zarr).")],
    memory_mode: Annotated[
        Literal["greedy", "conservative"],
        typer.Option(help="Memory mode: greedy (plasma-resident) or conservative (store-backed)."),
    ] = "greedy",
    n_iterations: Annotated[int, typer.Option(help="Number of synthetic solver iterations.")] = 10,
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
    """Transform coherencies into image-ready Stokes visibilities."""
    if backend == "native" or backend == "auto":
        try:
            from stokify.core.process import process as process_core  # noqa: E402

            process_core(output, input_data, memory_mode=memory_mode, n_iterations=n_iterations)
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
        process,
        dict(input_data=input_data, output=output, memory_mode=memory_mode, n_iterations=n_iterations),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
