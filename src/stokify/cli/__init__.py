"""CLI for stokify."""

import typer

app = typer.Typer(
    name="stokify",
    help="Demo the transpiler feature",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Demo the transpiler feature"""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from stokify.cli.image import image  # noqa: E402
from stokify.cli.init import init  # noqa: E402
from stokify.cli.onboard import onboard  # noqa: E402
from stokify.cli.process import process  # noqa: E402
from stokify.runtime.cli import monitor, run, watch  # noqa: E402

app.command(name="onboard")(onboard)
app.command(name="init")(init)
app.command(name="process")(process)
app.command(name="image")(image)
app.command(name="run")(run)
app.command(name="monitor")(monitor)
app.command(name="watch")(watch)

__all__ = ["app"]
