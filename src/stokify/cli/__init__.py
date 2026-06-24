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
from stokify.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

__all__ = ["app"]
