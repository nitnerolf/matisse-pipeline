"""
MATISSE visualisation interface based on mat_showOiData.py.
"""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import typer

from matisse.core.utils.oifits_reader import open_oifits
from matisse.viewer import viewer_plotly


def show(
    file: str = typer.Argument(..., help="Path to the OIFITS file"),
    save: str | None = typer.Option(
        None,
        "--save",
        help="Output filename (.png or .pdf)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Build a multi-BCD interactive figure from all files sharing the same TPL START.",
    ),
):
    """
    Display MATISSE OIFITS data.

    Use --interactive to generate figure with buttons to switch
    between BCD positions (IN_IN, OUT_OUT, …) and available bands (LM, N)
    for all files that share the same TPL START.
    """
    typer.echo("🔭 Launching MATISSE OIFITS viewer...")

    if interactive:
        typer.echo("🔀 Building multi-BCD figure (searching for BCD modes)...")
        fig = viewer_plotly.make_multi_bcd_plot(file)
        post_script = getattr(fig, "_matisse_post_script", "")
        viewer_plotly.show_plot(fig, post_script=post_script)
        return

    data = open_oifits(file)

    # Static mode
    fig = viewer_plotly.make_static_matisse_plot(data)
    if save:
        path = Path(save)
        ext = path.suffix.lower()
        if ext not in {".png", ".pdf"}:
            typer.echo(f"❌ Unsupported output format: {ext}. Use .png or .pdf.")
            raise typer.Exit(code=1)
        pio.write_image(fig, path, engine="kaleido")
        typer.echo(f"💾 Figure saved as {save}")
    else:
        typer.echo("👁️ Opening static figure...")
        viewer_plotly.show_plot(fig)
