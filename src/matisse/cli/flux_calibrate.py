"""
MATISSE spectrophotometric flux calibration CLI command.
"""

import os
from enum import Enum
from pathlib import Path

import typer

if os.environ.get("DISPLAY", "") == "":
    import matplotlib

    matplotlib.use("Agg")

from matplotlib import pyplot as plt

from matisse.core.flux.flux_calibration import (
    FluxCalibrationConfig,
    run_flux_calibration,
)
from matisse.core.utils.log_utils import console, log, section, set_verbosity


class SpectralBand(str, Enum):
    LM = "LM"
    N = "N"


class FluxMode(str, Enum):
    FLUX = "flux"
    CORRFLUX = "corrflux"
    BOTH = "both"


def flux_calibrate(
    datadir: Path = typer.Option(
        Path.cwd(),
        "--data-dir",
        "-d",
        help="Directory containing reduced MATISSE OIFITS files.",
    ),
    resultdir: Path | None = typer.Option(
        None,
        "--result-dir",
        "-r",
        help="Output directory for calibrated files (default: <datadir>/calflux or calcorrflux).",
    ),
    sci_name: str = typer.Option(
        "",
        "--sci-name",
        "-s",
        help="Science target name (substring matched in filenames).",
    ),
    cal_name: str = typer.Option(
        "",
        "--cal-name",
        "-c",
        help="Calibrator name (substring in filenames). If empty, closest in time is used.",
    ),
    mode: FluxMode = typer.Option(
        FluxMode.FLUX,
        "--mode",
        "-m",
        help="Calibration mode: 'flux' (total), 'corrflux' (correlated), or 'both'.",
    ),
    band: SpectralBand = typer.Option(
        SpectralBand.LM,
        "--band",
        "-b",
        help="Spectral band to calibrate ('LM' or 'N').",
    ),
    timespan: float = typer.Option(
        1.0,
        "--timespan",
        "-t",
        help="Maximum SCI–CAL time difference in hours.",
    ),
    airmass_correction: bool = typer.Option(
        False,
        "--airmass-corr/--no-airmass-corr",
        help="Apply airmass correction between SCI and CAL.",
    ),
    figdir: Path | None = typer.Option(
        "flux_diagnostics",
        "--fig-dir",
        "-f",
        help='Directory to save diagnostic plots (default: "flux_diagnostics"). Set to empty string to disable.',
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Show diagnostic plots interactively after processing each SCI–CAL pair.",
    ),
    spectral_features: bool = typer.Option(
        False,
        "--sf",
        help="Annotate common spectral features in diagnostic plots.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
):
    """
    Spectrophotometric calibration of MATISSE total and correlated spectra.

    Calibrates total flux (OI_FLUX / FLUXDATA) or correlated flux
    (OI_VIS / VISAMP) by dividing the observed spectrum by a transfer
    function derived from a spectrophotometric calibrator.

    Examples::

        # Calibrate total flux in LM band with automatic calibrator
        matisse flux_calibrate -d /path/to/reduced --sci-name='HD123' --band=LM

        # Calibrate correlated flux with a specific calibrator + airmass correction
        matisse flux_calibrate -d /path/to/reduced --sci-name='HD123' \\
            --cal-name='HD456' --mode=corrflux --airmass-corr
    """
    # --- 1. Verbosity and header ---
    section("MATISSE Flux Calibration")
    set_verbosity(log, verbose)

    # --- 2. Show configuration ---
    section("Configuration")
    console.print(f"[cyan]Input directory:[/]  {datadir.resolve()}")
    if resultdir:
        console.print(f"[cyan]Output directory:[/] {resultdir.resolve()}")
    else:
        console.print(
            "[cyan]Output directory:[/] auto (<datadir>/calflux or calcorrflux)"
        )
    console.print(f"[green]Science target:[/]  {sci_name or '(all)'}")
    console.print(f"[green]Calibrator:[/]      {cal_name or '(closest in time)'}")
    console.print(f"[magenta]Mode:[/]            {mode.value}")
    console.print(f"[magenta]Band:[/]            {band.value}")
    console.print(f"[magenta]Timespan:[/]        {timespan} h")
    console.print(f"[yellow]Airmass corr.:[/]  {'ON' if airmass_correction else 'OFF'}")
    console.print(f"[yellow]Diagnostics:[/]   {figdir.resolve() if figdir else 'OFF'}")
    console.print(f"[dim]Verbose:[/]         {'ON' if verbose else 'OFF'}")

    # --- 3. Build configuration and run ---
    config = FluxCalibrationConfig(
        input_dir=datadir,
        output_dir=resultdir,
        fig_dir=figdir,
        sci_name=sci_name,
        cal_name=cal_name,
        mode=mode.value,
        band=band.value,
        timespan=timespan,
        do_airmass_correction=airmass_correction,
        spectral_features=spectral_features,
    )

    try:
        run_flux_calibration(config)
        if show and plt.get_backend().lower() != "agg":
            plt.show()  # Show all diagnostics at the end of processing each pair
        console.rule("[bold green]Flux calibration completed[/]")
    except Exception as err:
        console.rule("[bold red]Flux calibration failed[/]")
        log.exception("Flux calibration execution failed.")
        typer.echo(f"[ERROR] Flux calibration failed: {err}")
        raise typer.Exit(code=1) from err
