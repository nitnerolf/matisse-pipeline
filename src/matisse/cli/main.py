import typer

from matisse.cli import (
    bcd,
    calibrate,
    doctor,
    format_results,
    reduce,
    show,
)

app = typer.Typer(help="MATISSE Data Reduction CLI")

app.command(name="reduce")(reduce.reduce)
app.command(name="calibrate")(calibrate.calibrate)
app.add_typer(bcd.app, name="bcd")
app.command(name="show")(show.show)
app.command(name="doctor")(doctor.doctor)
app.command(name="format")(format_results.format_results)


# -------------------------
# Main entrypoint
# -------------------------
def main():
    """CLI entrypoint for MATISSE pipeline."""
    app()


if __name__ == "__main__":
    main()
