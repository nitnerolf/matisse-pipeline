"""
Diagnostic plots for MATISSE spectrophotometric flux calibration.

Generates matplotlib figures saved to an output directory:

1. **Calibrator spectrum**: original model vs. resampled, plus
   correlated-flux × V_UD curves per baseline.
2. **Airmass correction factor**: wavelength-dependent ratio
   ``trans_cal / trans_sci``.

All functions accept a ``fig_dir`` path and silently skip plotting
when ``fig_dir`` is ``None``.

Ported from legacy ``libFluxCal_STARSFLUX.py`` (J. Varga, 2019-2022).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import transforms

from matisse.core.flux.transfer_function import uniform_disk_visibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Calibrator model spectrum + resampled + correlated flux per baseline
# ---------------------------------------------------------------------------


def plot_calibrator_spectrum(
    fig_dir: Path | None,
    cal_name: str,
    band: str,
    wav_model: np.ndarray,
    flux_model: np.ndarray,
    wav_obs: np.ndarray,
    spectrum_resampled: np.ndarray,
    *,
    hdul_cal: fits.HDUList | None = None,
    diameter_mas: float = 0.0,
    is_dense_model: bool = True,
) -> None:
    """Plot the calibrator model spectrum and its resampled version.

    Optionally overlays the correlated-flux model (``V_UD × S_model``)
    for each baseline if *hdul_cal* is provided.

    Parameters
    ----------
    fig_dir : Path | None
        Directory to save the figure. Skip if ``None``.
    cal_name : str
        Calibrator target name (used in title and filename).
    band : str
        Band identifier (``'IR-LM'`` or ``'IR-N'``).
    wav_model : np.ndarray
        Model wavelength grid in metres.
    flux_model : np.ndarray
        Model flux array (Jy).
    wav_obs : np.ndarray
        Observation wavelength grid in metres (already flipped for L if needed).
    spectrum_resampled : np.ndarray
        Resampled model flux on *wav_obs* grid (Jy).
    hdul_cal : fits.HDUList | None
        Calibrator FITS file (for OI_VIS baselines). Optional.
    diameter_mas : float
        Calibrator UD diameter in mas (for V_UD curves).
    is_dense_model : bool
        Whether the model is denser than the observation (affects line style).
    """
    if fig_dir is None:
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Original model spectrum
    wav_model_um = wav_model * 1e6
    wav_obs_um = wav_obs * 1e6

    if is_dense_model:
        ax.plot(
            wav_model_um,
            flux_model,
            "-",
            color="grey",
            label="Original sp.",
            lw=1.0,
            alpha=0.66,
        )
    else:
        ax.plot(
            wav_model_um,
            flux_model,
            "-o",
            color="grey",
            label="Original sp.",
            lw=1.0,
            alpha=0.66,
            markersize=3,
        )

    # Resampled total spectrum
    ax.plot(wav_obs_um, spectrum_resampled, "-r", label="Resampled total sp.", lw=1.5)

    # Correlated flux model per baseline (V_UD × S_model)
    if hdul_cal is not None and diameter_mas > 0.0:
        try:
            vis_data = hdul_cal["OI_VIS"].data
            for j in range(len(vis_data["VISAMP"])):
                uu = vis_data["UCOORD"][j]
                vv = vis_data["VCOORD"][j]
                bp = np.sqrt(uu**2 + vv**2)
                vis_cal = uniform_disk_visibility(diameter_mas, bp, wav_obs)
                ax.plot(
                    wav_obs_um,
                    vis_cal * spectrum_resampled,
                    "--",
                    label=f"Resampl. corr., B_p = {bp:.1f} m",
                    lw=1.0,
                )
        except KeyError:
            pass

    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel("Flux (Jy)")
    ax.set_title(cal_name)
    ax.set_xlim(np.min(wav_obs_um), np.max(wav_obs_um))
    ax.set_ylim(
        0.7 * np.nanmin(spectrum_resampled),
        1.1 * np.nanmax(spectrum_resampled),
    )
    ax.legend(loc="best", fontsize=7, fancybox=True, framealpha=0.5)
    fig.tight_layout()

    safe_name = cal_name.replace(" ", "_")
    band_tag = band.replace("IR-", "")
    fpath = fig_dir / f"calibrator_{band_tag}_{safe_name}_spectrum.png"
    fig.savefig(fpath, dpi=200)
    logger.info("Saved calibrator spectrum plot: %s", fpath.name)


# ---------------------------------------------------------------------------
# 2. Airmass correction factor
# ---------------------------------------------------------------------------


def plot_airmass_correction(
    fig_dir: Path | None,
    wav_sci_m: np.ndarray,
    airmass_correction: np.ndarray,
    output_tag: str,
) -> None:
    """Plot the airmass correction factor vs. wavelength.

    Parameters
    ----------
    fig_dir : Path | None
        Directory to save the figure. Skip if ``None``.
    wav_sci_m : np.ndarray
        Science wavelength grid in metres.
    airmass_correction : np.ndarray
        Correction factor array (``trans_cal / trans_sci``).
    output_tag : str
        Tag for the output filename.
    """
    if fig_dir is None:
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    wav_um = wav_sci_m * 1e6
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(wav_um, airmass_correction, lw=1.2)

    # Restrict y-axis to the useful range (avoid LM gap extremes)
    wl_min, wl_max = np.min(wav_um), np.max(wav_um)
    wl_range = wl_max - wl_min
    lm_gap = (wav_um > 4.1) & (wav_um < 4.6)
    useful = (wav_um > wl_min + 0.1 * wl_range) & (wav_um < wl_max - 0.1 * wl_range)
    range_idx = useful & ~lm_gap
    if np.any(range_idx):
        pmin = np.nanmin(airmass_correction[range_idx])
        pmax = np.nanmax(airmass_correction[range_idx])
        if pmin < pmax:
            ax.set_ylim(pmin, pmax)

    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel("Airmass correction factor")
    fig.tight_layout()

    fpath = fig_dir / f"skycalc_airmass_correction_factor_{output_tag}.png"
    fig.savefig(fpath, dpi=200)
    # plt.close(fig)
    logger.info("Saved airmass correction plot: %s", fpath.name)


# ---------------------------------------------------------------------------
# 3. Calibrated flux summary (total + correlated)
# ---------------------------------------------------------------------------


def plot_calibrated_flux(
    fig_dir: Path | None,
    hdul_out: fits.HDUList,
    cal_name: str,
    sci_name: str,
    mode: str,
    band: str,
    bcd: str,
    dark_mode: bool = True,
    spectral_features: bool = False,
) -> None:
    """Plot the calibrated total and/or correlated flux spectra.

    Parameters
    ----------
    fig_dir : Path | None
        Directory to save the figure. Skip if ``None``.
    hdul_out : fits.HDUList
        Output calibrated FITS file (already written).
    cal_name, sci_name : str
        Target names for titles.
    mode : str
        ``'flux'``, ``'corrflux'``, or ``'both'``.
    band : str
        Band identifier.
    spectral_features : bool
        Annotate common spectral features when ``True``.
    dark_mode : bool
        Use dark background styling when ``True``.
    """
    if fig_dir is None:
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Theme ---
    if dark_mode:
        bg_colour = "#121212"
        text_colour = "#E0E0E0"
        grid_colour = "#333333"
        line_colours = [
            "#00E5FF",
            "#FF9100",
            "#00E676",
            "#AA00FF",
            "#F50057",
            "#FFEA00",
        ]
    else:
        bg_colour = "#FFFFFF"
        text_colour = "#333333"
        grid_colour = "#E5E5E5"
        line_colours = [
            "#264653",
            "#2A9D8F",
            "#E9C46A",
            "#E76F51",
            "#8338EC",
            "#FB8500",
        ]

    safe_sci = sci_name.replace(" ", "_")
    safe_cal = cal_name.replace(" ", "_")
    band_tag = band.replace("IR-", "")

    # Wavelength grid from FITS
    wav_um = hdul_out["OI_WAVELENGTH"].data["eff_wave"] * 1e6

    # Telluric masks (LM band)
    tel_mask1 = (wav_um >= 4.07) & (wav_um <= 4.58)
    tel_mask2 = wav_um <= 2.85
    tel_mask = tel_mask1 | tel_mask2

    # Station-index → name lookup
    ref_station: dict[int, str] = {}
    try:
        arr_data = hdul_out["OI_ARRAY"].data
        for i in range(len(arr_data["STA_INDEX"])):
            ref_station[int(arr_data["STA_INDEX"][i])] = arr_data["STA_NAME"][i]
    except KeyError:
        pass

    n_panels = sum(m in ("flux", "both") for m in [mode]) + sum(
        m in ("corrflux", "both") for m in [mode]
    )
    if n_panels == 0:
        return

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(9 * n_panels, 5),
        squeeze=False,
        facecolor=bg_colour,
    )
    ax_idx = 0

    def _style_ax(ax: plt.Axes) -> None:
        ax.set_facecolor(bg_colour)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(text_colour)
            ax.spines[spine].set_linewidth(1.2)
        ax.grid(axis="y", color=grid_colour, linestyle="-", linewidth=0.7, zorder=0)
        ax.tick_params(axis="both", colors=text_colour, labelsize=10, width=1.2)
        ax.xaxis.label.set_color(text_colour)
        ax.yaxis.label.set_color(text_colour)
        ax.title.set_color(text_colour)

    # --- Total flux panel ---
    if mode in ("flux", "both"):
        ax = axes[0, ax_idx]
        ax_idx += 1
        _style_ax(ax)
        try:
            fluxdata = hdul_out["OI_FLUX"].data["FLUXDATA"]
            e_fluxdata = hdul_out["OI_FLUX"].data["FLUXERR"]

            aver_flux = np.nanmean(fluxdata, axis=0)
            sta_indices = hdul_out["OI_FLUX"].data["STA_INDEX"]
            l_max_flux = []
            for j in range(len(fluxdata)):
                sta_name = ref_station.get(int(sta_indices[j]), f"T{j + 1}")
                colour = line_colours[j % len(line_colours)]

                flux = fluxdata[j].copy()

                # Masked (telluric) plotted faint
                flux_clean = flux.copy()
                flux_clean[tel_mask1 | tel_mask2] = np.nan

                e_flux_clean = e_fluxdata[j].copy()
                e_flux_clean[tel_mask1 | tel_mask2] = np.nan

                aver_flux_clean = aver_flux.copy()
                aver_flux_clean[tel_mask1 | tel_mask2] = np.nan

                if j == 0:
                    ax.plot(
                        wav_um,
                        aver_flux_clean,
                        color="lightgrey",
                        lw=1.5,
                        alpha=0.9,
                        label="Average",
                    )

                ax.plot(
                    wav_um,
                    flux_clean,
                    color=colour,
                    lw=1.2,
                    alpha=0.85,
                    label=f"Obs {j + 1} ({sta_name})",
                )

                ax.fill_between(
                    wav_um,
                    flux_clean - e_flux_clean,
                    flux_clean + e_flux_clean,
                    color=colour,
                    alpha=0.3,
                    interpolate=True,
                )

                transy = transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )

                if spectral_features:
                    nanodiam_lines = (3.43, 3.53)  # nanodiamonds
                    hydro_lines = (3.296, 3.7405, 4.052279, 4.653)  # Pfd, Pfγ, Brα, Pfb
                    hydro_names = ("Pfδ", "Pfγ", "Brα", "Pfβ")
                    ch_lines = [3.4]  # aromatic CH stretch

                    water_band = [2.85, 3.2]

                    plt.axvspan(
                        water_band[0], water_band[1], color="steelblue", alpha=0.04
                    )

                    ax.text(
                        3.02,
                        0.9,
                        "Water band",
                        color="steelblue",
                        fontsize=8,
                        ha="center",
                        va="center",
                        rotation=0,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            fc="steelblue",
                            alpha=0.1,
                            lw=1,
                            edgecolor="steelblue",
                        ),
                        transform=transy,
                        alpha=0.2,
                    )

                    for x in nanodiam_lines:
                        ax.axvline(x=x, color="c", lw=0.8, ls="--", alpha=0.1)
                        ax.text(
                            x + 0.035,
                            0.9,
                            "Nano diamons",
                            color="c",
                            fontsize=8,
                            ha="center",
                            va="center",
                            rotation=90,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="c",
                                alpha=0.02,
                                lw=1,
                                edgecolor="c",
                            ),
                            transform=transy,
                            alpha=0.2,
                        )
                    for x in ch_lines:
                        ax.axvline(x=x, color="#AE9764", lw=0.8, ls="--", alpha=0.1)
                        ax.text(
                            x - 0.035,
                            0.9,
                            "aromatic CH",
                            color="#AE9764",
                            fontsize=8,
                            ha="center",
                            va="center",
                            rotation=90,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="#AE9764",
                                alpha=0.1,
                                lw=1,
                                edgecolor="#AE9764",
                            ),
                            transform=transy,
                            alpha=0.2,
                        )
                    for x in hydro_lines:
                        ax.axvline(x=x, color="lightcoral", lw=0.8, ls="--", alpha=0.1)
                        ax.text(
                            x - 0.035,
                            0.9,
                            hydro_names[hydro_lines.index(x)],
                            color="lightcoral",
                            fontsize=8,
                            ha="center",
                            va="center",
                            rotation=90,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                fc="lightcoral",
                                alpha=0.1,
                                lw=1,
                                edgecolor="lightcoral",
                            ),
                            transform=transy,
                            alpha=0.2,
                        )

                for mask in [tel_mask1, tel_mask2]:
                    ax.plot(
                        wav_um[mask],
                        flux[mask],
                        color="lightgrey",
                        lw=1.0,
                        alpha=0.15,
                    )
                l_max_flux.append(np.nanmax(flux_clean))
            if l_max_flux:
                ax.set_ylim(0, 1.2 * max(l_max_flux))
            ax.set_title(
                f"Total flux – {sci_name} (cal: {cal_name}), BCD: {bcd}",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(r"Wavelength (µm)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Flux (Jy)", fontsize=11, fontweight="bold")
            legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
            for t in legend.get_texts():
                t.set_color(text_colour)
        except KeyError:
            ax.text(
                0.5,
                0.5,
                "No OI_FLUX",
                transform=ax.transAxes,
                ha="center",
                color=text_colour,
            )

    # --- Correlated flux panel ---
    if mode in ("corrflux", "both"):
        ax = axes[0, ax_idx]
        _style_ax(ax)
        try:
            visamp = hdul_out["OI_VIS"].data["VISAMP"]
            sta_idx_arr = hdul_out["OI_VIS"].data["STA_INDEX"]
            l_max_vis = []
            for j in range(len(visamp)):
                vis = visamp[j].copy()
                s0 = ref_station.get(int(sta_idx_arr[j][0]), str(sta_idx_arr[j][0]))
                s1 = ref_station.get(int(sta_idx_arr[j][1]), str(sta_idx_arr[j][1]))
                colour = line_colours[j % len(line_colours)]
                vis_clean = np.where(tel_mask, np.nan, vis)
                ax.plot(
                    wav_um,
                    vis_clean,
                    color=colour,
                    lw=1.2,
                    alpha=0.85,
                    label=f"{s0}–{s1}",
                )
                ax.plot(
                    wav_um[tel_mask],
                    vis[tel_mask],
                    color="lightgrey",
                    lw=1.0,
                    alpha=0.15,
                )
                l_max_vis.append(np.nanmax(vis_clean))
            if l_max_vis:
                ax.set_ylim(0, 1.2 * max(l_max_vis))
            ax.set_title(
                f"Correlated flux – {sci_name} (cal: {cal_name}), BCD: {bcd}",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(r"Wavelength (µm)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Correlated flux (Jy)", fontsize=11, fontweight="bold")
            legend = ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
            for t in legend.get_texts():
                t.set_color(text_colour)
        except KeyError:
            ax.text(
                0.5,
                0.5,
                "No OI_VIS",
                transform=ax.transAxes,
                ha="center",
                color=text_colour,
            )

    fig.tight_layout()
    fpath = fig_dir / f"calibrated_{band_tag}_{safe_sci}_cal_{safe_cal}_bcd_{bcd}.png"
    fig.savefig(fpath, dpi=300, facecolor=bg_colour)
    logger.info("Saved calibrated flux plot: %s", fpath.name)
