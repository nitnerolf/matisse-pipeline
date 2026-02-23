"""
MATISSE BCD merge and removal tools.

This module provides functions for:
- Removing BCD (Beam Commuting Device) signatures from OIFITS files
- Merging multiple OIFITS exposures sharing the same TPL START
- Sorting OIFITS files by their template start timestamp

Original author: ame (Nov 2019, Observatoire de la Côte d'Azur)
Revised: Feb 2026 (aso) — modernised naming, typing, logging.

Copyright (C) 2017- Observatoire de la Côte d'Azur
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from scipy.stats import circvar

from matisse.core.utils.log_utils import log
from matisse.core.utils.oifits_reader import OIFitsReader

# ---------------------------------------------------------------------------
# BCD modes and positions
# ---------------------------------------------------------------------------

BCD_MODES_TO_CORRECT = ["IN_IN", "IN_OUT", "OUT_IN"]

BCD_POSITIONS: dict[str, list[int]] = {
    "OUT_OUT": [0, 1, 2, 3, 4, 5],
    "OUT_IN": [0, 1, 4, 5, 2, 3],
    "IN_OUT": [0, 1, 3, 2, 5, 4],
    "IN_IN": [0, 1, 5, 4, 3, 2],
}

# ---------------------------------------------------------------------------
# BCD baseline remapping tables
# ---------------------------------------------------------------------------
# Indexed by BCD state: 0=OUT-OUT, 1=OUT-IN, 2=IN-OUT, 3=IN-IN

BCD_BASELINE_REMAP: list[list[int]] = [
    [0, 1, 2, 3, 4, 5],  # OUT-OUT (0)
    [0, 1, 4, 5, 2, 3],  # OUT-IN  (1)
    [0, 1, 3, 2, 5, 4],  # IN-OUT  (2)
    [0, 1, 5, 4, 3, 2],  # IN-IN   (3)
]


BCD_SIGN: list[list[int]] = [
    [1, 1, 1, 1, 1, 1],  # OUT-OUT (0)
    [1, -1, 1, 1, 1, 1],  # OUT-IN  (1)
    [-1, 1, 1, 1, 1, 1],  # IN-OUT  (2)
    [-1, -1, 1, 1, 1, 1],  # IN-IN   (3)
]

BCD_CP_REMAP: list[list[int]] = [
    [0, 1, 2, 3],  # OUT-OUT (0)
    [3, 1, 2, 0],  # OUT-IN  (1)
    [0, 2, 1, 3],  # IN-OUT  (2)
    [3, 2, 1, 0],  # IN-IN   (3)
]

BCD_CP_SIGN: list[list[int]] = [
    [1, 1, 1, 1],  # OUT-OUT (0)
    [1, -1, -1, 1],  # OUT-IN  (1)
    [-1, 1, 1, -1],  # IN-OUT  (2)
    [-1, -1, -1, -1],  # IN-IN   (3)
]

BCD_FLUX_LM: list[list[int]] = [
    [0, 1, 3, 2],
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [1, 0, 2, 3],
]

BCD_FLUX_N: list[list[int]] = [
    [3, 2, 0, 1],
    [3, 2, 1, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
]

# Closure-phase station index triplets and UV coordinate indices
_STA_INDEX_CP: list[list[int]] = [[1, 2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3]]
_UV1: list[int] = [2, 1, 1, 4]
_UV2: list[int] = [0, 2, 3, 0]

# Number of baselines / closure-phase triangles for 4 telescopes
_N_BASELINES = 6
_N_CLOSURES = 4
_N_FLUXES = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sort_by_tpl_start(
    oifits_list: list[str] | list[fits.HDUList],
) -> tuple[list[str], list[list[fits.HDUList]]]:
    """
    Sort a list of OIFITS files (or HDULists) by their TPL START keyword.

    Parameters
    ----------
    oifits_list : list of str or list of HDUList
        Paths to OIFITS files, or already-opened HDULists.

    Returns
    -------
    tpl_starts : list of str
        Unique TPL START values in order of first appearance.
    sorted_data : list of list of HDUList
        For each TPL START, the list of corresponding HDULists.
    """
    data: list[fits.HDUList] = _ensure_hdulists(oifits_list)

    tpl_values = [d[0].header["ESO TPL START"] for d in data]

    tpl_starts: list[str] = []
    sorted_data: list[list[fits.HDUList]] = []

    for i, tpl in enumerate(tpl_values):
        if tpl not in tpl_starts:
            tpl_starts.append(tpl)
            sorted_data.append([])
        idx = tpl_starts.index(tpl)
        sorted_data[idx].append(data[i])

    return tpl_starts, sorted_data


def merge_oifits(oifits_list: list[str] | list[fits.HDUList]) -> fits.HDUList:
    """
    Merge a list of OIFITS files into one averaged HDUList.

    BCD signatures are removed before merging. The result is a single
    averaged file with combined error estimates (weighted mean +
    scatter contribution).

    Parameters
    ----------
    oifits_list : list of str or list of HDUList
        Paths to OIFITS files, or already-opened HDULists.

    Returns
    -------
    fits.HDUList
        Merged OIFITS HDUList.
    """
    data: list[fits.HDUList] = _ensure_hdulists(oifits_list)
    n_files = len(data)

    # Remove BCD signatures in-place
    for d in data:
        remove_bcd(d)

    extnames = [hdu.name for hdu in data[0]]
    merged = fits.HDUList([hdu.copy() for hdu in data[0]])

    if "OI_VIS2" in extnames:
        merged["OI_VIS2"] = _merge_vis2(data, n_files)

    if "OI_VIS" in extnames:
        merged["OI_VIS"] = _merge_vis(data, n_files)

    if "OI_T3" in extnames:
        merged["OI_T3"] = _merge_t3(data, n_files)

    if "OI_FLUX" in extnames:
        merged["OI_FLUX"] = _merge_flux(data, n_files)

    if "TF2" in extnames:
        merged["TF2"] = _merge_tf2(data, n_files)

    return merged


def remove_bcd(
    oifits: str | fits.HDUList,
    *,
    save: bool = False,
) -> fits.HDUList:
    """
    Remove BCD (Beam Commuting Device) signature from an OIFITS file.

    Re-orders baselines, flips signs, and rewrites headers so that the
    resulting file looks like an OUT-OUT observation.

    Parameters
    ----------
    oifits : str or HDUList
        Path to an OIFITS file, or an already-opened HDUList.
    save : bool
        If *True*, write the corrected data to ``<original>_noBCD.fits``.

    Returns
    -------
    fits.HDUList
        The modified HDUList (also modified in-place).
    """
    data: fits.HDUList = fits.open(oifits) if isinstance(oifits, str) else oifits

    bcd1 = data[0].header["ESO INS BCD1 NAME"]
    bcd2 = data[0].header["ESO INS BCD2 NAME"]
    bcd_state = (bcd2 == "IN") + 2 * (bcd1 == "IN")

    if bcd_state == 0:
        return data  # Already OUT-OUT

    # Update header to OUT-OUT
    hdr = data[0].header
    hdr["ESO INS BCD1 NAME"] = "OUT"
    hdr["ESO INS BCD2 NAME"] = "OUT"
    hdr["ESO INS BCD1 ID"] = "OUT"
    hdr["ESO INS BCD2 ID"] = "OUT"
    hdr["ESO INS BCD1 NO"] = 1
    hdr["ESO INS BCD1 NO"] = 1
    hdr["HIERARCH ESO DET BCD STATE"] = 0
    hdr["HIERARCH ESO CFG BCD MODE"] = "OUT-OUT"

    extnames = [hdu.name for hdu in data]

    if "OI_VIS2" in extnames:
        _remove_bcd_vis2(data, bcd_state)

    if "OI_VIS" in extnames:
        _remove_bcd_vis(data, bcd_state)

    if "OI_T3" in extnames:
        _remove_bcd_t3(data, bcd_state)

    if "OI_FLUX" in extnames:
        _remove_bcd_flux(data, bcd_state)

    if "TF2" in extnames:
        _remove_bcd_tf2(data, bcd_state)

    # Reset header (in case extensions overwrote it)
    hdr["ESO INS BCD1 NAME"] = "OUT"
    hdr["ESO INS BCD2 NAME"] = "OUT"

    if save:
        filename_in = data.filename()
        filename_out = filename_in.replace(".fits", "_noBCD.fits")
        data.writeto(filename_out, overwrite=True)
        log.info(f"Saved BCD-removed file to {Path(filename_out).name}")

    return data


def merge_by_tpl_start(
    source: str | list[str] | list[fits.HDUList],
    *,
    save: bool = False,
    output_dir: str | Path = "./MERGED",
    separate_chopping: bool = False,
) -> tuple[list[fits.HDUList], list[fits.HDUList]]:
    """
    Group OIFITS files by TPL START, then merge each group.

    Parameters
    ----------
    source : str, list of str, or list of HDUList
        A directory path, list of file paths, or list of HDULists.
    save : bool
        If *True*, write each merged file to *output_dir*.
    output_dir : str or Path
        Output directory (created if needed).
    separate_chopping : bool
        If *True*, keep chopped and non-chopped LM exposures separate.

    Returns
    -------
    merged_data : list of HDUList
        The merged HDULists.
    raw_data : list of HDUList
        The original (un-merged) HDULists.
    """
    output_path = Path(output_dir)

    # Resolve input to a list of HDULists
    if isinstance(source, str):
        source_dir = Path(source)
        if not source_dir.is_dir():
            log.error(f"{source} is not a directory")
            return [], []
        source_list: list[str] | list[fits.HDUList] = [
            str(f) for f in sorted(source_dir.iterdir()) if f.suffix == ".fits"
        ]
    else:
        source_list = source

    data: list[fits.HDUList] = _ensure_hdulists(source_list)
    tpl_starts, sorted_data = sort_by_tpl_start(data)
    n_tpl = len(tpl_starts)

    log.info(f"Number of TPL STARTs: {n_tpl}")

    merged_data: list[fits.HDUList] = []

    for i_tpl in range(n_tpl):
        band = np.array([d[0].header["ESO DET NAME"] for d in sorted_data[i_tpl]])
        chop = np.array([d[0].header["ESO ISS CHOP ST"] for d in sorted_data[i_tpl]])

        idx_n = np.where(band == "MATISSE-N")[0]

        if separate_chopping:
            idx_lm_chop = np.where((band == "MATISSE-LM") & (chop == "T"))[0]
            idx_lm_nochop = np.where((band == "MATISSE-LM") & (chop == "F"))[0]
            idx_lm = np.array([], dtype=int)
        else:
            idx_lm_chop = np.array([], dtype=int)
            idx_lm_nochop = np.array([], dtype=int)
            idx_lm = np.where(band == "MATISSE-LM")[0]

        log.info(
            f"({i_tpl + 1}/{n_tpl}) TPLSTART={tpl_starts[i_tpl]} — "
            f"LM={len(idx_lm)} "
            + (
                f"LM-chop={len(idx_lm_chop)} LM-nochop={len(idx_lm_nochop)} "
                if separate_chopping
                else ""
            )
            + f"N={len(idx_n)}"
        )

        for idx in [idx_lm, idx_lm_chop, idx_lm_nochop, idx_n]:
            if len(idx) == 0:
                continue

            sub_data = [sorted_data[i_tpl][j] for j in idx]
            merged = merge_oifits(sub_data)
            merged_data.append(merged)

            if save:
                output_path.mkdir(parents=True, exist_ok=True)
                base_name = Path(sorted_data[i_tpl][idx[0]].filename()).name
                file_out = _build_merged_filename(
                    base_name, separate_chopping=separate_chopping
                )
                out_file = output_path / file_out
                merged.writeto(str(out_file), overwrite=True)
                log.info(f"Saved merged file to {output_path.name}/{file_out}")

    return merged_data, data


# ---------------------------------------------------------------------------
# Private helpers — merge individual OIFITS extensions
# ---------------------------------------------------------------------------


def _ensure_hdulists(
    items: list[str] | list[fits.HDUList],
) -> list[fits.HDUList]:
    """Open file paths if needed, pass HDULists through."""
    if not items:
        return []
    if isinstance(items[0], str):
        return [fits.open(p) for p in items]
    return list(items)


def _hdu_cut_rows(hdu: Any, n_rows: int) -> fits.BinTableHDU:
    """Return a copy of *hdu* truncated to *n_rows* rows."""
    cols = hdu.data.columns
    new_cols = [
        fits.Column(
            name=c.name,
            array=hdu.data[c.name][:n_rows],
            unit=c.unit,
            format=c.format,
        )
        for c in cols
    ]
    new_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(new_cols))
    new_hdu.header = hdu.header
    return new_hdu


def _iter_exposures(
    data: list[fits.HDUList],
    ext_name: str,
    n_min: int,
) -> list[tuple[int, int, int]]:
    """
    Return (file_index, mod_index, n_min) tuples, skipping the first
    exposure of the first file (which seeds the accumulator).
    """
    result: list[tuple[int, int, int]] = []
    for i_file in range(len(data)):
        n_rows = len(data[i_file][ext_name].data)
        n_mod = n_rows // n_min
        for i_mod in range(n_mod):
            if i_file == 0 and i_mod == 0:
                continue  # seed already used
            result.append((i_file, i_mod, n_min))
    return result


def _accumulate_key(
    dest: NDArray[Any],
    src: NDArray[Any],
    norm: int,
) -> NDArray[Any]:
    """Running mean: ``(dest * norm + src) / (norm + 1)``."""
    return (dest * norm + src) / (norm + 1)


def _slice_key(
    data: fits.HDUList,
    ext: str,
    key: str,
    i_mod: int,
    n_min: int,
) -> NDArray[Any]:
    """Extract rows ``[i_mod*n_min : (i_mod+1)*n_min]`` from *data[ext][key]*."""
    arr = data[ext].data[key]
    lo = i_mod * n_min
    hi = (i_mod + 1) * n_min
    if arr.ndim == 2:
        return arr[lo:hi, :]
    return arr[lo:hi]


# ---------------------------------------------------------------------------
# Extension-specific merge routines
# ---------------------------------------------------------------------------


def _merge_vis2(data: list[fits.HDUList], n_files: int) -> fits.BinTableHDU:
    """Merge OI_VIS2 extension with weighted mean and scatter error."""
    n_min = _N_BASELINES
    temp = _hdu_cut_rows(data[0]["OI_VIS2"], n_min)

    vis2_sq = temp.data["VIS2DATA"] ** 2
    weight = 1.0 / temp.data["VIS2ERR"] ** 2
    vis2_weighted = temp.data["VIS2DATA"] / temp.data["VIS2ERR"] ** 2
    norm = 1

    keys_to_average = [
        "VIS2DATA",
        "VIS2ERR",
        "UCOORD",
        "VCOORD",
        "TIME",
        "MJD",
        "INT_TIME",
    ]

    for i_file, i_mod, n in _iter_exposures(data, "OI_VIS2", n_min):
        for key in keys_to_average:
            src = _slice_key(data[i_file], "OI_VIS2", key, i_mod, n)
            temp.data[key] = _accumulate_key(temp.data[key], src, norm)

        src_vis2 = _slice_key(data[i_file], "OI_VIS2", "VIS2DATA", i_mod, n)
        vis2_sq = _accumulate_key(vis2_sq, src_vis2**2, norm)

        src_err = _slice_key(data[i_file], "OI_VIS2", "VIS2ERR", i_mod, n)
        weight += 1.0 / src_err**2
        vis2_weighted += src_vis2 / src_err**2

        norm += 1

    temp.data["VIS2ERR"] = np.sqrt(
        1.0 / weight + np.abs(vis2_sq - temp.data["VIS2DATA"] ** 2)
    )
    temp.data["VIS2DATA"] = vis2_weighted / weight
    temp.data["INT_TIME"] *= norm

    return temp


def _merge_vis(data: list[fits.HDUList], n_files: int) -> fits.BinTableHDU:
    """Merge OI_VIS extension (amplitude + differential phase)."""
    n_min = _N_BASELINES
    temp = _hdu_cut_rows(data[0]["OI_VIS"], n_min)

    exp_visphi = np.exp(1j * np.deg2rad(temp.data["VISPHI"]))
    visamp_sq = temp.data["VISAMP"] ** 2
    weight = 1.0 / temp.data["VISAMPERR"] ** 2
    vis_weighted = temp.data["VISAMP"] / temp.data["VISAMPERR"] ** 2
    visphi_arr: list[NDArray[Any]] = [np.deg2rad(temp.data["VISPHI"])]
    norm = 1

    keys_to_average = [
        "VISAMP",
        "VISAMPERR",
        "VISPHIERR",
        "UCOORD",
        "VCOORD",
        "TIME",
        "MJD",
        "INT_TIME",
    ]

    for i_file, i_mod, n in _iter_exposures(data, "OI_VIS", n_min):
        for key in keys_to_average:
            src = _slice_key(data[i_file], "OI_VIS", key, i_mod, n)
            temp.data[key] = _accumulate_key(temp.data[key], src, norm)

        src_amp = _slice_key(data[i_file], "OI_VIS", "VISAMP", i_mod, n)
        visamp_sq = _accumulate_key(visamp_sq, src_amp**2, norm)

        src_err = _slice_key(data[i_file], "OI_VIS", "VISAMPERR", i_mod, n)
        weight += 1.0 / src_err**2
        vis_weighted += src_amp / src_err**2

        src_phi = np.deg2rad(_slice_key(data[i_file], "OI_VIS", "VISPHI", i_mod, n))
        exp_visphi += np.exp(1j * src_phi)
        visphi_arr.append(src_phi)

        norm += 1

    visphi_stack = np.array(visphi_arr)
    temp.data["VISPHI"] = np.rad2deg(np.angle(exp_visphi))
    temp.data["VISPHIERR"] = np.rad2deg(
        np.sqrt(
            np.deg2rad(temp.data["VISPHIERR"]) ** 2 / norm
            + circvar(visphi_stack, axis=0)
        )
    )
    temp.data["VISAMPERR"] = np.sqrt(
        1.0 / weight + np.abs(visamp_sq - temp.data["VISAMP"] ** 2)
    )
    temp.data["VISAMP"] = vis_weighted / weight
    temp.data["INT_TIME"] *= norm

    return temp


def _merge_t3(data: list[fits.HDUList], n_files: int) -> fits.BinTableHDU:
    """Merge OI_T3 extension (closure phases)."""
    n_min = _N_CLOSURES
    temp = _hdu_cut_rows(data[0]["OI_T3"], n_min)

    exp_t3phi = np.exp(1j * np.deg2rad(temp.data["T3PHI"]))
    t3phi_arr: list[NDArray[Any]] = [np.deg2rad(temp.data["T3PHI"])]
    norm = 1

    keys_to_average = [
        "T3PHIERR",
        "U1COORD",
        "V1COORD",
        "U2COORD",
        "V2COORD",
        "TIME",
        "MJD",
        "INT_TIME",
    ]

    for i_file, i_mod, n in _iter_exposures(data, "OI_T3", n_min):
        for key in keys_to_average:
            src = _slice_key(data[i_file], "OI_T3", key, i_mod, n)
            temp.data[key] = _accumulate_key(temp.data[key], src, norm)

        src_phi = np.deg2rad(_slice_key(data[i_file], "OI_T3", "T3PHI", i_mod, n))
        exp_t3phi += np.exp(1j * src_phi)
        t3phi_arr.append(src_phi)

        norm += 1

    t3phi_stack = np.array(t3phi_arr)
    temp.data["T3PHI"] = np.rad2deg(np.angle(exp_t3phi))
    temp.data["T3PHIERR"] = np.rad2deg(
        np.sqrt(
            np.deg2rad(temp.data["T3PHIERR"]) ** 2 / norm + circvar(t3phi_stack, axis=0)
        )
    )
    temp.data["INT_TIME"] *= norm

    return temp


def _merge_flux(data: list[fits.HDUList], n_files: int) -> fits.BinTableHDU:
    """Merge OI_FLUX extension."""
    n_min = _N_FLUXES
    temp = _hdu_cut_rows(data[0]["OI_FLUX"], n_min)

    flux_sq = temp.data["FLUXDATA"] ** 2
    weight = 1.0 / temp.data["FLUXERR"] ** 2
    flux_weighted = temp.data["FLUXDATA"] / temp.data["FLUXERR"] ** 2
    norm = 1

    keys_to_average = ["FLUXDATA", "FLUXERR", "TIME", "MJD", "INT_TIME"]

    for i_file, i_mod, n in _iter_exposures(data, "OI_FLUX", n_min):
        for key in keys_to_average:
            src = _slice_key(data[i_file], "OI_FLUX", key, i_mod, n)
            temp.data[key] = _accumulate_key(temp.data[key], src, norm)

        src_flux = _slice_key(data[i_file], "OI_FLUX", "FLUXDATA", i_mod, n)
        flux_sq = _accumulate_key(flux_sq, src_flux**2, norm)

        src_err = _slice_key(data[i_file], "OI_FLUX", "FLUXERR", i_mod, n)
        weight += 1.0 / src_err**2
        flux_weighted += src_flux / src_err**2

        norm += 1

    temp.data["FLUXERR"] = np.sqrt(
        1.0 / weight + np.abs(flux_sq - temp.data["FLUXDATA"] ** 2)
    )
    temp.data["FLUXDATA"] = flux_weighted / weight
    temp.data["INT_TIME"] *= norm

    return temp


def _merge_tf2(data: list[fits.HDUList], n_files: int) -> fits.BinTableHDU:
    """Merge TF2 extension (transfer function squared visibilities)."""
    n_min = _N_BASELINES
    temp = _hdu_cut_rows(data[0]["TF2"], n_min)

    tf2_sq = temp.data["TF2"] ** 2
    norm = 1

    keys_to_average = ["TF2", "TIME", "MJD", "INT_TIME"]

    for i_file, i_mod, n in _iter_exposures(data, "TF2", n_min):
        for key in keys_to_average:
            src = _slice_key(data[i_file], "TF2", key, i_mod, n)
            temp.data[key] = _accumulate_key(temp.data[key], src, norm)

        src_tf2 = _slice_key(data[i_file], "TF2", "TF2", i_mod, n)
        tf2_sq = _accumulate_key(tf2_sq, src_tf2**2, norm)

        norm += 1

    temp.data["TF2ERR"] = np.sqrt(
        temp.data["TF2ERR"] ** 2 / norm + np.abs(tf2_sq - temp.data["TF2"] ** 2)
    )
    temp.data["INT_TIME"] *= norm

    return temp


# ---------------------------------------------------------------------------
# Private helpers — BCD removal per extension
# ---------------------------------------------------------------------------


def _remove_bcd_vis2(data: fits.HDUList, bcd_state: int) -> None:
    """Remove BCD from OI_VIS2 extension in-place."""
    temp = data["OI_VIS2"].copy()
    n_rows = len(temp.data)
    remap = BCD_BASELINE_REMAP[bcd_state]
    sign = BCD_SIGN[bcd_state]

    for i in range(n_rows):
        i_bl = i % _N_BASELINES
        shift = (i // _N_BASELINES) * _N_BASELINES
        temp.data[i_bl + shift] = data["OI_VIS2"].data[remap[i_bl] + shift]
        temp.data[i_bl + shift]["UCOORD"] *= sign[i_bl]
        temp.data[i_bl + shift]["VCOORD"] *= sign[i_bl]
        if sign[i_bl] == -1:
            temp.data[i_bl + shift]["STA_INDEX"] = np.flip(
                temp.data[i_bl + shift]["STA_INDEX"], axis=0
            )

    data["OI_VIS2"] = temp


def _remove_bcd_vis(data: fits.HDUList, bcd_state: int) -> None:
    """Remove BCD from OI_VIS extension in-place."""
    temp = data["OI_VIS"].copy()
    n_rows = len(temp.data)
    remap = BCD_BASELINE_REMAP[bcd_state]
    sign = BCD_SIGN[bcd_state]

    for i in range(n_rows):
        i_bl = i % _N_BASELINES
        shift = (i // _N_BASELINES) * _N_BASELINES
        temp.data[i_bl + shift] = data["OI_VIS"].data[remap[i_bl] + shift]
        temp.data[i_bl + shift]["VISPHI"] *= sign[i_bl]
        temp.data[i_bl + shift]["UCOORD"] *= sign[i_bl]
        temp.data[i_bl + shift]["VCOORD"] *= sign[i_bl]
        if sign[i_bl] == -1:
            temp.data[i_bl + shift]["STA_INDEX"] = np.flip(
                temp.data[i_bl + shift]["STA_INDEX"], axis=0
            )

    data["OI_VIS"] = temp


def _remove_bcd_t3(data: fits.HDUList, bcd_state: int) -> None:
    """Remove BCD from OI_T3 extension in-place."""
    sta_index = data["OI_ARRAY"].data["STA_INDEX"]
    temp = data["OI_T3"].copy()
    n_rows = len(temp.data)
    cp_remap = BCD_CP_REMAP[bcd_state]
    cp_sign = BCD_CP_SIGN[bcd_state]

    for i in range(n_rows):
        i_cp = i % _N_CLOSURES
        shift_cp = (i // _N_CLOSURES) * _N_CLOSURES
        shift_bl = (i // _N_CLOSURES) * _N_BASELINES

        data["OI_T3"].data[cp_remap[i_cp] + shift_cp]["T3PHI"] *= cp_sign[i_cp]
        temp.data[i_cp + shift_cp] = data["OI_T3"].data[cp_remap[i_cp] + shift_cp]
        temp.data[i_cp + shift_cp]["U1COORD"] = data["OI_VIS2"].data["UCOORD"][
            _UV1[i_cp] + shift_bl
        ]
        temp.data[i_cp + shift_cp]["V1COORD"] = data["OI_VIS2"].data["VCOORD"][
            _UV1[i_cp] + shift_bl
        ]
        temp.data[i_cp + shift_cp]["U2COORD"] = data["OI_VIS2"].data["UCOORD"][
            _UV2[i_cp] + shift_bl
        ]
        temp.data[i_cp + shift_cp]["V2COORD"] = data["OI_VIS2"].data["VCOORD"][
            _UV2[i_cp] + shift_bl
        ]
        temp.data[i_cp + shift_cp]["STA_INDEX"] = np.array(
            [sta_index[_STA_INDEX_CP[i_cp][k]] for k in range(3)]
        )

    data["OI_T3"] = temp


def _remove_bcd_flux(data: fits.HDUList, bcd_state: int) -> None:
    """Remove BCD from OI_FLUX extension in-place."""
    chip = data[0].header["ESO DET CHIP NAME"]
    flux_remap = BCD_FLUX_LM if chip == "HAWAII-2RG" else BCD_FLUX_N

    temp = data["OI_FLUX"].copy()
    n_rows = len(temp.data)

    for i in range(n_rows):
        i_fl = i % _N_FLUXES
        shift = (i // _N_FLUXES) * _N_FLUXES
        temp.data[i_fl + shift] = data["OI_FLUX"].data[
            flux_remap[bcd_state][i_fl] + shift
        ]

    data["OI_FLUX"] = temp


def _remove_bcd_tf2(data: fits.HDUList, bcd_state: int) -> None:
    """Remove BCD from TF2 extension in-place."""
    temp = data["TF2"].copy()
    n_rows = len(temp.data)
    remap = BCD_BASELINE_REMAP[bcd_state]
    sign = BCD_SIGN[bcd_state]

    for i in range(n_rows):
        i_bl = i % _N_BASELINES
        shift = (i // _N_BASELINES) * _N_BASELINES
        temp.data[i_bl + shift] = data["TF2"].data[remap[i_bl] + shift]
        if sign[i_bl] == -1:
            temp.data[i_bl + shift]["STA_INDEX"] = np.flip(
                temp.data[i_bl + shift]["STA_INDEX"], axis=0
            )

    data["TF2"] = temp


def _build_merged_filename(base_name: str, *, separate_chopping: bool) -> str:
    """Clean up a filename for merged output."""
    name = base_name.replace("_OUT", "").replace("_IN", "")
    if not separate_chopping:
        name = name.replace("_noChop", "").replace("_Chop", "")
    return name


def find_sci_filename(data_dir, chopping=False, band=None):
    """Find filename bases in data_dir that are science."""
    data_dir = Path(data_dir)

    if chopping:
        suffix = "_Chop*.fits"
    else:
        suffix = "_noChop*.fits"

    list_scivis = []  # For tracking files of SCIENCE file
    for path in sorted(data_dir.glob(f"*IR-{band}_*{suffix}")):
        try:
            data = OIFitsReader(path).read()
        except Exception:
            continue
        if getattr(data, "category", "CAL") != "CAL":
            list_scivis.append(path)

    return list_scivis
