"""
Calibrator spectral databases download manager for MATISSE flux calibration.

Databases are hosted on Zenodo and downloaded/cached on first use via
`pooch <https://www.fatiando.org/pooch/>`_.  The local cache lives in the
platform-appropriate app-data directory (``~/.cache/matisse/cal_databases``
on Linux/macOS).

**During development** (before the Zenodo DOI is published) you can override
the download entirely by pointing the environment variable
``MATISSE_CAL_DB_PATH`` to a directory that already contains the three FITS
files::

    export MATISSE_CAL_DB_PATH=/path/to/local/calib_spec_databases

The module exposes a single public function:

.. code-block:: python

    from matisse.core.flux.databases import get_cal_databases_dir

    db_dir = get_cal_databases_dir()  # Path, guaranteed to exist

Version history (bump ``_DB_VERSION`` when new databases are published):

+-------+-------+----------------------------------------------------------------+
| ver.  | DOI   | databases                                                      |
+=======+=======+================================================================+
| 1     | TBD   | vBoekelDatabase, calib_spec_db_v10, calib_spec_db_v10_supp.   |
+-------+-------+----------------------------------------------------------------+
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version & Zenodo metadata
# ---------------------------------------------------------------------------

#: Bump this integer when a new set of databases is released on Zenodo.
_DB_VERSION = 1

#: Zenodo record URL template – replace ``{record_id}`` once the deposit is
#: published.  During development the placeholder keeps the code importable.
_ZENODO_RECORD_ID = "PENDING"  # e.g. "15012345"
_ZENODO_BASE_URL = f"https://zenodo.org/records/{_ZENODO_RECORD_ID}/files"

# SHA-256 hashes (computed from the canonical copies in legacy/).
# Update the hash whenever a database file is revised and bump _DB_VERSION.
_DB_REGISTRY: dict[str, str] = {
    "vBoekelDatabase.fits": (
        "sha256:b0a2e5e5b05bda32b035cb25e6d23e58715cc56032e6773e19caf02aaf646d95"
    ),
    "calib_spec_db_v10.fits": (
        "sha256:8e0a39c34b314cf1c3bf070755106dcdbf0afb410f6856903d30e37b0c445380"
    ),
    "calib_spec_db_v10_supplement.fits": (
        "sha256:ebb4b0bbd227d2bd4f57a2c06cd96170207aae784753e32a816435ac86dde42d"
    ),
}

# ---------------------------------------------------------------------------
# Fallback: legacy bundled copy (present in the git repo for dev convenience)
# ---------------------------------------------------------------------------

_LEGACY_BUNDLE = (
    Path(__file__).parent.parent.parent  # src/matisse/
    / "legacy"
    / "calib_spec_databases"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_cal_databases_dir() -> Path:
    """Return the directory that contains the calibrator FITS databases.

    Resolution order
    ----------------
    1. ``MATISSE_CAL_DB_PATH`` environment variable (any absolute path).
    2. ``pooch`` cache – downloads from Zenodo on first call, then reuses.
    3. Legacy bundled copy in ``src/matisse/legacy/calib_spec_databases/``
       (only available in developer installs, not in wheels).

    Returns
    -------
    Path
        Absolute path to a directory containing the three database files.

    Raises
    ------
    RuntimeError
        If no databases can be found or downloaded.
    """
    # ------------------------------------------------------------------
    # 1. Explicit local override (dev / pre-publication / offline use)
    # ------------------------------------------------------------------
    env_path = os.environ.get("MATISSE_CAL_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.is_dir():
            logger.info("Using calibrator databases from MATISSE_CAL_DB_PATH: %s", p)
            return p
        logger.warning("MATISSE_CAL_DB_PATH='%s' does not exist — ignoring.", env_path)

    # ------------------------------------------------------------------
    # 2. Pooch download / cache
    # ------------------------------------------------------------------
    try:
        db_dir = _ensure_pooch_cache()
        return db_dir
    except Exception as exc:
        logger.warning("pooch download failed (%s) — trying legacy bundle.", exc)

    # ------------------------------------------------------------------
    # 3. Legacy developer bundle
    # ------------------------------------------------------------------
    if _LEGACY_BUNDLE.is_dir():
        fits_found = list(_LEGACY_BUNDLE.glob("*.fits"))
        if fits_found:
            logger.info("Using legacy bundled calibrator databases: %s", _LEGACY_BUNDLE)
            return _LEGACY_BUNDLE

    raise RuntimeError(
        "Calibrator spectral databases not found.\n\n"
        "Options:\n"
        "  • Set MATISSE_CAL_DB_PATH=/path/to/local/calib_spec_databases\n"
        "  • Ensure internet access for automatic Zenodo download\n"
        "  • Use a developer install (git clone) which includes the legacy bundle\n"
    )


def _ensure_pooch_cache() -> Path:
    """Download all database files (if not already cached) and return the cache dir.

    Uses ``pooch`` with SHA-256 integrity verification.  Files are only
    downloaded once; subsequent calls return immediately from the cache.
    """
    try:
        import pooch
    except ImportError as exc:
        raise ImportError(
            "pooch is required to download calibrator databases.  "
            "Install it with:  pip install pooch"
        ) from exc

    if _ZENODO_RECORD_ID == "PENDING":
        raise RuntimeError(
            "Zenodo record not yet published (_ZENODO_RECORD_ID == 'PENDING'). "
            "Set MATISSE_CAL_DB_PATH to use a local copy in the meantime."
        )

    cache_dir = pooch.os_cache("matisse") / f"cal_databases_v{_DB_VERSION}"

    retriever = pooch.create(
        path=cache_dir,
        base_url=_ZENODO_BASE_URL + "/",
        version=f"v{_DB_VERSION}",
        version_dev="main",
        registry=_DB_REGISTRY,
        # Retry up to 3 times on transient network errors
        retry_if_failed=3,
    )

    for fname in _DB_REGISTRY:
        local_path = Path(retriever.fetch(fname))
        logger.debug("Database ready: %s", local_path)

    # All files land in the same directory
    return cache_dir


# ---------------------------------------------------------------------------
# CLI helper: download all files eagerly (matisse doctor / prefetch)
# ---------------------------------------------------------------------------


def prefetch_databases() -> Path:
    """Download all calibrator databases to the pooch cache and return the dir.

    Raises ``RuntimeError`` if the Zenodo record is not yet published or the
    download fails.  Useful to call from ``matisse doctor`` or a setup script.
    """
    if _ZENODO_RECORD_ID == "PENDING":
        raise RuntimeError(
            "Zenodo record not yet published.  "
            "Update _ZENODO_RECORD_ID in matisse/core/flux/databases.py."
        )
    return _ensure_pooch_cache()


def database_status() -> dict[str, str]:
    """Return a dict mapping each database filename to its local status.

    Possible statuses: ``"cached"``, ``"legacy_bundle"``, ``"missing"``,
    ``"missing (pooch not installed)"``, ``"local_override"``.
    Useful for ``matisse doctor``.
    """
    env_path = os.environ.get("MATISSE_CAL_DB_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        return {
            fname: "local_override" if (p / fname).exists() else "missing"
            for fname in _DB_REGISTRY
        }

    cache_dir: Path | None
    pooch_available = True
    try:
        import pooch

        cache_dir = pooch.os_cache("matisse") / f"cal_databases_v{_DB_VERSION}"
    except ImportError:
        pooch_available = False
        cache_dir = None

    status: dict[str, str] = {}
    for fname in _DB_REGISTRY:
        in_cache = cache_dir is not None and (cache_dir / fname).exists()
        in_legacy = (_LEGACY_BUNDLE / fname).exists()

        if in_cache:
            status[fname] = "cached"
        elif in_legacy:
            status[fname] = "legacy_bundle"
        elif not pooch_available:
            status[fname] = "missing (pooch not installed)"
        else:
            status[fname] = "missing"

    return status
