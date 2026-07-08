"""
retriever.py — Stage 1: MAST Catalogue Retriever
=================================================
Analogue of tsdat's retriever.yaml + reader layer.

In tsdat, the Retriever is the component that knows *where* raw data lives
and *how* to pull it into memory, but it applies no scientific logic: no
unit conversion beyond trivial column renaming, no SED fitting, no QC, no
derived quantities. Exactly one config-driven "how do I get bytes from the
archive onto disk, and bytes on disk into a DataFrame" responsibility.
``MASTRetriever`` below is that component for this pipeline: it fetches raw
JWST photometric catalogues (CEERS, JADES) from MAST / their public data
release pages, loads them into a `pandas.DataFrame` with normalised column
names, and validates that the minimal schema the rest of the pipeline
depends on is present. All science — SED fitting, residuals, anomaly
scoring — happens downstream in ``standardise.py`` and ``quality.py``.

Data provenance and fallback behaviour
---------------------------------------
CEERS and JADES DR1 catalogues are not exposed through a stable,
query-criteria-friendly `astroquery.mast.Observations` interface the way
single-visit HST/JWST observations are. This module therefore always
*attempts* an astroquery-based MAST lookup first (best-effort; the exact
proposal IDs / collections used are noted on each method), and if that
fails for any reason (import error, empty result set, network error) it
**falls back to a direct HTTPS download from the published data release
page** documented in the project README:

    CEERS DR1 : https://ceers.github.io/dr06.html
    JADES DR1 : https://archive.stsci.edu/hlsp/jades

Both of those URLs are HTML landing pages rather than direct file links, so
the fallback download is logged loudly (``logger.warning``) with a note
that the operator should confirm the retrieved payload is really the
catalogue file and not an HTML page, and update ``config/pipeline_config.yaml``
with the resolved direct-download URL once available.

Populated by: Claude Code Prompt 2.1
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests
from astropy.table import Table

logger = logging.getLogger(__name__)

# ── Module-level validation constants ────────────────────────────────────────

#: Columns that must be present (after snake_case normalisation) on every
#: catalogue for it to be usable downstream. Redshift is handled separately
#: via REDSHIFT_COLUMN_CANDIDATES since surveys name it differently.
REQUIRED_COLUMNS = ["ra", "dec"]

#: Acceptable column names for redshift; at least one must be present.
REDSHIFT_COLUMN_CANDIDATES = ["redshift", "z_phot"]

#: NIRCam/MIRI band prefixes used to identify photometric flux columns.
#: Mirrors config/pipeline_config.yaml -> retriever.band_prefixes.
BAND_PREFIXES = [
    "F090W",
    "F115W",
    "F150W",
    "F200W",
    "F277W",
    "F356W",
    "F410M",
    "F444W",
    "F770W",
]

#: Minimum number of distinct photometric bands required for a valid catalogue.
MIN_PHOTOMETRIC_BANDS = 4

#: Substrings that mark a column as *not* a bare flux measurement (e.g. its
#: uncertainty or a QC flag), even if it starts with a band prefix.
_FLUX_EXCLUDE_SUBSTRINGS = ("err", "unc", "flag", "wht", "weight")

# ── Data release locations (documented in README.md "Data Access") ──────────

#: CEERS DR1 published data release landing page (fallback download source).
CEERS_DR1_URL = "https://ceers.github.io/dr06.html"

#: JADES DR1 HLSP landing page on MAST (fallback download source).
JADES_DR1_URL = "https://archive.stsci.edu/hlsp/jades"

DEFAULT_RAW_DIR = Path("data/raw")


def _to_snake_case(name: str) -> str:
    """Normalise a catalogue column name to snake_case.

    Non-alphanumeric runs become a single underscore, and the result is
    lower-cased. This is intentionally simple (no camelCase splitting) so
    that already-delimited astronomical column names such as ``Z_PHOT`` or
    ``F090W_FLUX`` map predictably to ``z_phot`` / ``f090w_flux``.
    """
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def _infer_survey_from_path(filepath: Union[str, Path]) -> str:
    """Best-effort survey name inference from a catalogue filename."""
    stem = Path(filepath).stem.lower()
    for survey in ("ceers", "jades", "cosmos_web", "cosmos-web", "cosmos"):
        if survey.replace("-", "_") in stem.replace("-", "_"):
            return "cosmos_web" if "cosmos" in survey else survey
    return "unknown"


class MASTRetriever:
    """Fetches and lightly normalises raw JWST photometric catalogues.

    This is the tsdat "Retriever" analogue: it is responsible *solely* for
    getting raw bytes from MAST / a public data release onto disk, and raw
    tabular data into a `pandas.DataFrame` with sane column names. It must
    never perform scientific analysis (no SED fitting, no flagging, no
    derived quantities) — that is the responsibility of
    ``standardise.SEDStandardiser`` and ``quality`` stage components.
    """

    def __init__(self, raw_dir: Union[str, Path] = DEFAULT_RAW_DIR) -> None:
        self.raw_dir = Path(raw_dir)

    # ── Fetch methods ─────────────────────────────────────────────────────

    def fetch_ceers(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Download the CEERS DR1 photometric catalogue.

        Tries `astroquery.mast` first (CEERS is JWST proposal ID 1345); if
        that does not yield a usable product, falls back to a direct HTTPS
        download from the published data release page (`CEERS_DR1_URL`).

        Parameters
        ----------
        output_path : str | Path | None
            Destination file. Defaults to ``data/raw/ceers_dr1.fits``.

        Returns
        -------
        Path
            Path to the saved raw catalogue file.
        """
        output_path = Path(output_path) if output_path else self.raw_dir / "ceers_dr1.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._fetch_ceers_via_mast(output_path)
            logger.info("Fetched CEERS DR1 catalogue via astroquery.mast -> %s", output_path)
        except Exception as exc:  # noqa: BLE001 - any failure triggers documented fallback
            logger.warning(
                "astroquery.mast retrieval of CEERS DR1 failed (%s: %s). Falling back to "
                "direct HTTPS download from the published data release page %s. "
                "NOTE: that page is an HTML landing page, not a direct file link — verify "
                "the saved payload at %s is really the catalogue and update "
                "config/pipeline_config.yaml with the resolved direct-download URL once known.",
                type(exc).__name__,
                exc,
                CEERS_DR1_URL,
                output_path,
            )
            self._download_via_https(CEERS_DR1_URL, output_path)

        return output_path

    def fetch_jades(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Download the JADES DR1 photometric catalogue.

        Tries `astroquery.mast` first (JADES HLSP collection); if that does
        not yield a usable product, falls back to a direct HTTPS download
        from the published HLSP page (`JADES_DR1_URL`).

        Parameters
        ----------
        output_path : str | Path | None
            Destination file. Defaults to ``data/raw/jades_dr1.fits``.

        Returns
        -------
        Path
            Path to the saved raw catalogue file.
        """
        output_path = Path(output_path) if output_path else self.raw_dir / "jades_dr1.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._fetch_jades_via_mast(output_path)
            logger.info("Fetched JADES DR1 catalogue via astroquery.mast -> %s", output_path)
        except Exception as exc:  # noqa: BLE001 - any failure triggers documented fallback
            logger.warning(
                "astroquery.mast retrieval of JADES DR1 failed (%s: %s). Falling back to "
                "direct HTTPS download from the published HLSP page %s. "
                "NOTE: that page is an HTML landing page, not a direct file link — verify "
                "the saved payload at %s is really the catalogue and update "
                "config/pipeline_config.yaml with the resolved direct-download URL once known.",
                type(exc).__name__,
                exc,
                JADES_DR1_URL,
                output_path,
            )
            self._download_via_https(JADES_DR1_URL, output_path)

        return output_path

    # ── astroquery.mast helpers (best-effort primary path) ─────────────────

    def _fetch_ceers_via_mast(self, output_path: Path) -> None:
        """Best-effort CEERS retrieval via astroquery.mast.Observations.

        CEERS is JWST proposal ID 1345. This is a best-effort lookup: MAST's
        holdings and product naming for CEERS have changed across data
        releases, so any failure here (including no matching products) is
        treated as "astroquery access unavailable" by the caller, which
        falls back to the documented HTTPS download.
        """
        from astroquery.mast import Observations

        obs = Observations.query_criteria(obs_collection="JWST", proposal_id="1345")
        if len(obs) == 0:
            raise RuntimeError("astroquery.mast returned no CEERS (proposal 1345) observations")

        products = Observations.get_product_list(obs)
        catalog_products = products[
            [str(t).lower().endswith((".fits", ".csv")) for t in products["productFilename"]]
        ]
        if len(catalog_products) == 0:
            raise RuntimeError("astroquery.mast returned no downloadable CEERS catalog products")

        manifest = Observations.download_products(
            catalog_products[:1], download_dir=str(output_path.parent)
        )
        downloaded = Path(manifest["Local Path"][0])
        downloaded.replace(output_path)

    def _fetch_jades_via_mast(self, output_path: Path) -> None:
        """Best-effort JADES retrieval via astroquery.mast.Observations.

        JADES is published as a MAST High Level Science Product (HLSP)
        rather than a single proposal's raw observations, so this queries
        by HLSP collection name. As with CEERS, any failure here falls
        back to the documented HTTPS download.
        """
        from astroquery.mast import Observations

        obs = Observations.query_criteria(obs_collection="JADES", dataproduct_type="catalog")
        if len(obs) == 0:
            raise RuntimeError("astroquery.mast returned no JADES catalog observations")

        products = Observations.get_product_list(obs)
        catalog_products = products[
            [str(t).lower().endswith((".fits", ".csv")) for t in products["productFilename"]]
        ]
        if len(catalog_products) == 0:
            raise RuntimeError("astroquery.mast returned no downloadable JADES catalog products")

        manifest = Observations.download_products(
            catalog_products[:1], download_dir=str(output_path.parent)
        )
        downloaded = Path(manifest["Local Path"][0])
        downloaded.replace(output_path)

    @staticmethod
    def _download_via_https(url: str, output_path: Path, timeout: int = 60) -> None:
        """Stream a URL to disk. Shared fallback path for fetch_ceers/fetch_jades."""
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(output_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)

    # ── Load / validate ──────────────────────────────────────────────────

    def load_catalogue(
        self, filepath: Union[str, Path], survey: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a raw FITS or CSV catalogue into a normalised DataFrame.

        Column names are standardised to snake_case and a ``survey`` column
        is added (inferred from the filename if not given explicitly). No
        rows are dropped and no values are transformed — that belongs to
        ``standardise.SEDStandardiser``.

        Parameters
        ----------
        filepath : str | Path
            Path to a ``.fits``/``.fit`` or ``.csv`` catalogue file.
        survey : str | None
            Survey name to stamp on every row. Inferred from the filename
            (e.g. ``ceers_dr1.fits`` -> ``"ceers"``) if not given.

        Returns
        -------
        pd.DataFrame
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix in (".fits", ".fit"):
            table = Table.read(filepath)
            df = table.to_pandas()
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(
                f"Unsupported catalogue format {suffix!r} for {filepath}; expected "
                ".fits/.fit or .csv/.txt"
            )

        df = df.rename(columns={col: _to_snake_case(col) for col in df.columns})
        df["survey"] = survey if survey is not None else _infer_survey_from_path(filepath)

        return df

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """Check that a catalogue DataFrame has the minimal required schema.

        Required: ``ra``, ``dec``, one of ``redshift``/``z_phot``, and at
        least `MIN_PHOTOMETRIC_BANDS` distinct photometric flux columns
        (matched against `BAND_PREFIXES`).

        Returns
        -------
        dict
            ``{"valid": bool, "missing_columns": list[str], "n_sources": int,
            "redshift_range": tuple[float | None, float | None]}``
        """
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        redshift_col = next(
            (col for col in REDSHIFT_COLUMN_CANDIDATES if col in df.columns), None
        )
        if redshift_col is None:
            missing_columns.append("redshift|z_phot")

        bands_found = self._detect_flux_bands(df)
        if len(bands_found) < MIN_PHOTOMETRIC_BANDS:
            missing_columns.append(
                f"photometric_flux_columns(found={len(bands_found)}, "
                f"need>={MIN_PHOTOMETRIC_BANDS})"
            )

        n_sources = len(df)

        if redshift_col is not None and n_sources > 0:
            redshift_range = (
                float(df[redshift_col].min()),
                float(df[redshift_col].max()),
            )
        else:
            redshift_range = (None, None)

        return {
            "valid": len(missing_columns) == 0,
            "missing_columns": missing_columns,
            "n_sources": n_sources,
            "redshift_range": redshift_range,
        }

    @staticmethod
    def _detect_flux_bands(df: pd.DataFrame) -> set:
        """Return the set of BAND_PREFIXES represented by flux columns in df."""
        found = set()
        for col in df.columns:
            col_l = col.lower()
            if any(x in col_l for x in _FLUX_EXCLUDE_SUBSTRINGS):
                continue
            for band in BAND_PREFIXES:
                if col_l.startswith(band.lower()):
                    found.add(band)
                    break
        return found


def _print_validation_report(survey: str, path: Path, report: dict) -> None:
    print(f"\n--- {survey} ({path}) ---")
    print(f"  valid:           {report['valid']}")
    print(f"  n_sources:       {report['n_sources']}")
    print(f"  redshift_range:  {report['redshift_range']}")
    print(f"  missing_columns: {report['missing_columns']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = MASTRetriever()

    ceers_path = retriever.fetch_ceers()
    ceers_df = retriever.load_catalogue(ceers_path, survey="ceers")
    _print_validation_report("CEERS DR1", ceers_path, retriever.validate_schema(ceers_df))

    jades_path = retriever.fetch_jades()
    jades_df = retriever.load_catalogue(jades_path, survey="jades")
    _print_validation_report("JADES DR1", jades_path, retriever.validate_schema(jades_df))
