"""
standardise.py — Stage 2: SED Standardisation Layer
=====================================================
Analogue of tsdat's dataset.yaml + standardisation layer.

In tsdat, this stage turns a raw, loosely-structured input (whatever the
Retriever handed back) into a clean, fully-described xarray Dataset at a
named data level — variables renamed/typed to the target schema, units
normalised, and global/variable metadata attached. Nothing here decides
what counts as anomalous; that is `quality.py`'s job. This module's output
is data level **"a1"**: science-ready, but not yet quality-flagged.

Concretely, `SEDStandardiser` takes the DataFrame produced by
`pipeline.retriever.MASTRetriever.load_catalogue`, and:

1. `preprocess`      — filters to the science redshift window and minimum
                        photometric coverage, and normalises flux units.
2. `run_eazy_fit`     — obtains a best-fit photo-z / template / chi2 per
                        source by fitting with the real `eazy-py` Python API
                        (github.com/gbrammer/eazy-py) against the
                        gbrammer/eazy-photoz template and filter set, falling
                        back to a clearly-logged synthetic stub only if the
                        real fit cannot run (e.g. no network access to fetch
                        the template/filter data on first use).
3. `extract_residuals`— computes per-band SED residuals against EAZY's own
                        best-fit template photometry (`PhotoZ.fmodel`) when a
                        real fit was just run, packaging everything into an
                        xarray Dataset with tsdat-style global attributes. If
                        no real fit products are available (e.g. calling this
                        method standalone, as the unit tests do) it falls
                        back to a smooth log-log power-law continuum fit
                        through each source's own photometry as a stand-in
                        model.
4. `save`             — writes that Dataset to NetCDF using tsdat's
                        `{survey}.{level}.{date}.{time}.nc` naming convention.

Populated by: Claude Code Prompt 2.2
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from pipeline.retriever import BAND_PREFIXES

logger = logging.getLogger(__name__)

# ── Physical constants used to build the stand-in SED continuum model ───────

#: Approximate pivot wavelengths (microns) for the NIRCam/MIRI bands this
#: pipeline works with. Used by `extract_residuals` to fit a smooth
#: log-log power-law continuum per source as a stand-in for the true EAZY
#: best-fit template photometry (`obs_sed`), which a full implementation
#: would read directly from EAZY's output products.
BAND_PIVOT_WAVELENGTH_UM = {
    "F090W": 0.902,
    "F115W": 1.154,
    "F150W": 1.501,
    "F200W": 1.990,
    "F277W": 2.786,
    "F356W": 3.563,
    "F410M": 4.092,
    "F444W": 4.421,
    "F770W": 7.700,
}

#: Flux unit conversion factors to the pipeline's target unit (uJy),
#: keyed by the unit suffix a column name may carry (e.g. "f090w_flux_njy"),
#: or by a survey's configured `raw_flux_unit` (see `_convert_fluxes_to_ujy`).
_UNIT_CONVERSION_TO_UJY = {
    "njy": 1e-3,
    "ujy": 1.0,
    "mjy": 1e3,
    "jy": 1e6,
}

#: EAZY FILTER.RES.latest filter index for each JWST band this pipeline
#: works with, taken from gbrammer/eazy-photoz's
#: filters/FILTER.RES.latest.info (jwst_nircam_* / jwst_miri_* entries).
#: Used to build the `zphot.translate` file `eazy-py` needs to map this
#: pipeline's `<band>_flux(_err)` columns onto EAZY's filter curves.
EAZY_FILTER_INDEX = {
    "F090W": 363,
    "F115W": 364,
    "F150W": 365,
    "F200W": 366,
    "F277W": 375,
    "F356W": 376,
    "F410M": 383,
    "F444W": 377,
    "F770W": 396,
}

#: Fallback per-band flux-uncertainty column suffixes, checked in priority
#: order, when the canonical "<band>_flux_err" column is absent. CEERS DR1.0
#: (Cox et al. 2025) does not use "<band>_flux_err" at all; it publishes
#: "<band>_fluxerr_emp" (empirical, preferred) and "<band>_fluxerr_se"
#: (SExtractor formal error) instead. Same unit as "<band>_flux" (uJy),
#: verified via S/N sanity check against the real catalogue.
_FLUX_ERR_FALLBACK_SUFFIXES = ["fluxerr_emp", "fluxerr_se"]


def log_data_level_transition(from_level: str, to_level: str, n_in: int, n_out: int) -> None:
    """Log a standardised tsdat-style pipeline stage transition line.

    Uses the module logger (rather than a bare `print`) so verbosity is
    controlled by the caller's `logging` configuration, consistent with the
    rest of this pipeline (see `pipeline.retriever`).
    """
    pct_retained = (n_out / n_in * 100.0) if n_in else 0.0
    logger.info(
        "[%s -> %s] %d -> %d sources retained (%.1f%%, %d removed)",
        from_level,
        to_level,
        n_in,
        n_out,
        pct_retained,
        n_in - n_out,
    )


class SEDStandardiser:
    """Turns a raw photometric catalogue into a science-ready "a1" Dataset.

    This is the tsdat "standardisation" analogue: it renames/filters/units-
    normalises raw tabular input and produces a fully-described xarray
    Dataset. It does not flag or score anomalies (see `quality.py`).
    """

    def __init__(self, config_path: Union[str, Path], survey: Optional[str] = None) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as fh:
            self.config = yaml.safe_load(fh)

        retriever_cfg = self.config.get("retriever", {}) or {}
        standardiser_cfg = self.config.get("standardiser", {}) or {}

        # Survey name: explicit override, else the first survey configured
        # under retriever.surveys. A single SEDStandardiser instance can
        # still be reused across surveys by passing `survey=` to individual
        # calls where needed; this is just the constructor-time default.
        surveys = retriever_cfg.get("surveys") or []
        self.survey = survey or (surveys[0].get("name", "unknown") if surveys else "unknown")

        # Per-survey raw flux unit override (e.g. CEERS DR1.0's <band>_FLUX
        # columns carry no unit suffix but are actually nJy, not uJy -- see
        # `raw_flux_unit` in pipeline_config.yaml and `_convert_fluxes_to_ujy`).
        survey_cfg = next((s for s in surveys if s.get("name") == self.survey), {})
        self._raw_flux_unit: Optional[str] = survey_cfg.get("raw_flux_unit")

        self.band_list: List[str] = retriever_cfg.get("band_prefixes") or list(BAND_PREFIXES)
        self.redshift_range = tuple(standardiser_cfg.get("redshift_range", [0.5, 10.0]))
        self.min_photometric_bands: int = standardiser_cfg.get("min_photometric_bands", 4)
        self.flux_unit: str = standardiser_cfg.get("flux_unit", "uJy")
        self.sed_fitter: str = standardiser_cfg.get("sed_fitter", "eazy")
        self.eazy_config: dict = standardiser_cfg.get("eazy", {}) or {}
        self.output_filename_convention: str = standardiser_cfg.get(
            "output_filename_convention", "{survey}.a1.{date}.{time}.nc"
        )

        self._eazy_version = "unknown"
        #: Best-fit template photometry from the most recent real
        #: `run_eazy_fit` call (dict of `n`/`bands`/`fnu`/`efnu`/`fmodel`/
        #: `ok_data`), consumed by `extract_residuals` to compute real
        #: per-band residuals. `None` when no real fit has been run yet (or
        #: it fell back to the synthetic stub), in which case
        #: `extract_residuals` falls back to its power-law stand-in model.
        self._last_fit_products: Optional[dict] = None

    # ── preprocess ───────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to the science redshift window & minimum band coverage,
        and normalise flux units to `self.flux_unit`.

        Logs the number of sources removed at each step.
        """
        n0 = len(df)
        df = df.copy()

        redshift_col = next(
            (c for c in ("z_phot", "redshift", "lp_z_best", "lp_z_med") if c in df.columns),
            "redshift",
        )
        if redshift_col not in df.columns:
            raise ValueError(
                "preprocess() requires a 'z_phot', 'redshift', 'lp_z_best', or 'lp_z_med' column"
            )
        if redshift_col in ("lp_z_best", "lp_z_med"):
            # Normalise CEERS DR1.0 (Cox et al. 2025) LePHARE photo-z columns to
            # 'z_phot' so downstream stages (run_eazy_fit, extract_residuals, ...)
            # pick it up without needing to know about survey-specific naming.
            df = df.rename(columns={redshift_col: "z_phot"})
            redshift_col = "z_phot"

        z_min, z_max = self.redshift_range
        df = df[df[redshift_col].between(z_min, z_max)]
        n1 = len(df)
        logger.info(
            "preprocess: redshift filter [%s, %s] on '%s' removed %d/%d sources (%d remain)",
            z_min,
            z_max,
            redshift_col,
            n0 - n1,
            n0,
            n1,
        )

        bands = self._flux_bands_present(df)
        df = self._canonicalize_flux_err_columns(df, bands)
        df = self._convert_fluxes_to_ujy(df, bands)

        flux_cols = [f"{b.lower()}_flux" for b in bands]
        if flux_cols:
            valid_counts = ((df[flux_cols].notna()) & (df[flux_cols] > 0)).sum(axis=1)
        else:
            valid_counts = pd.Series(0, index=df.index)
        df = df[valid_counts >= self.min_photometric_bands]
        n2 = len(df)
        logger.info(
            "preprocess: minimum coverage filter (>=%d valid bands) removed %d/%d sources "
            "(%d remain)",
            self.min_photometric_bands,
            n1 - n2,
            n1,
            n2,
        )

        log_data_level_transition("raw", "a1-preprocessed", n0, n2)
        return df.reset_index(drop=True)

    def _flux_bands_present(self, df: pd.DataFrame) -> List[str]:
        """Bands from `self.band_list` that have a `<band>_flux` column in df."""
        return [b for b in self.band_list if f"{b.lower()}_flux" in df.columns]

    def _canonicalize_flux_err_columns(self, df: pd.DataFrame, bands: List[str]) -> pd.DataFrame:
        """Rename survey-specific per-band error columns to the canonical
        `<band>_flux_err` name that `_convert_fluxes_to_ujy` and
        `extract_residuals` expect.

        Without this, a survey whose error columns are never named
        `<band>_flux_err` (e.g. CEERS DR1.0's `<band>_fluxerr_emp`/`_se`)
        silently fails every per-band error lookup: no source ever has >=2
        bands with a valid error, so `extract_residuals` produces an all-NaN
        residual matrix and every downstream anomaly score collapses to a
        constant 0 -- verified against the real CEERS catalogue.
        """
        rename = {}
        renamed = []
        for band in bands:
            canonical = f"{band.lower()}_flux_err"
            if canonical in df.columns:
                continue
            for suffix in _FLUX_ERR_FALLBACK_SUFFIXES:
                candidate = f"{band.lower()}_{suffix}"
                if candidate in df.columns:
                    rename[candidate] = canonical
                    renamed.append(f"{canonical} (from {candidate})")
                    break
        if rename:
            df = df.rename(columns=rename)
            logger.info("preprocess: canonicalised per-band error columns: %s", renamed)
        return df

    def _convert_fluxes_to_ujy(self, df: pd.DataFrame, bands: List[str]) -> pd.DataFrame:
        """Normalise flux/flux_err columns to the target unit (uJy).

        Columns carrying an explicit unit suffix (e.g. `<band>_flux_njy`)
        are converted and renamed to drop the suffix. Columns already named
        `<band>_flux(_err)` (no unit suffix) are converted using this
        survey's configured `raw_flux_unit` (`self._raw_flux_unit`) if set
        (e.g. CEERS DR1.0's un-suffixed columns are actually nJy -- see
        `raw_flux_unit` in pipeline_config.yaml); otherwise they are assumed
        to already be in the target unit, matching the convention used by
        `MASTRetriever.load_catalogue`.
        """
        df = df.copy()
        raw_factor = _UNIT_CONVERSION_TO_UJY.get((self._raw_flux_unit or "").lower())
        converted, assumed = [], []
        for band in bands:
            for kind in ("flux", "flux_err"):
                base = f"{band.lower()}_{kind}"
                matched_suffix = None
                for suffix, factor in _UNIT_CONVERSION_TO_UJY.items():
                    candidate = f"{base}_{suffix}"
                    if candidate in df.columns:
                        df[base] = df.pop(candidate) * factor
                        matched_suffix = suffix
                        break
                if matched_suffix:
                    converted.append(f"{base} (from {matched_suffix})")
                elif base in df.columns and raw_factor is not None:
                    df[base] = df[base] * raw_factor
                    converted.append(f"{base} (from configured raw_flux_unit={self._raw_flux_unit})")
                elif base in df.columns:
                    assumed.append(base)

        if converted:
            logger.info("preprocess: converted flux columns to %s: %s", self.flux_unit, converted)
        if assumed:
            logger.debug(
                "preprocess: no unit suffix found; assuming already in %s: %s",
                self.flux_unit,
                assumed,
            )
        return df

    # ── EAZY fit ─────────────────────────────────────────────────────────

    def run_eazy_fit(self, df: pd.DataFrame, output_dir: Union[str, Path]) -> pd.DataFrame:
        """Fit photometric redshifts with EAZY and return best-fit params
        (`z_a`, `chi2`, `template_id`) merged with the input DataFrame.

        Writes EAZY's `zphot.cat`/`zphot.translate`/`zphot.param` input
        files, then fits with the real `eazy-py` Python API
        (`eazy.photoz.PhotoZ`) against the gbrammer/eazy-photoz template and
        filter set (fetched on first use if not already present locally —
        see `_ensure_eazy_templates`). Also stashes the fit's best-fit
        template photometry in `self._last_fit_products` so a subsequent
        `extract_residuals` call computes real per-band residuals against
        it. If the real fit cannot run for any reason (`eazy-py` not
        importable, template/filter data unreachable, a fit error), falls
        back to a synthetic stub — clearly logged — so the rest of the
        pipeline remains runnable during development.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = df.reset_index(drop=True)
        ids = np.arange(1, len(df) + 1)

        try:
            fit_df, version = self._fit_with_eazy_py(df, ids, output_dir)
            self._eazy_version = version
            logger.info(
                "EAZY fit completed for %d sources -> %s (%s)", len(df), output_dir, version
            )
        except Exception as exc:  # noqa: BLE001 - any failure triggers documented fallback
            logger.warning(
                "Real EAZY fit failed (%s: %s). Falling back to a SYNTHETIC STUB fit "
                "(lognormal chi2, jittered z_a, random template_id) so the pipeline remains "
                "runnable during development — these values are NOT suitable for science and "
                "must not be used to draw astrophysical conclusions.",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            fit_df = self._stub_eazy_fit(df, ids)
            self._eazy_version = "synthetic-stub"
            self._last_fit_products = None

        merged = df.copy()
        merged["id"] = ids
        merged = merged.merge(fit_df[["id", "z_a", "chi2", "template_id"]], on="id", how="left")
        return merged

    def _ensure_eazy_templates(self) -> Path:
        """Locate the gbrammer/eazy-photoz template & filter data `eazy-py`
        needs but does not ship on PyPI, fetching it via `git clone` on
        first use if necessary.

        Honours an existing `$EAZYCODE` if the caller has already set one
        (e.g. to share a cache across runs); otherwise defaults to
        `<repo_root>/EAZY`, which `.gitignore` already excludes.

        Returns
        -------
        Path
            The `eazy-photoz` checkout directory (contains `templates/` and
            `filters/`).
        """
        import eazy
        import eazy.utils as ezutils

        base = os.environ.get("EAZYCODE")
        if not base:
            base = str(Path(__file__).resolve().parent.parent / "EAZY")
            os.environ["EAZYCODE"] = base
        eazy.set_data_path(path=base)

        photoz_dir = Path(base) if Path(base).name == "eazy-photoz" else Path(base) / "eazy-photoz"
        if not (photoz_dir / "templates").exists():
            logger.info(
                "EAZY templates/filters not found at %s; fetching gbrammer/eazy-photoz "
                "from GitHub...",
                photoz_dir,
            )
            eazy.fetch_eazy_photoz()
        if not (photoz_dir / "templates").exists():
            raise RuntimeError(
                f"EAZY templates/filters still missing at {photoz_dir} after fetch attempt"
            )

        # `eazy.set_data_path()` only reassigns the `eazy` package's own
        # DATA_PATH global. Every module that actually resolves template/
        # filter/prior file paths (filters.py, templates.py, photoz.py,
        # param.py, sps.py) does so via `utils.DATA_PATH`, and `eazy/utils.py`
        # bound that name with `from . import DATA_PATH` at import time --
        # a static binding that a later `eazy.DATA_PATH = ...` does not
        # propagate to. Set it directly, or every relative template
        # component path (e.g. `tweak_fsps_QSF_12_v3_001.dat`) resolves
        # against the (missing) site-packages copy instead of this checkout.
        ezutils.DATA_PATH = str(photoz_dir)
        eazy.DATA_PATH = str(photoz_dir)

        return photoz_dir

    def _fit_with_eazy_py(
        self, df: pd.DataFrame, ids: np.ndarray, output_dir: Path
    ) -> tuple[pd.DataFrame, str]:
        """Fit real photometric redshifts using the `eazy-py` Python API.

        Returns the best-fit `(id, z_a, chi2, template_id)` DataFrame and an
        `eazy-py` version string. As a side effect, stores per-band
        best-fit template photometry in `self._last_fit_products` for
        `extract_residuals` to consume.
        """
        import eazy
        import eazy.photoz as ezphot

        photoz_dir = self._ensure_eazy_templates()

        bands = self._flux_bands_present(df)
        if not bands:
            raise ValueError("No recognised photometric flux columns present for EAZY fit")

        cat_path = self._write_eazy_catalog(df, ids, output_dir)
        translate_path = self._write_eazy_translate(output_dir, bands)

        template_set = self.eazy_config.get("template_set", "tweak_fsps_QSF_12_v3")
        params = {
            "CATALOG_FILE": str(cat_path),
            "CATALOG_FORMAT": "ascii.commented_header",
            "MAIN_OUTPUT_FILE": "photz",
            "OUTPUT_DIRECTORY": str(output_dir),
            "MW_EBV": 0.0,
            "Z_MIN": self.eazy_config.get("z_min", 0.01),
            "Z_MAX": self.eazy_config.get("z_max", 12.0),
            "Z_STEP": self.eazy_config.get("z_step", 0.01),
            "SYS_ERR": self.eazy_config.get("sys_err", 0.02),
            # preprocess() normalises catalogue fluxes to uJy; 23.9 is the
            # AB zeropoint for uJy, needed so EAZY's apparent-magnitude
            # prior is evaluated against the right flux scale.
            "PRIOR_ABZP": 23.9,
            "PRIOR_FILTER": 205,  # standard eazy-py F160W apparent-mag prior filter
            "PRIOR_FILE": str(photoz_dir / "templates" / "prior_F160W_TAO.dat"),
            "FILTERS_RES": str(photoz_dir / "filters" / "FILTER.RES.latest"),
            "TEMPLATES_FILE": str(photoz_dir / "templates" / "fsps_full" / f"{template_set}.param"),
            "TEMP_ERR_FILE": str(photoz_dir / "templates" / "uvista_nmf" / "template_error_10.def"),
            "WAVELENGTH_FILE": str(photoz_dir / "templates" / "uvista_nmf" / "lambda.def"),
        }

        ez = ezphot.PhotoZ(
            param_file=None,
            translate_file=str(translate_path),
            zeropoint_file=None,
            params=params,
            load_prior=True,
            load_products=False,
        )
        ez.param.write(str(output_dir / "zphot.param"))

        n_proc = self.eazy_config.get("n_proc", 0)
        ez.fit_catalog(n_proc=n_proc)
        ez.fit_at_zbest(n_proc=n_proc)

        # EAZY fits a non-negative linear combination of all templates
        # simultaneously (TEMPLATE_COMBOS='a'), so there's no single native
        # "template_id" the way the old eazy C-code .zout format had one;
        # take the most-heavily-weighted template in that combination as
        # the closest analogue.
        template_id = np.argmax(ez.coeffs_best, axis=1) + 1

        fit_df = pd.DataFrame(
            {
                "id": ids,
                "z_a": np.asarray(ez.zbest),
                "chi2": np.asarray(ez.chi2_best),
                "template_id": template_id,
            }
        )

        self._last_fit_products = {
            "n": len(df),
            "bands": bands,
            "fnu": np.asarray(ez.fnu),
            "efnu": np.asarray(ez.efnu),
            "fmodel": np.asarray(ez.fmodel),
            "ok_data": np.asarray(ez.ok_data),
        }
        return fit_df, f"eazy-py {eazy.__version__}"

    def _write_eazy_catalog(self, df: pd.DataFrame, ids: np.ndarray, output_dir: Path) -> Path:
        """Write EAZY's whitespace-delimited `zphot.cat` photometry catalogue."""
        bands = self._flux_bands_present(df)
        cat_path = output_dir / "zphot.cat"

        header_cols = ["id", "ra", "dec"]
        for band in bands:
            header_cols += [f"f_{band}", f"e_{band}"]

        lines = ["# " + " ".join(header_cols)]
        for i in range(len(df)):
            row = df.iloc[i]
            values = [str(ids[i]), f"{row.get('ra', -99.0):.6f}", f"{row.get('dec', -99.0):.6f}"]
            for band in bands:
                flux = row.get(f"{band.lower()}_flux", -99.0)
                err = row.get(f"{band.lower()}_flux_err", -99.0)
                values.append(f"{flux:.6f}" if pd.notna(flux) else "-99.000000")
                values.append(f"{err:.6f}" if pd.notna(err) else "-99.000000")
            lines.append(" ".join(values))

        cat_path.write_text("\n".join(lines) + "\n")
        return cat_path

    @staticmethod
    def _write_eazy_translate(output_dir: Path, bands: List[str]) -> Path:
        """Write EAZY's `zphot.translate` file mapping this pipeline's
        `f_<band>`/`e_<band>` catalogue columns onto EAZY filter numbers
        (`EAZY_FILTER_INDEX`)."""
        translate_path = output_dir / "zphot.translate"
        lines = []
        for band in bands:
            idx = EAZY_FILTER_INDEX.get(band)
            if idx is None:
                raise ValueError(f"No EAZY_FILTER_INDEX entry configured for band {band!r}")
            lines.append(f"f_{band} F{idx}")
            lines.append(f"e_{band} E{idx}")
        translate_path.write_text("\n".join(lines) + "\n")
        return translate_path

    def _stub_eazy_fit(self, df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        """Synthetic stand-in for a real EAZY fit, used when EAZY is unavailable.

        `chi2` is drawn from a lognormal distribution (as required); `z_a`
        jitters the input photo-z slightly (simulating a fit converging
        near the input estimate); `template_id` is drawn uniformly over the
        configured template set's approximate size.
        """
        n = len(df)
        rng = np.random.default_rng()

        redshift_col = "z_phot" if "z_phot" in df.columns else "redshift"
        if redshift_col in df.columns:
            z_a = df[redshift_col].to_numpy() * (1.0 + rng.normal(0, 0.02, size=n))
        else:
            z_a = rng.uniform(*self.redshift_range, size=n)

        chi2 = rng.lognormal(mean=0.5, sigma=0.7, size=n)

        template_set = str(self.eazy_config.get("template_set", ""))
        match = re.search(r"(\d+)", template_set)
        n_templates = int(match.group(1)) if match else 12
        template_id = rng.integers(1, max(n_templates, 1) + 1, size=n)

        return pd.DataFrame({"id": ids, "z_a": z_a, "chi2": chi2, "template_id": template_id})

    # ── residual extraction ─────────────────────────────────────────────

    def extract_residuals(self, df: pd.DataFrame, fit_df: pd.DataFrame) -> xr.Dataset:
        """Compute per-band SED residuals and package them as an "a1" Dataset.

        `df` and `fit_df` must be row-aligned (e.g. `fit_df` is the output
        of `run_eazy_fit(df, ...)`, which preserves row order).

        If `df`/`fit_df` are the same catalogue a real `run_eazy_fit` call
        just fit (checked via `self._last_fit_products`), each source's
        "model" flux is EAZY's own best-fit template photometry at that
        source's best-fit redshift (`PhotoZ.fmodel`) — the real EAZY output.
        Otherwise (e.g. this method is called standalone, as the unit tests
        do, or the most recent fit fell back to the synthetic stub) each
        source's model flux is instead a smooth log-log power-law fit
        through its own observed photometry, as a stand-in for the true
        EAZY template SED. Residuals are `(obs_flux - model_flux) /
        obs_flux_err` either way.
        """
        if len(df) != len(fit_df):
            raise ValueError(
                f"df and fit_df must be row-aligned; got {len(df)} vs {len(fit_df)} rows"
            )

        df = df.reset_index(drop=True)
        fit_df = fit_df.reset_index(drop=True)
        n = len(df)

        bands = self._flux_bands_present(df)
        if not bands:
            raise ValueError("extract_residuals: no recognised photometric flux columns in df")

        real = self._last_fit_products
        use_real = real is not None and real.get("n") == n and real.get("bands") == bands

        if use_real:
            with np.errstate(invalid="ignore", divide="ignore"):
                raw_residuals = (real["fnu"] - real["fmodel"]) / real["efnu"]
            residuals = np.where(real["ok_data"], raw_residuals, np.nan)
            logger.info(
                "extract_residuals: using real EAZY best-fit template photometry for %d sources",
                n,
            )
        else:
            residuals = np.full((n, len(bands)), np.nan)
            for i in range(n):
                # Per-source loop: each source can have a different subset
                # of valid bands, so the continuum fit is genuinely per-row
                # rather than a single vectorisable operation across the
                # catalogue.
                wavelengths, fluxes, errs, band_idx = [], [], [], []
                for j, band in enumerate(bands):
                    flux = df.at[i, f"{band.lower()}_flux"]
                    err_col = f"{band.lower()}_flux_err"
                    err = df.at[i, err_col] if err_col in df.columns else np.nan
                    if pd.notna(flux) and flux > 0 and pd.notna(err) and err > 0:
                        wavelengths.append(BAND_PIVOT_WAVELENGTH_UM[band])
                        fluxes.append(flux)
                        errs.append(err)
                        band_idx.append(j)

                if len(fluxes) >= 2:
                    log_w = np.log10(wavelengths)
                    log_f = np.log10(fluxes)
                    slope, intercept = np.polyfit(log_w, log_f, 1)
                    model_flux = 10 ** (intercept + slope * log_w)
                elif len(fluxes) == 1:
                    # A single valid band carries no shape information to
                    # compare against; treat it as its own model (zero residual).
                    model_flux = np.array(fluxes)
                else:
                    continue

                for k, j in enumerate(band_idx):
                    residuals[i, j] = (fluxes[k] - model_flux[k]) / errs[k]

        chi2 = fit_df["chi2"].to_numpy() if "chi2" in fit_df.columns else np.full(n, np.nan)
        z_a = fit_df["z_a"].to_numpy() if "z_a" in fit_df.columns else np.full(n, np.nan)
        template_id = (
            fit_df["template_id"].to_numpy() if "template_id" in fit_df.columns else np.full(n, -1)
        )
        redshift_col = "z_phot" if "z_phot" in df.columns else "redshift"
        z_phot = df[redshift_col].to_numpy() if redshift_col in df.columns else np.full(n, np.nan)
        ra = df["ra"].to_numpy() if "ra" in df.columns else np.full(n, np.nan)
        dec = df["dec"].to_numpy() if "dec" in df.columns else np.full(n, np.nan)

        ds = xr.Dataset(
            {
                "residuals": xr.DataArray(
                    residuals,
                    dims=["source_id", "band"],
                    attrs={
                        "units": "sigma",
                        "description": "Normalised SED residuals (obs_flux - model_flux) / obs_flux_err",
                    },
                ),
                "chi2_eazy": xr.DataArray(chi2, dims=["source_id"]),
                "z_a": xr.DataArray(z_a, dims=["source_id"]),
                "template_id": xr.DataArray(template_id, dims=["source_id"]),
                "z_phot": xr.DataArray(z_phot, dims=["source_id"]),
                "ra": xr.DataArray(ra, dims=["source_id"], attrs={"units": "deg"}),
                "dec": xr.DataArray(dec, dims=["source_id"], attrs={"units": "deg"}),
            },
            coords={"source_id": np.arange(n), "band": bands},
            attrs={
                "survey": self.survey,
                "data_level": "a1",
                "creation_timestamp": datetime.utcnow().isoformat(),
                "eazy_version": self._eazy_version,
                "n_sources": n,
                "n_bands": len(bands),
                "residual_model": "eazy_best_fit_template" if use_real else "powerlaw_stub",
            },
        )

        log_data_level_transition("a1-preprocessed", "a1", n, n)
        return ds

    # ── save ─────────────────────────────────────────────────────────────

    def save(self, ds: xr.Dataset, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Save a Dataset to NetCDF using tsdat's `{survey}.{level}.{date}.{time}.nc` convention.

        If `output_path` is a directory (or omitted), the standard filename
        is generated and placed inside it; if it is a full file path, it is
        used as-is.
        """
        now = datetime.utcnow()
        level = ds.attrs.get("data_level", "a1")
        survey = ds.attrs.get("survey", self.survey)
        filename = f"{survey}.{level}.{now:%Y%m%d}.{now:%H%M%S}.nc"

        if output_path is None:
            output_path = Path("data/processed") / filename
        else:
            output_path = Path(output_path)
            if output_path.suffix.lower() != ".nc":
                output_path = output_path / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        logger.info("Saved %s dataset -> %s", level, output_path)
        return output_path
