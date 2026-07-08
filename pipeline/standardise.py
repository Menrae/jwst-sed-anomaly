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
                        source (via the real EAZY CLI if installed, else a
                        clearly-logged synthetic stub for pipeline
                        development).
3. `extract_residuals`— computes per-band SED residuals against a smooth
                        continuum model and packages everything into an
                        xarray Dataset with tsdat-style global attributes.
4. `save`             — writes that Dataset to NetCDF using tsdat's
                        `{survey}.{level}.{date}.{time}.nc` naming convention.

Populated by: Claude Code Prompt 2.2
"""

from __future__ import annotations

import logging
import re
import subprocess
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
#: keyed by the unit suffix a column name may carry (e.g. "f090w_flux_njy").
_UNIT_CONVERSION_TO_UJY = {
    "njy": 1e-3,
    "ujy": 1.0,
    "mjy": 1e3,
    "jy": 1e6,
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

        Columns already named `<band>_flux(_err)` (no unit suffix) are
        assumed to already be in the target unit, matching the convention
        used by `MASTRetriever.load_catalogue`. Columns carrying an explicit
        unit suffix (e.g. `<band>_flux_njy`) are converted and renamed to
        drop the suffix.
        """
        df = df.copy()
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

        Writes EAZY's `zphot.cat` and `zphot.param` input files, then runs
        the `eazy` CLI as a subprocess. If EAZY is not installed (or the
        run fails), falls back to a synthetic stub — clearly logged — so
        the rest of the pipeline remains runnable during development.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = df.reset_index(drop=True)
        ids = np.arange(1, len(df) + 1)
        cat_path = self._write_eazy_catalog(df, ids, output_dir)
        param_path = self._write_eazy_param(output_dir, cat_path)

        try:
            subprocess.run(
                ["eazy", "-p", str(param_path)],
                cwd=output_dir,
                check=True,
                capture_output=True,
                timeout=600,
            )
            self._try_read_pz_file(output_dir)
            fit_df = self._read_eazy_outputs(output_dir)
            self._eazy_version = self._detect_eazy_version()
            logger.info("EAZY fit completed for %d sources -> %s", len(df), output_dir)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "EAZY executable not available or the fit failed (%s: %s). Falling back to a "
                "SYNTHETIC STUB fit (lognormal chi2, jittered z_a, random template_id) so the "
                "pipeline remains runnable during development — these values are NOT suitable "
                "for science and must not be used to draw astrophysical conclusions.",
                type(exc).__name__,
                exc,
            )
            fit_df = self._stub_eazy_fit(df, ids)
            self._eazy_version = "synthetic-stub"

        merged = df.copy()
        merged["id"] = ids
        merged = merged.merge(fit_df[["id", "z_a", "chi2", "template_id"]], on="id", how="left")
        return merged

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

    def _write_eazy_param(self, output_dir: Path, cat_path: Path) -> Path:
        """Write EAZY's `zphot.param` configuration file from `self.eazy_config`."""
        param_path = output_dir / "zphot.param"
        params = {
            "CATALOG_FILE": str(cat_path),
            "MAIN_OUTPUT_FILE": "photz",
            "OUTPUT_DIRECTORY": str(output_dir),
            "TEMPLATES_FILE": self.eazy_config.get("template_set", "tweak_fsps_QSF_12_v3"),
            "Z_MIN": self.eazy_config.get("z_min", 0.01),
            "Z_MAX": self.eazy_config.get("z_max", 12.0),
            "Z_STEP": self.eazy_config.get("z_step", 0.01),
        }
        param_path.write_text("\n".join(f"{k}  {v}" for k, v in params.items()) + "\n")
        return param_path

    def _read_eazy_outputs(self, output_dir: Path) -> pd.DataFrame:
        """Best-effort parser for EAZY's ASCII `.zout` best-fit output file.

        EAZY's exact `.zout` column layout has varied across versions/forks
        (the original `eazy` C binary vs. `eazy-py`); this reads whatever
        header EAZY wrote and falls back to the standard `photz` column
        names it has historically used.
        """
        zout_path = output_dir / "photz.zout"
        if not zout_path.exists():
            raise FileNotFoundError(f"Expected EAZY output not found: {zout_path}")

        header_line = None
        with open(zout_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    header_line = line.lstrip("#").split()
                else:
                    break

        columns = header_line if header_line else ["id", "z_spec", "z_a", "chi2a"]
        zout = pd.read_csv(zout_path, comment="#", sep=r"\s+", header=None, names=columns)

        if "chi2" not in zout.columns and "chi2a" in zout.columns:
            zout["chi2"] = zout["chi2a"]
        if "template_id" not in zout.columns:
            # Standard EAZY ASCII .zout does not report a per-source best
            # template index; a full implementation would read it from the
            # binary .tempfilt/.pz products via eazy-py.
            zout["template_id"] = -1

        return zout[["id", "z_a", "chi2", "template_id"]]

    def _try_read_pz_file(self, output_dir: Path) -> None:
        """Log whether EAZY's `.pz` p(z) grid file was produced.

        Full binary parsing of the p(z) grid is out of scope for this
        scaffold (use `eazy-py`'s readers for that); this only confirms
        the file exists so the fit is auditable.
        """
        pz_path = output_dir / "photz.pz"
        if pz_path.exists():
            logger.info(
                "EAZY p(z) grid file found at %s (not parsed further in this scaffold).", pz_path
            )
        else:
            logger.debug("No EAZY .pz file found at %s", pz_path)

    @staticmethod
    def _detect_eazy_version() -> str:
        try:
            result = subprocess.run(
                ["eazy", "--version"], capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() or result.stderr.strip() or "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

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

        Each source's "model" flux is a smooth log-log power-law fit through
        its own observed photometry — a stand-in for the true EAZY best-fit
        template SED, which a full implementation would read directly from
        EAZY's output products. Residuals are `(obs_flux - model_flux) /
        obs_flux_err`.
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

        residuals = np.full((n, len(bands)), np.nan)
        for i in range(n):
            # Per-source loop: each source can have a different subset of
            # valid bands, so the continuum fit is genuinely per-row rather
            # than a single vectorisable operation across the catalogue.
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
