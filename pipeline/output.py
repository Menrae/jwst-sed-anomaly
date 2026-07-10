"""
output.py — Stage 4: Catalogue Export & Diagnostics
====================================================
Analogue of tsdat's storage layer: it takes the fully processed **b1**
Dataset (from `pipeline.quality.apply_quality_pipeline`) and produces all
final science deliverables — a flat anomaly catalogue in FITS/CSV/NetCDF,
diagnostic figures, and a publication-ready LaTeX summary table. No
scientific analysis happens here; every quantity exported was already
computed upstream.

`run_full_pipeline` is the single entry point that chains all four stages
(retriever -> standardise -> quality -> output) end-to-end.

Entry point: python -m pipeline.output

Populated by: Claude Code Prompt 4.1
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from astropy.io import fits

import pipeline
from pipeline.quality import apply_quality_pipeline
from pipeline.retriever import MASTRetriever
from pipeline.standardise import SEDStandardiser, log_data_level_transition

logger = logging.getLogger(__name__)

#: The ensemble anomaly score variable name on a b1 Dataset, as appended by
#: `quality.apply_quality_pipeline`. Referred to as "ensemble_score" in the
#: Prompt 4.1 spec; this constant is the single source of truth for it.
ENSEMBLE_SCORE_VAR = "qc_anomaly_score"

#: Per-source scalar variables (as opposed to the per-band `residuals`
#: matrix) carried over into flat catalogue exports (FITS/CSV).
_SOURCE_METADATA_VARS = ["ra", "dec", "z_phot", "z_a", "chi2_eazy", "template_id"]


def _flatten_for_export(ds: xr.Dataset) -> pd.DataFrame:
    """Build a one-row-per-source DataFrame: source metadata + all qc_* flags.

    Deliberately excludes the 2D `residuals` (source x band) matrix — flat
    tabular formats (FITS BinTableHDU, CSV) are for per-source scalar
    quantities; the full-resolution residual matrix is only carried by
    `CatalogueExporter.to_netcdf`.
    """
    data = {"source_id": ds["source_id"].values}
    for var in _SOURCE_METADATA_VARS:
        if var in ds:
            data[var] = ds[var].values
    for var in sorted(v for v in ds.data_vars if v.startswith("qc_")):
        data[var] = ds[var].values
    return pd.DataFrame(data)


def _fits_column_format(series: pd.Series) -> str:
    """Map a pandas Series dtype to a FITS binary table TFORM code."""
    if pd.api.types.is_bool_dtype(series):
        return "L"
    if pd.api.types.is_integer_dtype(series):
        return "K"  # 64-bit integer
    if pd.api.types.is_float_dtype(series):
        return "D"  # 64-bit float
    maxlen = max((len(str(v)) for v in series), default=1)
    return f"{max(maxlen, 1)}A"


class CatalogueExporter:
    """Writes the b1 anomaly catalogue to disk in FITS, CSV, and NetCDF."""

    def to_fits(self, ds: xr.Dataset, output_path: Union[str, Path]) -> Path:
        """Write the flat anomaly catalogue as a FITS BinTableHDU.

        The primary HDU header carries pipeline-level metadata (survey,
        data_level, n_sources, creation_timestamp, pipeline_version); the
        extension HDU carries one row per source with all `qc_*` columns.
        """
        df = _flatten_for_export(ds)
        columns = [
            fits.Column(name=col, array=df[col].to_numpy(), format=_fits_column_format(df[col]))
            for col in df.columns
        ]
        table_hdu = fits.BinTableHDU.from_columns(columns, name="ANOMALY_CATALOGUE")

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["SURVEY"] = (ds.attrs.get("survey", "unknown"), "Source survey")
        primary_hdu.header["DATALVL"] = (ds.attrs.get("data_level", "unknown"), "tsdat-style data level")
        primary_hdu.header["NSOURCES"] = (ds.sizes.get("source_id", len(df)), "Number of sources")
        primary_hdu.header["CREATED"] = (
            ds.attrs.get("creation_timestamp", datetime.utcnow().isoformat()),
            "Dataset creation timestamp (UTC)",
        )
        primary_hdu.header["PIPEVERS"] = (pipeline.__version__, "jwst-sed-anomaly pipeline version")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary_hdu, table_hdu]).writeto(output_path, overwrite=True)
        logger.info("Wrote FITS anomaly catalogue (%d sources) -> %s", len(df), output_path)
        return output_path

    def to_csv(self, ds: xr.Dataset, output_path: Union[str, Path]) -> Path:
        """Write a clean, human-readable CSV: source metadata + all qc_* columns."""
        df = _flatten_for_export(ds)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Wrote CSV anomaly catalogue (%d sources) -> %s", len(df), output_path)
        return output_path

    def to_netcdf(self, ds: xr.Dataset, output_path: Union[str, Path]) -> Path:
        """Save the complete Dataset (including the full residual matrix) to
        NetCDF using tsdat's `{survey}.{level}.{YYYYMMDD}.{HHMMSS}.nc` convention.

        If `output_path` is a directory (or has no `.nc` suffix), the
        standard filename is generated inside it; otherwise it's used as-is.
        """
        now = datetime.utcnow()
        survey = ds.attrs.get("survey", "unknown")
        level = ds.attrs.get("data_level", "b1")
        filename = f"{survey}.{level}.{now:%Y%m%d}.{now:%H%M%S}.nc"

        output_path = Path(output_path)
        if output_path.suffix.lower() != ".nc":
            output_path = output_path / filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(output_path)
        logger.info("Wrote NetCDF %s dataset -> %s", level, output_path)
        return output_path


class DiagnosticPlotter:
    """Produces the diagnostic figures and summary table for the b1 catalogue."""

    def redshift_anomaly_rate(
        self, ds: xr.Dataset, output_dir: Union[str, Path], dz: float = 0.5, top_fraction: float = 0.02
    ) -> Path:
        """Bin sources into redshift shells of width `dz` and plot the anomaly
        rate (fraction scoring at or above the global top-`top_fraction`
        quantile of `ENSEMBLE_SCORE_VAR`) per bin, with Poisson error bars on
        the flagged count.

        Uses a quantile threshold rather than a fixed absolute one (e.g.
        "> 0.8") because `ENSEMBLE_SCORE_VAR` has no fixed scale: when
        UMAP+DBSCAN assigns no sources to its noise cluster, the ensemble
        score never exceeds 0.5, and an absolute threshold silently flags
        nothing. This mirrors the top-`outlier_fraction` convention used
        elsewhere in the pipeline (`sky_distribution`,
        `notebooks/04_interpretation.ipynb`). Sources excluded from scoring
        (`qc_eazy_fit_failure`, NaN score) are dropped before binning and
        before computing the threshold.

        Saves to `<output_dir>/anomaly_rate_vs_redshift.pdf`.
        """
        z = ds["z_phot"].values
        score = ds[ENSEMBLE_SCORE_VAR].values
        scored = ~np.isnan(score)
        z = z[scored]
        score = score[scored]
        threshold = np.quantile(score, 1 - top_fraction)

        z_min = np.floor(np.nanmin(z) / dz) * dz
        z_max = np.ceil(np.nanmax(z) / dz) * dz
        edges = np.arange(z_min, z_max + dz, dz)
        bin_idx = np.digitize(z, edges) - 1

        centers, rates, errs = [], [], []
        for b in range(len(edges) - 1):
            mask = bin_idx == b
            n = int(mask.sum())
            if n == 0:
                continue
            k = int((score[mask] >= threshold).sum())
            centers.append((edges[b] + edges[b + 1]) / 2)
            rates.append(k / n)
            errs.append(np.sqrt(k) / n)  # Poisson error on k, propagated to a rate

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(centers, rates, yerr=errs, fmt="o-", capsize=3, color="steelblue")
        ax.set_xlabel("Photometric redshift ($z_{phot}$)")
        ax.set_ylabel(f"Anomaly rate (top {top_fraction:.0%} of {ENSEMBLE_SCORE_VAR})")
        ax.set_title("Anomaly rate vs. redshift")
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        output_path = Path(output_dir) / "anomaly_rate_vs_redshift.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("Saved redshift anomaly-rate figure -> %s", output_path)
        return output_path

    def sky_distribution(
        self, ds: xr.Dataset, output_dir: Union[str, Path], top_fraction: float = 0.05
    ) -> Path:
        """Plot RA/Dec of all sources, highlighting the top `top_fraction` by
        ensemble score in a contrasting colour.

        Saves to `<output_dir>/sky_distribution.pdf`.
        """
        ra = ds["ra"].values
        dec = ds["dec"].values
        score = ds[ENSEMBLE_SCORE_VAR].values
        # NaN-safe: sources excluded from scoring (e.g. qc_eazy_fit_failure)
        # carry a NaN score and must not skew the quantile or be counted "top".
        threshold = np.nanquantile(score, 1 - top_fraction)
        is_top = score >= threshold

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(ra[~is_top], dec[~is_top], s=8, color="lightgray", label="all sources")
        ax.scatter(
            ra[is_top], dec[is_top], s=45, color="crimson", edgecolor="k", linewidths=0.5,
            label=f"top {top_fraction:.0%} anomalies",
        )
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title("Sky distribution")
        ax.invert_xaxis()  # RA increases to the east; astronomical plots flip the axis
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        output_path = Path(output_dir) / "sky_distribution.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("Saved sky distribution figure -> %s", output_path)
        return output_path

    def score_summary_table(
        self, ds: xr.Dataset, output_dir: Union[str, Path], n_top: int = 20
    ) -> Path:
        """Produce a LaTeX-formatted table of the top `n_top` *unexplained*
        outliers (qc_agn_match=False and qc_emission_line_flag=False), ranked
        by ensemble score.

        Saves to `<output_dir>/top_outliers.tex`.
        """
        df = _flatten_for_export(ds)
        # qc_* flags may arrive as int (0/1) rather than bool (e.g. NetCDF round-trips,
        # or the int-typed placeholders in tests/conftest.py's synthetic_residuals) —
        # cast explicitly so `~` is a logical, not bitwise, negation.
        agn_match = df["qc_agn_match"].astype(bool) if "qc_agn_match" in df else pd.Series(False, index=df.index)
        emission_flag = (
            df["qc_emission_line_flag"].astype(bool) if "qc_emission_line_flag" in df else pd.Series(False, index=df.index)
        )
        unexplained_mask = ~agn_match & ~emission_flag
        top = (
            df[unexplained_mask]
            .sort_values(ENSEMBLE_SCORE_VAR, ascending=False)
            .head(n_top)
            .reset_index(drop=True)
        )

        rename = {
            "source_id": "ID",
            "ra": "RA (deg)",
            "dec": "Dec (deg)",
            "z_phot": r"$z_{\mathrm{phot}}$",
            ENSEMBLE_SCORE_VAR: "Ensemble score",
            "qc_iso_forest_score": "IF score",
            "chi2_eazy": r"$\chi^2$",
        }
        display_cols = [c for c in rename if c in top.columns]
        table = top[display_cols].rename(columns=rename)
        for col, ndigits in (("RA (deg)", 5), ("Dec (deg)", 5), (r"$z_{\mathrm{phot}}$", 3),
                             ("Ensemble score", 3), ("IF score", 3), (r"$\chi^2$", 2)):
            if col in table.columns:
                table[col] = table[col].round(ndigits)

        latex = table.to_latex(
            index=False,
            escape=False,  # rename map already contains raw LaTeX math (\chi^2 etc.)
            caption=f"Top {len(table)} unexplained SED anomalies (no AGN or emission-line match).",
            label="tab:top_outliers",
            position="ht",
        )
        # 7 numeric columns don't fit a single column in a twocolumn AASTeX
        # document (e.g. results/paper/main.tex) -- span both columns.
        # table* is a no-op in single-column documents, so this is safe
        # everywhere this table gets \input'd.
        latex = latex.replace(r"\begin{table}", r"\begin{table*}").replace(
            r"\end{table}", r"\end{table*}"
        )

        output_path = Path(output_dir) / "top_outliers.tex"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex)
        logger.info("Saved top-outliers LaTeX table (%d rows) -> %s", len(table), output_path)
        return output_path


def run_full_pipeline(config_path: Union[str, Path], survey: Optional[str] = None) -> dict:
    """Orchestrate all four pipeline stages end-to-end: retriever -> standardise
    -> quality -> output.

    `survey` defaults to the first survey configured under `retriever.surveys`
    in `pipeline_config.yaml` (i.e. this call processes one survey per run;
    call again with a different `survey` to process another).

    Returns a dict with keys: n_sources_ingested, n_sources_processed,
    n_flagged, top_outlier_ids, output_paths.
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    surveys = config.get("retriever", {}).get("surveys") or []
    survey = survey or (surveys[0]["name"] if surveys else "ceers")
    survey_cfg = next((s for s in surveys if s.get("name") == survey), {})

    storage_cfg = config.get("storage", {}) or {}
    output_dir = Path(storage_cfg.get("output_dir", "data/flagged"))
    figures_dir = Path(storage_cfg.get("figures_dir", "results/figures"))
    tables_dir = Path(storage_cfg.get("tables_dir", "results/tables"))

    logger.info("=== run_full_pipeline: survey=%s ===", survey)

    # ── Stage 1: Retriever ──────────────────────────────────────────────
    retriever = MASTRetriever()
    raw_output_path = Path(survey_cfg.get("output_file", f"data/raw/{survey}_dr1.fits"))
    if survey == "ceers":
        raw_path = retriever.fetch_ceers(raw_output_path)
    elif survey == "jades":
        raw_path = retriever.fetch_jades(raw_output_path)
    else:
        raise ValueError(
            f"run_full_pipeline: no MASTRetriever fetch method configured for survey {survey!r} "
            "(expected 'ceers' or 'jades')"
        )
    raw_df = retriever.load_catalogue(raw_path, survey=survey)
    schema_report = retriever.validate_schema(raw_df)
    n_sources_ingested = schema_report["n_sources"]
    # No filtering happens at raw ingestion, so n_in == n_out here by convention.
    log_data_level_transition("mast", "raw", n_sources_ingested, n_sources_ingested)

    # ── Stage 2: Standardise ────────────────────────────────────────────
    standardiser = SEDStandardiser(config_path, survey=survey)
    pre_df = standardiser.preprocess(raw_df)
    n_sources_processed = len(pre_df)

    fit_df = standardiser.run_eazy_fit(pre_df, Path("data/processed") / f"{survey}_eazy")
    a1_ds = standardiser.extract_residuals(pre_df, fit_df)
    a1_path = standardiser.save(a1_ds, Path("data/processed"))

    # ── Stage 3: Quality ─────────────────────────────────────────────────
    quality_config_path = config_path.parent / "quality_config.yaml"
    b1_ds = apply_quality_pipeline(a1_ds, quality_config_path)

    ensemble_score = b1_ds[ENSEMBLE_SCORE_VAR].values
    outlier_fraction = config.get("quality", {}).get("outlier_fraction", 0.02)
    scored = ~np.isnan(ensemble_score)
    score_threshold = np.quantile(ensemble_score[scored], 1 - outlier_fraction)
    n_flagged = int((ensemble_score >= score_threshold).sum())
    top_n = min(20, b1_ds.sizes["source_id"])
    # NaN-safe descending order: sources excluded from scoring (e.g.
    # qc_eazy_fit_failure) carry a NaN score and must sort last, not first
    # -- plain np.argsort(...)[::-1] would put every NaN at the very front.
    score_for_ranking = np.where(np.isnan(ensemble_score), -np.inf, ensemble_score)
    top_order = np.argsort(score_for_ranking)[::-1][:top_n]
    top_outlier_ids = b1_ds["source_id"].values[top_order].tolist()

    # ── Stage 4: Output ──────────────────────────────────────────────────
    exporter = CatalogueExporter()
    plotter = DiagnosticPlotter()

    now = datetime.utcnow()
    catalogue_stem = f"{survey}.b1.anomalies.{now:%Y%m%d}"
    fits_path = exporter.to_fits(b1_ds, output_dir / f"{catalogue_stem}.fits")
    csv_path = exporter.to_csv(b1_ds, output_dir / f"{catalogue_stem}.csv")
    b1_netcdf_path = exporter.to_netcdf(b1_ds, Path("data/processed"))

    rate_plot_path = plotter.redshift_anomaly_rate(b1_ds, figures_dir)
    sky_plot_path = plotter.sky_distribution(b1_ds, figures_dir)
    table_path = plotter.score_summary_table(b1_ds, tables_dir)

    log_data_level_transition("b1", "output", b1_ds.sizes["source_id"], b1_ds.sizes["source_id"])

    return {
        "n_sources_ingested": n_sources_ingested,
        "n_sources_processed": n_sources_processed,
        "n_flagged": n_flagged,
        "top_outlier_ids": top_outlier_ids,
        "output_paths": {
            "a1_netcdf": str(a1_path),
            "b1_fits": str(fits_path),
            "b1_csv": str(csv_path),
            "b1_netcdf": str(b1_netcdf_path),
            "figure_redshift_anomaly_rate": str(rate_plot_path),
            "figure_sky_distribution": str(sky_plot_path),
            "table_top_outliers": str(table_path),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    default_config_path = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
    summary = run_full_pipeline(default_config_path)
    print(json.dumps(summary, indent=2, default=str))
