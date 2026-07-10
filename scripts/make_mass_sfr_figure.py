"""
make_mass_sfr_figure.py — Standalone regeneration of the stellar-mass vs.
SFR anomaly figure (Section 3 of notebooks/04_interpretation.ipynb).

Unlike the notebook's Section 3 (a hexbin colored by median ensemble score),
this script produces the paper's Figure 2: a plain-density hexbin of the
full clean sample with the top-2% anomalous sources overplotted as red
points, plus a running-median main-sequence trace and a Mann-Whitney U test
of whether anomalies sit preferentially above it.

`lp_mass_best`/`lp_sfr_best` (CEERS DR1.0 LePHARE fits) are not carried
through to the b1 NetCDF Dataset -- they only exist on the preprocessed
DataFrame (`SEDStandardiser.preprocess`'s output). `preprocess` is a pure,
deterministic function of the raw catalogue and pipeline_config.yaml, and
`extract_residuals` assigns b1's `source_id` coordinate as the positional
row order of that same DataFrame, so re-running `preprocess` on the raw
catalogue reproduces a frame that aligns 1:1 by position with the cached b1
Dataset (verified here by comparing ra/dec, which *are* carried through).

Run: python -m scripts.make_mass_sfr_figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats

from pipeline.output import ENSEMBLE_SCORE_VAR
from pipeline.retriever import MASTRetriever
from pipeline.standardise import SEDStandardiser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_CATALOGUE = ROOT / "data" / "raw" / "ceers_dr1.fits"
CONFIG_PATH = ROOT / "config" / "pipeline_config.yaml"
OUTPUT_PATH = ROOT / "results" / "figures" / "mass_sfr_anomalies.pdf"
TOP_FRACTION = 0.02  # same top-2% quantile convention as DiagnosticPlotter.redshift_anomaly_rate


def _latest_b1_netcdf() -> Path:
    candidates = sorted((ROOT / "data" / "processed").glob("ceers.b1.*.nc"))
    if not candidates:
        raise FileNotFoundError(
            "No cached ceers.b1.*.nc found in data/processed/ -- run "
            "pipeline.output.run_full_pipeline first."
        )
    return candidates[-1]


def main() -> Path:
    b1_path = _latest_b1_netcdf()
    b1 = xr.open_dataset(b1_path)
    logger.info("Loaded b1 dataset: %s (%d sources)", b1_path.name, b1.sizes["source_id"])

    retriever = MASTRetriever()
    raw_df = retriever.load_catalogue(RAW_CATALOGUE, survey="ceers")
    standardiser = SEDStandardiser(CONFIG_PATH, survey="ceers")
    pre_df = standardiser.preprocess(raw_df).reset_index(drop=True)

    if pre_df["ra"].to_numpy().shape != b1["ra"].values.shape or not np.allclose(
        pre_df["ra"].to_numpy(), b1["ra"].values, equal_nan=True
    ):
        raise RuntimeError(
            "pre_df/b1 row alignment check failed (ra mismatch) -- "
            "config/pipeline_config.yaml may have changed since the cached b1 was built."
        )

    eazy_fail = b1["qc_eazy_fit_failure"].values.astype(bool)
    clean = ~eazy_fail
    n_clean = int(clean.sum())
    logger.info("Clean (converged-EAZY-fit) sample: %d / %d", n_clean, b1.sizes["source_id"])

    score = b1[ENSEMBLE_SCORE_VAR].values[clean]
    threshold = np.quantile(score, 1 - TOP_FRACTION)
    is_anomaly = score >= threshold
    n_anomaly = int(is_anomaly.sum())
    logger.info("Top %.0f%% anomaly threshold: %.4f (%d / %d flagged)", TOP_FRACTION * 100, threshold, n_anomaly, n_clean)

    log_mass = pre_df["lp_mass_best"].to_numpy()[clean]
    log_sfr = pre_df["lp_sfr_best"].to_numpy()[clean]
    valid = (
        np.isfinite(log_mass) & np.isfinite(log_sfr)
        & (log_mass > 5) & (log_mass < 12)   # exclude -999 sentinel / catastrophic fits
        & (log_sfr > -6) & (log_sfr < 6)
    )
    n_valid = int(valid.sum())
    n_anomaly_valid = int((is_anomaly & valid).sum())
    logger.info(
        "%d / %d clean-sample sources have physically valid LePHARE mass/SFR fits "
        "(%d of the %d anomalous sources among them).",
        n_valid, n_clean, n_anomaly_valid, n_anomaly,
    )

    mass_v, sfr_v, score_v, anomaly_v = log_mass[valid], log_sfr[valid], score[valid], is_anomaly[valid]

    # Running-median main-sequence trace (0.5 dex mass bins, >=10 sources/bin).
    mbin_edges = np.arange(np.floor(mass_v.min() * 2) / 2, np.ceil(mass_v.max() * 2) / 2 + 0.5, 0.5)
    ms_x, ms_y = [], []
    for i in range(len(mbin_edges) - 1):
        m = (mass_v >= mbin_edges[i]) & (mass_v < mbin_edges[i + 1])
        if m.sum() >= 10:
            ms_x.append((mbin_edges[i] + mbin_edges[i + 1]) / 2)
            ms_y.append(np.median(sfr_v[m]))

    ms_interp = np.interp(mass_v, ms_x, ms_y)
    above_ms = (sfr_v - ms_interp) > 0.3
    below_ms = ~above_ms
    n1, n2 = int(above_ms.sum()), int(below_ms.sum())
    mw_stat, mw_p = stats.mannwhitneyu(score_v[above_ms], score_v[below_ms], alternative="two-sided")
    if mw_p == 0.0:
        # At this sample size (n1, n2 in the tens of thousands) the true
        # two-sided p-value underflows double precision; recover its order
        # of magnitude from the same normal approximation scipy uses
        # internally, via the log-survival function instead of exp/log(p).
        mu = n1 * n2 / 2
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = abs(mw_stat - mu) / sigma
        log10_p = (stats.norm.logsf(z) + np.log(2)) / np.log(10)
        mw_p_report = f"< 1e{int(np.ceil(log10_p))}"
    else:
        mw_p_report = f"= {mw_p:.3g}"
    logger.info(
        "Mann-Whitney U: n(above MS)=%d, n(at/below MS)=%d, p %s",
        n1, n2, mw_p_report,
    )

    # Clip axes to the 0.5-99.5 percentile range so a handful of extreme
    # LePHARE fits don't compress the bulk of the (dense, low-mass) sample
    # into a corner of the plot.
    xlim = np.percentile(mass_v, [0.5, 99.5])
    ylim = np.percentile(sfr_v, [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(mass_v, sfr_v, gridsize=50, cmap="Blues", bins="log", mincnt=1, extent=[*xlim, *ylim], zorder=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}(N)$ per hexbin")
    ax.plot(ms_x, ms_y, color="white", lw=3, ls="--", zorder=2)
    ax.plot(ms_x, ms_y, color="k", lw=1.3, ls="--", label="Running median (main sequence)", zorder=3)
    ax.scatter(
        mass_v[anomaly_v], sfr_v[anomaly_v], s=10, facecolors="none", edgecolors="crimson",
        linewidths=0.6, alpha=0.8, label=f"Top {TOP_FRACTION:.0%} anomalous ($N={n_anomaly_valid}$)", zorder=4,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(\mathrm{SFR} / M_\odot\,\mathrm{yr}^{-1})$")
    ax.set_title("Anomalous sources across the stellar-mass–SFR plane")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    logger.info("Saved -> %s", OUTPUT_PATH)
    logger.info("Mann-Whitney U p-value for caption: p %s", mw_p_report)
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
