"""
make_umap_figure.py — Standalone regeneration of the UMAP embedding figure
(paper Figure 3, Section 3.3).

Loads the real, cached b1 NetCDF, restricts to the clean (converged-EAZY-fit)
subsample, and re-runs `UMAPDBSCANChecker` with the exact configuration used
by `pipeline.quality.apply_quality_pipeline` (config/pipeline_config.yaml's
`quality.umap`/`quality.dbscan` blocks) to recover the 2D embedding
coordinates it stashes on `ds.attrs`. Points are colored by `qc_anomaly_score`
(already present on the b1 dataset) with the top-2% ensemble-score anomalies
overplotted, using the same quantile convention as
`DiagnosticPlotter.redshift_anomaly_rate` and `scripts/make_mass_sfr_figure.py`.

Run: python -m scripts.make_umap_figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from pipeline.output import ENSEMBLE_SCORE_VAR
from pipeline.quality import UMAPDBSCANChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dark, high-contrast plot style matching notebooks/04_interpretation.ipynb
# (same crimson highlight colour), bumped to print-quality DPI.
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "errorbar.capsize": 4,
})

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "results" / "figures" / "umap_embedding.pdf"
TOP_FRACTION = 0.02

# Mirrors config/pipeline_config.yaml's quality.umap / quality.dbscan blocks.
UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
DBSCAN_KWARGS = dict(eps=0.5, min_samples=5)


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

    eazy_fail = b1["qc_eazy_fit_failure"].values.astype(bool)
    clean_ds = b1.isel(source_id=~eazy_fail).load()
    n_clean = clean_ds.sizes["source_id"]
    logger.info("Clean (converged-EAZY-fit) sample: %d / %d", n_clean, b1.sizes["source_id"])

    checker = UMAPDBSCANChecker(**UMAP_KWARGS, **DBSCAN_KWARGS)
    dbscan_noise = checker.score(clean_ds)
    n_noise = int(dbscan_noise.sum())
    logger.info(
        "Re-ran UMAP+DBSCAN on the clean sample: %d / %d points assigned to the DBSCAN noise "
        "cluster.", n_noise, n_clean,
    )

    emb_x = np.array(clean_ds.attrs["umap_embedding_x"])
    emb_y = np.array(clean_ds.attrs["umap_embedding_y"])

    score = clean_ds[ENSEMBLE_SCORE_VAR].values
    threshold = np.quantile(score, 1 - TOP_FRACTION)
    is_anomaly = score >= threshold
    n_anomaly = int(is_anomaly.sum())
    logger.info("Top %.0f%% anomaly threshold: %.4f (%d / %d flagged)", TOP_FRACTION * 100, threshold, n_anomaly, n_clean)

    fig, ax = plt.subplots(figsize=(7, 6))
    sca = ax.scatter(
        emb_x, emb_y, c=score, s=6, cmap="viridis", alpha=0.6, linewidths=0, zorder=1,
    )
    cb = fig.colorbar(sca, ax=ax)
    cb.set_label(f"{ENSEMBLE_SCORE_VAR}")
    ax.scatter(
        emb_x[is_anomaly], emb_y[is_anomaly], s=22, facecolors="none", edgecolors="#DC143C",
        linewidths=0.8, label=f"Top {TOP_FRACTION:.0%} anomalous ($N={n_anomaly}$)", zorder=2,
    )
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_title("UMAP embedding of SED residuals")
    ax.legend(loc="best")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    logger.info("Saved -> %s", OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
