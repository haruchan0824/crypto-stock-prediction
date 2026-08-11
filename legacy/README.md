# Legacy and reference material

This directory preserves the original Colab/notebook-oriented experiments and
their environment notes. They are retained for historical interpretation, not
as executable entrypoints for the canonical local pipeline.

The canonical pipeline is:

* `scripts/fetch_eth_ohlcv.py` for local data acquisition
* `scripts/run_pipeline.py` for validation and preparation
* `scripts/train_lgbm.py` for one-fold and representative walk-forward training

Do not import `legacy/scripts/final.py` into the canonical pipeline. The legacy
files may contain Colab, Google Drive, GPU, and broad dependency assumptions.

`legacy/src/` contains the historical TFT, dataset, feature-engineering,
filtering, and optimization modules. `legacy/scripts/make_figures.py` belongs
to the earlier comparison-report workflow.
