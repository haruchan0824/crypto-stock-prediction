# Repository cleanup and reproduction workflow

## Current objective

The repository is being converted from a Colab/notebook-oriented research repository into a reproducible portfolio-quality time-series ML repository.

The current canonical local pipeline is the new local pipeline under `scripts/` and `src/`.

`scripts/final.py` is legacy/reference code. Do not treat it as the executable canonical pipeline and do not import it.

## General rules

* Work only on the current cleanup branch unless explicitly instructed otherwise.
* Keep changes small, reviewable, and reversible.
* Preserve the meaning of existing experiments unless a correctness bug requires a change.
* Prefer reproducibility, leakage prevention, and clarity over preserving notebook execution structure.
* Do not perform unrelated refactors.
* Do not delete files merely because they appear unused; verify their role first.
* Never commit secrets, API keys, raw data, generated reports, virtual environments, or large artifacts.
* Do not modify raw source data to improve apparent quality.
* Do not silently repair source OHLC anomalies.
* Use Git as the source of truth for code.
* Local raw data must remain Git-ignored.

## Canonical data

Representative dataset:

* asset: ETH
* quote: USD
* timeframe: 1h
* source: CryptoCompare / CCCAGG
* start: 2021-04-01T00:00:00Z
* end: 2024-12-31T23:00:00Z
* expected rows: 32904

Known source-data issue:

* 91 OHLC consistency anomalies exist in the source data.
* Preserve original values.
* Treat them as warnings and record them in metadata/profile.
* Do not drop or correct these rows automatically.

## Leakage rules

Never:

* use centered rolling windows
* use future data in features
* fit preprocessing on validation or test data
* use test data for early stopping
* use test data to select features
* use test data to fit label thresholds
* use test metrics to tune hyperparameters
* use `bfill` for time-series feature preparation
* compute global quantiles for fold-specific transformations

For forward labels, use exact timestamps rather than positional row shifts when gaps may exist.

For a 6-hour forward label, purge rows at split boundaries whose label timestamp crosses the boundary.

## Pipeline stages

The intended progression is:

1. validate
2. prepare
3. train-one-fold
4. representative-walk-forward
5. repository-cleanup
6. README-alignment
7. final-audit

Complete and verify each stage before continuing.

## Stage completion

After each meaningful stage:

1. run relevant syntax/tests
2. run the smallest representative execution
3. inspect `git diff --check`
4. inspect `git status`
5. ensure generated data/artifacts remain ignored
6. commit only if all checks succeed

Use descriptive commits.

Suggested examples:

* `feat: add reproducible LightGBM baseline`
* `feat: add representative walk-forward evaluation`
* `chore: remove obsolete notebook artifacts`
* `refactor: align repository with canonical pipeline`
* `docs: update README for reproducible pipeline`

## Stop conditions

STOP and report without committing if any of the following occurs:

### Environment

* no usable Python runtime
* required package cannot be installed or imported
* dependency resolution requires broad or risky version changes

### Data

* expected input data is missing
* input schema materially differs from the documented schema
* timestamp range or row count is materially inconsistent
* data corruption is suspected
* resolving a data issue would require modifying raw source values

### ML correctness

* leakage is discovered and fixing it materially changes experiment semantics
* label definition is ambiguous
* train/validation/test boundaries cannot be reconstructed confidently
* representative experiment choice is ambiguous
* test data would need to influence training decisions

### Repository semantics

* two implementations both appear potentially canonical
* deletion would remove potentially valuable experiment history
* a refactor would materially change behavior
* a file appears to contain unpublished or sensitive material
* a required design choice cannot be inferred from code or documentation

### Git

* unexpected uncommitted changes exist before a stage
* conflicts occur
* current branch is not the expected cleanup branch
* history rewriting would be required

### External services

* an API requires a new paid plan
* new credentials or secret handling decisions are required
* external service behavior prevents reproducible execution

## Do not stop for

Do not stop merely because:

* a small syntax or import bug is found
* a local path needs to become configurable
* a minor package is missing and installation is straightforward
* a test requires a small deterministic fixture
* logging/metadata needs a small correction
* a generated output directory already exists and a new run ID can be used
* a low-risk compatibility fix is required

Fix such issues, verify them, and continue.

## Legacy code

`scripts/final.py` and notebook-derived experimental code are reference material.

Do not:

* import them into the canonical pipeline
* wrap the entire legacy file in a new `main`
* spend time making every historical experiment executable

Use them only to verify historical experiment semantics.

## Repository cleanup order

After the representative experiment is reproduced:

1. identify canonical files
2. identify legacy/reference files
3. remove clearly disposable artifacts
4. commit
5. reorganize directories
6. commit
7. repair imports and paths
8. execute the representative pipeline
9. commit
10. update README from actual implementation
11. perform final read-only audit

Do not combine deletion, structural movement, path repair, and README rewriting into one large commit.

## Final success criteria

The repository is ready for final review when:

* one documented command obtains or prepares the required local data
* one documented command runs the representative experiment
* representative walk-forward results are reproducible
* no obvious time-series leakage remains
* configuration is externalized
* raw data and secrets are Git-ignored
* dependencies are documented
* repository structure reflects the canonical pipeline
* README matches actual code
* working tree is clean
* final audit finds no blocking issue
