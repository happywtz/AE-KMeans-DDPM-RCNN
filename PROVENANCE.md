# Implementation provenance

## Retained from the review deposit

`autoencoder.py`, `rcnn.py`, `unet1d.py`, `_ddpm_blocks.py`, `ddpm.py`, `kmeans.py`, `objectives.py`, `windows.py` and `tests/test_core.py` are retained from the existing repository. Their original tensor conventions and losses are preserved. In particular, the RCNN retains its LeakyReLU output; no nonnegative output clamp is added.

## MLP extraction

`routing.py` follows the archived `train_forecast_time_classifiers.py` and `train_forecast_time_fixed_mlp.py` implementation: past 12 Q + past 12 P + given future 24 P, a training-fitted `StandardScaler`, one 16-unit ReLU hidden layer, L-BFGS, alpha 0.01, at most 2,000 iterations, and training-only oversampling with the archived seed offset. It excludes classifier comparisons, study observations, fitted weights and local experiment paths.

## New integration interfaces

`preprocessing.py` expresses the training-only per-channel min–max operations as a reusable scaler. `training.py`, `inference.py`, `selection.py`, `evaluation.py`, examples, packaging metadata and the extended tests are newly separated interfaces for this release. They are not verbatim archived runners, and no claim of bitwise experiment reproduction is made. In particular, random candidate IDs are relative to the caller's supplied pool, not the paper's archived pool identities.

This release does not change the completed study results and does not rerun training on the study data.
