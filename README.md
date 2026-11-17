# AE–K-means–DDPM–MLP–RCNN

## Associated article

This repository contains the core algorithm implementations for:

> **Autoencoder-guided flood-event classification and diffusion augmentation for event-scale flood forecasting in a data-scarce basin**

The article has been accepted by *Journal of Hydrology*. This expanded deposit adds forecast-time MLP routing, training-side preprocessing, random candidate selection, component training interfaces, forecast dispatch and absolute evaluation metrics to the original review deposit.

## Implementation

Historical complete rainfall–discharge events are encoded by a convolutional autoencoder and grouped using K-means. Class-specific DDPMs generate paired trajectories for RCNN training. At forecast time, an MLP uses the preceding 12 h of discharge and rainfall and the given next 24 h of rainfall to select an RCNN expert. Future observed discharge is not an input to routing or forecasting.

| Module | Contents |
| --- | --- |
| `autoencoder.py` | Archived convolutional AE and five-dimensional latent code |
| `kmeans.py` | K-means++, repeated fitting, Euclidean distances and silhouettes |
| `unet1d.py`, `_ddpm_blocks.py`, `ddpm.py` | Archived 1D U-Net and diffusion loss/sampling |
| `rcnn.py` | Archived dual-branch residual convolutional forecaster |
| `routing.py` | Standardized 48-feature MLP, training-only oversampling and class assignment |
| `preprocessing.py` | Training-fitted Q/P min–max scaling and inverse scaling |
| `selection.py` | Random selection with recorded zero-based candidate IDs |
| `windows.py`, `objectives.py` | Thirteen 12-to-24 h windows per event and component losses |
| `training.py` | Explicit optimizer-update interfaces for AE, DDPM and RCNN |
| `inference.py` | MLP-to-expert dispatch with physical-unit discharge output |
| `evaluation.py` | Window NSE, RMSE, MAE and Pearson R |

All Python modules above are in `src/flood_core`. Model definitions retain the original review-deposit implementations. See `PROVENANCE.md` for the distinction between archived code and the new component interfaces.

## Installation and checks

Python 3.10 or later is required. Install a PyTorch build suitable for your system, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -B -m unittest discover -s tests -v
python -B examples/check_interfaces.py
```

The tests and interface example use artificial tensors only. They neither train the study models nor reproduce the article's scores.

## Data conventions

Q denotes discharge and P denotes rainfall. User-supplied event arrays have shape `(events, 48, 2)`, ordered Q then P. AE and DDPM tensors use `(batch, 2, 48)`; AE latent tensors have shape `(batch, 5, 1)`. Flatten the latter two axes before K-means. Cluster labels have no intrinsic order and must not be assumed to match across folds.

`training_windows` creates 13 overlapping windows from each 48-observation event. Its outputs are history `(batch, 2, 12)`, future rainfall `(batch, 1, 24)` and target discharge `(batch, 24)`. The first forecast target begins immediately after the twelve history observations.

Fit each scaler, AE, cluster solution, router and expert on the training events permitted by the evaluation protocol. Fit the MLP on physical-unit training windows; it has its own training-fitted `StandardScaler`. Preserve the Q/P scaler associated with each trained RCNN. Held-out future Q must not be used for preprocessing, selection, routing or expert fitting. The caller remains responsible for constructing independent event splits.

## Component usage

```python
import numpy as np
import torch
from flood_core import Autoencoder, RCNN
from flood_core.preprocessing import QPScaler
from flood_core.routing import fit_router
from flood_core.training import train_autoencoder, train_rcnn

# Supply your own permitted training events; no study observations are bundled.
events = np.load("training_events.npy", allow_pickle=False)
scaler = QPScaler.fit(events)
normalized = scaler.transform(events)
ae = Autoencoder()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)
# Explicitly choose the update budget for your own run.
# train_autoencoder(ae, optimizer, normalized.transpose(0, 2, 1), updates=...)

# After training-side clustering, labels refer to those same training events.
# router = fit_router(events, event_labels, seed=42)
# For each class, fit its DDPM and RCNN using the permitted class data.
# rcnn = RCNN(256, 24)
# train_rcnn(rcnn, optimizer_for_rcnn, class_training_events, updates=...)
```

The commented calls require explicit training budgets and class data, not bundled weights or hidden downloads. `examples/train_forecaster.py` provides a runnable RCNN entry point on a user-supplied training array. The generic training interfaces do not implement the archived checkpoint-selection, EMA, accumulation, multi-seed or fold-scheduling procedures. They must not be used to claim an exact rerun of the published experiments.

Given already fitted components:

```python
from flood_core.inference import ForecastExpert, forecast
from flood_core.evaluation import mean_window_metrics

# experts = {label: ForecastExpert(trained_rcnn, its_training_scaler), ...}
# prediction_q, class_ids = forecast(router, experts, history_qp, future_p)
# scores = mean_window_metrics(observed_future_q, prediction_q)
```

`forecast` accepts only historical Q/P and given future P. It does not receive future Q. Its predictions are not clipped to nonnegative discharge. Score physical-scale predictions without post hoc window removal. Mean-window NSE is the arithmetic mean of the individual window NSEs, not NSE on concatenated windows. Undefined NSE or R is returned as NaN rather than silently discarded.

## Release scope and availability

This is an expanded core implementation, not the complete archived experimental reproduction package. It includes executable algorithm components and their integration interfaces. It does not include confidential hydrological observations, observed or predicted study hydrographs, trained checkpoints, fitted normalizers or centers, archived generated candidate pools, complete experiment orchestration, all comparison/selection variants, or the scripts that assemble the article's figures and tables. The exact article results cannot be regenerated from this repository alone. See `AVAILABILITY.md`.

The project-specific code is not offered under a project-wide open-source license. The upstream-derived diffusion code retains its MIT terms; see `THIRD_PARTY_DDPM_LICENSE`. A public repository makes source visible but does not make confidential study data available. See `RIGHTS.md`.
