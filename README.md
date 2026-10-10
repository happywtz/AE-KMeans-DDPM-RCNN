# AE–K-means–DDPM–RCNN

Core code for the rainfall–runoff forecasting study. This is a small code deposit, not the complete experiment package.

## Files

- `autoencoder.py`: convolutional AE with a five-dimensional latent code.
- `kmeans.py`: NumPy K-means, Euclidean distances and silhouette calculation.
- `unet1d.py`, `_ddpm_blocks.py`, `ddpm.py`: one-dimensional U-Net, diffusion loss and sampling.
- `rcnn.py`: residual convolutional forecasting network.
- `windows.py`, `objectives.py`: tensor interfaces and losses separated for this package.

The model definitions and K-means functions retain the bodies of the project source. Imports and module boundaries have been reorganized. The tensor helpers and tests are new; they are not the original experiment runner. The RCNN output retains its original LeakyReLU, without an added non-negative clamp.

## Inputs

Q denotes discharge and P denotes rainfall. The AE and DDPM take tensors of shape `(batch, 2, 48)`, with channel order Q, P. The AE returns the reconstruction and a latent tensor of shape `(batch, 5, 1)`. Flatten the last two dimensions before K-means. Cluster numbers are arbitrary labels, not severity rankings.

For the 12 h history / 24 h forecast setting, instantiate `RCNN(256, 24)` and call `model(history, future_p, 12, 24)`. History has shape `(batch, 2, 12)`, given future rainfall `(batch, 1, 24)`, and the predicted discharge `(batch, 24)`. The other branches in the original network are retained but are not covered by the interface tests.

Fit normalization and cluster centers using the permitted training events only. Full training-event Q/P sequences can be used for AE fitting and DDPM training. A held-out event's future observed Q is a target for evaluation, not a forecasting input. Full-process AE clustering alone does not implement forecast-time routing; that additional classifier is outside this deposit.

## Check the code

From this folder:

~~~bash
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
~~~

The tests use artificial CPU tensors, including a four-step diffusion sampling check. They do not train a model or measure hydrological skill. `requirements.txt` lists the imports needed by these components; it is not a lockfile for the original experiments.

No observations, predictions, fitted normalization values, cluster centers, model weights, candidate pools or experiment tables are included. Experiment scheduling, candidate selection and result aggregation are also outside this package.

The repository is intended to remain private during review. The four core components are planned for public release upon manuscript acceptance; the hydrological observations remain restricted. See `AVAILABILITY.md` for the proposed statement. No open-source license has yet been chosen for the project-specific code. The DDPM-derived code retains its upstream MIT notice in `THIRD_PARTY_DDPM_LICENSE`.
