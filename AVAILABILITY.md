# Data and code availability

This repository accompanies the accepted *Journal of Hydrology* article:

> **Autoencoder-guided flood-event classification and diffusion augmentation for event-scale flood forecasting in a data-scarce basin**

The release contains the core AE, K-means, DDPM, forecast-time MLP and RCNN implementations, with training-side normalization, window construction, component training interfaces, random candidate selection, routed forecasting and evaluation metrics.

The hydrological observations used in the study are subject to confidentiality restrictions and are not included. Neither the corresponding study predictions, trained checkpoints, fitted preprocessing parameters, fitted clustering centers nor archived generated candidate pools are distributed here.

This deposit is an expanded core implementation, not the complete experimental code archive. Experiment scheduling, archived stopping/checkpoint rules, all baseline and selection variants, and paper-specific table/figure production are outside its scope. It is not sufficient by itself to reproduce the published numerical results. The artificial-input tests check interfaces, not hydrological forecasting performance.

The source is provided through this GitHub repository. Project-specific rights and third-party terms are described in `RIGHTS.md` and `THIRD_PARTY_DDPM_LICENSE`.
