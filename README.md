# Back to Basics: Embracing Heteroscedasticity for Robust Probabilistic Time
Series Forecasting

LSG-VAE: A variational autoencoder-based approach for probabilistic time series forecasting with location-scale Gaussian observation modeling.

## Architecture Overview

**Encoder**: `x → patches → encoder → (μ_z, logvar_z)`
**Decoder**: `z → decoder → (μ_out, logvar_out)` with RevIN denormalization

## Files

| File                      | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `LSG_VAE_nn.py`         | Core VAE architecture implementation                    |
| `LSG_VAE_Forecaster.py` | ProbTS forecaster wrapper with training/inference logic |
| `LSG_VAE.yaml`          | Configuration file for training                         |
| `run.sh`                | Training script for multiple prediction lengths         |

## Installation

This implementation is built on top of **ProbTS** and **K²VAE**. Place the files in the appropriate directories following the K²VAE structure:

```
probts/model/nn/prob/k2VAE/LSG_VAE_nn.py
probts/model/forecaster/prob_forecaster/LSG_VAE_Forecaster.py
```

## Usage

```bash
bash run.sh
```

## Configuration

Key hyperparameters in `LSG_VAE.yaml`:

- `patch_len`: Patch size (default: 24)
- `dynamic_dim`: Latent dimension (default: 128)
- `hidden_layers`: Number of hidden layers (default: 3)
- `hidden_dim`: Hidden layer dimension (default: 256)
- `weight_beta`: KL divergence weight (default: 0.01)

## Acknowledgments

Special thanks to the excellent open-source repositories:

- ProbTS - Unified probabilistic time series forecasting framework
- K²VAE - Koopman-based VAE for time series
