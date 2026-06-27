# Architecture Notes

## POC-1: residual U-Net detailer

Input:

- coarse/upscaled normalized height
- dx derivative
- dz derivative
- Laplacian/curvature

Output:

- high-resolution residual height

Loss:

- height L1
- gradient L1
- Laplacian L1
- spectral L1
- seam edge L1

## POC-2: conditional diffusion residual model

After the residual U-Net proves the dataset and rendering pipeline, move to diffusion over residuals.

Useful condition channels:

- coarse height
- ridge mask
- valley/channel mask
- target slope band
- terrain class embedding

## POC-3: latent diffusion

Train a terrain-specific autoencoder or VAE for heightmaps. Then train diffusion in latent space.

I'll get around to this soon™. A bad VAE will blur ridges and spit out a bunch of useless bullshit.
