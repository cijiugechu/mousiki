# mousiki-dred-weights

Build-time generator for DRED weights and stats tables used by mousiki.

## Features

- `fetch`: allow downloading the model tarball from Xiph during build.
  When disabled, you must provide a local path via `DRED_WEIGHTS_PATH`.

The main crate exposes this as `dred_fetch`.

## Build inputs

- `DRED_WEIGHTS_PATH`: Local directory or tarball containing
  `dred_rdovae_dec_data.c` and `dred_rdovae_stats_data.c`.
- `DRED_WEIGHTS_URL`: Override the default Xiph model URL.
- `DRED_WEIGHTS_SHA256`: Override the expected tarball SHA-256.

Proxy environment variables (`ALL_PROXY`, `HTTPS_PROXY`, `HTTP_PROXY`) are
honored when downloading.
