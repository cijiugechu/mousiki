# mousiki-deep-plc-weights

Builds a deep PLC/DRED weight blob at compile time and exposes it via
`DNN_BLOB`.

## Build inputs

- `DNN_WEIGHTS_PATH`: directory or tarball containing `dnn/*_data.c`
- `DNN_WEIGHTS_URL`: optional override for the tarball URL
- `DNN_WEIGHTS_SHA256`: optional override for the tarball checksum

If `DNN_WEIGHTS_PATH` is not set, enable the `fetch` feature to download the
model tarball automatically.
