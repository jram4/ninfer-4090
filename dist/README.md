# RTX 4090 Linux release bundle

The first public `v0.2.0-rtx4090-v1` release is source-only. No prebuilt binary from the private
qualification checkout is distributed. The packaging workflow below is reserved for a future
release after a clean isolated RTX 4090 rebuild and acceptance run.

The current package identity is
`ninfer-rtx4090-qwen3.6-27b-linux-x64-0.2.0-rtx4090-v1`.

After that acceptance gate, `scripts/package-release.ps1` packages Linux products from
`build-sm89`:

- `ninfer`
- `ninfer-serve`
- `ninfer_bench`
- `VERSION`
- `LICENSE`
- the RTX 4090 Linux qualification report
- per-file and archive SHA-256 manifests

The script does not create or publish a release and does not package model weights. Run it only
after the complete `sm_89` acceptance suite passes. Model artifacts remain separate downloads
whose hashes are documented in the project README.

Earlier `ninfer-rtx3090-*` archive metadata and reports are historical and are not produced by the
current package script. Native Windows is not a verified target of
`0.2.0-rtx4090-v1`.
