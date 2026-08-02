# RTX 4090 qualification evidence

These reports are curated from the local RTX 4090 qualification run described in
[`docs/rtx-4090-linux.md`](../../docs/rtx-4090-linux.md). They contain configuration, hardware,
memory, and timing data but no prompts, generated text, credentials, request logs, model weights,
or raw profiler traces.

## End-to-end reports

- `rtx4090-27b-initial-mtp0.json` and `rtx4090-27b-final-mtp0.json`: ordinary control/final.
- `rtx4090-27b-initial-mtp3.json` and `rtx4090-27b-definitive-mtp3.json`: MTP-3 control/final.
- `rtx4090-27b-bitexact-final-k{0,3}-ctx131072.json`: later 131,072-capacity source evidence.

## Operator evidence

- `rtx4090-sm89-gdn-*.csvlog`: measured GDN route boundaries.
- `bitexact-control-q5-*.csv` and `bitexact-candidate-q5-*.csv`: slower eight-row projection
  candidates that were rejected in favor of the inherited 16-row route.
- `bitexact-{control,candidate,retained}-q5add-5120x6144.csv`: the retained cache-read residual
  candidate.

The first public release is source-only. These historical measurements do not certify a bundled
binary, and no model artifact is distributed from this repository.
