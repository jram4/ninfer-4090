# Contributing

Contributions are welcome when they preserve the supported product contract: 64-bit Linux,
one NVIDIA GeForce RTX 4090 (`sm_89`), and Qwen3.6-27B.

## Before opening a pull request

1. Open an issue for architecture, artifact-format, public API, or dispatch-policy changes.
2. Keep inherited attribution and the Apache-2.0 license intact.
3. Do not commit model weights, source checkpoints, build products, profiler traces, request logs,
   credentials, or machine-specific paths.
4. Run `python3 scripts/check_public_tree.py` and the focused tests for the behavior changed.

## Verification expectations

- Documentation: run the public-tree/link check and `git diff --check`.
- Python tooling: run the affected `pytest` targets on Python 3.11.
- C++ runtime: build Release with CUDA 13 and run the affected native tests.
- CUDA math: provide an independent numerical or bit-exact comparison at the supported shapes.
- Performance: report GPU, driver, CUDA, compiler, command, warmups, repetitions, cache policy,
  control median, candidate median, and end-to-end impact.

Performance changes must be measured on an RTX 4090 and must not be justified by a warm-cache
microbenchmark alone. Regressions or candidates below the documented retention threshold should
remain out of production dispatch.

## Pull requests

Use a focused branch and Conventional Commit-style subjects (`build`, `fix`, `perf`, `test`,
`docs`, or `chore`). Complete the pull-request template, include the exact checks run, and state
anything that could not be verified. Maintainers may ask for a clean hardware rerun before a
kernel or release claim is accepted.
