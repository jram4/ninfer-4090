# NInfer research bench

This is the control and evidence tooling for isolated NInfer research. Pop owns
submissions, queue state, epochs, and integration. Target GPU measurements belong
on fnn's RTX 4090. The tools do not provide a Python model-inference route.

The implementation recovers useful primitives from the preserved E0007/E0008
work, replacing its one-pair bring-up workflow rather than adding another queue.
Historical research state is evidence, not current runtime configuration.

## Operating boundaries

- Submit immutable source snapshots, not moving branches or build directories.
- Keep model, compression, tokenizer, template, and capability locks fixed.
- A worker proposes changes; trusted evaluator policy owns commands and gates.
- An exploratory win is not qualification. Missing quality or held-out evidence
  blocks qualification even when measured throughput increases.
- Promotion is a shadow plan by default. A qualified result is not a deployment.
- Worktrees isolate source changes, not GPU access or all operating-system rights.
  Workers must not own the judge, production controls, or promotion credentials.
- Raw request and response bodies stay in private storage outside Git. Public
  descriptors and reports contain identities and operational measurements only.
- A GPU lock coordinates researchers, not a production service. Never interrupt
  production merely to obtain an uncontended measurement.

## Target identity

[`examples/fnn-4090.contract.json`](examples/fnn-4090.contract.json) records the
artifact and template identities freshly hashed on 2026-10-20, plus the observed
production startup configuration. Tokenizer resources are embedded in the same
artifact (`src/models/qwen3_5/load/resources.cpp`); the artifact hash locks their
bytes. Changed tokenizer implementation still needs behavioral qualification.

This is a **configuration reference, not a qualified performance baseline**.
Per-request sampler settings, budgets, effective seeds, and cache recipes must
also match the declared comparison. Serialized requests from one session are not
independent experimental samples.

## Current host limitation

On 2026-10-20, fnn's production service was active. Its NVML userspace/kernel
driver mismatch remained present, and a fresh driver initialization probe returned
`804 / CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE` before any context, allocation,
or model load. This prevents new candidate GPU qualification. The existing service
was not interrupted. Its activity is not proof that a fresh process can run.

No driver repair, reboot, production restart, or candidate deployment is performed
by this tooling. Existing-endpoint capture is a separate, explicitly labelled
operation and cannot establish isolated candidate performance.

## Command entrypoint

From the repository root, use Python 3.11:

```sh
python3.11 -m tools.dreamrsi --help
```

Each subcommand provides its own `--help`. Operator workflows and validation
results are completed after the isolated implementation lanes are reconciled.
