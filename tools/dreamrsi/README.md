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

The mismatch is one reboot away: the on-disk kernel module is already `580.178.04`
and matches the userspace; only the running module is the older `580.173.02`. The
host has no passwordless sudo and polkit refuses a non-root reboot, so the owner
must run the reboot. `linger` is enabled and the production unit is enabled and
wanted by `default.target`, so it returns by itself after the reboot.

No driver repair, reboot, production restart, or candidate deployment is performed
by this tooling. Existing-endpoint capture is a separate, explicitly labelled
operation and cannot establish isolated candidate performance.

## Command entrypoint

From the repository root, use Python 3.11:

```sh
python3.11 -m tools.dreamrsi --help
```

Each subcommand provides its own `--help`.

## Operator workflow

### 1. Build the trusted policy

The policy is the only place that decides what a candidate must pass. It lives
outside candidate source. Build it from a specification so the content identities
are computed rather than hand-copied:

```sh
python3.11 -m tools.dreamrsi policy build \
  --spec tools/dreamrsi/examples/fnn-4090.policy-spec.json \
  --output ~/.local/share/ninfer-bench/policies/fnn-4090.json
python3.11 -m tools.dreamrsi policy show --policy ~/.local/share/ninfer-bench/policies/fnn-4090.json
```

Qualification stays disabled until the policy evidence is trustworthy. The
example specification points `build`, `correctness`, `screen`, and `full` at
project stage scripts under `research/` on fnn; those scripts must publish the
evaluator report schema described in `tools/dreamrsi/stage.py`.

### 2. Submit an immutable candidate

A worker freezes its own checkpoint from its worktree:

```sh
python3.11 -m tools.dreamrsi submit \
  --source ~/ninfer-4090/.slim/worktrees/<lane> \
  --package-dir ~/.local/share/ninfer-bench/packages \
  --epoch-id <epoch> --worker-id <worker> --hypothesis-id <hypothesis> \
  --contract-id <contract> --policy-id <policy> --ref HEAD
```

A dirty worktree is refused unless `--include-dirty` is given, and a dirty
snapshot can never be promoted. The submission identity covers the source
manifest, not the archive path, so the same bytes can be evaluated on fnn.

### 3. Queue and evaluate

The queue database is Pop-local and authoritative. The coordinator claims one
item at a time, evaluates it under the trusted policy, and finishes it with a
fenced receipt:

```sh
python3.11 -m tools.dreamrsi coordinator \
  --queue-db ~/.local/share/ninfer-bench/queue.sqlite3 \
  --policy ~/.local/share/ninfer-bench/policies/fnn-4090.json \
  --run-root ~/.local/share/ninfer-bench/runs \
  --epoch-db ~/.local/share/ninfer-bench/epochs.sqlite3 \
  --evaluator-id fnn-gpu0 --max-items 1
```

For measurement on fnn, the transport installs a content-addressed trusted
harness and runs the evaluator there. This command only prints the plan; it
opens no SSH session and changes nothing:

```sh
python3.11 -m tools.dreamrsi transport \
  --remote-root /home/aramirezfamily/.local/share/ninfer-bench/runs \
  --trusted-harness tools/dreamrsi
```

The plan names the exact `ssh`/`scp` commands an operator runs to install the
harness under a content-addressed `TRUSTEDHARNESS/<harness-id>` directory, and
the coordinator verifies those files by hash before using them. The trusted
harness is installed separately from the candidate archive and is read-only
there. A candidate can never supply the evaluator, the policy, the corpus, or
the thresholds.

### 4. Promotion

```sh
python3.11 -m tools.dreamrsi promote --repo ~/ninfer-4090 \
  --candidate-commit <commit> --expected-main <commit> \
  --receipt <receipt.json>
```

The default is a shadow plan that never changes a ref. An actual promotion
requires a qualified receipt, evidence that still hashes correctly, a trusted
policy outside the repository, and an explicit `--authorize`; it updates one ref
by compare-and-swap and never pushes or deploys.

### 5. Capture and compare real traffic

Capture runs on fnn in a labelled window. One window is one independent coding
session, which is what lets the planner draw a holdout group:

```sh
bash tools/dreamrsi/ops/capture_window.sh start my-session-01
# ... run the coding client against http://100.119.241.20:18082 ...
bash tools/dreamrsi/ops/capture_window.sh stop
```

The window binds a private tailnet port, forwards to the production service
unchanged, and writes owner-only capsules under
`/home/aramirezfamily/.local/share/ninfer-bench/live-capture/<label>/` on fnn.
Captured bodies never enter Git. Descriptors carry hashes and operational labels
only.

`start` refuses to run while another window holds the capture port, and it only
reports success once the proxy actually owns that port. `windows` lists every
window with its request count, and `merge` combines the windows into one
descriptor manifest:

```sh
bash tools/dreamrsi/ops/capture_window.sh windows
bash tools/dreamrsi/ops/capture_window.sh merge ~/corpus-manifest.jsonl
```

The corpus planner samples whole sessions, keeps disjoint holdout groups,
replays each selected session's captured prefix with explicit warmups, and never
rewrites a request body or its output budget:

```sh
python3.11 -m tools.dreamrsi corpus plan --manifest ~/corpus-manifest.jsonl \
  --seed 41 --request-budget 64 --cache-recipe-id prefix-reuse-default-v1 \
  --output <plan.json>
```

A comparison is inconclusive when the data is insufficient; it is never a
release signal on its own. A plan built from a single session reports
`inconclusive_insufficient_corpus` by design.

### 6. Epochs and workers

An epoch is a user-controlled authorization envelope. It is created explicitly,
bounded, and cancelled explicitly:

```sh
python3.11 -m tools.dreamrsi epoch start --epoch-id <epoch> --budgets <budgets.json>
python3.11 -m tools.dreamrsi worker register --epoch-id <epoch> ...
```

Workers are bound to one exact existing worktree, base commit, model, and path
allowlist. Callback delivery is a durable outbox drained by software; there is no
claimed host wake hook, and the tooling does not poll on an agent's behalf.

## Stage ladder and who writes each stage

The six stages run in order: `admission`, `build`, `correctness`, `smoke`,
`screen`, `full`.

| Stage | Program | What it proves |
|---|---|---|
| admission | `stage.py toolchain` | The host really has the pinned CMake and CUDA toolkit. |
| build | `project_stages.py build` | The frozen source configures and compiles for the target GPU, and the evaluator hashes the produced binary itself. |
| correctness | `project_stages.py correctness` | The affected contract and numerical tests pass. |
| smoke | `project_stages.py smoke` | The candidate's own Serve starts, serves a request, and reports the runtime contract it actually loaded. |
| screen | `project_stages.py screen` | The candidate replays the screen split and the paired reduction passes. |
| full | `project_stages.py full` | The holdout split passes on unseen sessions. |

A stage that reports a failing gate is a **candidate** failure. A stage whose
evidence is missing, stale, or unverifiable is **inconclusive** and blocks
qualification. Infrastructure faults — a broken host, a missing dependency, a
toolchain that is not the pinned one — are **blocked**, never a candidate
verdict.

The build stage publishes the path of the binary it produced. The evaluator
hashes that file itself and refuses the run if the stage's claim and the file
disagree. The smoke stage reports the contract observed in the candidate's own
`server_start` event, including the hash of the binary that served the request.
The expected contract therefore carries no candidate identity: the model
artifact, tokenizer, template, KV format, context capacity, and sampling
defaults are invariant, while the binary and source commit are per-run
evidence.

## What this tooling does not do

- It does not run a model in Python, change the model artifact, or change
  compression, precision, context capacity, or sampling defaults.
- It does not stop, restart, or reconfigure the production service.
- It does not push to GitHub or deploy a release.
- It does not treat a throughput increase as proof of task quality.
- It does not replace a failed gate with a rerun.
