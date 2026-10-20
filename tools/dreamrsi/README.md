# Bounded FNN evaluator

This directory contains a single-owner FIFO submitter and evaluator for isolated NInfer
research runs on FNN. It is not part of the production serving path. The evaluator uses the
public `ninfer-serve` route, keeps one model resident per arm, serializes requests, and holds
the shared FNN GPU lease for the full paired run.

## Submission and queue

Freeze the assigned checkout, including its tracked and non-ignored untracked source, then enqueue
both arms in FIFO order. Use absolute paths and the explicit artifact path and SHA-256:

```sh
python3.11 tools/dreamrsi/submit.py freeze \
  --source "$CHECKOUT" --package-dir "$RESEARCH_ROOT/packages" \
  --role baseline --baseline-id "$PAIR_ID" \
  --artifact-path "$MODEL_ARTIFACT" --artifact-sha256 "$MODEL_SHA256" \
  --runtime-id "$RUNTIME_ID"
```

Repeat with `--role control` and the same pair ID, checkout, artifact, runtime ID, and overrides.
Use the manifest and archive paths returned by `freeze` with `submit.py enqueue`, first for the
baseline and then for the control. The FIFO stores immutable source archives and manifests in
owner-only directories. The worker rejects a pair whose runtime, artifact, overrides, source
snapshot, or Git commit differ.

Project-generated research material is stored beneath an owner-only research root. Request text
must not be placed in the repository, submission, logs, or command arguments. Create a short,
non-sensitive S0 probe in a mode-0600 file under the research root; the worker reads and deletes it
before starting. Captured Claude request capsules are the sole exception: the capture proxy stores
the bounded request and response bodies in mode-0700 directories with mode-0400 files so the same
request can be replayed. Result files contain request configuration and metrics, not prompt or
completion text. Temporary replay responses are deleted after extracting metadata.

## Paired worker

Install the current `capture.py`, `replay.py`, and `quick_eval.py` scripts in owner-only research
storage. Enqueue exactly a baseline followed by a control, then run:

```sh
python3.11 tools/dreamrsi/evaluator.py worker \
  --queue-root "$QUEUE_ROOT" --research-root "$RESEARCH_ROOT" \
  --lock-path /home/aramirezfamily/.local/share \
  --tools-dir "$CAPTURE_TOOLS" --capture-dir "$CAPTURE_DIR" \
  --descriptors "$CAPTURE_DIR/descriptors.jsonl" \
  --quick-eval "$CAPTURE_TOOLS/quick_eval.py" \
  --probe-text-file "$RESEARCH_ROOT/probe.txt" \
  --require-baseline-control
```

The baseline starts the isolated build and server with the fixed Qwen3.8 27B profile, runs a
one-token warmup and a four-token S0 plumbing check, and opens a temporary capture endpoint bound
to the FNN tailnet address. The endpoint accepts at most one request, for at most fifteen minutes
by default (configurable from 60 to 900 seconds), and rejects output budgets over 128 tokens before
forwarding. It shuts down before capsule replay.
The control reuses the same build, runtime settings, artifact, plan, and captured capsule(s). Each
arm is sampled serially with the same evaluator generation-seed override. The observed source,
artifact, runtime, server settings, cache recipe, capsule hash, and Serve request telemetry are
retained with each result.

S0 only verifies the route. It does not establish useful generation performance. S1 replay results
are reduced from Serve's request-level timing, completion, speculative-decoding, and memory
telemetry. The sampler may produce fewer than three paired observations when the bounded capture
contains fewer usable capsules; such output is exploratory and cannot be called a reliable win.

The evaluator checks the production user services and research ports, requires the expected GPU at
CUDA device 0 with no competing compute process, validates the artifact SHA-256, and audits process,
listener, and GPU-memory cleanup. It records free memory after model startup against a 1 GiB
threshold; it does not reduce context or disable the configured decoding features to force startup.
A failed startup, request, or cleanup moves that queue item to `failed` and stops the pair.

## Files

- `submit.py freeze`: package a reproducible checkout snapshot.
- `submit.py enqueue`: atomically append one immutable item to the FIFO.
- `evaluator.py worker`: run one bounded baseline/control queue pair.
- `lifecycle.py`: shared GPU lease, child-process-group, readiness, and preflight helpers.
- `tests/test_worker.py`: focused checks for source freezing, FIFO integrity, process cleanup,
  environment isolation, artifact identity, and Serve telemetry correlation.
