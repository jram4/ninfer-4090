# Candidate evaluation runbook (FNN)

This is the exact sequence for evaluating one frozen candidate against the
captured corpus. It is written for an operator, and every command is
copy-pasteable.

## What the run costs

| Step | Needs the GPU | Time |
|---|---|---|
| freeze the candidate | no | seconds |
| admission + build | no | ~5 min |
| correctness, smoke, screen | **yes** | ~35 min |
| full (only if the screen passes) | **yes** | ~25 min |

The evaluation host has one 24 GB RTX 4090. The production service holds
~22.5 GB of it, so **the GPU stages cannot run while production is serving**.
Stop production for the GPU window and start it again afterwards. The build
stage does not need the GPU and can be run beforehand.

The baseline arm is measured separately, by replaying the same plan against the
running production release, so production does not have to be stopped to obtain
the baseline.

## 1. Prepare on Pop

```sh
REPO=~/ninfer-4090/.slim/worktrees/ninfer-bench-integration
cd "$REPO"

# the corpus plans and the policy live outside the repository
CORPUS=~/.local/share/ninfer-bench/corpus
python3.11 tools/dreamrsi/ops/build_fnn_policy.py \
  --contract tools/dreamrsi/examples/fnn-4090.contract.json \
  --plan "$CORPUS/plan-screen.json" \
  --baseline-rows /home/aramirezfamily/.local/share/ninfer-bench/baseline/screen-baseline.jsonl \
  --harness <TRUSTEDHARNESS-ID> \
  --output "$CORPUS/policy-screen.json"
```

The trusted harness id is the content-addressed directory on fnn. Install or
refresh it with the transport's installation plan:

```sh
python3.11 -m tools.dreamrsi transport \
  --remote-root /home/aramirezfamily/.local/share/ninfer-bench/runs \
  --trusted-harness tools/dreamrsi
```

It prints the exact `ssh`/`scp` commands and the harness id. Run them.

## 2. Freeze the candidate and queue it

```sh
python3.11 -m tools.dreamrsi submit \
  --source <worktree> --package-dir ~/.local/share/ninfer-bench/packages \
  --epoch-id <epoch> --worker-id <worker> --hypothesis-id <hypothesis> \
  --contract-id <contract> --policy-id <policy> --ref HEAD
```

## 3. Measure the baseline arm (production stays up)

```sh
ssh fnn "cd /tmp && $REMOTE_PYTHON <HARNESS>/project_stages.py replay-to \
  --plan /home/aramirezfamily/.local/share/ninfer-bench/corpus-plan-screen.json \
  --split screen --capsule-root / \
  --upstream http://100.119.241.20:8080 \
  --rows /home/aramirezfamily/.local/share/ninfer-bench/baseline/screen-baseline.jsonl"
```

## 4. The GPU window

```sh
# stop production
ssh fnn 'systemctl --user stop cinference-qwen38-4090.service'
ssh fnn 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'   # must be small

# run the candidate through the evaluator on fnn
python3.11 -m tools.dreamrsi coordinator \
  --queue-db ~/.local/share/ninfer-bench/queue.sqlite3 \
  --policy ~/.local/share/ninfer-bench/corpus/policy-screen.json \
  --run-root ~/.local/share/ninfer-bench/runs \
  --epoch-db ~/.local/share/ninfer-bench/epochs.sqlite3 \
  --evaluator-id fnn-gpu0 --max-items 1 \
  --transport ssh \
  --remote-root /home/aramirezfamily/.local/share/ninfer-bench/runs \
  --trusted-harness tools/dreamrsi

# restore production
ssh fnn 'systemctl --user start cinference-qwen38-4090.service'
ssh fnn 'curl -s -o /dev/null -w "%{http_code}\n" http://100.119.241.20:8080/v1/models'
```

The evaluator stops at the first failing stage, so a candidate that fails the
screen never reaches the holdout.

## 5. Read the result

The receipt and every evidence artifact come back to Pop under
`~/.local/share/ninfer-bench/runs/trusted-receipts/`. The verdict vocabulary is
`rejected`, `regression`, `inconclusive`, `blocked`, `promising`, `qualified`.

- `regression` means the candidate ran and failed a gate.
- `blocked` means the host was not fit (for example the GPU could not be
  initialised). It is never a candidate verdict.
- `inconclusive` means the evidence was missing, noisy, or too small.
- `promising` means every configured stage passed with qualification disabled.
- `qualified` additionally means the policy's qualification gates were met.

Promotion is a separate, explicit step and never happens as part of the run.
