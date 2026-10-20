# E0007 sparse quick evaluation

`quick_eval.py` creates a seeded sparse sample from capture-observed bucket labels and reduces paired
baseline/candidate results from real Serve schema-v21 request telemetry. It does not start Serve or
make HTTP requests; the evaluator owns endpoint lifecycle and capsule replay.

## Capture descriptor JSONL

Each descriptor line contains:

```json
{"schema_version":1,"capsule_id":"cap-001","capsule_path":"/external/capsules/cap-001","bucket_labels":["short","tool"],"request_sha256":"<64 hex digits>","output_budget":256}
```

`capsule_path` points to an immutable external capture directory. The descriptor and generated plan
must not contain request bodies, prompts, credentials, cookies, or arbitrary metadata. The sampler
copies only the ID, path, observed labels, request digest, and optional captured output budget.

An explicitly transformed exploratory capsule may declare provenance in its descriptor `metadata`:
`semantic_transform:true`, `production_compatible:false`, `parent_raw_request_sha256`,
`transformed_body_sha256`, and `diff`. A single-field transform uses
`diff:{path,from,to}`; a two-field transform uses an ordered two-entry array of such objects. The
sampler preserves entry order, validates distinct paths, checks that the derived digest matches
`request_sha256`, and copies only those allowlisted provenance fields. A semantic transform cannot
be marked production-compatible.

Create a deterministic plan with an exact shared output budget and cache recipe identity:

```sh
python3.11 tools/dreamrsi/quick_eval.py sample \
  --manifest /external/capture/descriptors.jsonl \
  --seed 7007 --per-bucket 1 \
  --cache-recipe-id prefix-reuse-disabled-v1 \
  --output-budget 256 \
  --output /external/results/e0007-s1-plan.json
```

The plan contains its selected capsule IDs, observed labels, descriptor paths and request digests,
seed, sampler version, output budget, cache recipe ID, and a content-derived `plan_id`. Capsules
with a known captured output budget different from `--output-budget` are excluded. Selection is
unique by capsule ID and deterministic for a fixed manifest and seed. A selected multi-label capsule
can satisfy more than one observed stratum. Any requested label absent from the eligible manifest is
rejected.
Transform flags and provenance are included in each selected capsule and covered by the `plan_id`.

## Evaluator result JSONL

The evaluator replays the same plan serially against each arm and writes one wrapper row per planned
capsule. This maps Serve's request-local monotonic request ID back to the capsule without altering
the Serve schema. A completed row has this shape:

```json
{
  "schema_version":1,
  "plan_id":"<plan id>",
  "capsule_id":"cap-001",
  "cache_recipe_id":"prefix-reuse-disabled-v1",
  "request_start":{"schema_version":21,"event":"request_start","server_instance_id":"serve-1","request":{"request_id":1,"requested_output_tokens":256}},
  "terminal":{"schema_version":21,"event":"request_done","server_instance_id":"serve-1","request":{"request_id":1},"result":{"completion_tokens":80,"finish_reason":"output_limit"},"timings_seconds":{"ttft":0.18,"decode":1.2,"total":1.5},"speculative":{"rounds":35,"drafted_tokens":90,"accepted_tokens":45}},
  "client":{"complete":true,"finish_reason":"max_tokens","received_completion_tokens":80,"ttft_seconds":0.20,"wall_seconds":1.6},
  "vram_peak_bytes":null,
  "identity":{"artifact_sha256":"<artifact digest>","runtime_id":"<runtime identity>"},
  "semantic_transform":true,
  "production_compatible":false
}
```

`terminal` is the unmodified schema-v21 `request_done`, `request_error`, or `request_rejected` event
when one was observed. `request_start` may be null for a preparation rejection or when no request
reached Serve. `terminal` may be null when no terminal event was observed; the reducer reports that
sample as censored. Each row must include a nonempty `identity` object with exact artifact/runtime
identity. `client.complete` must be true for a sample to count as completed. A per-request or per-arm
`vram_peak_bytes` is optional; error messages and response text are not copied to reports.
The result-row transform flags are optional assertions; when present, they must match the plan.
Endpoint base URLs and credentials stay in the evaluator process and do not belong in the plan or
result records.

Reduce a paired run:

```sh
python3.11 tools/dreamrsi/quick_eval.py reduce \
  --plan /external/results/e0007-s1-plan.json \
  --baseline-results /external/results/baseline.jsonl \
  --candidate-results /external/results/candidate.jsonl \
  --output /external/results/e0007-s1-report.json
```

The reducer checks the top-level cache recipe against the plan, checks the request output budget,
and compares the full logged request configuration (including generation sampling settings and seed)
between arms. A pair with a different cache recipe or request config is excluded from classification.

The report includes per-capsule and aggregate TTFT, decode tok/s, generated and committed output
tokens, decode-committed tokens, speculative rounds and tokens per round, wall time, finish reasons,
tool-call parse counters, errors/censoring, optional VRAM, exact supplied identities, and paired
uncertainty. TTFT uses the client stream measurement when both arms supply it; the Serve-internal
`timings_seconds.ttft` is always retained separately. Decode tok/s is
`max(completion_tokens - 1, 0) / timings_seconds.decode`, matching the Serve performance definition.

Classification is a quick screen. A promising result requires at least three complete pairs and a
paired percentile bootstrap 95% interval of the mean log ratio. A material change is 5%, and a
promising result must clear an improvement threshold without a material regression in the other
primary metric. Any missing, failed, censored, shorter candidate output, or finish mismatch blocks a
promising label. With fewer than three pairs, only a measured slowdown above 30% without a
compensating primary-metric gain can be labeled an obvious regression; its bootstrap interval stays
unavailable. Other small samples remain flat/noisy. This is S1 screening only; it does not replace
S2/S3 qualification or a content-quality review.
For a semantically transformed plan, the report retains the paired statistical result as
`exploratory_classification` while setting `semantic_transform:true`, `production_compatible:false`,
and `production_classification:"ineligible"`. If that screen would otherwise be promising, the main
`classification` is `exploratory only`; transformed pairs never receive `candidate_win_eligible:true`.
Baseline and candidate request configurations must still match exactly for measurements to pair.

Focused tests:

```sh
python3.11 -m unittest discover -s tools/dreamrsi -p 'test_*.py'
```
