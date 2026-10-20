# E0005-LW01 — Q0 canary result

Status: terminal; stopped at the OFF-arm startup headroom gate before the public fixture session. No OFF/ON fixture pair was completed.

## Outcome

The frozen candidate passed the source, fixture, artifact, resource, binary, and runtime-library preflights. Serve loaded the model and completed its built-in warmup. The startup check then measured 1,045,561,344 bytes free (997.1 MiB), below the required 1,073,741,824 bytes (1,024 MiB) by 28,180,480 bytes (26.875 MiB). The runner stopped before launching the public fixture harness. The ON arm was not started.

The OFF identity sidecar contains the startup warmup only: a prompt identity with 53 tokens and a committed-token identity with 4 tokens. Its request log contains only `server_start`; no fixture telemetry was produced. There is no public-request result, OFF/ON output comparison, or model-backed Q0 overlap/support measurement.

E0005-H01 remains unresolved. The warmup shows that the identity hook recorded one internal Engine request, but the fixed public coding-agent request and observer ON/OFF invariance criterion were not exercised. This run neither supports nor falsifies the hypothesis.

## Frozen inputs and runtime

- Plan: `/Users/josh/Documents/ninfer-4090/Dream-RSI/experiments/E0004-Q0-canary-plan.md`.
- Artifact: `/home/aramirezfamily/josh/cinference-4090/models/qwen3_8_27b.ablit-wxp.ninfer`, 18,210,704,128 bytes, SHA-256 `b7c6693244cf7ab08554fff41c4f3c67695b6ec388c99eed6f12e0e71046fda8`.
- Tokenizer SHA-256: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`; chat-template SHA-256: `a497db9e663941e6f7a05307c2bafa8374c12e142ad1fbea6229888d41f1ab16`.
- Public fixture: `bench/fixtures/dreamrsi/merge_ranges/`, accepted package SHA-256 `ad91917862163854f4428b0f18efeda82b2d9414372e32ec6b74aab430341a49`; seed 101 and frozen sampling/tool limits were retained.
- Isolated FNN root: `/home/aramirezfamily/.local/share/ninfer-e0005-lw01-q0canary-20260927-01a0e223` (owner-only). Source: `work/source`, tree SHA-256 `fa909264ab8b90418c7690feab0360b34545748ba4f38a5e28c4ee70ccc22b55`. Serve binary: `work/build/apps/ninfer-serve`, SHA-256 `f5ae0a6b449a895418856a7f99c008c5e3127ea97dcc663f803d3439a29e1981`.
- Runtime: NVIDIA GeForce RTX 4090, driver 580.173.02, CUDA 13.0, compute capability 8.9; MTP K7 with optimized indexed LM-head, K8V4, explicit context/KV capacity 188,416, concurrency 1, CUDA Graphs enabled. Production units remained inactive.
- The initial GPU reading was 170 MiB used / 23,890 MiB free. After shutdown it returned to 170 MiB used / 23,890 MiB free.

## Execution and stop evidence

An initial server launch exited before `main` because `libswresample.so.7` was not on its loader path. Its exact loader error was preserved privately as `evidence/off.initial_loader_failure.log`. No model process or request began in that launch.

While checking the library-path correction, the full private runner was accidentally invoked once without the shared lock. The prior failed launch had left an empty `output/off.work` directory. The runner requires that directory to be newly created and performs this check before opening the server log or spawning Serve. The invocation therefore stopped before model startup. No request, warmup, generated output, sidecar, or telemetry resulted from that accidental call. The stale empty directory was removed under the lock after confirming its contents; the previous loader error was preserved.

The single corrected runner invocation then ran under `flock -x /home/aramirezfamily/.local/share`. The server log records successful model load, Engine readiness, and listen on `127.0.0.1:18945`. The private runtime record reports the insufficient headroom above. After the stop, the runner's port rebind check returned `loopback_port_not_free`. A locked terminal audit found no process tied to this research root, no listener on port 18945, all four production units inactive, and GPU memory back at its initial reading. The port-check error is retained as a cleanup anomaly; the canary was not retried.

Private evidence is under the FNN root: `evidence/off.runtime.json`, `evidence/off.server.log`, `output/off.request.jsonl`, and `output/off.q0.jsonl`. These files are owner-only. The report contains no prompt text, completion text, or raw tool arguments/results.

## Checks and changed files

On FNN, `cmake --build /home/aramirezfamily/.local/share/ninfer-e0005-lw01-q0canary-20260927-01a0e223/work/build -j --target ninfer-serve ninfer_qwen3_5_runtime_mechanisms_test` completed. `ctest --test-dir /home/aramirezfamily/.local/share/ninfer-e0005-lw01-q0canary-20260927-01a0e223/work/build -R '^ninfer_qwen3_5_runtime_mechanisms_test$' --output-on-failure` passed 1/1. Python 3.11 syntax compilation of the private runner/helpers, the Serve `--help` control check, and `git diff --check` passed. Build and test execution used the shared lock. These checks validate the candidate and identity hook; they do not replace the stopped model-backed fixture gate.

The cumulative worktree changes comprise:

- Q0 observation and identity: `include/ninfer/ops/speculative_round.h`; `src/models/qwen3_5/frontend/digest.cpp`, `digest.h`; `src/models/qwen3_5/program/context.h`, `decode.cpp`, `graphs.cpp`, `prefill.cpp`, `program.cpp`, `program.h`, `program_impl.cpp`, `program_impl.h`, `round_buffers.cpp`, `round_buffers.h`, `speculative/mtp.cpp`, `speculative/q0_observer.h`, `speculative/target_verification.cpp`, `storage/context.cpp`, `transactions/commit.cpp`; `src/ops/kernel/speculative_round.cuh`, `launcher/speculative_round.cu`, `launcher/speculative_round.h`, `wrapper/speculative_round.cpp`; `src/runtime/engine/engine_core.h`, `q0_observer_sink.h`, `request_record.h`; `tests/models/qwen3_5/test_runtime_mechanisms.cpp`, `tests/ops/test_speculative_round.cpp`, `tests/ops/tests.cmake`.
- Frozen public fixture: `bench/fixtures/dreamrsi/merge_ranges/agent_harness.py`, `checker.py`, `manifest.json`, `starter/merge_ranges.py`, `system_prompt.txt`, `task.md`, `test_agent_harness.py`, `test_checker.py`.
- This report: `Dream-RSI/experiments/E0005-LW01-q0-canary.md`.

No prompt, seed, budget, sampler, or model settings were retuned. No ON run, wider seed screen, performance comparison, Q1–Q4 experiment, production action, or deployment occurred.
