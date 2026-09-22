# Cinference → RTX 4090 port staging

This branch is a staging home for the RTX 4090 port because the connected GitHub
integration can write repositories and branches but cannot create/fork a repository.

## Source basis

- Cinference repository: `satellitedown/cinference`
- Pinned source revision: `2205806b3e9d12b23673a146eab1d36755ebc3a0`
- Upstream NInfer basis recorded by Cinference: `9e163eee4b8acec21ab0ac765107b6a3f287b217`
- Intended GPU: RTX 4090 / Ada / compute capability 8.9 / `sm_89`

## Port design

The port deliberately starts from current Cinference, **not** the older public
`jram4/ninfer-4090` source tree. Run `apply_port.py` from a clean checkout at
the pinned Cinference revision.

The overlay:

1. retargets CMake and runtime validation from `sm_120a` / CC 12.0 to
   `sm_89` / CC 8.9;
2. preserves Cinference MTP-10 and its expanded CPU/GPU speculative buffers;
3. preserves capture-derived CUDA Graph topology reuse;
4. changes the FP8 E4M3 MMA spelling to the native Ada SM89 instruction form,
   allowing the K8V4 attention path to remain available;
5. preserves groupwise Q4/Q5/Q6/Q8/BF16 weight execution;
6. removes compilation of Blackwell-only NVFP4 TMA / `setmaxnreg` units and
   supplies fail-closed ABI stubs for those launch routes;
7. does **not** requantize model weights.

## Intended artifact

The target artifact is the existing groupwise Qwen3.8-27B family used by the
4090 runtime, not Cinference's published NVFP4 Huihui artifact. Native NVFP4
weight execution remains Blackwell-only in this first port.

## Verification still required on a 4090

After applying the overlay on the target machine:

```bash
cmake -S . -B build-sm89 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-sm89 -j

# Confirm MTP-10 is exposed.
./build-sm89/apps/ninfer-cli --help | grep -E 'draft|mtp'

# Then qualify K=3/4/5/7/10 against the same groupwise Qwen3.8 artifact.
```

Do not call the port production-ready until it builds on the RTX 4090 and exact
greedy-output equivalence plus representative throughput/acceptance tests pass.
