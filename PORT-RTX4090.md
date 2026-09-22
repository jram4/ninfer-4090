# Cinference RTX 4090 port

Base: `satellitedown/cinference@2205806b3e9d12b23673a146eab1d36755ebc3a0`.

This port keeps Cinference's MTP-10, expanded speculative round buffers, and
capture-derived CUDA Graph topology reuse. It retargets the runtime to NVIDIA
Ada compute capability 8.9.

Supported first target:
- Qwen3.8-27B groupwise Q4/Q5/Q6/Q8/BF16 NInfer artifacts
- MTP windows 1..10
- BF16 / INT8 / FP8-K+V4 (K8V4) KV paths
- software E2M1 pack/decode on Ada for the V4 cache plane
- Ada fallback for the sm_90+ T=4 PDL/griddepcontrol GDN overlap
- single RTX 4090, Linux, CUDA 13.x

Not supported on Ada:
- native NVFP4 W4A4 model-weight execution
- Blackwell TMA/setmaxnreg NVFP4 prompt kernels

The unsupported Blackwell launch entry points fail closed. No model
requantization is performed by this port.
