<!-- Modified by satellitedown for Cinference. Upstream attribution is retained in NOTICE. -->

# Cinference

**A custom C++/CUDA inference engine for Qwen on a single RTX 5090 (32 GB), with 256K context.**

Built from [NInfer](https://github.com/Neroued/ninfer), with source-level changes to speculative decoding, CPU/GPU round buffers, and CUDA Graph management. Cinference modifies the native engine itself, not just its launch flags.

## What changed

- **MTP-10 decoding:** raised the draft window from 5 to 10 tokens. Longer proposals let the engine emit more tokens per verification round when the drafts are accepted.
- **Capture-based CUDA Graph reuse:** reworked MTP graph matching to use the captured node types and kernel functions. Profiles with matching signatures and batch sizes share an executable, rather than relying only on planned context ranges.
- **Expanded CPU/GPU round handling:** enlarged draft and token-position buffers and updated native validation for the longer windows. This carries MTP-10 through the decoding path, not just the command-line options.
- **Ready-to-run Huihui setup:** published the NVFP4 v3 model and a matching installer. Users download one ~21.5 GB model file, without a local conversion step.

## Run

For [Huihui Qwen3.8-27B Abliterated NVFP4](https://huggingface.co/satellitedown/Huihui-Qwen3.8-27B-abliterated-NVFP4-NInfer-v3), use the [one-menu installer](https://github.com/satellitedown/fast-long-context-cinference):

```bash
git clone https://github.com/satellitedown/fast-long-context-cinference.git
cd fast-long-context-cinference
bash setup.sh
```

Requires Linux and working NVIDIA drivers. Choose **1** to install, then **3** to start.

## Performance

| Prompt tokens | Tokens/s |
|---:|---:|
| 8,192 | 450.78 |
| 32,768 | 432.60 |
| 131,072 | 364.84 |
| 260,000 | 299.64 |

Huihui Qwen3.8-27B Abliterated NVFP4, MTP-10, K8V4. Generation speed on synthetic recall, thinking off. [Measurements](results/rtx5090-archive-recall.json).

[Build from source](docs/maintainer/build-system.md) · [Technical docs](docs/README.md) · [Attribution](NOTICE)
