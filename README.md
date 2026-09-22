<!-- Modified by satellitedown for Cinference. Upstream attribution is retained in NOTICE. -->

# Cinference

**Fast local Qwen inference on a single RTX 5090 (32 GB), with 256K context.**

Built on [NInfer](https://github.com/Neroued/ninfer), with **MTP-10 speculative decoding** to generate more tokens per step.

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
