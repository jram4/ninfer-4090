# NInfer documentation

- [RTX 4090 Linux qualification and measured configuration](rtx-4090-linux.md)
- [Linux release bundle metadata](../dist/README.md)

Start with the [project README](../README.md) to build NInfer, download the registered Qwen3.6-27B
artifact, and run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Chat Completions, Anthropic Messages, streaming, token counting, authentication, and tool calls |
| [RTX 4090 qualification](rtx-4090-linux.md) | current sm_89 build, tests, route evidence, and MTP-on/off baselines |
| [Performance](performance.md) | historical upstream RTX 5090 results and reproduction context |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Download | Versioned model card source |
|---|---|---|
| Qwen3.6-27B | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Curated RTX 4090 evidence](../benchmark-results/rtx4090/README.md)
- [Tests](../tests/README.md)
- [Maintainer tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Historical port reports

These reports are retained as evidence from the earlier RTX 3090 port. They are not part of the
current RTX 4090 product contract or dispatch path.

- [RTX 3090 / WSL2 port](rtx-3090-wsl.md)
- [RTX 3090 / native Windows](rtx-3090-windows.md)
- [RTX 3090 ordinary inference](rtx-3090-normal-inference.md)
- [RTX 3090 Qwen3.6-35B-A3B](rtx-3090-35b-a3b.md)

## Maintainer references

The files under [`maintainer/`](maintainer/) record the current artifact formats, exact model and
artifact contracts, and Op-development rules used for ongoing project maintenance. They are not
additional user workflows or installed API documentation.
