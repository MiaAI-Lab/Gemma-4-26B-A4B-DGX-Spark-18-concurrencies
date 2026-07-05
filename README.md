# Gemma-4-26B-A4B Super Easy vLLM recipe | DGX Spark

This project starts and stops a Dockerized vLLM OpenAI-compatible API server for `nvidia/Gemma-4-26B-A4B-NVFP4`.

The included `start.sh` script runs vLLM with the local `gemma4.py` patch mounted into the container, exposes the API on port `8888`, and stores runtime state in local cache, log, and pid files. The `stop.sh` script stops and removes the vLLM container and cleans up the pid file while preserving the log.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>


## Capacity

The server is configured for up to 18 concurrent sessions. With MTP/speculative decoding enabled, measured cumulative throughput scales with concurrency as shown below.

Benchmarks were run with `./tok.py` against a live server on `spark2`, with `max_tokens: 1500` per request (each session allowed up to 1500 generated tokens). Each row runs *N* concurrent sessions completing one generation each; **cumulative tok/s** is total generated tokens divided by wall-clock time for the full batch.

| Concurrent sessions | Wall clock (s) | Cumulative gen tok/s | Cumulative e2e tok/s | Avg gen tok/s / session | Avg e2e tok/s / session |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.95 | 29.77 | 30.87 | 29.77 | 30.87 |
| 2 | 25.02 | 62.72 | 65.19 | 32.19 | 33.46 |
| 4 | 29.27 | 108.91 | 113.15 | 28.39 | 29.49 |
| 8 | 36.29 | 181.97 | 188.80 | 24.50 | 25.42 |
| 18 | 53.84 | 271.55 | 281.91 | 16.18 | 16.80 |

At 18 concurrent sessions, cumulative end-to-end throughput reaches **~282 tok/s**. Per-session latency increases under load, but total throughput continues to climb. Actual results will vary with prompt size, generation length, GPU availability, cache state, and runtime settings.

## Runtime Configuration

`start.sh` launches `vllm/vllm-openai:nightly` with these model and serving settings:

| Setting | Value |
| --- | --- |
| Primary model | `nvidia/Gemma-4-26B-A4B-NVFP4` |
| MTP assistant model | `google/gemma-4-26B-A4B-it-assistant` |
| Container name | `gemma-4-26b-nvfp4-vllm` |
| Host / port | `0.0.0.0:8888` |
| Tensor parallel size | `1` |
| Quantization | `modelopt` |
| GPU memory utilization | `0.70` |
| Max context length | `262144` tokens |
| Max concurrent sequences | `18` |
| Max batched tokens | `8192` |
| KV cache dtype | `fp8` |
| Attention backend | `triton_attn` |
| Load format | `fastsafetensors` |
| Chunked prefill | enabled |
| Prefix caching | enabled |
| Trust remote code | enabled |
| Reasoning parser | `gemma4` |
| Tool call parser | `gemma4` |
| Auto tool choice | enabled |
| Thinking | enabled by default |
| MTP/speculative decoding | enabled by default |
| Speculative tokens | `1` |
| Images per prompt | `1` |
| Multimodal encoder TP mode | `weights` |

Default generation config:

```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0
}
```

## Requirements

- Docker
- NVIDIA GPU support for Docker
- `curl`
- Hugging Face access to the configured model
- Optional: `HF_TOKEN` exported in the shell if the model download requires authentication

## Start

```bash
./start.sh
```

When ready, the server listens at:

```text
http://0.0.0.0:8888/v1
```

Local health and API endpoints:

```text
http://127.0.0.1:8888/v1/models
http://127.0.0.1:8888/v1/chat/completions
```

Runtime logs are written to `.vllm.log`.

## Stop

```bash
./stop.sh
```

This stops and removes the `gemma-4-26b-nvfp4-vllm` container.

## Configuration

The scripts support these environment variables:

```bash
GPU_DEVICES=0 ./start.sh
ENABLE_MTP=0 ./start.sh
ENABLE_THINKING=0 ./start.sh
HF_TOKEN=... ./start.sh
```

- `GPU_DEVICES`: Docker GPU device selection. Defaults to `0`.
- `ENABLE_MTP`: Enables speculative decoding with the configured assistant model. Defaults to `1`; set to `0` to disable MTP.
- `ENABLE_THINKING`: Passes `enable_thinking` to the chat template. Defaults to `1`; set to `0` to disable thinking.
- `HF_TOKEN`: Hugging Face token passed into the container.

## Git Upload Contents

The repository is intentionally restricted for upload. A normal `git add .` should include only:

- `README.md`
- `.gitignore`
- `start.sh`
- `stop.sh`

Model weights, caches, logs, pid files, patched Python files, templates, and generated runtime artifacts are ignored.
