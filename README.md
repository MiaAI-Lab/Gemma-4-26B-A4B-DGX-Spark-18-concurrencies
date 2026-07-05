# Gemma 4 26B vLLM Server

This project starts and stops a Dockerized vLLM OpenAI-compatible API server for `nvidia/Gemma-4-26B-A4B-NVFP4`.

The included `start.sh` script runs vLLM with the local `gemma4.py` patch mounted into the container, exposes the API on port `8888`, and stores runtime state in local cache, log, and pid files. The `stop.sh` script stops and removes the vLLM container and cleans up the pid file while preserving the log.

## Capacity

The server is configured for up to 18 concurrent sessions. With MTP/speculative decoding enabled, cumulative throughput can reach up to 300 tokens per second, depending on prompt size, generation length, GPU availability, cache state, and runtime settings.

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
