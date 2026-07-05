#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="nvidia/Gemma-4-26B-A4B-NVFP4"
ASSISTANT_MODEL_ID="google/gemma-4-26B-A4B-it-assistant"
IMAGE="vllm/vllm-openai:nightly"
CONTAINER_NAME="gemma-4-26b-nvfp4-vllm"

HOST="0.0.0.0"
PORT="8888"

PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"

GPU_DEVICES="${GPU_DEVICES:-0}"

ENABLE_MTP="${ENABLE_MTP:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"

# Safer MTP config. Do not use google/gemma-4-26B-A4B-it as the draft model.
SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"model\":\"${ASSISTANT_MODEL_ID}\",\"num_speculative_tokens\":1}"

RUN_USER="${SUDO_USER:-$(id -un)}"
USER_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6 || true)"

HF_HOME_HOST="${USER_HOME:-${HOME}}/.cache/huggingface"
HF_HOME_CONTAINER="/root/.cache/huggingface"

CACHE_ROOT="${WORK_DIR}/.cache"
TRITON_CACHE_HOST="${CACHE_ROOT}/triton"
TORCHINDUCTOR_CACHE_HOST="${CACHE_ROOT}/torchinductor"
XDG_CACHE_HOST="${CACHE_ROOT}/xdg"
VLLM_CACHE_HOST="${XDG_CACHE_HOST}/vllm"

TRITON_CACHE_CONTAINER="/tmp/triton-cache"
TORCHINDUCTOR_CACHE_CONTAINER="/tmp/torchinductor-cache"
XDG_CACHE_CONTAINER="/tmp/.cache"
VLLM_CACHE_CONTAINER="${XDG_CACHE_CONTAINER}/vllm"

READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

if [[ ! -f "${WORK_DIR}/gemma4.py" ]]; then
  echo "Missing patched gemma4.py in ${WORK_DIR}"
  exit 1
fi

mkdir -p \
  "${HF_HOME_HOST}" \
  "${TRITON_CACHE_HOST}" \
  "${TORCHINDUCTOR_CACHE_HOST}" \
  "${XDG_CACHE_HOST}" \
  "${VLLM_CACHE_HOST}"

# Triton/TorchInductor failed before because the container could not write /root/.triton.
# Keep these cache dirs permissive to avoid host/container UID mismatches.
chmod -R 777 "${CACHE_ROOT}" >/dev/null 2>&1 || true

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"
echo "MTP/speculative decoding enabled: ${ENABLE_MTP}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container
EOF

VLLM_ARGS=(
  "${MODEL_ID}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --trust-remote-code
  --quantization modelopt
  # DGX Spark-safe starting point.
  # Increase after it boots cleanly.
  --gpu-memory-utilization 0.70
  --max-model-len 262144
  --max-num-seqs 18
  --max-num-batched-tokens 8192
  --kv-cache-dtype fp8
  --enable-chunked-prefill
  --enable-prefix-caching
  --attention-backend triton_attn
  --load-format fastsafetensors
  --reasoning-parser gemma4
  --tool-call-parser gemma4
  --enable-auto-tool-choice
  --default-chat-template-kwargs "{\"enable_thinking\":${ENABLE_THINKING}}"
  --limit-mm-per-prompt '{"image": 1}'
  --allowed-media-domains '*'
  --mm-encoder-tp-mode weights
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
)

if [[ "${ENABLE_MTP}" == "1" ]]; then
  VLLM_ARGS+=(
    --speculative-config "${SPECULATIVE_CONFIG}"
  )
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus "device=${GPU_DEVICES}" \
  -e VLLM_TARGET_DEVICE=cuda \
  -e HF_HOME="${HF_HOME_CONTAINER}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TRITON_CACHE_DIR="${TRITON_CACHE_CONTAINER}" \
  -e TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_CONTAINER}" \
  -e XDG_CACHE_HOME="${XDG_CACHE_CONTAINER}" \
  -e VLLM_CACHE_ROOT="${VLLM_CACHE_CONTAINER}" \
  -v "${HF_HOME_HOST}:${HF_HOME_CONTAINER}" \
  -v "${TRITON_CACHE_HOST}:${TRITON_CACHE_CONTAINER}" \
  -v "${TORCHINDUCTOR_CACHE_HOST}:${TORCHINDUCTOR_CACHE_CONTAINER}" \
  -v "${XDG_CACHE_HOST}:${XDG_CACHE_CONTAINER}" \
  -v "${WORK_DIR}/gemma4.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py:ro" \
  -v "${WORK_DIR}:/workspace" \
  "${IMAGE}" \
  "${VLLM_ARGS[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1 &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"

until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Models endpoint: ${READY_URL}"
echo "Chat endpoint: ${CHAT_URL}"
echo "Log: ${LOG_FILE}"
echo "vLLM is ready and responding; shell is now free."
