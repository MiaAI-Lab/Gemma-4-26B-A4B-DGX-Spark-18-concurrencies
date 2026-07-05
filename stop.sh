#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONTAINER_NAME="gemma-4-26b-nvfp4-vllm"
PORT="8888"
PID_FILE=".vllm.pid"
LOG_PID_FILE=".vllm.log.pid"
LOG_FILE=".vllm.log"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

stop_log_followers() {
  if [[ -f "${LOG_PID_FILE}" ]]; then
    log_pid="$(tr -d '[:space:]' < "${LOG_PID_FILE}")"
    if [[ -n "${log_pid}" ]] && kill -0 "${log_pid}" 2>/dev/null; then
      kill "${log_pid}" 2>/dev/null || true
    fi
    rm -f "${LOG_PID_FILE}"
  fi

  pkill -f "docker logs -f ${CONTAINER_NAME}" 2>/dev/null || true
}

resolve_container_ref() {
  if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "${CONTAINER_NAME}"
    return 0
  fi

  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi

  container_id="$(tr -d '[:space:]' < "${PID_FILE}")"
  container_id="${container_id#sha256:}"
  if [[ -z "${container_id}" ]]; then
    return 1
  fi

  docker ps -aq --no-trunc --filter "id=${container_id}" | head -n 1
}

stop_log_followers

container_ref=""
if container_ref="$(resolve_container_ref)"; then
  :
else
  container_ref=""
fi

stopped=0

if [[ -n "${container_ref}" ]]; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Stopping vLLM container ${CONTAINER_NAME}"
    docker stop -t 60 "${container_ref}" >/dev/null 2>&1 || true
  else
    echo "Removing vLLM container ${container_ref}"
  fi

  docker rm -f "${container_ref}" >/dev/null 2>&1 || true
  stopped=1
else
  if [[ -f "${PID_FILE}" ]]; then
    echo "No running container for ID $(tr -d '[:space:]' < "${PID_FILE}")"
  else
    echo "No container found for ${CONTAINER_NAME}"
  fi
fi

rm -f "${PID_FILE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} still present; forcing removal"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  stopped=1
fi

if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "Warning: port ${PORT} is still listening after stop"
  ss -tlnp 2>/dev/null | grep ":${PORT} " || true
fi

if [[ "${stopped}" -eq 1 ]]; then
  echo "Stopped. Log preserved at ${LOG_FILE}"
else
  echo "Nothing to stop. Log preserved at ${LOG_FILE}"
fi