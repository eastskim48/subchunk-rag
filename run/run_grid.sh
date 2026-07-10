#!/usr/bin/env bash
set -u

usage() {
    cat <<'EOF'
Usage:
  run/run_grid.sh [OPTIONS] GRID.yaml [GRID2.yaml ...]
  run/run_grid.sh [OPTIONS] -- COMMAND [ARG ...]

Options:
  --gpu ID              Physical GPU id to reserve. Default: 0.
  --wait-gpu           Wait until the GPU has no compute process before running.
  --interval SECONDS   Poll interval for --wait-gpu. Default: 15.
  --wait-lock          Wait for this script's cooperative lock instead of failing fast.
  --allow-busy         Do not abort when the GPU is already busy.
  --no-exclusive       Do not enable NVIDIA EXCLUSIVE_PROCESS mode.
  --lock-dir DIR       Directory for cooperative lock files. Default: /tmp.
  -h, --help           Show this help.

Grid mode:
  Each GRID.yaml is run as:
    python run/grid_search/eval.py GRID.yaml
  Later grids continue even if an earlier grid fails. The final exit code is 1
  if any grid failed.

Command mode:
  Everything after -- is run as a single command under the GPU lock.

Notes:
  This is a cooperative lock. It prevents overlap between jobs that use this
  wrapper. By default it also enables NVIDIA EXCLUSIVE_PROCESS mode for the
  duration of the run, then restores the previous compute mode on exit.
  Use --no-exclusive to skip sudo and compute-mode changes. Use --wait-gpu to
  wait until the GPU is idle before trying to enter the run.
EOF
}

log() {
    printf '[gpu-lock] %s\n' "$*" >&2
}

die() {
    printf '[gpu-lock] error: %s\n' "$*" >&2
    exit 2
}

gpu_compute_pids() {
    nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {print $1}'
}

print_gpu_status() {
    nvidia-smi -i "$GPU_ID" >&2 || true
    nvidia-smi -i "$GPU_ID" \
        --query-compute-apps=pid,process_name,used_memory \
        --format=csv >&2 || true
}

query_compute_mode() {
    nvidia-smi -i "$GPU_ID" --query-gpu=compute_mode --format=csv,noheader,nounits \
        | head -n 1 \
        | tr -d '[:space:]'
}

normalize_compute_mode() {
    case "$1" in
        Default|DEFAULT)
            printf 'DEFAULT'
            ;;
        Exclusive_Process|EXCLUSIVE_PROCESS|ExclusiveProcess)
            printf 'EXCLUSIVE_PROCESS'
            ;;
        Prohibited|PROHIBITED)
            printf 'PROHIBITED'
            ;;
        *)
            return 1
            ;;
    esac
}

wait_for_gpu_idle() {
    while true; do
        local pids
        pids="$(gpu_compute_pids || true)"
        if [ -z "${pids//[[:space:]]/}" ]; then
            log "GPU ${GPU_ID} is idle"
            return 0
        fi
        log "GPU ${GPU_ID} is busy; waiting ${POLL_INTERVAL}s"
        print_gpu_status
        sleep "$POLL_INTERVAL"
    done
}

acquire_lock() {
    mkdir -p "$LOCK_DIR" || die "unable to create lock dir: $LOCK_DIR"
    LOCK_FILE="${LOCK_DIR%/}/subchunk-gpu${GPU_ID}.lock"
    exec 9>"$LOCK_FILE" || die "unable to open lock file: $LOCK_FILE"

    if [ "$WAIT_LOCK" = "1" ]; then
        log "waiting for cooperative lock: $LOCK_FILE"
        flock 9 || die "failed to acquire lock: $LOCK_FILE"
    else
        flock -n 9 || die "GPU ${GPU_ID} lock is already held: $LOCK_FILE"
    fi
    log "acquired cooperative lock: $LOCK_FILE"
}

require_sudo() {
    command -v sudo >/dev/null 2>&1 || die "sudo not found; use --no-exclusive or install sudo"
    log "checking sudo access for NVIDIA compute-mode changes"
    sudo -v || die "sudo authentication failed; use --no-exclusive or fix sudo access"
    start_sudo_keepalive
}

start_sudo_keepalive() {
    if [ -n "${SUDO_KEEPALIVE_PID:-}" ]; then
        return 0
    fi
    (
        while true; do
            sleep 60
            sudo -n -v >/dev/null 2>&1 || exit 0
        done
    ) &
    SUDO_KEEPALIVE_PID="$!"
    log "started sudo keepalive pid=${SUDO_KEEPALIVE_PID}"
}

stop_sudo_keepalive() {
    if [ -z "${SUDO_KEEPALIVE_PID:-}" ]; then
        return 0
    fi
    kill "$SUDO_KEEPALIVE_PID" >/dev/null 2>&1 || true
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    SUDO_KEEPALIVE_PID=""
}

enable_exclusive_process() {
    PREVIOUS_COMPUTE_MODE_RAW="$(query_compute_mode)" || die "failed to query GPU compute mode"
    PREVIOUS_COMPUTE_MODE="$(normalize_compute_mode "$PREVIOUS_COMPUTE_MODE_RAW")" \
        || die "unsupported current GPU compute mode: $PREVIOUS_COMPUTE_MODE_RAW"
    log "current GPU ${GPU_ID} compute mode: $PREVIOUS_COMPUTE_MODE"

    if [ "$PREVIOUS_COMPUTE_MODE" = "EXCLUSIVE_PROCESS" ]; then
        log "GPU ${GPU_ID} is already in EXCLUSIVE_PROCESS mode"
        EXCLUSIVE_ENABLED="0"
        return 0
    fi

    log "setting GPU ${GPU_ID} compute mode to EXCLUSIVE_PROCESS"
    sudo nvidia-smi -i "$GPU_ID" -c EXCLUSIVE_PROCESS \
        || die "failed to set EXCLUSIVE_PROCESS mode; GPU may still be in use"
    EXCLUSIVE_ENABLED="1"
}

restore_compute_mode() {
    if [ "${EXCLUSIVE_ENABLED:-0}" != "1" ]; then
        return 0
    fi
    if [ -z "${PREVIOUS_COMPUTE_MODE:-}" ]; then
        return 0
    fi
    log "restoring GPU ${GPU_ID} compute mode to ${PREVIOUS_COMPUTE_MODE}"
    sudo nvidia-smi -i "$GPU_ID" -c "$PREVIOUS_COMPUTE_MODE" >&2 || true
}

cleanup() {
    restore_compute_mode
    stop_sudo_keepalive
}

run_grid() {
    local grid="$1"
    if [ ! -f "$grid" ]; then
        log "[failed] missing grid file: $grid"
        return 1
    fi

    log "[start] grid=$grid"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python run/grid_search/eval.py "$grid"
    local status=$?
    if [ "$status" -eq 0 ]; then
        log "[done] grid=$grid"
    else
        log "[failed] grid=$grid status=$status"
    fi
    return "$status"
}

GPU_ID="0"
POLL_INTERVAL="15"
WAIT_GPU="0"
WAIT_LOCK="0"
ALLOW_BUSY="0"
EXCLUSIVE_PROCESS="1"
EXCLUSIVE_ENABLED="0"
PREVIOUS_COMPUTE_MODE=""
SUDO_KEEPALIVE_PID=""
LOCK_DIR="/tmp"
COMMAND_MODE="0"
ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            [ "$#" -ge 2 ] || die "--gpu requires an argument"
            GPU_ID="$2"
            shift 2
            ;;
        --interval)
            [ "$#" -ge 2 ] || die "--interval requires an argument"
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --wait-gpu)
            WAIT_GPU="1"
            shift
            ;;
        --wait-lock)
            WAIT_LOCK="1"
            shift
            ;;
        --allow-busy)
            ALLOW_BUSY="1"
            WAIT_GPU="0"
            shift
            ;;
        --no-exclusive)
            EXCLUSIVE_PROCESS="0"
            shift
            ;;
        --lock-dir)
            [ "$#" -ge 2 ] || die "--lock-dir requires an argument"
            LOCK_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            COMMAND_MODE="1"
            shift
            ARGS=("$@")
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

[ "${#ARGS[@]}" -gt 0 ] || {
    usage
    exit 2
}

case "$POLL_INTERVAL" in
    ''|*[!0-9]*)
        die "--interval must be a positive integer"
        ;;
esac
[ "$POLL_INTERVAL" -gt 0 ] || die "--interval must be positive"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
command -v flock >/dev/null 2>&1 || die "flock not found"

if [ "$EXCLUSIVE_PROCESS" = "1" ] && [ "$ALLOW_BUSY" = "1" ]; then
    die "--allow-busy requires --no-exclusive"
fi

if [ "$EXCLUSIVE_PROCESS" = "1" ]; then
    require_sudo
fi

trap cleanup EXIT

acquire_lock

log "initial GPU ${GPU_ID} status"
print_gpu_status

if [ "$WAIT_GPU" = "1" ]; then
    wait_for_gpu_idle
fi

if [ "$EXCLUSIVE_PROCESS" = "1" ]; then
    enable_exclusive_process
fi

if [ "$WAIT_GPU" != "1" ] && [ "$ALLOW_BUSY" != "1" ]; then
    busy_pids="$(gpu_compute_pids || true)"
    if [ -n "${busy_pids//[[:space:]]/}" ]; then
        log "GPU ${GPU_ID} already has compute processes; aborting"
        print_gpu_status
        exit 1
    fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ "$COMMAND_MODE" = "1" ]; then
    log "[start] command=${ARGS[*]}"
    "${ARGS[@]}"
    status=$?
    if [ "$status" -eq 0 ]; then
        log "[done] command=${ARGS[*]}"
    else
        log "[failed] command=${ARGS[*]} status=$status"
    fi
    log "final GPU ${GPU_ID} status"
    print_gpu_status
    exit "$status"
fi

overall_status=0
for grid in "${ARGS[@]}"; do
    run_grid "$grid"
    status=$?
    if [ "$status" -ne 0 ]; then
        overall_status=1
    fi
done

log "final GPU ${GPU_ID} status"
print_gpu_status

if [ "$overall_status" -eq 0 ]; then
    log "all grids completed successfully"
else
    log "one or more grids failed"
fi
exit "$overall_status"
