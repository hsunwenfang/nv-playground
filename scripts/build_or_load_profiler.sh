#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="nv-mnist-profiler:latest"
HASH_DIR=".image-cache"
HASH_FILE="${HASH_DIR}/profiler.hash"
FILES=(Dockerfile.profiler requirements.txt profile.py app.py)

log() { echo "[build-or-load] $*"; }

missing=()
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || missing+=("$f")
done
if ((${#missing[@]})); then
  log "WARNING: missing files skipped in hash: ${missing[*]}"
fi

calc_hash() {
  sha256sum "${FILES[@]}" 2>/dev/null | sha256sum | cut -d' ' -f1
}

current_hash=$(calc_hash || echo "none")
mkdir -p "$HASH_DIR"
previous_hash=""
[[ -f "$HASH_FILE" ]] && previous_hash=$(<"$HASH_FILE")

need_build=0
if [[ "$current_hash" != "$previous_hash" ]]; then
  log "Hash changed (old=${previous_hash:-<none>} new=$current_hash) -> will build"
  need_build=1
else
  if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    log "Hash same but local image missing -> will build"
    need_build=1
  fi
fi

if ((need_build)); then
  log "Building $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" -f Dockerfile.profiler .
  echo "$current_hash" > "$HASH_FILE"
else
  log "Skipping build (no changes)"
fi

# Decide if we must load into minikube
need_load=0
if ((need_build)); then
  need_load=1
else
  # Try to list images inside minikube; fallback to always load if command fails
  if ! minikube image ls | grep -q "nv-mnist-profiler"; then
    log "Image not present in minikube -> will load"
    need_load=1
  fi
fi

if ((need_load)); then
  log "Loading image into minikube"
  minikube image load "$IMAGE_NAME"
else
  log "Skipping minikube image load"
fi

log "Done"
