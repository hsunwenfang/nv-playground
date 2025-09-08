#!/usr/bin/env bash
set -euo pipefail

# End-to-end reproduction of section "9. Reproduction Cheat Sheet":
# 1) Start (or verify) minikube with GPU
# 2) Enable NVIDIA device plugin
# 3) Build app + optional profiler image (in-cluster build cache)
# 4) Deploy tiny chat service
# 5) (Optional) Run profiler Job and collect artifacts

APP_IMAGE="tiny-chat:latest"; PROF_IMAGE="nv-profiler:latest"; JOB_NAME="profiler-job"
RUN_PROFILER=0; NAMESPACE="tiny-chat"; OUT_BASE="repro_artifacts"; SKIP_BUILD=0
BUILD_PROFILER=1; PROF_SAMPLES=2000; PROF_QUICK=1; PROF_OUTPUT_DIR="/outputs"
DRY_RUN_BUILD=0; NO_DEPLOY=0; UNIQUE_TAG=1; REBUILD_BASE=0

usage() {
  echo "Usage: $0 [--skip-build] [--no-profiler] [--run-profiler] \\
    [--app-image <tag>] [--prof-image <tag>] [--prof-samples <n>] [--prof-no-quick] \\
    [--dry-run-build] [--no-deploy] [--no-unique-tag]\n\nEnvironment overrides for large model path:\n  MODEL_NAME=EleutherAI/gpt-j-6B QUANTIZE=8 USE_TORCH=1 BATCH_MAX_SIZE=4 BATCH_TIMEOUT_MS=25\n  (QUANTIZE=4 for 4-bit if bitsandbytes supports it; ensure GPU memory sufficient)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --skip-build) SKIP_BUILD=1; shift ;;
  --no-unique-tag) UNIQUE_TAG=0; shift ;;
  --app-image) APP_IMAGE="$2"; shift 2 ;;
  --prof-image) PROF_IMAGE="$2"; shift 2 ;;
  --no-profiler) BUILD_PROFILER=0; shift ;;
  --run-profiler) RUN_PROFILER=1; shift ;;
  --prof-samples) PROF_SAMPLES="$2"; shift 2 ;;
  --prof-no-quick) PROF_QUICK=0; shift ;;
  --dry-run-build) DRY_RUN_BUILD=1; shift ;;
  --rebuild-base) REBUILD_BASE=1; shift ;;
  --no-deploy) NO_DEPLOY=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

need() { command -v "$1" >/dev/null || { echo "Missing required command: $1"; exit 2; }; }
for c in docker kubectl minikube; do need "$c"; done

# Auto-generate unique runtime image tag if enabled and user left default (helps avoid stale cached :latest)
if (( UNIQUE_TAG )) && [[ "$APP_IMAGE" == "tiny-chat:latest" ]]; then
  TS=$(date +%H%M%S)
  SHORT_SHA=$(sha1sum requirements.txt app.py 2>/dev/null | sha1sum | cut -c1-8)
  APP_IMAGE="tiny-chat:${SHORT_SHA}-${TS}"
  echo "[tag] Using unique image tag: $APP_IMAGE"
fi

echo "[1] Assuming existing minikube cluster is already running (use run-fresh-cluster.sh for full reset)."
if (( ! NO_DEPLOY )); then
  if ! minikube status >/dev/null 2>&1; then
    echo "[ERROR] minikube not running. Start it manually or use run-fresh-cluster.sh" >&2
    exit 1
  fi
else
  echo "[dry-run] Skipping minikube status check due to --no-deploy"
fi

echo "[2] (Skipping NVIDIA device plugin – not required for tiny chat service)"

echo "[3] Pre-cache heavy base image inside minikube (one-time, harmless if repeated)"
minikube cache add tensorflow/tensorflow:2.12.0-gpu || true

echo "[4] Rebuilding images (always) with change detection (use --dry-run-build to skip actual docker build)"
RUNTIME_CHANGED=0
if (( DRY_RUN_BUILD )); then
  echo "[dry-run] Skipping real docker builds"
else
  if (( ! SKIP_BUILD )); then
    eval "$(minikube docker-env)"
    echo "Using in-cluster docker daemon: $(docker info --format '{{.Name}}' 2>/dev/null || echo unknown)"
  fi
fi

CACHE_DIR=".image-cache"; mkdir -p "$CACHE_DIR"
REQ_SHA=$(sha1sum requirements.txt | awk '{print $1}')
BASE_HASH_FILE="$CACHE_DIR/base.hash"
RUNTIME_ID_FILE="$CACHE_DIR/app-image.id"

if (( DRY_RUN_BUILD )); then
  echo "[dry-run] requirements.txt sha1: $REQ_SHA (not triggering base build)"
else
  echo "requirements.txt sha1: $REQ_SHA"
fi

NEED_BASE=1
if [[ -f "$BASE_HASH_FILE" ]] && (( ! REBUILD_BASE )); then
  PREV_BASE_HASH=$(<"$BASE_HASH_FILE")
  if [[ "$PREV_BASE_HASH" == "$REQ_SHA" ]] && { (( DRY_RUN_BUILD )) || docker image inspect tiny-chat:base >/dev/null 2>&1; }; then
    NEED_BASE=0
    echo "[base] Reusing tiny-chat:base (no requirements change)"
  fi
fi
if (( REBUILD_BASE )); then
  echo "[base] Forced rebuild requested (--rebuild-base)"
  NEED_BASE=1
fi
if (( NEED_BASE )); then
  if (( DRY_RUN_BUILD )); then
    echo "[base] (dry-run) Would build tiny-chat:base"
  elif (( ! SKIP_BUILD )); then
    echo "[base] Building tiny-chat:base"
    DOCKER_BUILDKIT=1 docker build -t tiny-chat:base -f Dockerfile.base .
    echo "$REQ_SHA" > "$BASE_HASH_FILE"
  fi
fi

# Always (re)build runtime image when not skipping build (ensures unique tag image exists)
if (( ! DRY_RUN_BUILD )) && (( ! SKIP_BUILD )); then
  echo "[runtime] Building runtime image $APP_IMAGE"
  DOCKER_BUILDKIT=1 docker build -t "$APP_IMAGE" -f Dockerfile.runtime .
fi

if [[ -n "${FORCE_NEW_RUNTIME_ID:-}" ]]; then
  NEW_RUNTIME_ID="$FORCE_NEW_RUNTIME_ID"
  echo "[runtime] Using forced image ID: $NEW_RUNTIME_ID"
else
  if (( DRY_RUN_BUILD )); then
    NEW_RUNTIME_ID="dryrun-$(date +%s)"
  else
    NEW_RUNTIME_ID=$(docker image inspect --format='{{.Id}}' "$APP_IMAGE" 2>/dev/null || echo unknown)
  fi
fi
PREV_RUNTIME_ID="none"; [[ -f "$RUNTIME_ID_FILE" ]] && PREV_RUNTIME_ID=$(<"$RUNTIME_ID_FILE")
if [[ "$NEW_RUNTIME_ID" != "$PREV_RUNTIME_ID" ]]; then
  echo "[runtime] Image ID changed: $PREV_RUNTIME_ID -> $NEW_RUNTIME_ID"
  echo "$NEW_RUNTIME_ID" > "$RUNTIME_ID_FILE"
  RUNTIME_CHANGED=1
else
  echo "[runtime] Image ID unchanged"
fi

# Profiler image change detection
if (( BUILD_PROFILER )) && [[ -f Dockerfile.profiler ]]; then
  PROF_HASH_FILE="$CACHE_DIR/profiler.hash"
  PROF_SRC=(Dockerfile.profiler requirements.txt profile.py app.py)
  HAVE_SRC=()
  for f in "${PROF_SRC[@]}"; do [[ -f "$f" ]] && HAVE_SRC+=("$f"); done
  if ((${#HAVE_SRC[@]})); then
    CURR_PH=$(sha256sum "${HAVE_SRC[@]}" | sha256sum | cut -d' ' -f1)
    PREV_PH=""; [[ -f "$PROF_HASH_FILE" ]] && PREV_PH=$(<"$PROF_HASH_FILE")
    PROF_IMAGE_MISSING=0; (( DRY_RUN_BUILD )) || { docker image inspect "$PROF_IMAGE" >/dev/null 2>&1 || PROF_IMAGE_MISSING=1; }
    if [[ "$CURR_PH" != "$PREV_PH" || $PROF_IMAGE_MISSING -eq 1 ]]; then
      if (( DRY_RUN_BUILD )); then
        echo "[profiler] (dry-run) Would build $PROF_IMAGE"
        echo "$CURR_PH" > "$PROF_HASH_FILE"
      elif (( ! SKIP_BUILD )); then
        echo "[profiler] Building $PROF_IMAGE"
        if DOCKER_BUILDKIT=1 docker build -t "$PROF_IMAGE" -f Dockerfile.profiler .; then
          echo "$CURR_PH" > "$PROF_HASH_FILE"
        else
          echo "[profiler] WARN build failed" >&2
        fi
      fi
    else
      echo "[profiler] No change; reusing image"
    fi
  else
    echo "[profiler] No source files; skipping"
  fi
else
  echo "[profiler] Skipped (disabled or missing Dockerfile.profiler)"
fi

if (( ! DRY_RUN_BUILD )) && (( ! SKIP_BUILD )); then
  eval "$(minikube docker-env -u)"
fi

echo "[5] Deployment phase"

echo "[6] Skipping GPU checks (fresh flow handles GPU init)"

if (( NO_DEPLOY )); then
  echo "[deploy] Skipped due to --no-deploy"
else
  echo "[7] Deploy tiny chat service"
  kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"
  TMP_DEPLOY=$(mktemp)
  cp k8s-deployment.yaml "$TMP_DEPLOY"
  # Inject unique tag (replace only default image occurrence)
  if grep -q 'image: tiny-chat:latest' "$TMP_DEPLOY" && [[ "$APP_IMAGE" != "tiny-chat:latest" ]]; then
    sed -i "s#image: tiny-chat:latest#image: ${APP_IMAGE}#" "$TMP_DEPLOY"
  fi
  # Inject forced torch env vars if missing
  if ! grep -q 'FORCE_TORCH' "$TMP_DEPLOY"; then
    awk '/env:/ && !f {print;print "        - name: FORCE_TORCH\n          value: \"1\"\n        - name: USE_TORCH\n          value: \"1\"\n        - name: ENABLE_BATCHING\n          value: \"1\"\n        - name: QUANTIZE\n          value: \"8\"\n        - name: BATCH_MAX_SIZE\n          value: \"4\"";f=1;next}1' "$TMP_DEPLOY" > "$TMP_DEPLOY.tmp" && mv "$TMP_DEPLOY.tmp" "$TMP_DEPLOY"
  fi
  echo "[deploy] Applying unified manifest (single RS expected)"
  kubectl apply -f "$TMP_DEPLOY"
  rm -f "$TMP_DEPLOY"
  echo "[8] Wait for deployment ready"
  kubectl -n "$NAMESPACE" rollout status deploy/tiny-chat --timeout=180s
  echo "[9] Service endpoints"
  kubectl -n "$NAMESPACE" get svc tiny-chat -o wide || true
  POD=$(kubectl -n "$NAMESPACE" get pods -l app=tiny-chat -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  [[ -n "$POD" ]] && echo "Pod: $POD" || echo "No pod yet"
  if [[ -n "$POD" ]]; then
    echo "[10] Test internal request (kubectl exec)"
    kubectl -n "$NAMESPACE" exec "$POD" -- curl -s http://localhost:8000/healthz || true
  fi
  echo "[11] Port-forward hint: kubectl -n $NAMESPACE port-forward svc/tiny-chat 8000:8000"
fi

if (( RUN_PROFILER )) && (( ! NO_DEPLOY )); then
  echo "[12] Running profiler Job ($JOB_NAME)"
  kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"
  TMP_MANIFEST=$(mktemp)
  echo "Generating profiler Job manifest (samples=$PROF_SAMPLES quick=$PROF_QUICK image=$PROF_IMAGE)"
  {
    echo "apiVersion: batch/v1"
    echo "kind: Job"
    echo "metadata:"; echo "  name: $JOB_NAME"; echo "  namespace: $NAMESPACE"
    echo "spec:"; echo "  backoffLimit: 0"; echo "  template:"; echo "    metadata:"; echo "      labels:"; echo "        job-name: $JOB_NAME"; echo "    spec:"; echo "      restartPolicy: Never"; echo "      containers:"; echo "      - name: profiler"; echo "        image: $PROF_IMAGE"; echo "        imagePullPolicy: IfNotPresent"; echo "        command: [\"python\",\"/app/profile.py\"]"; echo "        args:";
    if (( PROF_QUICK )); then echo "          - --quick"; fi
    echo "          - --samples"; echo "          - $PROF_SAMPLES"; echo "          - --output-dir"; echo "          - $PROF_OUTPUT_DIR";
    echo "        resources:"; echo "          limits:"; echo "            nvidia.com/gpu: 1"; echo "          requests:"; echo "            nvidia.com/gpu: 1";
    echo "        volumeMounts:"; echo "        - name: outputs"; echo "          mountPath: $PROF_OUTPUT_DIR";
    echo "      volumes:"; echo "      - name: outputs"; echo "        emptyDir: {}";
  } > "$TMP_MANIFEST"
  kubectl apply -f "$TMP_MANIFEST"
  rm -f "$TMP_MANIFEST"
  echo "Waiting for profiler Job completion (10m timeout)"
  if ! kubectl -n "$NAMESPACE" wait --for=condition=complete job/$JOB_NAME --timeout=10m; then
    echo "Profiler job failed or timed out" >&2
    kubectl -n "$NAMESPACE" logs job/$JOB_NAME || true
  else
    echo "Profiler job completed. Gathering artifacts."
    PODP=$(kubectl -n "$NAMESPACE" get pods -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "$PODP" ]]; then
      OUT_DIR="${OUT_BASE}/profiler-$(date +%Y%m%d_%H%M%S)"
      mkdir -p "$OUT_DIR"
      kubectl -n "$NAMESPACE" logs "$PODP" > "$OUT_DIR/job.log" 2>&1 || true
      if kubectl -n "$NAMESPACE" exec "$PODP" -- test -d "$PROF_OUTPUT_DIR" 2>/dev/null; then
        kubectl -n "$NAMESPACE" cp "$PODP:$PROF_OUTPUT_DIR" "$OUT_DIR/outputs" || true
      fi
      if [[ -d "$OUT_DIR/outputs" ]]; then
        ( cd "$OUT_DIR" && tar -czf outputs.tar.gz outputs ) || true
      fi
      echo "Profiler artifacts stored in $OUT_DIR"
    fi
  fi
fi

echo "[Done] Tiny chat deployment (and profiler if requested) complete with layered base/runtime build."
