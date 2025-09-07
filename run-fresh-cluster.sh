#!/usr/bin/env bash
set -euo pipefail

# Fresh cluster workflow: ALWAYS recreate minikube (deletes etcd + images), then build & deploy.
# Intended when you want a totally clean environment.
#
# Steps:
# 1) Delete existing minikube profile if present
# 2) Start new minikube with GPU (or fallback to CPU if requested)
# 3) Pre-cache base image
# 4) Build app (base + runtime) and optional profiler image inside cluster
# 5) Deploy tiny-chat
# 6) (Optional) Run profiler job
#
# Flags kept minimal & explicit for fresh flow.
#
# Usage examples:
#   bash run-fresh-cluster.sh
#   bash run-fresh-cluster.sh --cpus 6 --memory 12g --run-profiler --prof-samples 500
#   bash run-fresh-cluster.sh --fallback-no-gpu

APP_IMAGE="tiny-chat:latest"
PROF_IMAGE="nv-profiler:latest"
NAMESPACE="tiny-chat"
RUN_PROFILER=0
BUILD_PROFILER=1
PROF_SAMPLES=2000
PROF_QUICK=1
PROF_OUTPUT_DIR="/outputs"
START_CPUS=""
START_MEMORY=""
FALLBACK_NO_GPU=0

usage(){
  echo "Usage: $0 [--cpus <n>] [--memory <MB|GB>] [--fallback-no-gpu] [--run-profiler] [--no-profiler] \\
    [--prof-samples <n>] [--prof-no-quick] [--app-image <tag>] [--prof-image <tag>]"; exit 1;
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) START_CPUS="$2"; shift 2 ;;
    --memory) START_MEMORY="$2"; shift 2 ;;
    --fallback-no-gpu) FALLBACK_NO_GPU=1; shift ;;
    --run-profiler) RUN_PROFILER=1; shift ;;
    --no-profiler) BUILD_PROFILER=0; shift ;;
    --prof-samples) PROF_SAMPLES="$2"; shift 2 ;;
    --prof-no-quick) PROF_QUICK=0; shift ;;
    --app-image) APP_IMAGE="$2"; shift 2 ;;
    --prof-image) PROF_IMAGE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

need(){ command -v "$1" >/dev/null || { echo "Missing command: $1" >&2; exit 2; }; }
for c in docker kubectl minikube; do need "$c"; done

echo "[Fresh 1] Deleting any existing minikube profile"
minikube delete || true

echo "[Fresh 2] Starting new minikube (GPU first, optional fallback)"
START_ARGS=(--driver=docker --container-runtime=docker --gpus=all)
[[ -n "$START_CPUS" ]] && START_ARGS+=(--cpus="$START_CPUS")
[[ -n "$START_MEMORY" ]] && START_ARGS+=(--memory="$START_MEMORY")
echo "minikube start ${START_ARGS[*]}"
if ! minikube start "${START_ARGS[@]}"; then
  if (( FALLBACK_NO_GPU )); then
    echo "[WARN] GPU start failed; retrying without GPU" >&2
    START_ARGS_NO_GPU=(--driver=docker --container-runtime=docker)
    [[ -n "$START_CPUS" ]] && START_ARGS_NO_GPU+=(--cpus="$START_CPUS")
    [[ -n "$START_MEMORY" ]] && START_ARGS_NO_GPU+=(--memory="$START_MEMORY")
    echo "minikube start ${START_ARGS_NO_GPU[*]}"
    minikube start "${START_ARGS_NO_GPU[@]}"
  else
    echo "[ERROR] GPU start failed (add --fallback-no-gpu to retry without)" >&2
    exit 1
  fi
fi

echo "[Fresh 3] Pre-caching base TF image"
minikube cache add tensorflow/tensorflow:2.12.0-gpu || true

echo "[Fresh 4] Building images in-cluster"
eval "$(minikube docker-env)"
REQ_SHA=$(sha1sum requirements.txt | awk '{print $1}')
DOCKER_BUILDKIT=1 docker build --build-arg REQ_SHA="$REQ_SHA" -t tiny-chat:base -f Dockerfile.base .
DOCKER_BUILDKIT=1 docker build -t "$APP_IMAGE" -f Dockerfile.runtime .
if (( BUILD_PROFILER )) && [[ -f Dockerfile.profiler ]]; then
  DOCKER_BUILDKIT=1 docker build -t "$PROF_IMAGE" -f Dockerfile.profiler . || echo "[profiler] build failed (continuing)" >&2
fi
eval "$(minikube docker-env -u)"

echo "[Fresh 5] Deploying tiny-chat"
kubectl get ns "$NAMESPACE" >/dev/null 2>&1 || kubectl create ns "$NAMESPACE"
kubectl apply -f k8s-deployment.yaml

echo "[Fresh 6] Waiting for deployment rollout"
if ! kubectl -n "$NAMESPACE" rollout status deploy/tiny-chat --timeout=240s; then
  echo "[ERROR] Deployment failed to become ready" >&2
  kubectl -n "$NAMESPACE" get pods -o wide || true
  exit 1
fi

POD=$(kubectl -n "$NAMESPACE" get pods -l app=tiny-chat -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
kubectl -n "$NAMESPACE" get svc tiny-chat -o wide || true

if (( RUN_PROFILER )); then
  echo "[Fresh 7] Running profiler job"
  TMP=$(mktemp)
  cat > "$TMP" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: profiler-job
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: profiler
        image: $PROF_IMAGE
        imagePullPolicy: IfNotPresent
        command: ["python","/app/profile.py"]
        args:
$( (( PROF_QUICK )) && echo '          - "--quick"' )
          - "--samples"
          - "$PROF_SAMPLES"
          - "--output-dir"
          - "$PROF_OUTPUT_DIR"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: out
          mountPath: $PROF_OUTPUT_DIR
      volumes:
      - name: out
        emptyDir: {}
EOF
  kubectl apply -f "$TMP"
  rm -f "$TMP"
  kubectl -n "$NAMESPACE" wait --for=condition=complete job/profiler-job --timeout=15m || true
fi

echo "[Fresh Done]"
