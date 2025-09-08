#!/usr/bin/env bash
set -euo pipefail
# Standalone utility to submit a one-off GPU profiling Job (training) OR deploy a continuous
# metrics-only monitor (no GPU resource request) using the existing profile.py script.
# Extracted from run-full-repro.sh profiling block.
#
# Modes:
#  1) Job (default)  : Runs training profile (optionally --quick) and exits
#  2) Monitor Deploy : Continuous nvidia-smi / dmon / (optional) DCGM sampling without reserving GPU
#
# Examples:
#  Run training profiler quick sample job:
#    ./scripts/run-profiler-job.sh --namespace tiny-chat --image nv-profiler:latest --samples 5000 --quick
#  Run full (not quick) profiler job with model samples:
#    ./scripts/run-profiler-job.sh --namespace tiny-chat --image nv-profiler:latest --no-quick --samples 20000
#  Deploy continuous metrics-only monitor (Deployment) w/out GPU reservation:
#    ./scripts/run-profiler-job.sh --namespace tiny-chat --monitor --monitor-image nv-profiler:latest \
#       --sample-interval 5 --enable-dmon
#  Deploy metrics monitor with DCGM fields (requires dcgmi in image & DCGM host engine):
#    ./scripts/run-profiler-job.sh --namespace tiny-chat --monitor --enable-dcgm --dcgm-fields 100,101,150
#
# Cleanup:
#  kubectl -n <ns> delete job <job-name>
#  kubectl -n <ns> delete deploy gpu-profiler-monitor

NS=tiny-chat
JOB_NAME=profiler-job
IMAGE="nv-profiler:latest"
MONITOR_IMAGE="nv-profiler:latest"
OUTPUT_DIR="/outputs"
SAMPLES=2000
QUICK=1
RUN_MONITOR=0
REQUEST_GPU=1
DRY_RUN=0
RESTART_POLICY=Never
MONITOR_INTERVAL=5
MONITOR_DURATION=0   # 0 = infinite (until pod deleted)
ENABLE_DMON=0
ENABLE_DCGM=0
DCGM_FIELDS=""
DMON_SELECT="pucvmet"
EXTRA_ARGS=()
JOB_BACKOFF=0
GPU_LIMIT=1
MODEL_SIZES="small"
BATCH_SIZES="32,64,128"
EPOCHS=1
PREFETCH=1
PIPELINE=standard

usage(){
  cat <<EOF
Usage: $0 [--namespace <ns>] [--image <profiler-img>] [--samples N] [--quick|--no-quick] \
          [--job-name name] [--output-dir /path] [--no-gpu] [--gpu-limit N] \
          [--monitor] [--monitor-image <img>] [--sample-interval SEC] [--monitor-duration SEC] \
          [--enable-dmon] [--dmon-select flags] [--enable-dcgm] [--dcgm-fields ids] \
          [--batch-sizes list] [--model-sizes list] [--epochs N] [--prefetch N] [--data-pipeline standard|optimized|parallel] \
          [--dry-run] [--extra-arg <arg> ...]

Modes:
  (default) Job: one-off training profile (requests GPU unless --no-gpu)
  --monitor        Deploy continuous metrics-only Deployment (no GPU by default)

Examples:
  $0 --namespace tiny-chat --image nv-profiler:latest --samples 5000 --quick
  $0 --namespace tiny-chat --monitor --enable-dmon --sample-interval 10
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) NS="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --monitor-image) MONITOR_IMAGE="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --no-quick) QUICK=0; shift ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --monitor) RUN_MONITOR=1; shift ;;
    --no-gpu) REQUEST_GPU=0; shift ;;
    --gpu-limit) GPU_LIMIT="$2"; shift 2 ;;
    --sample-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
    --monitor-duration) MONITOR_DURATION="$2"; shift 2 ;;
    --enable-dmon) ENABLE_DMON=1; shift ;;
    --enable-dcgm) ENABLE_DCGM=1; shift ;;
    --dcgm-fields) DCGM_FIELDS="$2"; shift 2 ;;
    --dmon-select) DMON_SELECT="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --model-sizes) MODEL_SIZES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --prefetch) PREFETCH="$2"; shift 2 ;;
    --data-pipeline) PIPELINE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --extra-arg) [[ $# -lt 2 ]] && usage; EXTRA_ARGS+=("$2"); shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

need(){ command -v "$1" >/dev/null || { echo "Missing required command: $1" >&2; exit 2; }; }
for c in kubectl; do need "$c"; done

kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"

if (( RUN_MONITOR )); then
  echo "[monitor] Deploying continuous metrics Deployment in namespace $NS"
  NAME=gpu-profiler-monitor
  TMP=$(mktemp)
  cat > "$TMP" <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $NAME
  namespace: $NS
spec:
  replicas: 1
  selector:
    matchLabels:
      app: $NAME
  template:
    metadata:
      labels:
        app: $NAME
    spec:
      restartPolicy: Always
      containers:
      - name: profiler-monitor
        image: $MONITOR_IMAGE
        imagePullPolicy: IfNotPresent
        command: ["python","/app/profile.py"]
        args:
          - "--metrics-only"
          - "--sample-interval"
          - "$MONITOR_INTERVAL"
          - "--duration"
          - "$MONITOR_DURATION"
          - "--enable-dmon"
          - "--dmon-select"
          - "$DMON_SELECT"
$( (( ENABLE_DCGM )) && echo '          - "--enable-dcgm"' )
$( [[ -n "$DCGM_FIELDS" ]] && echo '          - "--dcgm-fields"' )
$( [[ -n "$DCGM_FIELDS" ]] && printf '          - "%s"\n' "$DCGM_FIELDS" )
          - "--output-dir"
          - "/tmp"
        # No GPU resources requested; relies on node driver visibility for /dev/nvidia*
        resources: {}
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
EOF
  if (( DRY_RUN )); then
    echo "[dry-run] Would apply Deployment:"; cat "$TMP"
  else
    kubectl apply -f "$TMP"
    echo "[monitor] Deployment applied. Logs: kubectl -n $NS logs deploy/$NAME -f"
  fi
  rm -f "$TMP"
  exit 0
fi

# Build profiler job manifest
echo "[job] Creating profiler job manifest (samples=$SAMPLES quick=$QUICK gpu=$REQUEST_GPU)"
TMP=$(mktemp)
cat > "$TMP" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  namespace: $NS
spec:
  backoffLimit: $JOB_BACKOFF
  template:
    metadata:
      labels:
        job-name: $JOB_NAME
    spec:
      restartPolicy: $RESTART_POLICY
      containers:
      - name: profiler
        image: $IMAGE
        imagePullPolicy: IfNotPresent
        command: ["python","/app/profile.py"]
  args:
EOF

if (( QUICK )); then echo '          - "--quick"' >> "$TMP"; fi
cat >> "$TMP" <<EOF
          - "--samples"
          - "$SAMPLES"
          - "--model-sizes"
          - "$MODEL_SIZES"
          - "--batch-sizes"
          - "$BATCH_SIZES"
          - "--epochs"
          - "$EPOCHS"
          - "--prefetch-values"
          - "$PREFETCH"
          - "--data-pipelines"
          - "$PIPELINE"
          - "--output-dir"
          - "$OUTPUT_DIR"
EOF
for a in "${EXTRA_ARGS[@]}"; do
  printf '          - %q\n' "$a" >> "$TMP"
done

if (( REQUEST_GPU )); then
cat >> "$TMP" <<EOF
        resources:
          limits:
            nvidia.com/gpu: $GPU_LIMIT
          requests:
            nvidia.com/gpu: $GPU_LIMIT
EOF
else
  echo '        resources: {}' >> "$TMP"
fi

cat >> "$TMP" <<EOF
        volumeMounts:
        - name: outputs
          mountPath: $OUTPUT_DIR
      volumes:
      - name: outputs
        emptyDir: {}
EOF

if (( DRY_RUN )); then
  echo "[dry-run] Would apply Job:"; cat "$TMP"
else
  kubectl apply -f "$TMP"
  echo "[job] Applied. Waiting for completion..."
  set +e
  kubectl -n "$NS" wait --for=condition=complete job/"$JOB_NAME" --timeout=30m
  STATUS=$?
  set -e
  if (( STATUS != 0 )); then
    echo "[job] Job did not complete successfully (status=$STATUS). Logs:" >&2
    kubectl -n "$NS" logs job/"$JOB_NAME" || true
    exit $STATUS
  else
    echo "[job] Completed. Retrieve logs: kubectl -n $NS logs job/$JOB_NAME"
    POD=$(kubectl -n "$NS" get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
    if [[ -n "$POD" ]]; then
      echo "[job] Copying output dir $OUTPUT_DIR from pod $POD to local ./profiler_outputs"
      mkdir -p profiler_outputs
      kubectl -n "$NS" cp "$POD:$OUTPUT_DIR" "profiler_outputs/${JOB_NAME}-$(date +%Y%m%d_%H%M%S)" || true
    fi
  fi
fi
rm -f "$TMP"
