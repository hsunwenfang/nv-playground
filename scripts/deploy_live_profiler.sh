#!/usr/bin/env bash
set -euo pipefail

NS="mnist-gpu-demo"
DEPLOY="mnist-live-profiler"

info(){ echo "[live-profiler] $*"; }

info "Ensure namespace"
kubectl get ns "$NS" >/dev/null 2>&1 || kubectl create ns "$NS"

info "Conditional build/load"
"$(dirname "$0")/build_or_load_profiler.sh"

info "Apply deployment"
kubectl apply -f live-profiler-deployment.yaml

info "Wait for pod Ready (timeout 600s)"
SECS=0
while (( SECS < 600 )); do
  ready=$(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
  if [[ "$ready" == "1" ]]; then
    info "Deployment ready"
    break
  fi
  sleep 5; SECS=$((SECS+5))
DoneLoop=$SECS
done

info "Current pods:" 
kubectl -n "$NS" get pods -l app="$DEPLOY" -o wide

cat <<EOF
To stream logs:
  kubectl -n $NS logs -f deploy/$DEPLOY

To verify still running after profile completes (look for PROFILE_COMPLETE flag):
  kubectl -n $NS exec \$(kubectl -n $NS get pod -l app=$DEPLOY -o jsonpath='{.items[0].metadata.name}') -- ls -1 /outputs

To copy report:
  POD=\$(kubectl -n $NS get pod -l app=$DEPLOY -o jsonpath='{.items[0].metadata.name}')
  kubectl -n $NS cp $POD:/outputs/profile_report.md profile/profile_report_live.md
EOF
