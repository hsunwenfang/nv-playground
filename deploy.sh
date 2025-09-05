#!/usr/bin/env bash
set -euo pipefail

# Simple deploy helper for this demo. Prompts for confirmation before network actions.

ACR_NAME=hsunacr
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"
IMAGE_TAG=nv-mnist:latest
NAMESPACE=mnist-gpu-demo


check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command '$1' not found in PATH."
    return 1
  fi
}

MODE="acr"
show_help() {
  cat <<EOF
usage: $0 [--local]

By default the script builds the image into ACR (requires 'az').
Use --local to build locally and load the image into minikube (no ACR required).
EOF
}

# parse args
if [ "$#" -gt 0 ]; then
  case "$1" in
    --local)
      MODE="local"
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      show_help
      exit 1
      ;;
  esac
fi

echo "Running preflight checks for mode: $MODE"
MISSING=()
if [ "$MODE" = "acr" ]; then
  for cmd in az kubectl minikube; do
    if ! check_cmd "$cmd"; then
      MISSING+=("$cmd")
    fi
  done
else
  # local mode: only require docker, kubectl, minikube
  for cmd in docker kubectl minikube; do
    if ! check_cmd "$cmd"; then
      MISSING+=("$cmd")
    fi
  done
fi

if [ ${#MISSING[@]} -ne 0 ]; then
  echo
  echo "One or more required CLIs are missing: ${MISSING[*]}"
  echo "Install them and ensure they are in your PATH. Helpful links:"
  echo "  az: https://aka.ms/azure-cli-install"
  echo "  kubectl: https://kubernetes.io/docs/tasks/tools/"
  echo "  minikube: https://minikube.sigs.k8s.io/docs/start/"
  echo "  docker: https://docs.docker.com/get-docker/"
  exit 2
fi

if [ "$MODE" = "acr" ]; then
  # Check az login
  if ! az account show >/dev/null 2>&1; then
    echo "ERROR: Azure CLI not logged in. Run 'az login' and ensure you have access to the ACR resource '${ACR_NAME}'."
    exit 3
  fi
fi

# Check kubectl connectivity
if ! kubectl version --short >/dev/null 2>&1; then
  echo "ERROR: kubectl cannot connect to a cluster. Ensure your kubeconfig is correct (e.g. 'minikube start')."
  exit 4
fi

echo "Preflight checks passed."

echo "This script will ${MODE:-acr} deploy the image and apply the k8s manifest which requests 1 GPU."
read -p "Proceed? [y/N] " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

if [ "$MODE" = "acr" ]; then
  echo "Logging into ACR..."
  az acr login -n $ACR_NAME

  echo "Starting ACR build: ${IMAGE_TAG}"
  az acr build -r $ACR_NAME -t ${IMAGE_TAG} .
else
  echo "Building local Docker image: ${IMAGE_TAG}"
  docker build -t ${IMAGE_TAG} .

  echo "Loading image into minikube: ${IMAGE_TAG}"
  minikube image load ${IMAGE_TAG}
fi

echo "Ensure namespace exists"
kubectl get namespace $NAMESPACE >/dev/null 2>&1 || kubectl create namespace $NAMESPACE

if [ "$MODE" = "acr" ]; then
  echo "Creating/Updating imagePullSecret in namespace ${NAMESPACE}"
  ACR_USER=$(az acr credential show -n $ACR_NAME --query username -o tsv)
  ACR_PASS=$(az acr credential show -n $ACR_NAME --query "passwords[0].value" -o tsv)

  kubectl get secret acr-secret -n $NAMESPACE >/dev/null 2>&1 && \
    kubectl delete secret acr-secret -n $NAMESPACE || true

  kubectl create secret docker-registry acr-secret \
    --docker-server=$ACR_LOGIN_SERVER \
    --docker-username=$ACR_USER \
    --docker-password=$ACR_PASS \
    --namespace=$NAMESPACE
else
  echo "Local mode: skipping imagePullSecret creation (using preloaded image)."
fi

# Ensure kubectl context is set to minikube
if command -v minikube &> /dev/null; then
    echo "INFO: Ensuring kubectl context is set to minikube..."
    minikube update-context
fi

echo "Applying k8s manifest"
kubectl apply -f k8s-deployment.yaml

if [ "$MODE" = "local" ]; then
  echo "Patching deployment image to use local ${IMAGE_TAG}"
  kubectl -n $NAMESPACE set image deployment/mnist-gpu mnist-gpu=${IMAGE_TAG}
fi

echo "Done. Watch pods with: kubectl -n ${NAMESPACE} get pods -w"
