

https://www.databasemart.com/blog/ollama-gpu-benchmark-v100

This repo contains a minimal TensorFlow GPU demo that trains/evaluates a tiny MNIST model.

Files added:
- `app.py` - minimal TF app that prints available GPUs and runs a short train/eval
- `Dockerfile` - uses `tensorflow/tensorflow:2.12.0-gpu` base image
- `requirements.txt` - tiny requirements file
- `k8s-deployment.yaml` - namespace + deployment requesting 1 GPU from `nvidia.com/gpu`

Quick steps (adjust placeholders where noted):

1) Start minikube (example):

```bash
minikube start --driver=docker --memory=8g --cpus=4
```

2) Deploy NVIDIA device plugin so the cluster exposes `nvidia.com/gpu` (necessary for scheduling):

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
```

3) Build and push the image to Azure Container Registry using `az acr build` (or use `--local` deploy mode to load into Minikube):

```bash
AZ_ACR_NAME=<your-acr-name>
AZ_ACR_LOGIN_SERVER="${AZ_ACR_NAME}.azurecr.io"

az acr login -n $AZ_ACR_NAME
az acr build -r $AZ_ACR_NAME -t nv-mnist:latest .

Or to test locally without ACR, run the deploy script with `--local` which will build the image locally and load it into minikube:

```bash
./deploy.sh --local
```
```

4) Create an imagePullSecret in Kubernetes so the cluster can pull from ACR (replace variables as needed):

```bash
ACR_USER=$(az acr credential show -n $AZ_ACR_NAME --query username -o tsv)
ACR_PASS=$(az acr credential show -n $AZ_ACR_NAME --query "passwords[0].value" -o tsv)

kubectl create secret docker-registry acr-secret \
  --docker-server=$AZ_ACR_LOGIN_SERVER \
  --docker-username=$ACR_USER \
  --docker-password=$ACR_PASS \
  --namespace=mnist-gpu-demo
```

5) Edit `k8s-deployment.yaml` and replace `<ACR_LOGIN_SERVER>/nv-mnist:latest` with `$AZ_ACR_LOGIN_SERVER/nv-mnist:latest`, then deploy:

```bash
kubectl apply -f k8s-deployment.yaml
kubectl -n mnist-gpu-demo get pods -w
```

6) Check pod logs for GPU detection and inference output:

```bash
POD=$(kubectl -n mnist-gpu-demo get pods -l app=mnist-gpu -o jsonpath='{.items[0].metadata.name}')
kubectl -n mnist-gpu-demo logs -f $POD
```

Verification inside the running container:

```bash
# Exec into the pod and run a TF GPU check
kubectl -n mnist-gpu-demo exec -it $POD -- python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Notes:
- By default Kubernetes requires integer GPU requests (e.g., `nvidia.com/gpu: 1`). Fractional GPUs require special device plugins or MIG on supported GPUs.
- Minikube GPU support depends on host drivers and the minikube driver. If minikube can't access your GPU, deploy the same manifest on a real k8s cluster with GPU nodes.

## Profiler & Continuous GPU Metrics Monitoring

This repo also provides a comprehensive profiling script `profile.py` and helper artifacts to either:

1. Launch a one-off profiling Job that trains a small model and collects GPU metrics, plots, and a markdown report.
2. Deploy a continuous *metrics-only* monitor (no GPU resource request) that samples `nvidia-smi`, optional `nvidia-smi dmon`, and (if available) DCGM fields.

### One-Off Profiling Job

Use the helper script:

```bash
./scripts/run-profiler-job.sh --namespace mnist-gpu-demo --image nv-profiler:latest --samples 5000 --quick
```

Key options:
- `--samples N`            Number of synthetic samples per run (quick mode uses fewer epochs)
- `--quick | --no-quick`   Toggle reduced training duration
- `--no-gpu`               Run the profiler code without requesting a GPU (will still try to observe metrics if device is visible)
- `--batch-sizes` / `--model-sizes`  Comma lists for comparative profiling

Artifacts are copied locally into `./profiler_outputs/<job-name>-TIMESTAMP` after the job completes.

### Continuous Metrics Monitor Deployment

Deploy a lightweight monitor (no GPU request) that continuously samples GPU metrics:

```bash
./scripts/run-profiler-job.sh --namespace mnist-gpu-demo --monitor --sample-interval 5 --enable-dmon
```

Or apply the provided Deployment manifest directly (edit image as needed):

```bash
kubectl apply -f monitor-profiler-deployment.yaml -n mnist-gpu-demo
```

### Node-wide Monitoring (DaemonSet + Prometheus)

For per-node GPU visibility and Prometheus scraping, deploy the DaemonSet (one pod per GPU node):

```bash
kubectl apply -f monitor-profiler-daemonset.yaml -n mnist-gpu-demo
```

Each pod exposes metrics on `:9400/metrics` when launched with `--prometheus`. The manifest includes standard scrape annotations (`prometheus.io/scrape: "true"`). Adjust your Prometheus config to discover pods in this namespace or rely on annotation-based auto-discovery.

#### Deployment vs DaemonSet vs Job
- Job: point-in-time profiling run producing artifacts; good for deep dive performance analysis.
- Deployment: single long-running collector (cluster-wide aggregate view but may run on a non-GPU node lacking device files depending on scheduling).
- DaemonSet: one collector per node ensures local `/dev/nvidia*` access and granular metrics; preferred for fleet monitoring & Prometheus.

If you only need high-level aggregate metrics, Deployment is simpler. For scalable observability and per-node power/utilization data, use the DaemonSet.

Command line flags exposed to the monitor container (forwarded to `profile.py`):
- `--sample-interval SEC`  Interval between metric samples
- `--duration 0`           0 = run indefinitely (until pod deleted)
- `--enable-dmon`          Include `nvidia-smi dmon` sampling (aggregate per-GPU metrics)
- `--dmon-select pucvmet`  Select columns (power, util, clock, etc.)
- `--enable-dcgm`          Attempt DCGM `dcgmi dmon` parsing (requires DCGM host engine & binary in image)
- `--dcgm-fields 100,101`  Explicit DCGM field IDs (optional)

Logs show rolling JSON lines; you can aggregate externally (e.g., send to Loki or Promtail sidecar). For persistent retention, adapt the Deployment to mount a PVC at `/tmp` or export metrics via a Prometheus sidecar (future enhancement).

### Troubleshooting

- If `nvidia-smi` not found inside the monitor pod, ensure the base image includes NVIDIA drivers utilities or use an image built FROM an NVIDIA CUDA runtime.
- If DCGM sampling fails, the monitor will log a warning and continue with standard sampling.
- To remove the monitor: `kubectl -n mnist-gpu-demo delete deploy/gpu-profiler-monitor`.
 - To remove the DaemonSet: `kubectl -n mnist-gpu-demo delete ds/gpu-profiler-monitor`.

### Security / Permissions

Accessing `/dev/nvidia*` without requesting a GPU relies on cluster policy: some setups require an explicit resource request for device plugin to inject device files. If you don't see devices, you may need to allow privileged or add a minimal GPU request (e.g., for one monitor per node consider a DaemonSet with `resources.limits.nvidia.com/gpu: 1` but using small container). Adjust according to your governance rules.

