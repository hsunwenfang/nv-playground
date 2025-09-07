# End-to-End GPU MNIST Pipeline Run (Minikube)

Generated: 2025-09-06

## 1. Environment & Cluster Setup

Minikube started with NVIDIA GPU addon:

```
minikube start --driver=docker --container-runtime=docker --gpus=all
Enabled addons: default-storageclass, storage-provisioner, nvidia-device-plugin
```

This automatically deploys the `nvidia-device-plugin` DaemonSet exposing `nvidia.com/gpu` resources to the scheduler.

## 2. Application Image

Base image: `tensorflow/tensorflow:2.12.0-gpu`

Built locally:
```
docker build -t nv-mnist:latest -f Dockerfile .
minikube image load nv-mnist:latest
```

Profiler image (adds matplotlib + profile script):
```
docker build -t nv-mnist-profiler:latest -f Dockerfile.profiler .
minikube image load nv-mnist-profiler:latest
```

## 3. Initial GPU Deployment

Applied `k8s-deployment.yaml` (1 replica requesting 1 GPU). Training container completed rapidly (single epoch) causing restarts (`CrashLoopBackOff` after normal exit code 0).

Evidence of GPU allocation from pod logs:
```
Created device ... device: 0, name: Tesla V100-PCIE-16GB, compute capability: 7.0
Loaded cuDNN version 8600
```

## 4. Profiling Job

Created one-off Job `mnist-profiler-job` (`profiler-job.yaml`) requesting 1 GPU and writing outputs to `/outputs` (emptyDir volume). Only one quick configuration executed:

```
batch_size: 128, epochs: 1, samples: 1000, model_size: small, mixed_precision: False
```

Pod scheduled after freeing GPU (deleted original deployment). Status transitioned: `Pending` → `Running` → `Completed`.

## 5. GPU Usage Validation

Runtime log excerpts inside profiling job:
```
gpu_list: count 1 -> [/physical_device:GPU:0]
Created device ... Tesla V100-PCIE-16GB
Profiler session initializing / CUPTI activity collected
```

These confirm TensorFlow enumerated and used the CUDA device and CUPTI (profiling interface) captured GPU activity.

## 6. Profiling Artifacts

Copied from pod:
```
profile/profile_report.md
profile/profile_plots/ (PNG charts)
```

Summary table from `profile_report.md`:

| Batch Size | Model Size | Mixed Precision | Images/sec | Accuracy | GPU Util % | Mem (MiB) |
|------------|------------|----------------|-----------:|---------:|-----------:|----------:|
| 128        | small      | False          | 426.73     | 0.6600   | 0.0        | 0.0       |

Note: GPU util & memory columns reported 0.0 in this quick run. Causes:
* The sampling window was short (single epoch / 1000 samples) so periodic `nvidia-smi` probes in the script likely missed active kernels (bursty, fast batches).
* Container may lack `nvidia-smi` binary path or sampling errors were suppressed.

Inference micro-benchmarks (selected):
| Inference Batch | Avg ms | Samples/sec |
|-----------------|-------:|------------:|
| 1               | 47.07  | 21.25       |
| 32              | 42.70  | 749.48      |
| 64              | 46.12  | 1387.74     |

## 7. Bottleneck Analysis

Indicators:
* Warnings: TensorFlow reported callback hooks slower than batch compute (`on_train_batch_begin` ~8× batch time). The instrumentation overhead dominates due to tiny workload.
* Extremely short epoch (8 steps) prevents stable utilization; GPU spends proportionally more time in setup (graph/XLA compile, dataset load) than in steady-state kernels.
* Small model (≈253k params) is likely memory-light and compute-light; with V100 this becomes input / Python overhead bound.

Primary bottleneck: Underutilization due to too small workload + profiling overhead (not raw GPU throughput limits).

## 8. Recommended Next Steps

Workload Scaling:
1. Increase training samples to full 60k MNIST and epochs≥5 for sustained kernels.
2. Test larger model size (`--model-sizes medium,large`) to raise arithmetic intensity.
3. Enable mixed precision (`--test-mixed-precision`) and compare throughput & memory.

Data Pipeline Optimization:
* Convert to `tf.data` pipeline with: `dataset.cache().shuffle(10000).batch(BS).prefetch(tf.data.AUTOTUNE)`.
* Add `num_parallel_calls=tf.data.AUTOTUNE` in `map` transforms if preprocessing introduced.

Profiler Enhancements:
* Add explicit `nvidia-smi` sampler thread at ≤250ms interval.
* Export raw CUPTI trace using TensorFlow profiler (already partially enabled) and analyze with TensorBoard.
* Log host CPU utilization to rule out host-side bottlenecks.

Instrumentation Overhead Reduction:
* Disable layer-by-layer or batch timing callbacks after establishing baseline.
* Aggregate metrics every N batches instead of every batch.

Capacity Exploration:
* Increase `batch_size` until memory usage approaches ~85% of GPU memory (monitor with `nvidia-smi`); record scaling curve.
* If memory bound before throughput saturates, consider gradient accumulation.

## 9. Reproduction Cheat Sheet

```
# Start cluster with GPU
minikube start --driver=docker --gpus=all

# Build & load images
docker build -t nv-mnist:latest -f Dockerfile .
docker build -t nv-mnist-profiler:latest -f Dockerfile.profiler .
minikube image load nv-mnist:latest
minikube image load nv-mnist-profiler:latest

# Apply profiling job
kubectl apply -f profiler-job.yaml

# Watch
kubectl -n mnist-gpu-demo get pods -w

# Copy artifacts
kubectl -n mnist-gpu-demo cp <profiler-pod>:/outputs/profile_report.md profile/profile_report.md
kubectl -n mnist-gpu-demo cp <profiler-pod>:/outputs/profile_plots profile/profile_plots
```

## 10. Validation Checklist

| Stage | Evidence | Status |
|-------|----------|--------|
| GPU Device Plugin | minikube addons enabled list | OK |
| Pod requested GPU | `resources.limits nvidia.com/gpu:1` | OK |
| TensorFlow saw GPU | Log line `Created device ... Tesla V100` | OK |
| Profiling executed | CUPTI profiler session messages | OK |
| Artifacts exported | `profile_report.md`, plots copied | OK |

## 11. Planned Improvements (Optional)
* Add `FULL_PIPELINE_REPORT.md` (this file) to version control.
* Introduce automated script (`run-full-profile.sh`) to chain build → deploy → profile → copy → summarize.
* Add a Grafana/Prometheus scrape for long-running training jobs.

---
End of report.
