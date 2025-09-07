# GPU Profiling for TensorFlow MNIST

This directory contains enhanced profiling tools to analyze the performance of the TensorFlow MNIST application with GPU acceleration.

## Available Profiling Tools

### 1. Enhanced `profile.py` Script

The main profiling script `profile.py` is a comprehensive tool that:

- Tests various model configurations (batch sizes, model sizes, data pipelines)
- Collects detailed GPU metrics (utilization, memory, temperature, power)
- Analyzes layer-by-layer performance
- Tests inference with different batch sizes
- Generates visualizations of performance metrics
- Produces a detailed markdown report with recommendations

### 2. Kubernetes Deployment Options

There are two deployment YAML files to run profiling in Kubernetes:

- `profiler-deployment.yaml`: Comprehensive profiling with multiple configurations
- `quick-profiler.yaml`: Quick profiling for a fast performance snapshot

## How to Run the Profiler

### Local Execution

You can run the profiler directly on a machine with NVIDIA GPU:

```bash
# Simple run with default settings
python profile.py

# Quick profile
python profile.py --quick

# Custom profile with specific batch sizes
python profile.py --batch-sizes 64,128,256 --epochs 1

# Test mixed precision
python profile.py --test-mixed-precision

# Comprehensive profile
python profile.py --comprehensive
```

### Kubernetes Deployment

To run in Kubernetes with minikube:

```bash
# Ensure minikube is running with GPU support
minikube start --driver=docker --memory=8g --cpus=4 --addons=nvidia-device-plugin

# Build and load the Docker image
docker build -t nv-mnist:latest .
minikube image load nv-mnist:latest

# Create namespace if needed
kubectl create namespace mnist-gpu-demo

# Run quick profiling
kubectl apply -f quick-profiler.yaml

# Or run comprehensive profiling (takes longer)
kubectl apply -f profiler-deployment.yaml

# Follow logs
kubectl -n mnist-gpu-demo logs -f $(kubectl -n mnist-gpu-demo get pods -l app=mnist-quick-profiler -o jsonpath='{.items[0].metadata.name}')
```

## Interpreting Results

The profiler generates multiple outputs:

1. **JSON log entries**: Structured log entries with performance metrics
2. **Performance summary table**: Shows images/sec, GPU utilization, and memory usage
3. **Visualization plots**: PNG files showing GPU utilization and memory over time
4. **Markdown report**: Comprehensive analysis and recommendations in `/tmp/profile_report.md`

## Common Bottlenecks to Look For

- **Low GPU utilization (<70%)**: CPU bottleneck or inefficient data pipeline
- **High GPU utilization (>90%)**: GPU compute bound - good!
- **Memory approaching capacity**: Reduce batch size or model complexity
- **Slow scaling with batch size**: Data pipeline bottleneck
- **Poor mixed precision speedup**: Operations not optimized for Tensor Cores

## Exporting Results

To copy the profile report and plots from the Kubernetes pod:

```bash
# Get pod name
POD=$(kubectl -n mnist-gpu-demo get pods -l app=mnist-profiler -o jsonpath='{.items[0].metadata.name}')

# Copy report
kubectl -n mnist-gpu-demo cp $POD:/tmp/profile_report.md ./profile_report.md

# Copy plots (if generated)
kubectl -n mnist-gpu-demo cp $POD:/tmp/profile_plots ./profile_plots
```
