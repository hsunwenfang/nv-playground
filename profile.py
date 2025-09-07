#!/usr/bin/env python3
"""
Enhanced GPU profiling script for MNIST TensorFlow workload.
- Runs training with different batch sizes and epochs
- Captures detailed GPU metrics during training
- Records execution time and throughput with tensor operation breakdowns
- Analyzes memory usage patterns and allocation tracking
- Tests data pipeline optimizations (prefetch, cache, parallel)
- Visualizes utilization and memory patterns for easy interpretation
- Includes model complexity analysis and layer-by-layer profiling
"""
import os
import sys
import time
import json
import subprocess
import platform
import argparse
import threading
import queue
import math
from datetime import datetime, timezone

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import mnist
    import numpy as np
    # Import profiler utilities if available
    try:
        from tensorflow.profiler import experimental as tf_profiler
        from tensorflow.python.eager import profiler_client
        PROFILER_AVAILABLE = True
    except ImportError:
        PROFILER_AVAILABLE = False
    # Check if we can generate plots
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for container use
        import matplotlib.pyplot as plt
        CAN_PLOT = True
    except ImportError:
        CAN_PLOT = False
    # Optional NVML (pynvml) for direct GPU metrics
    try:
        import pynvml
        pynvml.nvmlInit()
        NVML_AVAILABLE = True
    except Exception:
        NVML_AVAILABLE = False
except Exception as e:
    print(f"Error importing TensorFlow: {e}")
    sys.exit(1)

def log(event, level="INFO", **fields):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event}
    
    # Convert NumPy types to Python native types
    for key, value in fields.items():
        if isinstance(value, (np.integer, np.int64, np.int32)):
            fields[key] = int(value)
        elif isinstance(value, (np.floating, )):
            fields[key] = float(value)
        elif isinstance(value, np.ndarray):
            fields[key] = value.tolist()
    
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False))

def see_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    log("gpu_list", count=len(gpus), gpus=[g.name for g in gpus])
    return len(gpus) > 0

def capture_nvidia_smi(query_options=None):
    """Capture enhanced GPU metrics using nvidia-smi with customizable fields"""
    if query_options is None:
        # Default query includes detailed metrics
        query_options = [
            "index", "name", "temperature.gpu", "utilization.gpu", "utilization.memory",
            "memory.used", "memory.free", "memory.total", "power.draw", "clocks.current.sm",
            "clocks.current.memory"
        ]
    
    query_str = ",".join(query_options)
    
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_str}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            # Parse all returned lines (supports multi-GPU)
            all_gpus = []
            for line in result.stdout.strip().split('\n'):
                values = [x.strip() for x in line.split(',')]
                if len(values) >= len(query_options):
                    gpu_data = {opt: values[i] for i, opt in enumerate(query_options)}
                    # Convert numeric values
                    for key, val in gpu_data.items():
                        try:
                            if '.' in val:
                                gpu_data[key] = float(val)
                            else:
                                gpu_data[key] = int(val)
                        except ValueError:
                            pass  # Keep as string if not convertible
                    all_gpus.append(gpu_data)
            return all_gpus
    except Exception as e:
        log("nvidia_smi_error", error=str(e), level="WARN")
    return None

def capture_nvml():
    """Capture GPU metrics using NVML if available."""
    if not NVML_AVAILABLE:
        return None
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            clocks_sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            clocks_mem = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            power = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = 0.0
            gpus.append({
                "index": i,
                "name": pynvml.nvmlDeviceGetName(h).decode() if isinstance(pynvml.nvmlDeviceGetName(h), bytes) else pynvml.nvmlDeviceGetName(h),
                "temperature.gpu": temp,
                "utilization.gpu": util.gpu,
                "utilization.memory": util.memory,
                "memory.used": int(mem.used / (1024*1024)),
                "memory.free": int(mem.free / (1024*1024)),
                "memory.total": int(mem.total / (1024*1024)),
                "power.draw": power,
                "clocks.current.sm": clocks_sm,
                "clocks.current.memory": clocks_mem
            })
        return gpus
    except Exception as e:
        log("nvml_error", error=str(e), level="WARN")
        return None

def start_gpu_sampler(interval, stop_event, sample_list):
    """Thread target to continuously sample GPU metrics."""
    log("gpu_sampler_start", interval=interval, nvml=NVML_AVAILABLE)
    while not stop_event.is_set():
        metrics = capture_nvml() or capture_nvidia_smi()
        if metrics:
            ts = time.time()
            # unify first GPU only for summary
            first = metrics[0] if isinstance(metrics, list) else metrics
            # Normalize keys to expected names used in plots
            normalized = {
                "ts": ts,
                "gpu_util_percent": first.get("utilization.gpu") if "utilization.gpu" in first else first.get("gpu_util_percent"),
                "mem_used_mib": first.get("memory.used") if "memory.used" in first else first.get("mem_used_mib"),
                "mem_total_mib": first.get("memory.total") if "memory.total" in first else first.get("mem_total_mib"),
                "power_watts": first.get("power.draw"),
                "sm_clock_mhz": first.get("clocks.current.sm"),
                "mem_clock_mhz": first.get("clocks.current.memory")
            }
            sample_list.append(normalized)
        stop_event.wait(interval)
    log("gpu_sampler_stop")

def build_model(model_size='small'):
    """Build model with configurable complexity"""
    if model_size == 'tiny':
        filters1, filters2, dense_units = 8, 16, 32
    elif model_size == 'small':
        filters1, filters2, dense_units = 16, 32, 64
    elif model_size == 'medium':
        filters1, filters2, dense_units = 32, 64, 128
    elif model_size == 'large':
        filters1, filters2, dense_units = 64, 128, 256
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters1, 3, activation='relu', name='conv1'),
        layers.MaxPooling2D(name='pool1'),
        layers.Conv2D(filters2, 3, activation='relu', name='conv2'),
        layers.Flatten(name='flatten'),
        layers.Dense(dense_units, activation='relu', name='dense1'),
        layers.Dense(10, activation='softmax', name='output')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Calculate model complexity metrics
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    
    # Calculate FLOPs for forward pass (rough estimate)
    input_shape = (1, 28, 28, 1)  # Batch size of 1
    flops = estimate_flops(model, input_shape)
    
    log("model_complexity", 
        model_size=model_size,
        total_params=total_params, 
        trainable_params=trainable_params,
        estimated_flops_per_inference=flops,
        filters1=filters1,
        filters2=filters2,
        dense_units=dense_units
    )
    
    return model

def estimate_flops(model, input_shape):
    """Estimate FLOPs for a Keras model (very rough approximation)"""
    # This is a simplified approach and doesn't account for all operations
    flops = 0
    prev_units = np.prod(input_shape[1:])  # Skip batch dimension
    
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # Conv2D FLOPs = 2 * H * W * Cin * Cout * K^2
            kernel_size = layer.kernel_size[0]  # Assuming square kernel
            output_height = prev_units / (layer.input_shape[-1] * layer.strides[0])
            output_width = output_height  # Assuming square inputs/outputs
            flops_layer = 2 * output_height * output_width * layer.input_shape[-1] * layer.filters * kernel_size**2
            flops += flops_layer
            prev_units = layer.output_shape[1] * layer.output_shape[2] * layer.output_shape[3]
        
        elif isinstance(layer, layers.Dense):
            # Dense FLOPs = 2 * input_units * output_units
            flops_layer = 2 * prev_units * layer.units
            flops += flops_layer
            prev_units = layer.units
    
    return int(flops)

def profile_training(batch_size=128, epochs=1, mixed_precision=False, samples=5000, 
                    model_size='small', data_pipeline='standard', prefetch=1, output_dir="/tmp",
                    gpu_sample_interval=0.25, enable_gpu_sampler=True):
    """Run training with enhanced metrics collection and return results"""
    log("profile_training_start", 
        batch_size=batch_size, 
        epochs=epochs, 
        mixed_precision=mixed_precision, 
        samples=samples,
        model_size=model_size,
        data_pipeline=data_pipeline,
        prefetch=prefetch,
        output_dir=output_dir
    )
    
    # Enable memory growth to avoid allocating all GPU memory at once
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            log("memory_growth_enabled")
        except Exception as e:
            log("memory_growth_failed", error=str(e), level="WARN")
    
    # Enable mixed precision if requested
    if mixed_precision:
        try:
            # Import with alias to avoid shadowing the boolean parameter name
            from tensorflow.keras import mixed_precision as tf_mixed_precision
            tf_mixed_precision.set_global_policy('mixed_float16')
            log("mixed_precision_enabled")
        except Exception as e:
            log("mixed_precision_failed", error=str(e), level="WARN")
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Subsample for profiling
    train_x = x_train[:samples]
    train_y = y_train[:samples]
    
    # Build model with specified complexity
    model = build_model(model_size=model_size)
    
    # Create tf.data pipeline with various optimizations based on strategy
    if data_pipeline == 'standard':
        ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(prefetch)
    elif data_pipeline == 'optimized':
        ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        ds = ds.shuffle(buffer_size=1000).cache()
        ds = ds.batch(batch_size).prefetch(prefetch)
    elif data_pipeline == 'parallel':
        ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        ds = ds.shuffle(buffer_size=1000)
        # Use parallel map for preprocessing (not needed for MNIST but demonstrates concept)
        ds = ds.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(prefetch)
    else:
        raise ValueError(f"Unknown data pipeline strategy: {data_pipeline}")
    
    # Prepare metrics collection
    metrics = []
    interval_secs = 0.5  # Collect metrics every 0.5 seconds
    
    # Start GPU sampler thread (before warmup for baseline)
    gpu_samples = []
    sampler_stop = threading.Event()
    sampler_thread = None
    if enable_gpu_sampler:
        try:
            sampler_thread = threading.Thread(target=start_gpu_sampler, args=(gpu_sample_interval, sampler_stop, gpu_samples), daemon=True)
            sampler_thread.start()
        except Exception as e:
            log("gpu_sampler_thread_error", error=str(e), level="WARN")

    # Warmup pass to compile graphs and initialize memory
    log("warmup_start")
    model.predict(x_test[:batch_size])
    log("warmup_complete")
    
    # Training with enhanced metrics collection
    start_time = time.time()
    last_metric_time = start_time
    
    # Collect memory stats before training to establish baseline
    baseline_memory = None
    if len(physical_devices) > 0:
        try:
            baseline_memory = tf.config.experimental.get_memory_info('GPU:0')
            log("baseline_gpu_memory", **baseline_memory)
        except Exception as e:
            log("memory_info_error", error=str(e), level="WARN")
    
    # Setup TF profiler if available
    profile_logdir = '/tmp/tf_profile'
    profiler_enabled = False
    if PROFILER_AVAILABLE:
        try:
            os.makedirs(profile_logdir, exist_ok=True)
            # We'll enable the profiler during specific iterations
            profiler_enabled = True
            log("tf_profiler_ready", logdir=profile_logdir)
        except Exception as e:
            log("tf_profiler_setup_error", error=str(e), level="WARN")
            profiler_enabled = False
            
    # Custom callback for detailed metrics collection
    class EnhancedMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.batch_times = []
            self.profiling_active = False
            self.profile_batch = None
            
        def on_train_begin(self, logs=None):
            self.train_start = time.time()
            
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start = time.time()
            log("epoch_begin", epoch=epoch)
            
        def on_batch_begin(self, batch, logs=None):
            self.batch_start = time.time()
            
            # Enable TF profiler for one specific batch in the middle of training
            if (profiler_enabled and 
                batch == min(10, samples // batch_size // 2) and 
                not self.profiling_active):
                try:
                    log("enabling_tf_profiler", batch=batch)
                    tf_profiler.start(logdir=profile_logdir)
                    self.profiling_active = True
                    self.profile_batch = batch
                except Exception as e:
                    log("profiler_start_error", error=str(e), level="WARN")
            
        def on_batch_end(self, batch, logs=None):
            batch_time = time.time() - self.batch_start
            self.batch_times.append(batch_time)
            
            # Disable TF profiler if it was enabled for this batch
            if self.profiling_active and batch == self.profile_batch:
                try:
                    tf_profiler.stop()
                    self.profiling_active = False
                    log("tf_profiler_complete", batch=batch)
                except Exception as e:
                    log("profiler_stop_error", error=str(e), level="WARN")
            
            nonlocal last_metric_time
            current_time = time.time()
            
            # Collect metrics at regular intervals, not every batch
            if current_time - last_metric_time >= interval_secs:
                # Collect GPU metrics
                gpu_metrics = capture_nvidia_smi()
                
                # Get memory info
                memory_info = None
                if len(physical_devices) > 0:
                    try:
                        memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    except Exception:
                        pass
                
                if gpu_metrics or memory_info:
                    elapsed = current_time - start_time
                    step_metrics = {
                        "elapsed_sec": elapsed,
                        "batch": batch,
                        "batch_time_sec": batch_time,
                        "samples_in_batch": batch_size
                    }
                    
                    # Add GPU metrics from nvidia-smi
                    if gpu_metrics:
                        # Just use the first GPU for simplicity in multi-GPU setups
                        step_metrics["gpu_metrics"] = gpu_metrics[0] if isinstance(gpu_metrics, list) else gpu_metrics
                    
                    # Add TF memory info
                    if memory_info:
                        step_metrics["tf_memory"] = memory_info
                    
                    # Add training metrics
                    if logs:
                        step_metrics.update({k: float(v) for k, v in logs.items()})
                    
                    metrics.append(step_metrics)
                    last_metric_time = current_time
            
        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start
            avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
            
            log("epoch_complete", 
                epoch=epoch, 
                epoch_time_sec=epoch_time,
                avg_batch_time_sec=avg_batch_time,
                batch_time_p50=np.percentile(self.batch_times, 50) if self.batch_times else 0,
                batch_time_p95=np.percentile(self.batch_times, 95) if self.batch_times else 0,
                batch_time_min=min(self.batch_times) if self.batch_times else 0,
                batch_time_max=max(self.batch_times) if self.batch_times else 0,
                **{k: float(v) for k, v in logs.items() if logs}
            )
            self.batch_times = []
    
    # Run training with enhanced metrics
    callback = EnhancedMetricsCallback()
    t0 = time.time()
    history = model.fit(ds, epochs=epochs, verbose=2, callbacks=[callback])
    train_seconds = time.time() - t0

    # Stop GPU sampler
    if enable_gpu_sampler and sampler_thread:
        sampler_stop.set()
        sampler_thread.join(timeout=2)

    # If metrics_series lack gpu util (all zero) but we have sampler data, inject synthetic time series
    has_nonzero = any(m.get("gpu_metrics", {}).get("gpu_util_percent", 0) > 0 for m in metrics)
    if gpu_samples and not has_nonzero:
        start_ts = gpu_samples[0]["ts"]
        for s in gpu_samples:
            metrics.append({
                "elapsed_sec": s["ts"] - start_ts,
                "gpu_metrics": {
                    "gpu_util_percent": s.get("gpu_util_percent", 0),
                    "mem_used_mib": s.get("mem_used_mib", 0),
                    "mem_total_mib": s.get("mem_total_mib", 0)
                }
            })
        log("synthetic_gpu_metrics_injected", count=len(gpu_samples))
    
    # Enhanced evaluation with per-layer profiling
    eval_t0 = time.time()
    
    # Profile per-layer inference time
    layer_times = {}
    if len(physical_devices) > 0:  # Only do layer profiling with GPU
        log("layer_profiling_start")
        # Create a separate model that outputs each layer for analysis
        layer_outputs = [layer.output for layer in model.layers]
        layer_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Run inference with timing for each layer
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'name'):
                layer_name = layer.name
                # Create a model just up to this layer
                temp_outputs = layer_outputs[:i+1]
                temp_model = tf.keras.Model(inputs=model.input, outputs=temp_outputs)
                
                # Time inference
                layer_start = time.time()
                _ = temp_model.predict(x_test[:100], verbose=0)
                layer_end = time.time()
                
                layer_times[layer_name] = {
                    "index": i,
                    "time_ms": (layer_end - layer_start) * 1000,
                    "layer_type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape)
                }
        
        log("layer_profiling_complete", layer_count=len(layer_times))
    
    # Standard evaluation
    loss, acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
    eval_seconds = time.time() - eval_t0
    
    # Run inference benchmark
    inference_batch_sizes = [1, 8, 16, 32, 64]
    inference_benchmarks = []
    
    for inf_batch in inference_batch_sizes:
        # Prepare test batch
        test_batch = x_test[:inf_batch]
        
        # Warmup
        _ = model.predict(test_batch, verbose=0)
        
        # Timed runs
        num_runs = 10
        inf_times = []
        for _ in range(num_runs):
            inf_start = time.time()
            _ = model.predict(test_batch, verbose=0)
            inf_times.append((time.time() - inf_start) * 1000)  # ms
        
        # Calculate stats
        avg_time = sum(inf_times) / len(inf_times)
        inference_benchmarks.append({
            "batch_size": inf_batch,
            "avg_time_ms": avg_time,
            "min_time_ms": min(inf_times),
            "max_time_ms": max(inf_times),
            "p95_time_ms": np.percentile(inf_times, 95),
            "throughput_samples_per_sec": (inf_batch * 1000) / avg_time
        })
    
    log("inference_benchmark_complete", benchmarks=inference_benchmarks)
    
    # Calculate throughput
    images_processed = samples * epochs
    throughput = images_processed / train_seconds
    
    # Generate plots if matplotlib is available
    plot_paths = {}
    if CAN_PLOT:
        plot_paths = generate_profile_plots(metrics, model_name=f"bs{batch_size}_mp{1 if mixed_precision else 0}", output_dir=output_dir)
    
    # Summarize results with enhanced metrics
    result = {
        "batch_size": batch_size,
        "epochs": epochs,
        "mixed_precision": mixed_precision,
        "model_size": model_size,
        "data_pipeline": data_pipeline,
        "prefetch": prefetch,
        "samples": samples,
        "total_images": images_processed,
        "training_seconds": train_seconds,
        "images_per_second": throughput,
        "batches_per_second": throughput / batch_size,
        "eval_seconds": eval_seconds,
        "final_accuracy": float(acc),
        "final_loss": float(loss),
        "layer_profiling": layer_times,
        "inference_benchmarks": inference_benchmarks,
        "plot_paths": plot_paths,
    "metrics_series": metrics,
    "gpu_samples": gpu_samples
    }
    
    log("profile_training_complete", **{k: v for k, v in result.items() 
                                     if k not in ["metrics_series", "layer_profiling", "plot_paths"]})
    return result

def generate_profile_plots(metrics, model_name="model", output_dir="/tmp"):
    """Generate visualization plots from profiling data"""
    plot_paths = {}
    if not CAN_PLOT or not metrics:
        return plot_paths
    
    try:
        # Create output directory
        plot_dir = os.path.join(output_dir, "profile_plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Extract time series data
        times = [m.get("elapsed_sec", 0) for m in metrics]
        gpu_utils = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics]
        mem_utils = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in metrics]
        
        # Plot 1: GPU Utilization over time
        plt.figure(figsize=(10, 6))
        plt.plot(times, gpu_utils, 'b-', linewidth=2)
        plt.title(f"GPU Utilization During Training - {model_name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("GPU Utilization (%)")
        plt.grid(True)
        util_path = f"{plot_dir}/gpu_util_{model_name}.png"
        plt.savefig(util_path)
        plt.close()
        plot_paths["gpu_utilization"] = util_path
        
        # Plot 2: Memory Usage over time
        plt.figure(figsize=(10, 6))
        plt.plot(times, mem_utils, 'r-', linewidth=2)
        plt.title(f"GPU Memory Usage During Training - {model_name}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Used (MiB)")
        plt.grid(True)
        mem_path = f"{plot_dir}/mem_usage_{model_name}.png"
        plt.savefig(mem_path)
        plt.close()
        plot_paths["memory_usage"] = mem_path
        
        # Extract batch times if available
        if "batch_time_sec" in metrics[0]:
            batch_times = [m.get("batch_time_sec", 0) * 1000 for m in metrics]  # Convert to ms
            batch_nums = range(len(batch_times))
            
            # Plot 3: Batch processing time
            plt.figure(figsize=(10, 6))
            plt.plot(batch_nums, batch_times, 'g-', linewidth=1)
            plt.title(f"Batch Processing Time - {model_name}")
            plt.xlabel("Batch Number")
            plt.ylabel("Processing Time (ms)")
            plt.grid(True)
            batch_path = f"{plot_dir}/batch_time_{model_name}.png"
            plt.savefig(batch_path)
            plt.close()
            plot_paths["batch_times"] = batch_path
        
        log("plots_generated", plot_dir=plot_dir, paths=plot_paths)
        return plot_paths
    
    except Exception as e:
        log("plot_generation_error", error=str(e), level="WARN")
        return {}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced GPU Profiling for TensorFlow MNIST')
    parser.add_argument('--batch-sizes', type=str, default='32,64,128,256',
                      help='Comma-separated list of batch sizes to test')
    parser.add_argument('--model-sizes', type=str, default='small',
                      help='Comma-separated list of model sizes to test (tiny, small, medium, large)')
    parser.add_argument('--prefetch-values', type=str, default='1',
                      help='Comma-separated list of prefetch values to test')
    parser.add_argument('--data-pipelines', type=str, default='standard',
                      help='Comma-separated list of data pipeline strategies to test (standard, optimized, parallel)')
    parser.add_argument('--epochs', type=int, default=2,
                      help='Number of epochs for each test')
    parser.add_argument('--samples', type=int, default=5000,
                      help='Number of samples to use for training')
    parser.add_argument('--test-mixed-precision', action='store_true',
                      help='Test mixed precision training')
    parser.add_argument('--quick', action='store_true',
                      help='Run a quick profile with fewer configurations')
    parser.add_argument('--comprehensive', action='store_true',
                      help='Run a comprehensive profile with many configurations')
    parser.add_argument('--output-dir', type=str, default='/tmp',
                      help='Directory to store profile reports and plots')
    parser.add_argument('--gpu-sample-interval', type=float, default=0.25,
                      help='Interval (seconds) for high-frequency GPU sampler thread')
    parser.add_argument('--disable-gpu-sampler', action='store_true',
                      help='Disable high-frequency GPU sampler thread')
    args = parser.parse_args()
    
    # Initial GPU check
    see_gpus()
    
    # Parse configuration parameters
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    model_sizes = args.model_sizes.split(',')
    prefetch_values = [int(x) for x in args.prefetch_values.split(',')]
    data_pipelines = args.data_pipelines.split(',')
    
    # Define profiles to run based on args
    if args.quick:
        # Quick profile with minimal configs
        configs = [
            {"batch_size": 128, "epochs": 1, "mixed_precision": False, "model_size": "small", 
             "data_pipeline": "standard", "prefetch": 1, "samples": min(1000, args.samples)}
        ]
    elif args.comprehensive:
        # Comprehensive profile with many combinations
        configs = []
        for bs in batch_sizes:
            for ms in model_sizes:
                for dp in data_pipelines:
                    for pf in prefetch_values:
                        # Standard precision
                        configs.append({
                            "batch_size": bs, "epochs": args.epochs, "mixed_precision": False,
                            "model_size": ms, "data_pipeline": dp, "prefetch": pf, "samples": args.samples
                        })
                        # Mixed precision if requested
                        if args.test_mixed_precision:
                            configs.append({
                                "batch_size": bs, "epochs": args.epochs, "mixed_precision": True,
                                "model_size": ms, "data_pipeline": dp, "prefetch": pf, "samples": args.samples
                            })
    else:
        # Default profile: batch sizes with default settings
        configs = []
        for bs in batch_sizes:
            configs.append({
                "batch_size": bs, "epochs": args.epochs, "mixed_precision": False,
                "model_size": model_sizes[0], "data_pipeline": data_pipelines[0], 
                "prefetch": prefetch_values[0], "samples": args.samples
            })
        
        # Add mixed precision test if requested
        if args.test_mixed_precision:
            for bs in [batch_sizes[-1]]:  # Just test largest batch size
                configs.append({
                    "batch_size": bs, "epochs": args.epochs, "mixed_precision": True,
                    "model_size": model_sizes[0], "data_pipeline": data_pipelines[0], 
                    "prefetch": prefetch_values[0], "samples": args.samples
                })
    
    # Log configuration summary
    log("profile_configurations", count=len(configs), configs=configs)
    
    # Run each profile configuration
    results = []
    for i, config in enumerate(configs):
        log("profile_config_start", config_index=i, **config)
        # Pass output_dir to profile_training
        config['output_dir'] = args.output_dir
        config_kwargs = dict(config)
        config_kwargs['gpu_sample_interval'] = args.gpu_sample_interval
        config_kwargs['enable_gpu_sampler'] = not args.disable_gpu_sampler
        result = profile_training(**config_kwargs)
        results.append(result)
        log("profile_config_complete", config_index=i, **config)
        # Short cooldown between runs
        if i < len(configs) - 1:
            time.sleep(5)
    
    # Generate comparison report in markdown format
    md_report = generate_comparison_report(results)
    report_path = os.path.join(args.output_dir, "profile_report.md")
    with open(report_path, "w") as f:
        f.write(md_report)
    log("report_generated", path=report_path)
    
    # Print summary comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*50)
    print("batch_size | model_size | mixed_precision | data_pipeline | prefetch | imgs/sec | GPU Util% | Mem Used")
    print("-"*100)
    for r in results:
        # Calculate average GPU utilization
        gpu_utils = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in r["metrics_series"]]
        mem_used = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in r["metrics_series"]]
        avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        avg_mem = sum(mem_used) / len(mem_used) if mem_used else 0
        
        print(f"{r['batch_size']:9d} | {r['model_size']:10s} | {str(r['mixed_precision']):15s} | " +
              f"{r['data_pipeline']:12s} | {r['prefetch']:8d} | {r['images_per_second']:7.1f} | " +
              f"{avg_util:8.1f} | {avg_mem:8.1f}")
    
    # Keep alive loop for inspection
    log("profile_complete", total_configs=len(configs))
    log("entering_keep_alive")
    
    try:
        while True:
            # Periodic GPU status
            gpus = see_gpus()
            nvidia_stats = capture_nvidia_smi()
            if nvidia_stats:
                log("keep_alive_status", gpu_metrics=nvidia_stats)
            time.sleep(10)
    except KeyboardInterrupt:
        log("profile_interrupted")

def generate_comparison_report(results):
    """Generate a detailed markdown report comparing all profiling runs"""
    if not results:
        return "No profiling results to report."
    
    md = "# TensorFlow GPU Profiling Report\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # System information
    md += "## System Information\n\n"
    md += f"* Platform: {platform.platform()}\n"
    md += f"* Python: {platform.python_version()}\n"
    md += f"* TensorFlow: {tf.__version__}\n"
    
    # GPU information
    gpus = tf.config.list_physical_devices('GPU')
    md += f"* GPUs: {len(gpus)}\n"
    for i, gpu in enumerate(gpus):
        md += f"  * GPU {i}: {gpu.name}\n"
    
    # Summary table
    md += "\n## Performance Summary\n\n"
    md += "| Batch Size | Model Size | Mixed Precision | Data Pipeline | Prefetch | Images/sec | Accuracy | GPU Util % | Memory (MiB) |\n"
    md += "|------------|------------|----------------|---------------|----------|------------|----------|------------|-------------|\n"
    
    for r in results:
        # Calculate averages
        metrics = r.get("metrics_series", [])
        gpu_utils = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics]
        mem_used = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in metrics]
        avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        avg_mem = sum(mem_used) / len(mem_used) if mem_used else 0
        
        md += f"| {r['batch_size']} | {r['model_size']} | {r['mixed_precision']} | {r['data_pipeline']} | "
        md += f"{r['prefetch']} | {r['images_per_second']:.2f} | {r['final_accuracy']:.4f} | "
        md += f"{avg_util:.1f} | {avg_mem:.1f} |\n"
    
    # Detailed analysis
    md += "\n## Detailed Analysis\n\n"
    
    # Batch size impact
    md += "### Impact of Batch Size\n\n"
    md += "Larger batch sizes typically improve throughput up to a point, then can cause memory pressure:\n\n"
    
    # Group by everything except batch size for fair comparison
    batch_size_groups = {}
    for r in results:
        key = (r['model_size'], r['mixed_precision'], r['data_pipeline'], r['prefetch'])
        if key not in batch_size_groups:
            batch_size_groups[key] = []
        batch_size_groups[key].append(r)
    
    for key, group in batch_size_groups.items():
        if len(group) > 1:  # Only report groups with multiple batch sizes
            model_size, mixed_precision, data_pipeline, prefetch = key
            md += f"Configuration: {model_size} model, mixed_precision={mixed_precision}, {data_pipeline} pipeline, prefetch={prefetch}\n\n"
            md += "| Batch Size | Images/sec | Speedup | GPU Util % | Memory (MiB) |\n"
            md += "|------------|------------|---------|------------|-------------|\n"
            
            # Sort by batch size
            group.sort(key=lambda x: x['batch_size'])
            baseline = group[0]['images_per_second']
            
            for r in group:
                # Calculate averages
                metrics = r.get("metrics_series", [])
                gpu_utils = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics]
                mem_used = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in metrics]
                avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                avg_mem = sum(mem_used) / len(mem_used) if mem_used else 0
                
                speedup = r['images_per_second'] / baseline if baseline > 0 else 0
                md += f"| {r['batch_size']} | {r['images_per_second']:.2f} | {speedup:.2f}x | {avg_util:.1f} | {avg_mem:.1f} |\n"
            
            md += "\n"
    
    # Mixed precision impact
    md += "### Impact of Mixed Precision\n\n"
    md += "Mixed precision (FP16) can significantly improve performance, especially for compute-bound workloads:\n\n"
    
    # Group by everything except mixed precision for fair comparison
    mp_groups = {}
    for r in results:
        key = (r['batch_size'], r['model_size'], r['data_pipeline'], r['prefetch'])
        if key not in mp_groups:
            mp_groups[key] = []
        mp_groups[key].append(r)
    
    for key, group in mp_groups.items():
        if len(group) > 1:  # Only report groups with both precisions
            batch_size, model_size, data_pipeline, prefetch = key
            md += f"Configuration: batch_size={batch_size}, {model_size} model, {data_pipeline} pipeline, prefetch={prefetch}\n\n"
            md += "| Mixed Precision | Images/sec | Speedup | GPU Util % | Memory (MiB) |\n"
            md += "|----------------|------------|---------|------------|-------------|\n"
            
            # Get FP32 (False) as baseline
            fp32 = next((r for r in group if not r['mixed_precision']), None)
            fp16 = next((r for r in group if r['mixed_precision']), None)
            
            if fp32 and fp16:
                # Calculate averages for FP32
                metrics_fp32 = fp32.get("metrics_series", [])
                gpu_utils_fp32 = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics_fp32]
                mem_used_fp32 = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in metrics_fp32]
                avg_util_fp32 = sum(gpu_utils_fp32) / len(gpu_utils_fp32) if gpu_utils_fp32 else 0
                avg_mem_fp32 = sum(mem_used_fp32) / len(mem_used_fp32) if mem_used_fp32 else 0
                
                # Calculate averages for FP16
                metrics_fp16 = fp16.get("metrics_series", [])
                gpu_utils_fp16 = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics_fp16]
                mem_used_fp16 = [m.get("gpu_metrics", {}).get("mem_used_mib", 0) for m in metrics_fp16]
                avg_util_fp16 = sum(gpu_utils_fp16) / len(gpu_utils_fp16) if gpu_utils_fp16 else 0
                avg_mem_fp16 = sum(mem_used_fp16) / len(mem_used_fp16) if mem_used_fp16 else 0
                
                baseline = fp32['images_per_second']
                speedup = fp16['images_per_second'] / baseline if baseline > 0 else 0
                
                md += f"| False (FP32) | {fp32['images_per_second']:.2f} | 1.00x | {avg_util_fp32:.1f} | {avg_mem_fp32:.1f} |\n"
                md += f"| True (FP16) | {fp16['images_per_second']:.2f} | {speedup:.2f}x | {avg_util_fp16:.1f} | {avg_mem_fp16:.1f} |\n"
            
            md += "\n"
    
    # Inference benchmarks
    md += "### Inference Performance\n\n"
    md += "Inference performance with different batch sizes:\n\n"
    
    for i, r in enumerate(results):
        if "inference_benchmarks" in r and r["inference_benchmarks"]:
            md += f"Configuration {i+1}: batch_size={r['batch_size']}, {r['model_size']} model, mixed_precision={r['mixed_precision']}\n\n"
            md += "| Inference Batch | Avg Time (ms) | Min Time (ms) | P95 Time (ms) | Samples/sec |\n"
            md += "|----------------|---------------|---------------|---------------|-------------|\n"
            
            for bench in r["inference_benchmarks"]:
                md += f"| {bench['batch_size']} | {bench['avg_time_ms']:.2f} | {bench['min_time_ms']:.2f} | "
                md += f"{bench['p95_time_ms']:.2f} | {bench['throughput_samples_per_sec']:.2f} |\n"
            
            md += "\n"
    
    # Recommendations based on profiling
    md += "## Recommendations\n\n"
    
    # Find best batch size
    best_batch = max(results, key=lambda x: x['images_per_second'])
    md += f"* **Optimal batch size**: {best_batch['batch_size']} (achieves {best_batch['images_per_second']:.2f} images/sec)\n"
    
    # Mixed precision recommendation
    mp_pairs = []
    for key, group in mp_groups.items():
        if len(group) > 1:
            fp32 = next((r for r in group if not r['mixed_precision']), None)
            fp16 = next((r for r in group if r['mixed_precision']), None)
            if fp32 and fp16:
                speedup = fp16['images_per_second'] / fp32['images_per_second'] if fp32['images_per_second'] > 0 else 0
                mp_pairs.append((speedup, fp32, fp16))
    
    if mp_pairs:
        max_speedup_pair = max(mp_pairs, key=lambda x: x[0])
        speedup, fp32, fp16 = max_speedup_pair
        md += f"* **Mixed precision**: {'Recommended' if speedup > 1.1 else 'Not significantly beneficial'}"
        md += f" (up to {speedup:.2f}x speedup observed with batch size {fp16['batch_size']})\n"
    
    # Memory recommendations
    high_mem_configs = [r for r in results if any(m.get("gpu_metrics", {}).get("mem_used_mib", 0) > 
                                                 0.9 * m.get("gpu_metrics", {}).get("mem_total_mib", 1) 
                                                 for m in r.get("metrics_series", []))]
    if high_mem_configs:
        md += "* **Memory pressure detected**: Consider reducing batch size for the following configurations:\n"
        for r in high_mem_configs:
            md += f"  * batch_size={r['batch_size']}, {r['model_size']} model, mixed_precision={r['mixed_precision']}\n"
    
    # Utilization recommendations
    low_util_configs = []
    for r in results:
        metrics_series = r.get("metrics_series", [])
        if metrics_series:
            gpu_utils = [m.get("gpu_metrics", {}).get("gpu_util_percent", 0) for m in metrics_series]
            avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
            if avg_util < 50:
                low_util_configs.append(r)
    
    if low_util_configs:
        md += "* **Low GPU utilization detected**: Consider optimizing the data pipeline or increasing model complexity:\n"
        for r in low_util_configs:
            md += f"  * batch_size={r['batch_size']}, {r['model_size']} model, mixed_precision={r['mixed_precision']}\n"
    
    return md


if __name__ == "__main__":
    main()
