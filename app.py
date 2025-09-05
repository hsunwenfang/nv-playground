import os
import sys
import time
import json
from datetime import datetime, timezone

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import mnist
    import numpy as np
except Exception as e:
    # If TensorFlow isn't available locally, emit structured logs and exit gracefully.
    def log(event, level="INFO", **fields):
        record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event}
        record.update(fields)
        print(json.dumps(record, ensure_ascii=False))
    log("tensorflow_import_missing", error=str(e), hint="Run inside the provided Docker image for GPU support")
    sys.exit(0)


def log(event, level="INFO", **fields):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event}
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False))


def see_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    log("gpu_list", count=len(gpus), gpus=[g.name for g in gpus])
    return len(gpus) > 0


def wait_for_gpu(timeout_secs: int = 60, interval_secs: int = 2) -> bool:
    """Poll for at least one visible GPU until timeout.

    Returns True if a GPU becomes available, False otherwise.
    Uses TF device listing so it reflects container runtime visibility.
    """
    start = time.time()
    while time.time() - start < timeout_secs:
        if see_gpus():
            log("gpu_ready")
            return True
        remaining = timeout_secs - int(time.time() - start)
        log("gpu_wait_retry", remaining_seconds=remaining, interval=interval_secs)
        time.sleep(interval_secs)
    log("gpu_wait_timeout", timeout=timeout_secs, level="WARN")
    return False


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    # Optionally wait for GPU readiness
    wait_timeout = int(os.environ.get("GPU_WAIT_TIMEOUT", "60"))
    wait_interval = int(os.environ.get("GPU_WAIT_INTERVAL", "2"))
    waited_gpu = wait_for_gpu(timeout_secs=wait_timeout, interval_secs=wait_interval)
    log("gpu_usage_decided", using_gpu=waited_gpu)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # training parameters (tunable)
    epochs = int(os.environ.get("EPOCHS", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "128"))
    prefetch = int(os.environ.get("PREFETCH", "1"))
    mixed_precision = os.environ.get("MIXED_PRECISION", "0") == "1"

    # Enable mixed precision if requested
    if mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            log("mixed_precision_enabled")
        except Exception as e:
            log("mixed_precision_failed", error=str(e), level="WARN")

    model = build_model()

    # Build tf.data pipeline for training subset to better control throughput
    train_x = x_train[:5000]
    train_y = y_train[:5000]
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    ds = ds.shuffle(buffer_size=1000).batch(batch_size).prefetch(prefetch)

    log("training_start", epochs=epochs, samples=5000, batch_size=batch_size, prefetch=prefetch, mixed_precision=mixed_precision)
    t0 = time.time()
    model.fit(ds, epochs=epochs, verbose=2)
    train_seconds = time.time() - t0
    log("training_complete", training_seconds=round(train_seconds, 3))

    loss, acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
    log("evaluation", subset_size=1000, loss=round(float(loss), 6), accuracy=round(float(acc), 6))

    preds = model.predict(x_test[:10])
    pred_labels = preds.argmax(axis=-1)
    log("prediction_sample", count=10, predictions=pred_labels.tolist(), ground_truth=y_test[:10].tolist())

    # Optional keep-alive loop for benchmarking and inspection.
    # Set KEEP_ALIVE=1 to keep running indefinitely, or set KEEP_ALIVE_SECONDS=<n> to run for n seconds.
    keep_alive = os.environ.get("KEEP_ALIVE", "0")
    keep_alive_seconds = int(os.environ.get("KEEP_ALIVE_SECONDS", "0"))
    bench_interval = int(os.environ.get("BENCH_SLEEP_SECONDS", "10"))

    if keep_alive == "1" or keep_alive_seconds > 0:
        import signal
        stop = False

        def _signal_handler(signum, frame):
            nonlocal stop
            log("signal_received", signal=signum)
            stop = True

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        start = time.time()
        log("bench_start", keep_alive=keep_alive, keep_alive_seconds=keep_alive_seconds, interval=bench_interval)

        while not stop:
            # Periodic status log: GPU list + optional nvidia-smi snapshot if available
            gpus = tf.config.list_physical_devices('GPU')
            gpu_names = [g.name for g in gpus]
            info = {"using_gpu": len(gpus) > 0, "gpus": gpu_names}
            # Try to gather lightweight nvidia-smi stats if available
            try:
                import subprocess
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL)
                lines = out.decode().strip().splitlines()
                metrics = []
                for ln in lines:
                    util, mem = [x.strip() for x in ln.split(',')]
                    metrics.append({"util_percent": int(util), "mem_mib": int(mem)})
                info["nvidia_smi"] = metrics
            except Exception:
                # nvidia-smi may not be present in all images; ignore if not available
                info["nvidia_smi"] = None

            # Log heartbeat
            log("bench_heartbeat", **info)

            # Check timeout for KEEP_ALIVE_SECONDS
            if keep_alive_seconds > 0 and time.time() - start > keep_alive_seconds:
                log("bench_timeout_reached", elapsed=int(time.time() - start))
                break

            # Sleep until next heartbeat or until signaled
            for _ in range(bench_interval):
                if stop:
                    break
                time.sleep(1)

        log("bench_exit", reason=("signal" if stop else "timeout"))

    # --- Minimal benchmarking harness & metrics endpoint ---
    # Export a Prometheus metrics endpoint with simple timings and gauges.
    try:
        from prometheus_client import start_http_server, Gauge, Summary
        METRICS_PORT = int(os.environ.get("METRICS_PORT", "8000"))

        train_time = Summary('mnist_train_seconds', 'Time spent training the model')
        inference_time = Summary('mnist_inference_seconds', 'Time spent running inference')
        gpu_gauge = Gauge('mnist_gpu_count', 'Number of GPUs visible to the process')

        # Start metrics server in background
        start_http_server(METRICS_PORT)
        log("metrics_server_started", port=METRICS_PORT)

        # Record training time & inference time (if we want to re-run quickly)
        # Note: we already trained above, but we can run a quick timed inference as a metric.
        @inference_time.time()
        def _record_inference():
            _ = model.predict(x_test[:100])

        gpu_gauge.set(len(tf.config.list_physical_devices('GPU')))
        _record_inference()
    except Exception as e:
        log("metrics_setup_failed", error=str(e), level="WARN")


if __name__ == '__main__':
    main()
