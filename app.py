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

    model = build_model()

    # keep training short for demo
    epochs = int(os.environ.get("EPOCHS", "1"))

    # train on a smaller subset so the demo runs quickly in CI/demo
    log("training_start", epochs=epochs, samples=5000, batch_size=128)
    model.fit(x_train[:5000], y_train[:5000], epochs=epochs, batch_size=128, verbose=2, callbacks=[])
    log("training_complete")

    loss, acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=0)
    log("evaluation", subset_size=1000, loss=round(float(loss), 6), accuracy=round(float(acc), 6))

    preds = model.predict(x_test[:10])
    pred_labels = preds.argmax(axis=-1)
    log("prediction_sample", count=10, predictions=pred_labels.tolist(), ground_truth=y_test[:10].tolist())


if __name__ == '__main__':
    main()
