# Troubleshooting Minikube GPU Issues

This document outlines the steps taken to troubleshoot and resolve issues with getting a GPU-accelerated application to run on a Minikube cluster.

## Initial Problem

The initial problem was that a Kubernetes pod requesting a GPU resource was stuck in a `Pending` state. The error message from `kubectl describe pod` was `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`.

## Investigation

The investigation involved a series of checks to diagnose the problem:

1.  **Verify Minikube and NVIDIA Device Plugin Status:**
    *   `minikube status`: Confirmed that the Minikube cluster was running.
    *   `kubectl get pods -n kube-system | grep nvidia`: Confirmed that the NVIDIA device plugin pod was running.

2.  **Check for Allocatable GPU Resources:**
    *   `kubectl describe node minikube | grep nvidia.com/gpu`: This command showed that there were no allocatable GPU resources on the node, which was the root cause of the scheduling failure.

3.  **Verify Host and Minikube GPU Visibility:**
    *   `nvidia-smi` on the host machine confirmed that the GPU was available and the NVIDIA drivers were installed correctly.
    *   `minikube ssh -- nvidia-smi`: This command failed, indicating that the NVIDIA drivers were not available *inside* the Minikube node's environment.

4.  **Check Device Plugin Logs:**
    *   `kubectl logs <nvidia-device-plugin-pod> -n kube-system`: The logs for the device plugin showed an error: `Incompatible strategy detected auto` and `No devices found`. This confirmed that the plugin was not able to detect the GPU.

## Resolution

The resolution involved a combination of starting with a clean slate, using a specific Kubernetes version, and ensuring the Docker image was available to the Minikube environment.

1.  **Clean Minikube Environment:**
    *   The existing Minikube cluster was deleted to ensure a clean start:
        ```bash
        minikube delete
        ```

2.  **Start Minikube with a Specific Kubernetes Version:**
    *   A new Minikube cluster was started with a specific, known-stable version of Kubernetes (`v1.28.3`) and with GPU support enabled:
        ```bash
        minikube start --kubernetes-version=v1.28.3 --driver=docker --gpus=all
        ```

3.  **Verify GPU Access Inside Minikube:**
    *   After the cluster was started, we verified that the GPU was now visible from within the Minikube node:
        ```bash
        minikube ssh -- sudo nvidia-smi
        ```
    *   This command was now successful, indicating that the drivers were correctly installed inside the Minikube node.

4.  **Resolve Image Pulling Issue:**
    *   When the application was deployed, it failed with an `ErrImageNeverPull` error. This was because the Docker image was built on the host and was not available to the Docker daemon running inside the Minikube cluster.
    *   To resolve this, we first configured the local Docker client to use the Docker daemon inside the Minikube cluster:
        ```bash
        eval $(minikube -p minikube docker-env)
        ```
    *   Then, we rebuilt the Docker image. This made the image available to the Minikube cluster:
        ```bash
        docker build -t nv-mnist:latest .
        ```

5.  **Update and Redeploy the Application:**
    *   The `k8s-deployment.yaml` file was updated to use the locally built image and to set the `imagePullPolicy` to `IfNotPresent`:
        ```yaml
        # ...
        spec:
          containers:
          - name: mnist-gpu
            image: nv-mnist:latest
            imagePullPolicy: IfNotPresent
        # ...
        ```
    *   The deployment was then deleted and reapplied:
        ```bash
        kubectl delete deployment mnist-gpu -n mnist-gpu-demo
        kubectl apply -f /home/azureuser/nv-playground/k8s-deployment.yaml
        ```

After these steps, the pod started successfully and was able to access the GPU.
