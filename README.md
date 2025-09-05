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
