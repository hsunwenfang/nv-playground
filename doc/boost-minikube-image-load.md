# Speeding Up `minikube image load` (tiny-chat:latest)

Goal: Eliminate or hide slow `minikube image load` by building directly inside the Minikube node, shrinking layers, and maximizing cache reuse.

## Quick Win Order

1. Add a `.dockerignore` (reduce build context):
```
__pycache__/
*.pyc
outputs/
profile/
advanced_profile_*/
*.png
*.tar.gz
.git
```

2. Split heavy deps from fast‑changing code:

`Dockerfile.base`
```dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu AS deps
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt
```

`Dockerfile.runtime`
```dockerfile
FROM deps AS runtime
WORKDIR /app
COPY app.py .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000 9100
CMD ["python","app.py"]
```

3. Pre-cache the large base once:
```bash
minikube cache add tensorflow/tensorflow:2.12.0-gpu
```

4. Build inside Minikube (skip `minikube image load` entirely):
```bash
eval $(minikube docker-env)
DOCKER_BUILDKIT=1 docker build -t tiny-chat:base -f Dockerfile.base .
DOCKER_BUILDKIT=1 docker build -t tiny-chat:latest -f Dockerfile.runtime .
eval $(minikube docker-env -u)
```

5. Ensure `imagePullPolicy: IfNotPresent` (reuse cached node image):
```bash
kubectl patch deployment tiny-chat -p '{"spec":{"template":{"spec":{"containers":[{"name":"tiny-chat","imagePullPolicy":"IfNotPresent"}]}}}}'
```

6. Fast edit loop (only code changed):
```bash
eval $(minikube docker-env)
DOCKER_BUILDKIT=1 docker build -t tiny-chat:latest -f Dockerfile.runtime .
eval $(minikube docker-env -u)
kubectl rollout restart deployment/tiny-chat
```

7. Optional: Local registry (useful for multi-node or future clusters):
```bash
minikube addons enable registry
docker tag tiny-chat:latest localhost:5000/tiny-chat:latest
docker push localhost:5000/tiny-chat:latest
# Then set image: localhost:5000/tiny-chat:latest
```

8. Fallback (external build → import quickly):
```bash
docker save tiny-chat:latest | (eval $(minikube docker-env); docker load)
```

9. Remove unused deps to shrink layers (edit `requirements.txt`):
- Drop `transformers` if not currently used.
- Pin versions to stabilize caching.

10. Inspect layer cache efficiency:
```bash
docker history tiny-chat:latest
docker image inspect tiny-chat:latest --format='Size: {{.Size}}'
```

## Optional Automation Script

`rebuild.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
eval $(minikube docker-env)
DOCKER_BUILDKIT=1 docker build -t tiny-chat:latest -f Dockerfile.runtime .
eval $(minikube docker-env -u)
kubectl rollout restart deployment/tiny-chat
```

## Why This Works

| Problem | Mitigation |
|---------|------------|
| Slow tar streaming via `minikube image load` | Build directly inside node daemon |
| Large unchanging dependencies rebuilt | Layer split (`deps` vs runtime) |
| Bloated context | `.dockerignore` |
| Repeated base fetch | `minikube cache add` |
| Slow pushes/pulls | Local registry / stable tag reuse |
| Cache bust on code change | Keep code in final COPY layer only |

## Decision Tree

1. Need fastest inner loop? → Build in-node (Step 4).
2. Still slow? → Split Dockerfiles + .dockerignore.
3. Team/CI integration? → Local registry.
4. Multi-node / scale out later? → Pre-cache + registry mirror.

## Summary

Eliminate `minikube image load` by building inside Minikube’s Docker, isolate heavy deps in a cached base layer, shrink context, and restart deployment with cached images. Use a local registry only if scaling beyond a single node or integrating external