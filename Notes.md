

# Q

- Improve ETL pipeline for better GPU utilization

- How is minikube consuming GPU by nvidia-device-plugin ds

- Why is docker build for GPU image very slow
   - Can it be GPU enhanced
   - Will az acr build save it

- Image Load is often slow
   - Can build a smaller image
   - Can minikube cache the base image

- How does containerd decide which layer to cache

- Parallelize the image pull with GPU

- 1 pod 1 GPU or fractional GPU

- Monitor GPU with nvidia-smi or DCGM or GPU Operator
- Monitor GPU with metric-server or Prometheus/Grafana

- Setup chat services to serve multiple requests in parralell

- pod termniation seems to have issue

# I

- Can have better analysis with GRPC reflection (no need to maintain proto files)
- nvidia-device-plugin



- Design, implement, and maintain ML pipelines for automated training, testing, and deployment of machine learning models, ensuring scalability and efficiency.
- Work collaboratively with ML engineers to troubleshoot and optimize model performance, ensuring models are production-ready and meet defined SLAs
- Manage and monitor Kubernetes clusters and related infrastructure to support high-volume ML workloads, implementing best practices for security and resilience.
- Develop and maintain documentation on ML infrastructure, tools, and best practices, providing guidance and support to ML teams.
- Continuously evaluate and incorporate new technologies and tools to enhance the ML platform's capabilities and performance.

- Experience: 3 years or more of experience in MLOps, with a proven track record of managing ML infrastructure
- Kubernetes Proficiency: Deep understanding of Kubernetes (K8s) infrastructure and its application in managing ML workloads
- Programming Skills: Proficiency in Python or Golang
- Proven experience with Linux OS, with the ability to maintain system performance, ensure proper configuration, and leverage tools to troubleshoot software, hardware, and network-related issues
- Education: Bachelor’s or higher degree in Computer Science, Engineering, or a related technical discipline
- Strong communication and teamwork skills
- Passion for technology and solving challenging problems

- Familiarity with ML frameworks (e.g., TensorFlow, PyTorch) and CUDA
- CI/CD Tools: Experience with CI/CD tools (e.g., GitHub Actions, Jenkins, GitLab CI) and container technologies (e.g., Docker)
- Experience training large models, including LLMs


# First Prompt

## Infra
- build a local k8s cluster with node leverage GPU using minikube
- build the python application into a container by az acr build
- pull the container image from ACR and run it in local k8s

## Application
- Build me a simplest gpu python code that identify the numbers of a MNIST dataset with tensorflow

# Improvement

## Idenfity the Bottle Neck is INPUT PIPELINE

Short answer: for MNIST–style inference, the **biggest, easy-to-fix bottleneck is almost always the CPU side of the input pipeline**—per-image preprocessing (normalize, resize/pad/center, NHWC→NCHW), plus host↔device copies and too many tiny GPU kernels. Moving those steps into **one fused CUDA kernel** and exposing it to TensorRT as a **plugin** typically yields the largest win.

What to optimize (ranked by impact):

1. **Preprocessing & transfers (CPU→GPU):**

   * Normalize/scalings, padding/centering 28×28, grayscale to 1×H×W, NHWC→NCHW.
   * Use **pinned (page-locked) memory** and **cudaMemcpyAsync** on a **non-default stream**.
   * Fuse all preprocessing into **one CUDA kernel** that writes the final FP16/INT8 tensor layout expected by TensorRT.

2. **Kernel-launch overhead from many tiny ops:**

   * For MNIST nets the math is small; launching separate kernels for bias, BN, ReLU, softmax, argmax can dominate time.
   * Implement a **fused elementwise CUDA kernel** (e.g., bias+activation, or softmax+argmax) as a **TensorRT plugin layer** so it runs as a single op.

3. **Batching & precision:**

   * Prefer **batch>1** during throughput benchmarks (e.g., 32/64).
   * Build the engine in **FP16** or **INT8** (with a quick calibration set). This reduces bandwidth and speeds up conv/FC—TensorRT already does the heavy lifting.

4. **Avoid Python loops in the hot path:**

   * Don’t loop over images in Python to preprocess; pass whole batches to your CUDA kernel.
   * Keep preprocessing → inference on the **same CUDA stream** to maximize overlap.

Concrete pattern (what you’d write):

* A CUDA kernel (or Numba/CuPy prototype) that: loads uint8 MNIST image(s) → converts to FP16/INT8 → normalizes → centers/pads if needed → writes directly into **NCHW** contiguous buffer.
* A small **IPluginV2DynamicExt** (TensorRT plugin) wrapping that kernel so it appears as a layer in the network (e.g., `PreprocessPlugin`).
* Optional second plugin for **fused softmax+argmax** if argmax is needed on-GPU.

Typical Python flow (high-level):

```python
# Pseudocode sketch
stream = cudaStreamCreate()
host_batch = allocate_pinned_batch_uint8(B, 28, 28)      # fill from dataset
dev_in = cudaMalloc(B*1*28*28*dtype_size)                # final NCHW fp16/int8
dev_out_bindings = [dev_in, ...]                         # other TRT bindings

# 1) Async H2D of raw bytes
cudaMemcpyAsync(dev_raw_u8, host_batch, size, H2D, stream)

# 2) Fused preprocessing (normalize + NHWC->NCHW + cast) -> writes into dev_in
launch_preprocess_kernel(dev_raw_u8, dev_in, B, 28, 28, mean, std, stream)

# 3) Execute TensorRT engine on the same stream
context.execute_async_v2(dev_out_bindings, stream)

# 4) (Optional) fused softmax+argmax plugin in the engine; otherwise D2H
```

Why this works for MNIST: the network is tiny (e.g., 2 convs + 1 FC), so pure compute is **not** the limiter; **data movement and many micro-ops** are. Fusing preprocessing and small elementwise ops into **one or two CUDA kernels** and letting TensorRT handle the heavy layers (conv/FC) usually gives the best speedup.
