# GPU Memory Sizing for LLM Inference

This note explains how to estimate GPU RAM needs for deploying language models. Useful to decide the largest model that fits a single 16 GB GPU and what optimization lever to pull next.

---

## 1. Main Memory Components (Inference)

| Component | What | Scales With | Notes |
|-----------|------|-------------|-------|
| Weights (parameters) | Model tensor values | Param count × bytes/param | Dominant term at rest |
| KV Cache (attention cache) | Keys + Values per generated (or input) token | Layers × SeqLen × Hidden × 2 × bytes | Grows with context + generated tokens |
| Activations / Workspace | Temporary intermediates per forward step | Batch × Layers × Hidden | Mostly transient; allocator may keep peak |
| Framework / Runtime Overhead | CUDA context, cuBLAS/cuDNN handles, graph metadata | Constant-ish | 200–600 MB typical |
| Fragmentation / Safety Headroom | Unused gaps + future growth | — | Keep 5–15% free to avoid OOM |

(Training adds Optimizer States (≈2× or 4× weights) + Gradients, ignored here.)

---

## 2. Bytes per Parameter

| Precision | Bytes / Param | Relative |
|-----------|---------------|----------|
| FP32      | 4             | 1.0× |
| FP16 / bfloat16 | 2      | 0.5× |
| INT8 (quantized) | 1     | 0.25× |
| INT4 (quantized) | 0.5   | 0.125× |

Approx weight memory (GB) = (Params_in_Billions × BytesPerParam) / (1024^3 / 10^9 ≈ 1.073).  
Simplify: 1B params ≈  
- FP16: ~2.0 GB (actually ~1.9)  
- INT8: ~1.0 GB  
- INT4: ~0.5 GB  

Rule of thumb quick mental math: FP16 GB ≈ 2 × (B params).

---

## 3. KV Cache Estimation (Decoder-Only GPT)

Per token per layer (FP16):
KV bytes ≈ 2 × HiddenDim × BytesPerElem.

For all tokens so far (sequence length S):
KV_total ≈ Layers × S × 2 × HiddenDim × BytesPerElem.

Convert to GB: divide by 1,073,741,824.

Example (GPT-J 6B style: 28 layers, hidden 4096, FP16):
Per token:
2 × 4096 × 2 B × 28 = 917,504 B ≈ 0.875 MB / token.
At 1024 tokens → ~896 MB cache.
(Observation: large contexts substantially inflate memory beyond just weights.)

Tokenizer *input* prompt tokens count toward cache the same way as generated tokens (unless cache reuse disabled).

---

## 4. Putting It Together

Total ≈ Weights + KV Cache (current context) + Activations Peak + Overhead + Headroom.

Activations (inference, batch=1) are usually << weights + KV (especially with attention cache dominating long prompts). A conservative lump: 5–10% of weight size for typical decoder-only inference.

Overhead+allocator safety: reserve ~0.5–1.0 GB.

---

## 5. Example Budgets (Single 16 GB Card, ~14 GB Practical Usable)

(Assume FP16 unless noted; context length = 1024; batch=1.)

| Model | Params | Weights (FP16 GB) | KV @1024 (GB est) | Overhead + Act (GB) | Total (GB) | Fits? |
|-------|--------|------------------:|------------------:|--------------------:|-----------:|-------|
| GPT‑2 XL | 1.5B | ~3.0 | ~0.22 | ~0.6 | ~3.8 | Easy |
| GPT‑Neo 2.7B | 2.7B | ~5.4 | ~0.45 | ~0.7 | ~6.6 | Easy |
| GPT‑J 6B | 6.0B | ~12.0 | ~0.90 | ~0.9 | ~13.8 | Tight (OK) |
| Llama‑2 7B | 7.0B | ~14.0 | ~0.80 | ~1.0 | ~15.8 | Marginal / Risk OOM |
| Llama‑2 7B (INT8) | 7.0B | ~7.0 | ~0.80 | ~0.9 | ~8.7 | Comfortable |
| Llama‑2 13B (INT8) | 13B | ~13.0 | ~1.3 | ~1.0 | ~15.3 | Tight borderline |
| Llama‑2 13B (INT4) | 13B | ~6.5 | ~1.3 | ~0.9 | ~8.7 | Fits well |

Notes:
- Different architectures (hidden size, layer count) shift KV share: higher layer count increases multiplier linearly.
- Some models (Llama) have hidden dims = 4096 (7B) / 5120 (13B), so KV sizes differ slightly from GPT-J.

---

## 6. When You Hit OOM Sooner Than Estimates

Common causes:
1. Larger effective context length than assumed (e.g., 2k vs 1k tokens). KV scales linearly with S.
2. Batch size > 1 multiplies activation + partial KV growth (if you prefill multi prompts).
3. Flash attention disabled → extra temporary buffers.
4. Mixed precision disabled (falling back to FP32).
5. Quantization runtime keeping both original + quantized weights temporarily during load.
6. Fragmentation in long-lived processes (allocator growth).

Mitigations:
- Quantize (8‑bit, then 4‑bit).
- Reduce max context length (truncate history).
- Early-free / offload past layers (advanced).
- Use inference libraries with paged attention (vLLM, TensorRT-LLM) to reduce KV overhead.
- Enforce batch=1 if memory-bound.

---

## 7. Quick Capacity Formula (Cheat Sheet)

Given:
- P = params (billions)
- Bp = bytes per param (2 for FP16, 1 for INT8, 0.5 for INT4)
- L = number of layers
- H = hidden size
- S = current (prompt + generated) tokens
- Be = bytes per element (usually = Bp for FP16 weight scenario; KV typically same precision)

Weights_GB ≈ (P × Bp) / (1e9 / 1.073)  (≈ P × Bp / 0.933)
KV_GB ≈ (L × S × 2 × H × Be) / 1,073,741,824
Total_GB ≈ Weights_GB + KV_GB + 0.6 (overhead) + 0.05 × Weights_GB (activations buffer)

Keep Total_GB < (GPU_RAM_GB × 0.90) for safety.

---

## 8. Practical Selection for 16 GB GPU

Tiered recommendation:
| Goal | Recommendation |
|------|----------------|
| Max accuracy within fp16 only | GPT-J 6B (or similar ~6B) |
| Slightly larger but safer | Llama‑2 7B INT8 |
| Bigger (context heavy) with margin | Llama‑2 13B INT4 |
| Fast latency / many concurrent users | Smaller (2.7B) + batching |

If planning for *training* (even LoRA fine-tuning), reduce param target drastically or add gradient checkpointing & mixed precision.

---

## 9. Checklist Before Deploying a Larger Model

1. Confirm `nvidia-smi` free memory.
2. Load model with `low_cpu_mem_usage=True` (HF) to stream weights.
3. Run a dry-run generation at target max context to observe peak memory.
4. Enable quantization *before* warmup (bitsandbytes / GPTQ / AWQ).
5. Monitor with `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv -l 1`.

---

## 10. Quick Examples

### Example A: Can I run a 7B FP16 model with 2048 context?
Weights ≈ 14 GB; KV (assume 32 layers, H=4096, S=2048):  
KV ≈ 32 × 2048 × 2 × 4096 × 2 / 1,073,741,824 ≈ 0.97 GB  
Total ≈ 14 + 0.97 + 0.6 + 0.7 ≈ 16.3 GB → No (too tight). Use INT8 or reduce context.

### Example B: 13B INT4, context 1024:
Weights ≈ 6.5 GB; Suppose 40 layers, H=5120:  
KV ≈ 40 × 1024 × 2 × 5120 × 1 (INT4 stored often unpacked to 1 byte for KV) / 1,073,741,824 ≈ 0.39 GB  
Total ≈ 6.5 + 0.39 + 0.6 + 0.33 ≈ 7.8 GB → Comfortable.

---

## 11. Common Pitfalls

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Forget quantization | OOM earlier | Apply 8‑bit/4‑bit weights |
| High max_new_tokens | Large KV | Cap generation length |
| Retaining long chat history | KV grows | Summarize / truncate |
| Batch explosion | Activations blow up | Use micro-batching |
| Mixed precision off | Doubles memory | Ensure fp16 enabled |

---

## 12. Decision Flow (Single GPU)

1. Target param count?  
2. If >6B → plan quantization.  
3. Required context length? If large (≥2k), favor smaller or quantized model.  
4. Need concurrency? Prefer smaller model + batching rather than one huge model.  
5. Perform dry run → measure → adjust.

---

## 13. TL;DR

- Rough FP16 weight GB ≈ 2 × (B params).  
- KV cache can reach ~1 GB for 6–7B at 1k context.  
- 16 GB card: 6B fp16 safe, 7B fp16 tight, 7B int8 comfy, 13B needs int4/8-bit.  
- enc/dec (video engines) irrelevant for text; focus on SM + memory.  
- Quantize + limit context + batch to stretch capacity.

---
