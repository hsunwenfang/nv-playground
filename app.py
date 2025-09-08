"""Adaptive Chat Service (Tiny GPT-2 TF path OR Large 6B–7B Torch path).

Features:
 - Dual backend: default lightweight TensorFlow causal LM; optional large PyTorch model (e.g., GPT-J 6B) with bitsandbytes quantization.
 - Quantization: 8-bit or 4-bit (env QUANTIZE=8|4) when torch + bitsandbytes available.
 - Context management: token-based truncation to MAX_CONTEXT_TOKENS (default 1024) retaining most recent dialogue.
 - Micro-batching (torch path): aggregate concurrent requests up to BATCH_MAX_SIZE or BATCH_TIMEOUT_MS for better GPU utilization.
 - GPU enforcement controls as before (REQUIRE_GPU / ALLOW_CPU_FALLBACK).
"""

import os, json, time, hashlib, threading
# NOTE: Default FORCE_TORCH behavior is enabled (default='1') to avoid fragile TF path in container.
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoConfig, set_seed, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROM_ENABLED = True
except Exception:
    PROM_ENABLED = False


def log(event: str, level: str = "INFO", **fields):
    record = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "event": event}
    record.update(fields)
    print(json.dumps(record, ensure_ascii=False))


MODEL_NAME = os.environ.get('MODEL_NAME', os.environ.get('LARGE_MODEL', 'distilgpt2'))  # allow LARGE_MODEL override
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1"
ALLOW_CPU_FALLBACK = os.environ.get("ALLOW_CPU_FALLBACK", "0") == "1"
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "4"))
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "1024"))  # hard cap on tokenized context (prompt + new)
GEN_DEFAULTS = {
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "40")),
    "temperature": float(os.environ.get("TEMPERATURE", "0.8")),
    "top_p": float(os.environ.get("TOP_P", "0.95")),
    "do_sample": True,
}

# Torch / quantization controls
QUANTIZE = os.environ.get("QUANTIZE")  # "8" or "4" supported
USE_TORCH = True  # forced torch-only mode
BATCH_MAX_SIZE = int(os.environ.get("BATCH_MAX_SIZE", "4"))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "25"))  # flush window
ENABLE_BATCHING = os.environ.get("ENABLE_BATCHING", "1") == "1"

torch_available = False
bitsandbytes_available = False
torch = None  # type: ignore
if USE_TORCH:
    try:
        import torch  # type: ignore
        torch_available = True
        try:
            import bitsandbytes as bnb  # noqa: F401
            bitsandbytes_available = True
        except Exception:
            bitsandbytes_available = False
    except Exception:
        torch_available = False
        bitsandbytes_available = False

if QUANTIZE and QUANTIZE not in ("8", "4"):
    log("invalid_quantize_value", level="WARN", value=QUANTIZE)
    QUANTIZE = None

gpu_devices = []
try:
    import torch as _torch_check  # type: ignore
    if _torch_check.cuda.is_available():
        gpu_devices = [f"cuda:{i}" for i in range(_torch_check.cuda.device_count())]
        log("startup_gpu_list", gpus=gpu_devices)
    elif REQUIRE_GPU and not ALLOW_CPU_FALLBACK:
        log("startup_no_gpu", level="ERROR", require_gpu=REQUIRE_GPU)
        raise SystemExit("GPU required but not present.")
    else:
        log("startup_gpu_absent_fallback", level="WARN", fallback=ALLOW_CPU_FALLBACK)
except Exception as _gpu_e:
    log("gpu_detect_error", error=str(_gpu_e), level="ERROR")

log("model_load_start", model=MODEL_NAME, torch=torch_available, quantize=QUANTIZE, bitsandbytes=bitsandbytes_available)
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Ensure a pad token exists (needed for batching). If missing, add one derived from EOS.
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = AutoConfig.from_pretrained(MODEL_NAME)
is_enc_dec = getattr(config, 'is_encoder_decoder', False)
model = None
pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

def load_tf_model():
    raise RuntimeError("TensorFlow backend removed (torch-only build)")

def load_torch_model():
    global model, pad_id
    load_kwargs: Dict[str, Any] = {}
    if QUANTIZE and bitsandbytes_available:
        if QUANTIZE == '8':
            load_kwargs['load_in_8bit'] = True
        elif QUANTIZE == '4':
            load_kwargs['load_in_4bit'] = True
        load_kwargs['device_map'] = 'auto'
    else:
        # full precision (may be large!)
        load_kwargs['device_map'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    pad_id_local = tokenizer.pad_token_id or tokenizer.eos_token_id
    return pad_id_local

if torch_available and (QUANTIZE or '6B' in MODEL_NAME or '7B' in MODEL_NAME or os.environ.get('FORCE_TORCH','1')=='1'):
    try:
        pad_id = load_torch_model()
        BACKEND = 'torch'
    except Exception as e:
        log("torch_model_load_failed", error=str(e), level="ERROR")
        if REQUIRE_GPU and not ALLOW_CPU_FALLBACK:
            raise
    raise SystemExit("Torch model load failed in torch-only mode")
else:
    raise SystemExit("Torch not available in torch-only mode")

if pad_id is None:
    pad_id = tokenizer.eos_token_id

# Torch device validation
if REQUIRE_GPU:
    has_cuda_param = any(p.is_cuda for p in model.parameters())
    if not has_cuda_param and not ALLOW_CPU_FALLBACK:
        raise SystemExit("Torch model not loaded on CUDA; aborting.")
log("torch_model_devices", devices=list({str(p.device) for p in model.parameters()}))

def warmup():
    try:
        t_w0 = time.time()
        warm_ids = tokenizer("Warmup", return_tensors='pt').input_ids
        if torch_available:
            warm_ids = warm_ids.to(next(model.parameters()).device)
        _ = model.generate(warm_ids, max_new_tokens=4, pad_token_id=pad_id, eos_token_id=pad_id)
        log("gpu_warmup_complete", ms=int((time.time()-t_w0)*1000), backend=BACKEND)
    except Exception as e:
        log("gpu_warmup_error", error=str(e), backend=BACKEND, level="WARN")

warmup()

log("model_load_complete", model=MODEL_NAME, pad_id=pad_id, encoder_decoder=is_enc_dec, backend=BACKEND)

history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY * 2))
_gen_lock = threading.Lock()  # used for TF path (serial) or as fallback

# =============== Torch Micro-batching Infrastructure ==================
class _GenRequest:
    __slots__ = ("prompt", "event", "gen_cfg", "result", "error", "enqueue_time")
    def __init__(self, prompt: str, gen_cfg: Dict[str, Any]):
        self.prompt = prompt
        self.gen_cfg = gen_cfg
        self.event = threading.Event()
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.enqueue_time = time.time()

_batch_queue: List[_GenRequest] = []
_queue_lock = threading.Lock()

def _batch_worker():
    while True:
        try:
            time.sleep(BATCH_TIMEOUT_MS / 1000.0)
            to_process: List[_GenRequest] = []
            with _queue_lock:
                if _batch_queue and (len(_batch_queue) >= BATCH_MAX_SIZE or (time.time() - _batch_queue[0].enqueue_time) * 1000.0 >= BATCH_TIMEOUT_MS):
                    while _batch_queue and len(to_process) < BATCH_MAX_SIZE:
                        to_process.append(_batch_queue.pop(0))
            if not to_process:
                continue
            prompts = [r.prompt for r in to_process]
            try:
                tokenized = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_CONTEXT_TOKENS)
            except Exception as tok_e:
                # Attempt to set pad token dynamically then retry once
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    try:
                        tokenized = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_CONTEXT_TOKENS)
                    except Exception:
                        raise tok_e
                else:
                    raise tok_e
            input_ids = tokenized.input_ids
            attn_mask = tokenized.attention_mask
            if torch_available:
                dev = next(model.parameters()).device
                input_ids = input_ids.to(dev)
                attn_mask = attn_mask.to(dev)
            gen_cfg_ref = to_process[0].gen_cfg
            try:
                with torch.inference_mode():  # type: ignore
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=gen_cfg_ref["max_new_tokens"],
                        temperature=gen_cfg_ref["temperature"],
                        top_p=gen_cfg_ref["top_p"],
                        do_sample=gen_cfg_ref["do_sample"],
                        pad_token_id=pad_id,
                        eos_token_id=pad_id,
                    )
                for r, seq in zip(to_process, outputs):
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    reply = text.split("Assistant:")[-1].strip()
                    if "\n\n" in reply:
                        reply = reply.split("\n\n")[0].strip()
                    r.result = reply
                    r.event.set()
                if PROM_ENABLED:
                    # simple gauge via counter label reuse (not ideal but keeps footprint low)
                    REQ_COUNTER.labels(endpoint="batched").inc(len(to_process))
                log("batch_generation_complete", batch_size=len(to_process))
            except Exception as gen_e:
                err_str = str(gen_e)
                log("batch_generation_error", error=err_str, level="ERROR")
                for r in to_process:
                    r.error = err_str
                    r.event.set()
        except Exception as loop_e:
            log("batch_worker_loop_error", error=str(loop_e), level="ERROR")

if BACKEND == 'torch':
    # Only start batch worker if enabled and batch size >1
    if ENABLE_BATCHING and BATCH_MAX_SIZE > 1:
        threading.Thread(target=_batch_worker, daemon=True).start()
    else:
        log("batching_disabled", reason="config", enable_batching=ENABLE_BATCHING, batch_max=BATCH_MAX_SIZE)
    if ENABLE_BATCHING and BATCH_MAX_SIZE > 1:
        threading.Thread(target=_batch_worker, daemon=True).start()
    else:
        log("batching_disabled", reason="config", enable_batching=ENABLE_BATCHING, batch_max=BATCH_MAX_SIZE)
    if getattr(model.config, 'pad_token_id', None) is None and pad_id is not None:
        model.config.pad_token_id = pad_id

# ==================== Context Construction / Truncation =====================
def build_context_tokens(session_id: str, new_msg: str) -> str:
    """Assemble recent dialogue ensuring total tokens <= MAX_CONTEXT_TOKENS before generation.
    Strategy: walk history from newest to oldest, prepend until limit would be exceeded, then reverse.
    """
    messages = list(history[session_id]) + [("user", new_msg)]
    # Convert to role-tagged lines
    lines = [f"{role.title()}: {content}" for role, content in messages]
    # We'll accumulate tokens in reverse
    kept = []
    total_tokens = 0
    for line in reversed(lines):
        # rough token count via tokenizer (could cache)
        tok_len = len(tokenizer.tokenize(line))
        if total_tokens + tok_len > MAX_CONTEXT_TOKENS - GEN_DEFAULTS["max_new_tokens"]:
            break
        kept.append(line)
        total_tokens += tok_len
    kept.reverse()
    kept.append("Assistant:")
    return "\n".join(kept)

if PROM_ENABLED:
    REQ_COUNTER = Counter("tiny_gpt2_requests_total", "Chat requests", ["endpoint"])
    ERR_COUNTER = Counter("tiny_gpt2_errors_total", "Errors", ["endpoint", "type"])
    LATENCY = Histogram("tiny_gpt2_latency_seconds", "Latency", buckets=(0.05,0.1,0.2,0.5,1,2,5))
    ACTIVE = Gauge("tiny_gpt2_active_sessions", "Active sessions")
    if os.environ.get("DISABLE_METRICS", "0") != "1":
        try:
            start_http_server(int(os.environ.get("METRICS_PORT", "9100")))
            log("metrics_server_started", port=int(os.environ.get("METRICS_PORT", "9100")))
        except OSError as e:
            log("metrics_server_port_in_use", level="WARN", error=str(e))

app = FastAPI(title="Tiny GPT-2 Chat", version="0.3.0")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None)
    max_new_tokens: Optional[int] = Field(None, ge=1, le=256)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    model_latency_ms: int
    history: List[Dict[str, Any]]


@app.get("/healthz")
def health():
    if PROM_ENABLED:
        ACTIVE.set(len(history))
    return {"status": "ok", "model": MODEL_NAME, "time": datetime.utcnow().isoformat()}


def build_prompt(session_id: str, new_msg: str) -> str:
    # For compatibility keep name, delegate to token-aware version.
    return build_context_tokens(session_id, new_msg)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()
    if PROM_ENABLED:
        REQ_COUNTER.labels(endpoint="chat").inc()
    session_id = req.session_id or hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:12]
    try:
        history[session_id].append(("user", req.message))
        gen_cfg = GEN_DEFAULTS.copy()
        if req.max_new_tokens is not None:
            gen_cfg["max_new_tokens"] = req.max_new_tokens
        if req.temperature is not None:
            gen_cfg["temperature"] = req.temperature
        if req.top_p is not None:
            gen_cfg["top_p"] = req.top_p
        prompt = build_prompt(session_id, req.message)

        if BACKEND == 'torch':
            if ENABLE_BATCHING and BATCH_MAX_SIZE > 1:
                # enqueue for batch worker
                gr = _GenRequest(prompt, gen_cfg)
                with _queue_lock:
                    _batch_queue.append(gr)
                if not gr.event.wait(timeout=30):
                    with _queue_lock:
                        if gr in _batch_queue:
                            _batch_queue.remove(gr)
                    raise HTTPException(status_code=504, detail="generation timeout")
                if gr.error:
                    raise HTTPException(status_code=500, detail=gr.error)
                reply = gr.result or ""
            else:
                # direct single inference
                try:
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(model.parameters()).device)
                    with torch.inference_mode():  # type: ignore
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=gen_cfg["max_new_tokens"],
                            temperature=gen_cfg["temperature"],
                            top_p=gen_cfg["top_p"],
                            do_sample=gen_cfg["do_sample"],
                            pad_token_id=pad_id,
                            eos_token_id=pad_id,
                        )
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reply = text.split("Assistant:")[-1].strip()
                except Exception as se:
                    raise HTTPException(status_code=500, detail=f"generation failed: {se}")
        else:
            raise HTTPException(status_code=500, detail="Torch backend not initialized")

        if "\n\n" in reply:
            reply = reply.split("\n\n")[0].strip()
        history[session_id].append(("assistant", reply))
        latency_ms = int((time.time() - t0) * 1000)
        if PROM_ENABLED:
            LATENCY.observe(latency_ms / 1000.0)
        log("chat_generated", session_id=session_id, latency_ms=latency_ms, new_tokens=gen_cfg["max_new_tokens"], backend=BACKEND)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            model_latency_ms=latency_ms,
            history=[{"role": r, "content": c} for r, c in history[session_id]],
        )
    except HTTPException:
        raise
    except Exception as e:
        if PROM_ENABLED:
            ERR_COUNTER.labels(endpoint="chat", type=type(e).__name__).inc()
        log("chat_error", error=str(e), level="ERROR")
        raise HTTPException(status_code=500, detail="internal error")


def run():
    import uvicorn
    port = int(os.environ.get("APP_PORT", "8000"))
    host = os.environ.get("APP_HOST", "0.0.0.0")
    log("service_start", host=host, port=port, model=MODEL_NAME)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
