"""Tiny GPT-2 (TF backend) Chat Service.

Provides /chat and /healthz endpoints. Uses HuggingFace sshleifer/tiny-gpt2
loaded via TensorFlow (no torch). Keeps a small per-session rolling context.
"""

import os, json, time, hashlib, threading
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional

import tensorflow as tf
from transformers import AutoTokenizer, AutoConfig, TFAutoModelForCausalLM, set_seed
try:
    from transformers import TFAutoModelForSeq2SeqLM  # available for encoder-decoder models (e.g., T5)
except Exception:  # older versions may not have this symbol
    TFAutoModelForSeq2SeqLM = None  # type: ignore
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


MODEL_NAME = os.environ.get('MODEL_NAME', 'sshleifer/tiny-gpt2')  # override via env; default tiny causal LM
REQUIRE_GPU = os.environ.get("REQUIRE_GPU", "1") == "1"
ALLOW_CPU_FALLBACK = os.environ.get("ALLOW_CPU_FALLBACK", "0") == "1"
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "4"))
GEN_DEFAULTS = {
    "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "40")),
    "temperature": float(os.environ.get("TEMPERATURE", "0.8")),
    "top_p": float(os.environ.get("TOP_P", "0.95")),
    "do_sample": True,
}

gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
    msg = "No GPU detected for TensorFlow runtime"
    if REQUIRE_GPU and not ALLOW_CPU_FALLBACK:
        log("startup_no_gpu", level="ERROR", require_gpu=REQUIRE_GPU)
        raise SystemExit(msg)
    else:
        log("startup_gpu_absent_fallback", level="WARN", fallback=ALLOW_CPU_FALLBACK)
else:
    for g in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    log("startup_gpu_list", gpus=[d.name for d in gpu_devices])

log("model_load_start", model=MODEL_NAME)
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
config = AutoConfig.from_pretrained(MODEL_NAME)
is_enc_dec = getattr(config, 'is_encoder_decoder', False)
model = None
pad_id = None
try:
    if is_enc_dec:
        if TFAutoModelForSeq2SeqLM is None:
            raise ValueError("Encoder-decoder model requested but TFAutoModelForSeq2SeqLM not available in installed transformers.")
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    else:
        model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # Some GPT2 variants have no explicit pad token; fall back to eos
        pad_id = tokenizer.eos_token_id
except ValueError as e:
    # Automatic fallback: if causal loader failed due to config mismatch, retry seq2seq
    if 'Unrecognized configuration class' in str(e) and TFAutoModelForSeq2SeqLM is not None:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    else:
        raise

if pad_id is None:
    pad_id = tokenizer.eos_token_id

log("model_load_complete", model=MODEL_NAME, pad_id=pad_id, encoder_decoder=is_enc_dec)

history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY * 2))
_gen_lock = threading.Lock()

if PROM_ENABLED:
    REQ_COUNTER = Counter("tiny_gpt2_requests_total", "Chat requests", ["endpoint"])
    ERR_COUNTER = Counter("tiny_gpt2_errors_total", "Errors", ["endpoint", "type"])
    LATENCY = Histogram("tiny_gpt2_latency_seconds", "Latency", buckets=(0.05,0.1,0.2,0.5,1,2,5))
    ACTIVE = Gauge("tiny_gpt2_active_sessions", "Active sessions")
    start_http_server(int(os.environ.get("METRICS_PORT", "9100")))
    log("metrics_server_started", port=int(os.environ.get("METRICS_PORT", "9100")))

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
    parts = []
    for role, content in history[session_id]:
        parts.append(f"{role.title()}: {content}")
    parts.append(f"User: {new_msg}")
    parts.append("Assistant:")
    return "\n".join(parts)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
        t0 = time.time()
        if PROM_ENABLED:
            REQ_COUNTER.labels(endpoint="chat").inc()
        session_id = req.session_id or hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:12]
        try:
            history[session_id].append(("user", req.message))
            prompt = build_prompt(session_id, req.message)
            gen_cfg = GEN_DEFAULTS.copy()
            if req.max_new_tokens is not None:
                gen_cfg["max_new_tokens"] = req.max_new_tokens
            if req.temperature is not None:
                gen_cfg["temperature"] = req.temperature
            if req.top_p is not None:
                gen_cfg["top_p"] = req.top_p
            input_ids = tokenizer(prompt, return_tensors='tf').input_ids
            with _gen_lock:
                # Force GPU placement if available
                if gpu_devices:
                    with tf.device('/GPU:0'):
                        outputs = model.generate(
                            input_ids,
                            max_new_tokens=gen_cfg["max_new_tokens"],
                            temperature=gen_cfg["temperature"],
                            top_p=gen_cfg["top_p"],
                            do_sample=gen_cfg["do_sample"],
                            pad_token_id=pad_id,
                            eos_token_id=pad_id,
                        )
                else:
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
            if "\n\n" in reply:
                reply = reply.split("\n\n")[0].strip()
            history[session_id].append(("assistant", reply))
            latency_ms = int((time.time() - t0) * 1000)
            if PROM_ENABLED:
                LATENCY.observe(latency_ms / 1000.0)
            log("chat_generated", session_id=session_id, latency_ms=latency_ms, new_tokens=gen_cfg["max_new_tokens"])
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
