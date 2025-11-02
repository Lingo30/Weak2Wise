"""
llm_client.py - Multi-backend LLM client

Supports:
  - OpenAI-compatible remote endpoints (type: "openai")
    -> Uses OpenAI-compatible client (from openai import OpenAI) or openai.ChatCompletion
  - Transformers local models (type: "transformers")
    -> Uses transformers.AutoTokenizer and AutoModelForCausalLM

Public API:
  - call_llm_chat(model_key, messages, temperature, max_tokens) -> dict {content, reasoning_content?, raw}
  - call_llm_text(model_key, prompt, temperature, max_tokens) -> str
"""

import os
import yaml
import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# load configuration
_config = None
def load_config(path: str = "config.yaml"):
    global _config
    if _config is None:
        with open(path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config

# try import openai & OpenAI class
try:
    import openai
    try:
        from openai import OpenAI as OpenAIClient
    except Exception:
        OpenAIClient = None
    _OPENAI_PRESENT = True
except Exception:
    openai = None
    OpenAIClient = None
    _OPENAI_PRESENT = False

# try import transformers for local models
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch
    _TRANSFORMERS_PRESENT = True
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    GenerationConfig = None
    torch = None
    _TRANSFORMERS_PRESENT = False

# cache for loaded transformers models to avoid reload
_TRANSFORMERS_MODELS: Dict[str, Dict[str, Any]] = {}

def _get_endpoint_cfg(endpoint_key: str) -> Optional[Dict[str, Any]]:
    cfg = load_config()
    endpoints = cfg.get("model_endpoints", {}) or {}
    return endpoints.get(endpoint_key)

def _create_openai_client_for(endpoint_cfg: Dict[str,str]):
    api_key_env = endpoint_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else os.getenv("OPENAI_API_KEY")
    base_url = endpoint_cfg.get("base_url")
    if not api_key:
        raise RuntimeError(f"API key for endpoint not found. Expected env var: {api_key_env} or OPENAI_API_KEY.")
    if OpenAIClient is not None:
        return OpenAIClient(api_key=api_key, base_url=base_url)
    if _OPENAI_PRESENT:
        # set openai module keys as fallback
        openai.api_key = api_key
        openai.api_base = base_url
        return openai
    raise RuntimeError("OpenAI client not available; install openai package.")

def _load_transformers_model(endpoint_key: str, endpoint_cfg: Dict[str, Any]):
    """
    Lazy-load and cache transformers tokenizer and model.
    endpoint_cfg should include:
      - model_name (str)
      - device (str): "cuda" or "cpu"
      - tokenizer_kwargs, generation_kwargs (optional)
    """
    if endpoint_key in _TRANSFORMERS_MODELS:
        return _TRANSFORMERS_MODELS[endpoint_key]

    if not _TRANSFORMERS_PRESENT:
        raise RuntimeError("transformers not installed. Please install transformers and accelerate.")

    model_name = endpoint_cfg.get("model_name")
    device = endpoint_cfg.get("device", "cpu")
    tok_kwargs = endpoint_cfg.get("tokenizer_kwargs", {}) or {}
    gen_kwargs_cfg = endpoint_cfg.get("generation_kwargs", {}) or {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model.to(device)

    entry = {"tokenizer": tokenizer, "model": model, "device": device, "gen_kwargs": gen_kwargs_cfg}
    _TRANSFORMERS_MODELS[endpoint_key] = entry
    return entry

# -------------------------
# Public API
# -------------------------

def call_llm_chat(model: str, messages: List[Dict[str,str]], temperature: float = 0.6, max_tokens: int = 8192) -> Dict[str, Any]:
    """
    Chat-style call. Returns:
      {"content": str, "reasoning_content": Optional[str], "raw": raw_response}
    model can be:
      - a key in config.model_endpoints (preferred)
      - or a direct model id used by global openai fallback
    """
    cfg = load_config()
    endpoint_cfg = _get_endpoint_cfg(model)

    if endpoint_cfg:
        typ = endpoint_cfg.get("type", "openai")
        if typ == "openai":
            client = _create_openai_client_for(endpoint_cfg)
            model_id = endpoint_cfg.get("model_id")
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            # robust extraction of content and reasoning_content
            try:
                choice0 = resp["choices"][0] if isinstance(resp, dict) and "choices" in resp and resp["choices"] else resp.choices[0]
                msg = choice0.get("message", {}) if isinstance(choice0, dict) else getattr(choice0, "message", None)
                if isinstance(msg, dict):
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning_content")
                else:
                    content = getattr(msg, "content", "") or ""
                    reasoning = getattr(msg, "reasoning_content", None)
            except Exception:
                content = ""
                reasoning = None
            return {"content": content, "reasoning_content": reasoning, "raw": resp}
        elif typ == "transformers":
            entry = _load_transformers_model(model, endpoint_cfg)
            tokenizer = entry["tokenizer"]
            lm = entry["model"]
            device = entry["device"]
            gen_kwargs_conf = entry.get("gen_kwargs", {})

            prompt = tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            if device.startswith("cuda"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = dict(gen_kwargs_conf)
            gen_kwargs.setdefault("max_new_tokens", 8192)
            if temperature is not None:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                out_ids = lm.generate(**inputs, **gen_kwargs)
            generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"content": generated, "reasoning_content": generated, "raw": out_ids}
        else:
            raise NotImplementedError(f"Unknown endpoint type: {typ}")
    else:
        # fallback to global openai.ChatCompletion if available
        if not _OPENAI_PRESENT:
            raise NotImplementedError("Model endpoint not configured and openai package not available.")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        content = resp["choices"][0]["message"]["content"]
        return {"content": content, "reasoning_content": None, "raw": resp}

def call_llm_text(model: str, prompt: str, temperature: float = 0.6, max_tokens: int = 8192) -> str:
    """
    Text-style call that returns a string.
    """
    cfg = load_config()
    endpoint_cfg = _get_endpoint_cfg(model)
    if endpoint_cfg:
        typ = endpoint_cfg.get("type", "openai")
        if typ == "openai":
            client = _create_openai_client_for(endpoint_cfg)
            model_id = endpoint_cfg.get("model_id")
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            try:
                choice0 = resp["choices"][0] if isinstance(resp, dict) and "choices" in resp and resp["choices"] else resp.choices[0]
                msg = choice0.get("message", {}) if isinstance(choice0, dict) else getattr(choice0, "message", None)
                if isinstance(msg, dict):
                    return msg.get("content", "")
                else:
                    return getattr(msg, "content", "") or ""
            except Exception:
                return str(resp)
        elif typ == "transformers":
            entry = _load_transformers_model(model, endpoint_cfg)
            tokenizer = entry["tokenizer"]
            lm = entry["model"]
            device = entry["device"]
            gen_kwargs_conf = entry.get("gen_kwargs", {})

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            if device.startswith("cuda"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = dict(gen_kwargs_conf)
            gen_kwargs.setdefault("max_new_tokens", 8192)
            if temperature is not None:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                out_ids = lm.generate(**inputs, **gen_kwargs)
            generated = tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated
        else:
            raise NotImplementedError(f"Unknown endpoint type: {typ}")
    else:
        if not _OPENAI_PRESENT:
            raise NotImplementedError("call_llm_text fallback requires openai package.")
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return resp["choices"][0]["message"]["content"]
