"""inference.py — Load cached Qwen3-4B assets and expose predict functions.

This module is imported by server.py and test.py. The model is loaded
lazily on first call and cached for subsequent calls.
"""

import os
from typing import Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_CACHE_DIR = os.path.join(MODELS_DIR, "qwen3-4b")
MODEL_READY_PATH = os.path.join(MODELS_DIR, "model.ready")
THINK_END_TOKEN_ID = 151668

_model = None
_tokenizer = None


def _runtime_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _normalize_prompt(features: Any) -> str:
    if isinstance(features, str):
        prompt = features
    elif isinstance(features, list):
        prompt = " ".join(str(part) for part in features)
    else:
        raise ValueError("Features must be either a string or a list of values.")

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    return prompt


def load_model() -> Tuple[Any, Any]:
    """Load and cache tokenizer/model from disk. Raises FileNotFoundError if missing."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        if not (os.path.exists(MODEL_READY_PATH) and os.path.isdir(MODEL_CACHE_DIR)):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_CACHE_DIR}'. Run init.py first."
            )

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_CACHE_DIR,
            torch_dtype=_runtime_dtype(),
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if not torch.cuda.is_available():
            _model.to("cpu")

    return _tokenizer, _model


def predict(features: list) -> str:
    """Run text generation for a single prompt payload."""
    prompt = _normalize_prompt(features)
    tokenizer, model = load_model()

    messages = [{"role": "user", "content": prompt}]
    enable_thinking = os.environ.get("ENABLE_THINKING", "true").lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    model_inputs = tokenizer([text], return_tensors="pt", truncation=True)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        model_inputs = {k: v.to(target_device) for k, v in model_inputs.items()}
    else:
        fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inputs = {k: v.to(fallback_device) for k, v in model_inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "96")),
            do_sample=True,
            temperature=float(os.environ.get("TEMPERATURE", "0.7")),
            top_p=float(os.environ.get("TOP_P", "0.9")),
        )

    output_ids = output[0][len(model_inputs["input_ids"][0]):].tolist()

    try:
        think_end_idx = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        think_end_idx = 0

    content = tokenizer.decode(output_ids[think_end_idx:], skip_special_tokens=True).strip()
    if not content:
        content = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return content


def predict_batch(samples: list) -> list:
    """Run text generation for a batch of prompt payloads."""
    if not isinstance(samples, list) or not samples:
        raise ValueError("Batch features must be a non-empty list.")
    return [predict(sample) for sample in samples]
