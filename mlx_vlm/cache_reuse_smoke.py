import argparse
import json
import numbers
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import mlx.core as mx

from .generate import PromptCacheState, generate, normalize_resize_shape
from .prompt_utils import apply_chat_template
from .utils import load, prepare_inputs


@dataclass(frozen=True)
class PromptReusePlan:
    prompt_tokens: int
    reused_tokens: int
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool


@dataclass(frozen=True)
class CacheReuseSmokeReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    prompt_tokens: int
    reused_tokens: int
    boundary_count: int
    main_cache_bytes: int
    boundary_cache_bytes: int
    total_cache_bytes: int
    peak_memory_bytes: Optional[int]
    image_token_id: Optional[int]
    image_token_in_prompt: bool
    image_token_in_suffix: bool
    first_wall_time_s: float
    second_wall_time_s: float
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool
    image_state_contract: str
    first_text: str
    second_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CacheReuseParityReport:
    scenario: str
    model: str
    image: Optional[List[str]]
    top_k: int
    prompt_tokens: int
    reused_tokens: int
    boundary_count: int
    main_cache_bytes: int
    boundary_cache_bytes: int
    total_cache_bytes: int
    peak_memory_bytes: Optional[int]
    image_token_id: Optional[int]
    image_token_in_prompt: bool
    image_token_in_suffix: bool
    used_boundary_restore: bool
    unsafe_rewind_refused: bool
    generated_tail_removed: bool
    recoverable: bool
    cold_token: Optional[int]
    reused_token: Optional[int]
    tokens_match: bool
    parity: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cache_nbytes(value: Any) -> int:
    if value is None:
        return 0
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, numbers.Integral):
        return nbytes
    if isinstance(value, dict):
        return sum(cache_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(cache_nbytes(v) for v in value)
    return 0


def prompt_cache_state_metrics(state: PromptCacheState) -> Dict[str, int]:
    main_cache_bytes = cache_nbytes(state.cache)
    boundary_cache_bytes = cache_nbytes(state.boundary_cache)
    boundary_count = 1 if state.boundary_cache is not None else 0
    return {
        "boundary_count": boundary_count,
        "main_cache_bytes": main_cache_bytes,
        "boundary_cache_bytes": boundary_cache_bytes,
        "total_cache_bytes": main_cache_bytes + boundary_cache_bytes,
    }


def _logprob_values(logprobs: Any) -> List[float]:
    if logprobs is None:
        raise ValueError("Expected logprobs, got None.")
    if hasattr(logprobs, "flatten"):
        logprobs = logprobs.flatten().tolist()
    return [float(v) for v in logprobs]


def topk_logprob_snapshot(logprobs: Any, k: int) -> List[Dict[str, Union[int, float]]]:
    values = _logprob_values(logprobs)
    ranked = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    return [
        {"token_id": int(token_id), "logprob": float(logprob)}
        for token_id, logprob in ranked[:k]
    ]


def compare_topk_logprobs(cold_logprobs: Any, reused_logprobs: Any, k: int) -> Dict[str, Any]:
    cold_values = _logprob_values(cold_logprobs)
    reused_values = _logprob_values(reused_logprobs)
    if len(cold_values) != len(reused_values):
        raise ValueError(
            f"Logprob lengths differ: cold={len(cold_values)} reused={len(reused_values)}"
        )

    cold_topk = topk_logprob_snapshot(cold_values, k)
    reused_topk = topk_logprob_snapshot(reused_values, k)
    cold_ids = [entry["token_id"] for entry in cold_topk]
    reused_ids = [entry["token_id"] for entry in reused_topk]
    cold_argmax = cold_ids[0] if cold_ids else None
    reused_argmax = reused_ids[0] if reused_ids else None
    argmax_delta = (
        abs(float(cold_values[cold_argmax] - reused_values[cold_argmax]))
        if cold_argmax is not None and cold_argmax == reused_argmax
        else None
    )
    shared_ids = sorted(set(cold_ids) & set(reused_ids))
    shared_deltas = [
        {
            "token_id": int(token_id),
            "cold_logprob": float(cold_values[token_id]),
            "reused_logprob": float(reused_values[token_id]),
            "abs_delta": abs(float(cold_values[token_id] - reused_values[token_id])),
        }
        for token_id in shared_ids
    ]
    max_delta = max((entry["abs_delta"] for entry in shared_deltas), default=None)
    return {
        "cold_topk": cold_topk,
        "reused_topk": reused_topk,
        "cold_argmax_token_id": cold_argmax,
        "reused_argmax_token_id": reused_argmax,
        "argmax_token_id_match": cold_argmax == reused_argmax,
        "argmax_abs_logprob_delta": argmax_delta,
        "cold_topk_token_ids": cold_ids,
        "reused_topk_token_ids": reused_ids,
        "topk_token_ids_match": cold_ids == reused_ids,
        "shared_topk_token_count": len(shared_ids),
        "shared_topk_deltas": shared_deltas,
        "max_abs_logprob_delta": max_delta,
    }


def inspect_prompt_reuse(
    state: PromptCacheState, new_token_ids: Sequence[int]
) -> PromptReusePlan:
    new_token_ids = list(new_token_ids)
    prefix_len = state.find_prefix_length(new_token_ids)
    recovered = state.recover_prefix_cache(prefix_len)
    recoverable = recovered is not None
    boundary_match = (
        state.boundary_cache is not None
        and state.boundary_token_ids is not None
        and prefix_len == len(state.boundary_token_ids)
        and state.token_ids is not None
        and state.boundary_token_ids == state.token_ids[:prefix_len]
    )
    unsafe_rewind_refused = False
    if boundary_match and state.cache is not None and state.token_ids is not None:
        no_boundary = PromptCacheState()
        no_boundary.update(state.token_ids, state.cache)
        unsafe_rewind_refused = no_boundary.recover_prefix_cache(prefix_len) is None

    generated_tail_removed = (
        recoverable
        and boundary_match
        and state.token_ids is not None
        and prefix_len < len(state.token_ids)
    )
    return PromptReusePlan(
        prompt_tokens=len(new_token_ids),
        reused_tokens=prefix_len,
        used_boundary_restore=recoverable and boundary_match,
        unsafe_rewind_refused=unsafe_rewind_refused,
        generated_tail_removed=generated_tail_removed,
        recoverable=recoverable,
    )


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _config_dict(model: Any) -> Dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        return {}
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(getattr(config, "__dict__", {}))


def _add_special_tokens(model: Any, processor: Any) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in ["gemma3", "gemma3n", "gemma4"]:
        return getattr(processor, "chat_template", None) is None
    return True


def prepare_prompt_inputs(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: Optional[List[str]],
    resize_shape: Optional[Union[int, Sequence[int]]],
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return prepare_inputs(
        processor,
        images=image,
        prompts=prompt,
        image_token_index=getattr(
            getattr(model, "config", None), "image_token_index", None
        ),
        resize_shape=normalize_resize_shape(resize_shape),
        add_special_tokens=_add_special_tokens(model, processor),
        **(processor_kwargs or {}),
    )


def append_suffix_tokens_to_inputs(
    inputs: Dict[str, Any], suffix_tokens: Sequence[int]
) -> Dict[str, Any]:
    suffix_tokens = list(suffix_tokens)
    if not suffix_tokens:
        return dict(inputs)

    extended = dict(inputs)
    input_ids = inputs["input_ids"]
    suffix = mx.array([suffix_tokens], dtype=input_ids.dtype)
    extended["input_ids"] = mx.concatenate([input_ids, suffix], axis=1)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        suffix_mask = mx.ones(
            (attention_mask.shape[0], len(suffix_tokens)),
            dtype=attention_mask.dtype,
        )
        extended["attention_mask"] = mx.concatenate(
            [attention_mask, suffix_mask], axis=1
        )
    return extended


def append_suffix_tokens_to_cached_turn_inputs(
    inputs: Dict[str, Any],
    *,
    cached_token_ids: Sequence[int],
    suffix_tokens: Sequence[int],
) -> Dict[str, Any]:
    token_ids = list(cached_token_ids) + list(suffix_tokens)
    extended = dict(inputs)
    extended["input_ids"] = mx.array([token_ids], dtype=inputs["input_ids"].dtype)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        extended["attention_mask"] = mx.ones(
            (attention_mask.shape[0], len(token_ids)),
            dtype=attention_mask.dtype,
        )
    return extended


def generation_kwargs_from_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }
    kwargs["input_ids"] = inputs["input_ids"]
    kwargs["pixel_values"] = inputs.get("pixel_values", None)
    kwargs["mask"] = inputs.get("attention_mask", None)
    return kwargs


def _image_state_contract(model: Any, prompt: str, plan: PromptReusePlan) -> str:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    image_token_id = image_token_id or getattr(
        getattr(model, "config", None), "image_token_index", None
    )
    if image_token_id is None:
        return "no_image_token_id"
    if not plan.recoverable:
        return "not_recovered"
    return "reprimed_or_not_in_suffix"


def _image_token_id(model: Any) -> Optional[int]:
    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
    return image_token_id or getattr(
        getattr(model, "config", None), "image_token_index", None
    )


def run_image_prefix_smoke(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    max_tokens: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseSmokeReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    first_kwargs = generation_kwargs_from_inputs(first_inputs)
    second_kwargs = generation_kwargs_from_inputs(second_inputs)

    second_prompt_label = f"{first_prompt}{second_suffix}"

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    first_start = time.perf_counter()
    first_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **first_kwargs,
    )
    first_wall_time = time.perf_counter() - first_start

    plan = inspect_prompt_reuse(state, second_ids)

    second_start = time.perf_counter()
    second_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **second_kwargs,
    )
    second_wall_time = time.perf_counter() - second_start

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    return CacheReuseSmokeReport(
        scenario="image_prefix_diverged_suffix",
        model=model_path,
        image=images,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        first_wall_time_s=first_wall_time,
        second_wall_time_s=second_wall_time,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        image_state_contract=_image_state_contract(model, second_prompt_label, plan),
        first_text=first_result.text,
        second_text=second_result.text,
    )


def run_image_multiturn_smoke(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    max_tokens: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseSmokeReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    first_kwargs = generation_kwargs_from_inputs(first_inputs)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    first_start = time.perf_counter()
    first_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **first_kwargs,
    )
    first_wall_time = time.perf_counter() - first_start

    if state.token_ids is None:
        raise RuntimeError("Prompt cache state did not capture the first turn.")

    second_inputs = append_suffix_tokens_to_cached_turn_inputs(
        first_inputs,
        cached_token_ids=state.token_ids,
        suffix_tokens=suffix_tokens,
    )
    second_ids = second_inputs["input_ids"].flatten().tolist()
    second_kwargs = generation_kwargs_from_inputs(second_inputs)
    plan = inspect_prompt_reuse(state, second_ids)

    second_start = time.perf_counter()
    second_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=max_tokens,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **second_kwargs,
    )
    second_wall_time = time.perf_counter() - second_start

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    return CacheReuseSmokeReport(
        scenario="image_multiturn_text_followup",
        model=model_path,
        image=images,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        first_wall_time_s=first_wall_time,
        second_wall_time_s=second_wall_time,
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        image_state_contract=_image_state_contract(model, second_suffix, plan),
        first_text=first_result.text,
        second_text=second_result.text,
    )


def _token_value(token: Any) -> Optional[int]:
    if token is None:
        return None
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def run_image_prefix_parity(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseParityReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    second_inputs = append_suffix_tokens_to_inputs(first_inputs, suffix_tokens)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    second_ids = second_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(first_inputs),
    )

    plan = inspect_prompt_reuse(state, second_ids)
    cold_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        **generation_kwargs_from_inputs(second_inputs),
    )
    reused_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(second_inputs),
    )

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario="image_prefix_diverged_suffix_parity",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def run_image_multiturn_parity(
    *,
    model_path: str,
    image: Union[str, Sequence[str]],
    prefix_prompt: str,
    second_suffix: str,
    top_k: int,
    resize_shape: Optional[Union[int, Sequence[int]]] = None,
    trust_remote_code: bool = False,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> CacheReuseParityReport:
    images = _as_list(image)
    model, processor = load(model_path, trust_remote_code=trust_remote_code)
    config = _config_dict(model)
    num_images = len(images or [])
    first_prompt = apply_chat_template(
        processor, config, prefix_prompt, num_images=num_images
    )
    first_inputs = prepare_prompt_inputs(
        model,
        processor,
        first_prompt,
        image=images,
        resize_shape=resize_shape,
        processor_kwargs=processor_kwargs,
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    suffix_tokens = tokenizer.encode(second_suffix, add_special_tokens=False)
    first_ids = first_inputs["input_ids"].flatten().tolist()
    image_token_id = _image_token_id(model)

    state = PromptCacheState()
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(first_inputs),
    )
    if state.token_ids is None:
        raise RuntimeError("Prompt cache state did not capture the first turn.")

    second_inputs = append_suffix_tokens_to_cached_turn_inputs(
        first_inputs,
        cached_token_ids=state.token_ids,
        suffix_tokens=suffix_tokens,
    )
    second_ids = second_inputs["input_ids"].flatten().tolist()
    plan = inspect_prompt_reuse(state, second_ids)
    cold_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        **generation_kwargs_from_inputs(second_inputs),
    )
    reused_result = generate(
        model,
        processor,
        "",
        image=None,
        max_tokens=1,
        resize_shape=resize_shape,
        prompt_cache_state=state,
        **generation_kwargs_from_inputs(second_inputs),
    )

    metrics = prompt_cache_state_metrics(state)
    peak_memory_bytes = (
        int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else None
    )
    cold_token = _token_value(cold_result.token)
    reused_token = _token_value(reused_result.token)
    return CacheReuseParityReport(
        scenario="image_multiturn_text_followup_parity",
        model=model_path,
        image=images,
        top_k=top_k,
        prompt_tokens=plan.prompt_tokens,
        reused_tokens=plan.reused_tokens,
        boundary_count=metrics["boundary_count"],
        main_cache_bytes=metrics["main_cache_bytes"],
        boundary_cache_bytes=metrics["boundary_cache_bytes"],
        total_cache_bytes=metrics["total_cache_bytes"],
        peak_memory_bytes=peak_memory_bytes,
        image_token_id=image_token_id,
        image_token_in_prompt=(
            image_token_id is not None and image_token_id in first_ids
        ),
        image_token_in_suffix=(
            image_token_id is not None and image_token_id in suffix_tokens
        ),
        used_boundary_restore=plan.used_boundary_restore,
        unsafe_rewind_refused=plan.unsafe_rewind_refused,
        generated_tail_removed=plan.generated_tail_removed,
        recoverable=plan.recoverable,
        cold_token=cold_token,
        reused_token=reused_token,
        tokens_match=cold_token == reused_token,
        parity=compare_topk_logprobs(
            cold_result.logprobs, reused_result.logprobs, top_k
        ),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a structured VLM prompt-cache reuse smoke."
    )
    parser.add_argument("--model", required=True, help="Local path or HF repo id.")
    parser.add_argument(
        "--scenario",
        choices=[
            "image-prefix",
            "image-multiturn",
            "image-prefix-parity",
            "image-multiturn-parity",
        ],
        default="image-prefix",
    )
    parser.add_argument(
        "--image", nargs="+", required=True, help="Image path(s) or URL(s)."
    )
    parser.add_argument(
        "--prefix-prompt",
        default="Describe this image briefly.",
        help="Prompt shared by both turns.",
    )
    parser.add_argument(
        "--second-suffix",
        default="\nNow answer with one short additional detail.",
        help="Suffix appended to the second prompt after the shared prefix.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--resize-shape", type=int, nargs="+", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--processor-kwargs", type=json.loads, default={})
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scenario_runners = {
        "image-prefix": run_image_prefix_smoke,
        "image-multiturn": run_image_multiturn_smoke,
        "image-prefix-parity": run_image_prefix_parity,
        "image-multiturn-parity": run_image_multiturn_parity,
    }
    run_smoke = scenario_runners[args.scenario]
    common_kwargs = dict(
        model_path=args.model,
        image=args.image,
        prefix_prompt=args.prefix_prompt,
        second_suffix=args.second_suffix,
        resize_shape=args.resize_shape,
        trust_remote_code=args.trust_remote_code,
        processor_kwargs=args.processor_kwargs,
    )
    if args.scenario.endswith("-parity"):
        report = run_smoke(top_k=args.top_k, **common_kwargs)
    else:
        report = run_smoke(max_tokens=args.max_tokens, **common_kwargs)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
